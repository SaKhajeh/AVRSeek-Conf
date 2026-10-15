from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import get_settings
from app.schemas.reaction_route import (
    ExtractionResult,
    PipelineStage,
    PipelineStageStatus,
    SynthesisRoute,
)
from app.services.export import excel_bytes, write_batch_json, write_excel, write_result_json
from app.services.ocr import available_ocr_backends, available_ocr_languages
from app.services.pipeline import run_pipeline
from app.services.result_cache import (
    bootstrap_cache_index,
    is_pdf_cached,
    load_cached_result,
    register_cached_result,
)
from app.services.time_estimate import FileEstimate, estimate_batch, estimate_remaining_in_batch

settings = get_settings()
logger = logging.getLogger("avrseek")

app = FastAPI(
    title="AVRSeek",
    description="Antiviral Reaction Seek — literature/patent reaction knowledge extraction",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_cache_bootstrap() -> None:
    added = bootstrap_cache_index(settings.output_dir)
    if added:
        logger.info("Rebuilt result cache index with %s entries", added)


async def _save_uploads(files: list[UploadFile]) -> list[tuple[Path, str]]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")
    saved: list[tuple[Path, str]] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {file.filename or 'unnamed'}")
        original_name = Path(file.filename).name
        suffix = Path(file.filename).suffix or ".pdf"
        dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        try:
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            saved.append((dest, original_name))
        finally:
            await file.close()
    return saved


def _batch_dir(batch_id: str) -> Path:
    return settings.output_dir / "batches" / batch_id


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": "AVRSeek",
        "base_url": settings.openai_base_url,
        "model_text": settings.model_text,
        "model_vlm": settings.model_vlm,
        "model_standardize": settings.model_standardize,
        "api_key_configured": bool(settings.openai_api_key) and not settings.openai_api_key.startswith("YOUR_"),
        "port": settings.avrseek_port,
        "enable_ocr": settings.enable_ocr,
        "enable_vlm_page_read": settings.enable_vlm_page_read,
        "ocr_languages": settings.ocr_languages,
        "ocr_languages_installed": available_ocr_languages(),
        "ocr_backends": available_ocr_backends(),
        "features": [
            "single_extract",
            "batch_extract",
            "json_export",
            "excel_export",
            "time_estimate",
            "result_cache" if settings.enable_result_cache else "result_cache_disabled",
            "ocr",
            "vlm_page_read",
        ],
        "enable_result_cache": settings.enable_result_cache,
    }


@app.get("/api/config")
def public_config():
    return {
        "models": {
            "text": settings.model_text,
            "vlm": settings.model_vlm,
            "standardize": settings.model_standardize,
        },
        "base_url": settings.openai_base_url,
        "port": settings.avrseek_port,
        "max_batch_files": 20,
        "pipeline": [
            "parse_pdf",
            "text_extract",
            "table_extract",
            "image_scheme",
            "reconcile",
            "standardize",
        ],
    }


@app.post("/api/extract/batch/estimate")
async def batch_estimate(files: list[UploadFile] = File(...)):
    """Quick PDF scan (no LLM) → estimated processing time for a batch."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 PDFs per batch.")
    paths = await _save_uploads(files)
    try:
        est = await asyncio.to_thread(
            estimate_batch,
            [p for p, _ in paths],
            [name for _, name in paths],
        )
        per_file_out = []
        cached_count = 0
        adjusted_seconds = 0.0
        for file_est, (pdf_path, _) in zip(est.per_file, paths, strict=True):
            cached = False
            if settings.enable_result_cache:
                cached = await asyncio.to_thread(is_pdf_cached, pdf_path, settings.output_dir)
            if cached:
                cached_count += 1
            else:
                adjusted_seconds += file_est.estimated_seconds
            per_file_out.append(
                {
                    "filename": file_est.filename,
                    "page_count": file_est.page_count,
                    "figure_count": file_est.figure_count,
                    "table_hints": file_est.table_hints,
                    "estimated_seconds": 0 if cached else file_est.estimated_seconds,
                    "estimated_display": "cached" if cached else _format_duration(file_est.estimated_seconds),
                    "cached": cached,
                }
            )
        note = est.note
        if cached_count:
            note = (
                f"{cached_count} file(s) already processed — will load from cache. "
                + note
            )
        elif not settings.enable_result_cache:
            note = "Result cache is off — every upload is fully reprocessed. " + note
        return {
            "file_count": est.file_count,
            "total_pages": est.total_pages,
            "total_figures": est.total_figures,
            "total_tables": est.total_tables,
            "estimated_seconds": round(adjusted_seconds, 1),
            "estimated_minutes": round(adjusted_seconds / 60.0, 1),
            "estimated_display": _format_duration(adjusted_seconds),
            "cached_count": cached_count,
            "note": note,
            "per_file": per_file_out,
        }
    finally:
        for p, _ in paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _failed_result(filename: str, detail: str) -> ExtractionResult:
    job_id = uuid.uuid4().hex[:12]
    return ExtractionResult(
        job_id=job_id,
        filename=filename,
        stages=[
            PipelineStage(
                id="pipeline",
                label="Pipeline",
                status=PipelineStageStatus.error,
                detail=detail,
            )
        ],
        route=SynthesisRoute(title=filename, warnings=[detail]),
        error=detail,
    )


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"~{s}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"~{m}m {rem}s" if rem else f"~{m}m"
    h, m = divmod(m, 60)
    return f"~{h}h {m}m"


@app.post("/api/extract", response_model=ExtractionResult)
async def extract(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
):
    paths = await _save_uploads([file])
    pdf_path, display_name = paths[0]
    try:
        if settings.enable_result_cache:
            cached = await asyncio.to_thread(
                load_cached_result,
                pdf_path,
                settings.output_dir,
                display_filename=display_name,
            )
            if cached:
                write_result_json(cached, settings.output_dir / f"{cached.job_id}.json")
                return cached

        result = await run_pipeline(
            pdf_path,
            settings,
            model_override=model or None,
            display_filename=display_name,
        )
        if settings.enable_result_cache:
            await asyncio.to_thread(register_cached_result, pdf_path, result, settings.output_dir)
        else:
            write_result_json(result, settings.output_dir / f"{result.job_id}.json")
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/extract/batch/stream")
async def extract_batch_stream(
    files: list[UploadFile] = File(...),
    model: str | None = Form(default=None),
):
    """
    Process multiple PDFs sequentially. SSE events:
      estimate, file_start, stage_update, file_complete, batch_complete, error
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 PDFs per batch.")
    uploads = await _save_uploads(files)
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = _batch_dir(batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def run_batch() -> None:
        results: list[ExtractionResult] = []
        file_index_ctx = {"index": 0}
        try:
            est = await asyncio.to_thread(
                estimate_batch,
                [p for p, _ in uploads],
                [name for _, name in uploads],
            )
            per_file_estimates: list[FileEstimate] = []
            cached_flags: list[bool] = []
            adjusted_total = 0.0
            for file_est, (pdf_path, _) in zip(est.per_file, uploads, strict=True):
                cached = False
                if settings.enable_result_cache:
                    cached = await asyncio.to_thread(is_pdf_cached, pdf_path, settings.output_dir)
                cached_flags.append(cached)
                secs = 0.0 if cached else file_est.estimated_seconds
                adjusted_total += secs
                per_file_estimates.append(
                    FileEstimate(
                        filename=file_est.filename,
                        page_count=file_est.page_count,
                        figure_count=file_est.figure_count,
                        table_hints=file_est.table_hints,
                        estimated_seconds=secs,
                    )
                )

            await queue.put(
                {
                    "type": "estimate",
                    "batch_id": batch_id,
                    "estimated_seconds": round(adjusted_total, 1),
                    "estimated_display": _format_duration(adjusted_total),
                    "file_count": est.file_count,
                    "cached_count": sum(cached_flags),
                    "per_file": [
                        {
                            "filename": f.filename,
                            "estimated_seconds": f.estimated_seconds,
                            "estimated_display": "cached"
                            if cached_flags[i]
                            else _format_duration(f.estimated_seconds),
                            "page_count": f.page_count,
                            "figure_count": f.figure_count,
                            "cached": cached_flags[i],
                        }
                        for i, f in enumerate(per_file_estimates)
                    ],
                }
            )
            await asyncio.sleep(0)
            total = len(uploads)

            async def on_progress(event: dict) -> None:
                remaining = estimate_remaining_in_batch(
                    per_file_estimates,
                    file_index=file_index_ctx["index"],
                    current_stage_id=str(event.get("stage_id", "parse_pdf")),
                    image_index=int(event.get("image_index") or 0),
                    image_total=int(event.get("image_total") or 0),
                    entity_index=int(event.get("entity_index") or 0),
                    entity_total=int(event.get("entity_total") or 0),
                )
                await queue.put(
                    {
                        **event,
                        "batch_id": batch_id,
                        "index": file_index_ctx["index"],
                        "total": total,
                        "remaining_seconds": remaining,
                        "remaining_display": _format_duration(remaining),
                    }
                )

            for idx, (pdf_path, display_name) in enumerate(uploads, start=1):
                file_index_ctx["index"] = idx
                await queue.put(
                    {
                        "type": "file_start",
                        "batch_id": batch_id,
                        "index": idx,
                        "total": total,
                        "filename": display_name,
                        "display_filename": display_name,
                    }
                )
                await asyncio.sleep(0)
                logger.info("Batch %s: starting file %s/%s (%s)", batch_id, idx, total, display_name)
                try:
                    if settings.enable_result_cache:
                        cached_result = await asyncio.to_thread(
                            load_cached_result,
                            pdf_path,
                            settings.output_dir,
                            display_filename=display_name,
                        )
                        if cached_result:
                            write_result_json(
                                cached_result,
                                settings.output_dir / f"{cached_result.job_id}.json",
                            )
                            results.append(cached_result)
                            logger.info(
                                "Batch %s: cache hit file %s/%s source_job=%s",
                                batch_id,
                                idx,
                                total,
                                cached_result.source_job_id,
                            )
                            await queue.put(
                                {
                                    "type": "file_complete",
                                    "batch_id": batch_id,
                                    "index": idx,
                                    "total": total,
                                    "filename": cached_result.filename,
                                    "job_id": cached_result.job_id,
                                    "steps": len(cached_result.route.steps) if cached_result.route else 0,
                                    "error": cached_result.error,
                                    "from_cache": True,
                                    "source_job_id": cached_result.source_job_id,
                                    "result": cached_result.model_dump(mode="json"),
                                }
                            )
                            await asyncio.sleep(0)
                            continue

                    result = await run_pipeline(
                        pdf_path,
                        settings,
                        model_override=model or None,
                        display_filename=display_name,
                        on_progress=on_progress,
                    )
                    if settings.enable_result_cache:
                        await asyncio.to_thread(register_cached_result, pdf_path, result, settings.output_dir)
                    else:
                        write_result_json(result, settings.output_dir / f"{result.job_id}.json")
                    results.append(result)
                    logger.info(
                        "Batch %s: finished file %s/%s job=%s steps=%s",
                        batch_id,
                        idx,
                        total,
                        result.job_id,
                        len(result.route.steps) if result.route else 0,
                    )
                    await queue.put(
                        {
                            "type": "file_complete",
                            "batch_id": batch_id,
                            "index": idx,
                            "total": total,
                            "filename": result.filename,
                            "job_id": result.job_id,
                            "steps": len(result.route.steps) if result.route else 0,
                            "error": result.error,
                            "from_cache": False,
                            "result": result.model_dump(mode="json"),
                        }
                    )
                    await asyncio.sleep(0)
                except Exception as exc:
                    logger.exception("Batch %s: file %s failed", batch_id, display_name)
                    failed = _failed_result(display_name, str(exc))
                    write_result_json(failed, settings.output_dir / f"{failed.job_id}.json")
                    results.append(failed)
                    await queue.put(
                        {
                            "type": "file_error",
                            "batch_id": batch_id,
                            "index": idx,
                            "total": total,
                            "filename": display_name,
                            "detail": str(exc),
                            "job_id": failed.job_id,
                            "result": failed.model_dump(mode="json"),
                        }
                    )
                    await asyncio.sleep(0)
            write_batch_json(results, batch_id, batch_dir / "batch.json")
            write_excel(results, batch_dir / "batch.xlsx")
            await queue.put(
                {
                    "type": "batch_complete",
                    "batch_id": batch_id,
                    "file_count": len(results),
                    "total_steps": sum(len(r.route.steps) if r.route else 0 for r in results),
                    "downloads": {
                        "json": f"/api/batch/{batch_id}/download.json",
                        "excel": f"/api/batch/{batch_id}/download.xlsx",
                    },
                    "results": [r.model_dump(mode="json") for r in results],
                }
            )
        except Exception as exc:
            await queue.put({"type": "error", "detail": str(exc)})
        finally:
            await queue.put(None)

    async def event_gen():
        task = asyncio.create_task(run_batch())
        # Padding helps some proxies/browsers flush SSE immediately.
        yield ": " + " " * 2048 + "\n\n"
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}", response_model=ExtractionResult)
def get_job(job_id: str):
    path = settings.output_dir / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/download.json")
def download_job_json(job_id: str):
    path = settings.output_dir / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path, media_type="application/json", filename=f"avrseek_{job_id}.json")


@app.get("/api/jobs/{job_id}/download.xlsx")
def download_job_excel(job_id: str):
    path = settings.output_dir / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    result = ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))
    data = excel_bytes([result])
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="avrseek_{job_id}.xlsx"'},
    )


@app.get("/api/batch/{batch_id}/download.json")
def download_batch_json(batch_id: str):
    path = _batch_dir(batch_id) / "batch.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Batch not found")
    return FileResponse(path, media_type="application/json", filename=f"avrseek_batch_{batch_id}.json")


@app.get("/api/batch/{batch_id}/download.xlsx")
def download_batch_excel(batch_id: str):
    path = _batch_dir(batch_id) / "batch.xlsx"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Batch Excel not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"avrseek_batch_{batch_id}.xlsx",
    )

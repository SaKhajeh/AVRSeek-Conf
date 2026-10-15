from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.reaction_route import ExtractionResult, PipelineStage, PipelineStageStatus

CACHE_DIR_NAME = "cache"
INDEX_FILENAME = "index.json"


def pdf_content_hash(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_dir(output_dir: Path) -> Path:
    path = output_dir / CACHE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(output_dir: Path) -> Path:
    return _cache_dir(output_dir) / INDEX_FILENAME


def load_cache_index(output_dir: Path) -> dict[str, dict]:
    path = _index_path(output_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache_index(output_dir: Path, index: dict[str, dict]) -> None:
    path = _index_path(output_dir)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def lookup_cached_job_id(pdf_path: Path, output_dir: Path) -> str | None:
    content_hash = pdf_content_hash(pdf_path)
    entry = load_cache_index(output_dir).get(content_hash)
    if not entry:
        return None
    job_id = entry.get("job_id")
    if not job_id:
        return None
    job_path = output_dir / f"{job_id}.json"
    return job_id if job_path.exists() else None


def load_cached_result(
    pdf_path: Path,
    output_dir: Path,
    *,
    display_filename: str | None = None,
) -> ExtractionResult | None:
    content_hash = pdf_content_hash(pdf_path)
    index = load_cache_index(output_dir)
    entry = index.get(content_hash)
    if not entry:
        return None

    job_id = entry.get("job_id")
    if not job_id:
        return None
    job_path = output_dir / f"{job_id}.json"
    if not job_path.exists():
        index.pop(content_hash, None)
        save_cache_index(output_dir, index)
        return None

    result = ExtractionResult.model_validate_json(job_path.read_text(encoding="utf-8"))
    return adapt_cached_result(
        result,
        display_filename=display_filename or result.filename,
        content_hash=content_hash,
        source_job_id=job_id,
    )


def adapt_cached_result(
    result: ExtractionResult,
    *,
    display_filename: str,
    content_hash: str,
    source_job_id: str,
) -> ExtractionResult:
    """Return a copy suitable for a new batch run (preserve extraction, update display metadata)."""
    batch_job_id = uuid.uuid4().hex[:12]
    data = result.model_dump(mode="json")
    data["job_id"] = batch_job_id
    data["filename"] = display_filename
    data["from_cache"] = True
    data["content_hash"] = content_hash
    data["source_job_id"] = source_job_id
    if data.get("document"):
        data["document"]["filename"] = display_filename
    cached = ExtractionResult.model_validate(data)
    cached_stages = [
        PipelineStage(
            id="cache",
            label="Result cache",
            status=PipelineStageStatus.skipped,
            detail=f"Reused prior extraction (job {source_job_id}) — PDF unchanged",
            data={"source_job_id": source_job_id, "content_hash": content_hash},
        ),
        *cached.stages,
    ]
    return cached.model_copy(update={"stages": cached_stages})


def register_cached_result(pdf_path: Path, result: ExtractionResult, output_dir: Path) -> str:
    content_hash = pdf_content_hash(pdf_path)
    index = load_cache_index(output_dir)
    index[content_hash] = {
        "job_id": result.job_id,
        "filename": result.filename,
        "content_hash": content_hash,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "steps": len(result.route.steps) if result.route else 0,
    }
    save_cache_index(output_dir, index)

    enriched = result.model_copy(update={"from_cache": False, "content_hash": content_hash})
    job_path = output_dir / f"{result.job_id}.json"
    job_path.write_text(enriched.model_dump_json(indent=2), encoding="utf-8")
    return content_hash


def bootstrap_cache_index(output_dir: Path) -> int:
    """Rebuild index entries from saved job JSON files (e.g. after index loss)."""
    index = load_cache_index(output_dir)
    added = 0
    for path in output_dir.glob("*.json"):
        if "cache" in path.parts or "batches" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        content_hash = data.get("content_hash")
        job_id = data.get("job_id")
        if not content_hash or not job_id or content_hash in index:
            continue
        index[content_hash] = {
            "job_id": job_id,
            "filename": data.get("filename", ""),
            "content_hash": content_hash,
            "processed_at": data.get("processed_at", ""),
            "steps": len(data.get("route", {}).get("steps", [])),
        }
        added += 1
    if added:
        save_cache_index(output_dir, index)
    return added


def is_pdf_cached(pdf_path: Path, output_dir: Path) -> bool:
    if not load_cache_index(output_dir):
        bootstrap_cache_index(output_dir)
    return lookup_cached_job_id(pdf_path, output_dir) is not None

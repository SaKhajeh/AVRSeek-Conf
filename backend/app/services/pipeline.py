from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas.reaction_route import (
    ExtractionResult,
    PipelineStage,
    PipelineStageStatus,
)
from app.services.llm_client import extract_reactions_from_text
from app.services.evidence_retrieval import build_extraction_plan, plan_extraction_passes
from app.services.pdf_parse import (
    document_coverage_warning,
    document_text_warning,
    parse_pdf,
)
from app.services.reconcile import reconcile_route
from app.services.stubs import run_standardize
from app.services.image_extract import (
    detect_figure_scheme_pages,
    page_read_to_reaction_steps,
    run_image_scheme_extraction,
    schemes_to_reaction_steps,
)
from app.services.table_extract import run_table_extraction
from app.services.time_estimate import STAGE_LABELS, STAGE_ORDER

ProgressFn = Callable[[dict[str, Any]], Awaitable[None]]


async def _emit(on_progress: ProgressFn | None, payload: dict[str, Any]) -> None:
    if on_progress:
        await on_progress(payload)
        await asyncio.sleep(0)


def _stage_checklist(current_stage_id: str, stage_statuses: dict[str, str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    reached_current = False
    for stage_id in STAGE_ORDER:
        if stage_id == current_stage_id:
            reached_current = True
            status = stage_statuses.get(stage_id, "running")
        elif not reached_current:
            status = stage_statuses.get(stage_id, "done")
        else:
            status = stage_statuses.get(stage_id, "pending")
        items.append({"id": stage_id, "label": STAGE_LABELS[stage_id], "status": status})
    return items


async def run_pipeline(
    pdf_path: Path,
    settings: Settings,
    model_override: str | None = None,
    *,
    display_filename: str | None = None,
    on_progress: ProgressFn | None = None,
) -> ExtractionResult:
    job_id = uuid.uuid4().hex[:12]
    out_name = display_filename or pdf_path.name
    stages: list[PipelineStage] = []
    stage_statuses: dict[str, str] = {}
    parse_info: list[str] = []
    parse_warnings: list[str] = []
    extraction_debug: dict = {}
    extraction_plan = None

    async def stage_event(
        stage_id: str,
        status: str,
        detail: str = "",
        *,
        image_index: int = 0,
        image_total: int = 0,
        entity_index: int = 0,
        entity_total: int = 0,
    ) -> None:
        stage_statuses[stage_id] = status
        await _emit(
            on_progress,
            {
                "type": "stage_update",
                "stage_id": stage_id,
                "stage_label": STAGE_LABELS[stage_id],
                "stage_status": status,
                "detail": detail,
                "filename": out_name,
                "stages": _stage_checklist(stage_id, stage_statuses),
                "image_index": image_index,
                "image_total": image_total,
                "entity_index": entity_index,
                "entity_total": entity_total,
            },
        )

    # 1) Document understanding (+ OCR gate)
    stages.append(
        PipelineStage(
            id="parse_pdf",
            label=STAGE_LABELS["parse_pdf"],
            status=PipelineStageStatus.running,
        )
    )
    await stage_event("parse_pdf", "running", "Reading PDF (+ OCR if scanned)…")
    try:
        parsed = await asyncio.to_thread(
            parse_pdf,
            pdf_path,
            output_dir=settings.output_dir,
            enable_ocr=settings.enable_ocr,
        )
        stages[-1].status = PipelineStageStatus.done
        ocr_note = ""
        if parsed.segments and parsed.segments[0].meta.get("ocr_used"):
            ocr_note = f", OCR via {parsed.segments[0].meta.get('ocr_backend')}"
        detail = (
            f"{parsed.page_count} pages, {parsed.figure_count} images, "
            f"{len(parsed.table_hints)} table hints, {len(parsed.text or '')} text chars{ocr_note}"
        )
        stages[-1].detail = detail
        stages[-1].data = {
            "page_count": parsed.page_count,
            "figure_count": parsed.figure_count,
            "segments": len(parsed.segments),
            "text_chars": len(parsed.text or ""),
            "ocr_used": bool(parsed.segments and parsed.segments[0].meta.get("ocr_used")),
            "ocr_backend": (parsed.segments[0].meta.get("ocr_backend") if parsed.segments else ""),
            "document_type": (parsed.segments[0].meta.get("document_type") if parsed.segments else ""),
            "procedure_pages": (parsed.segments[0].meta.get("procedure_pages") if parsed.segments else []),
            "evidence_profile": (parsed.segments[0].meta.get("evidence_profile") if parsed.segments else {}),
        }
        await stage_event("parse_pdf", "done", detail)
        text_warn, parse_info = document_text_warning(parsed)
        if text_warn:
            parse_warnings.append(text_warn)
    except Exception as exc:
        stages[-1].status = PipelineStageStatus.error
        stages[-1].detail = str(exc)
        await stage_event("parse_pdf", "error", str(exc))
        return ExtractionResult(
            job_id=job_id,
            filename=out_name,
            stages=stages,
            error=f"PDF parse failed: {exc}",
        )

    text_chars = len(parsed.text or "")

    # 2) Text reaction extraction
    stages.append(
        PipelineStage(
            id="text_extract",
            label=STAGE_LABELS["text_extract"],
            status=PipelineStageStatus.running,
            detail=f"model={model_override or settings.model_text}",
        )
    )
    await stage_event(
        "text_extract",
        "running",
        f"Calling {model_override or settings.model_text} on document text…",
    )
    route = None
    raw_out = None
    text_route = None
    text_steps: list = []
    table_steps: list = []
    scheme_steps: list = []
    page_read_steps: list = []
    extraction_plan = None
    extraction_passes: list = []
    try:
        extraction_plan = build_extraction_plan(parsed)
        extraction_passes = plan_extraction_passes(parsed)
        extraction_debug = extraction_plan.to_debug_dict()
        extraction_debug["page_selection"] = {
            "candidate_procedure_pages": extraction_plan.candidate_procedure_pages,
            "mandatory_pages": extraction_plan.mandatory_pages,
            "pages_included": extraction_plan.pages_included,
            "pages_excluded": extraction_plan.pages_excluded,
            "exclusion_reasons": getattr(extraction_plan, "exclusion_reasons", {}),
            "pass_count": len(extraction_passes),
            "passes": [p.to_debug_dict() for p in extraction_passes],
        }
        for msg in parse_info:
            extraction_debug.setdefault("info", []).append(msg)
        extract_text = extraction_plan.text
        for w in extraction_plan.warnings:
            parse_warnings.append(w)
        title = out_name
        if parsed.segments and parsed.segments[0].text:
            first_line = parsed.segments[0].text.splitlines()[0].strip()
            if first_line:
                title = first_line[:200]
        doc_type = extraction_plan.profile.document_type if extraction_plan.profile else "unknown"
        ocr_used = bool(parsed.segments and parsed.segments[0].meta.get("ocr_used"))
        proc_pages = set(extraction_plan.profile.procedure_pages or []) if extraction_plan.profile else set()
        missing_proc = sorted(proc_pages - set(extraction_plan.pages_included))
        use_single_shot = (
            not ocr_used
            and text_chars <= settings.single_shot_max_chars
            and bool((parsed.text or "").strip())
            and not missing_proc
            and len(extraction_passes) == 1
            and not extraction_plan.truncated
        )
        if extract_text.strip() or use_single_shot:
            if use_single_shot:
                llm_text = parsed.text or ""
                llm_chunks = None
                route, raw_out = await asyncio.to_thread(
                    extract_reactions_from_text,
                    settings=settings,
                    title=title,
                    text=llm_text,
                    model=model_override,
                    text_chunks=llm_chunks,
                )
            elif len(extraction_passes) > 1:
                all_text_steps: list = []
                raw_parts: list[str] = []
                for i, plan in enumerate(extraction_passes, start=1):
                    if not plan.text.strip():
                        continue
                    part_route, part_raw = await asyncio.to_thread(
                        extract_reactions_from_text,
                        settings=settings,
                        title=f"{title} [pass {i}/{len(extraction_passes)}]",
                        text=plan.text,
                        model=model_override,
                        text_chunks=plan.chunks,
                    )
                    all_text_steps.extend(part_route.steps)
                    raw_parts.append(part_raw)
                reconcile_stats: dict = {}
                route = reconcile_route(
                    text_steps=all_text_steps,
                    title=title,
                    target_name=all_text_steps[0].step_label if all_text_steps else "",
                    target_class="unknown",
                    document_type=doc_type,
                    warnings=[f"Merged {len(extraction_passes)} evidence-based extraction passes."],
                    stats_out=reconcile_stats,
                )
                raw_out = "\n\n---PASS---\n\n".join(raw_parts)
                extraction_debug["multi_pass"] = True
                extraction_debug["text_extraction_passes"] = len(extraction_passes)
                extraction_debug["reconcile_preview"] = reconcile_stats
            else:
                plan = extraction_passes[0]
                route, raw_out = await asyncio.to_thread(
                    extract_reactions_from_text,
                    settings=settings,
                    title=title,
                    text=plan.text,
                    model=model_override,
                    text_chunks=plan.chunks,
                )
            text_route = route
            text_route.document_type = doc_type if doc_type != "unknown" else text_route.document_type
            text_steps = list(route.steps)
            stages[-1].status = PipelineStageStatus.done
            if use_single_shot:
                detail = (
                    f"Extracted {len(route.steps)} step(s) via single-shot full text "
                    f"({text_chars} chars, document_type={doc_type})"
                )
                extraction_debug["single_shot"] = True
            elif len(extraction_passes) > 1:
                detail = (
                    f"Extracted {len(route.steps)} step(s) via {len(extraction_passes)} pass(es) "
                    f"(pages covered: {extraction_plan.pages_included})"
                )
                extraction_debug["single_shot"] = False
            else:
                detail = (
                    f"Extracted {len(route.steps)} step(s) from {len(extraction_passes[0].chunks)} evidence chunk(s) "
                    f"({extraction_passes[0].chars_selected}/{extraction_passes[0].chars_total} chars, "
                    f"pages {extraction_passes[0].pages_included})"
                )
                extraction_debug["single_shot"] = False
        else:
            stages[-1].status = PipelineStageStatus.stubbed
            detail = "Skipped — no extractable text (awaiting OCR/VLM page-read)"
        stages[-1].detail = detail
        stages[-1].data = {
            "steps": len(text_steps),
            "target": text_route.target_name if text_route else None,
            "text_chars_sent": text_chars if use_single_shot else extraction_plan.chars_selected,
            "single_shot": use_single_shot if extract_text.strip() or use_single_shot else False,
            "evidence_chunks": 0 if use_single_shot else sum(len(p.chunks) for p in extraction_passes),
            "pages_included": list(range(1, parsed.page_count + 1)) if use_single_shot else extraction_plan.pages_included,
            "pages_excluded": [] if use_single_shot else extraction_plan.pages_excluded,
            "truncated": False if use_single_shot else extraction_plan.truncated,
            "extraction_pass_count": len(extraction_passes),
            "document_type": doc_type,
            "extraction_plan": extraction_debug.get("page_selection"),
        }
        await stage_event("text_extract", str(stages[-1].status.value), detail)
    except Exception as exc:
        stages[-1].status = PipelineStageStatus.error
        stages[-1].detail = str(exc)
        await stage_event("text_extract", "error", str(exc))

    # 3) Table extraction
    stages.append(
        PipelineStage(
            id="table_extract",
            label=STAGE_LABELS["table_extract"],
            status=PipelineStageStatus.running,
            detail="Extracting reaction data from PDF tables…",
        )
    )
    await stage_event("table_extract", "running", "Scanning tables with pdfplumber + LLM…")
    try:
        table_stage, table_steps = await run_table_extraction(pdf_path, settings, model_override)
        stages[-1] = table_stage
        status = "done" if table_stage.status == PipelineStageStatus.done else "stubbed"
        if table_stage.status == PipelineStageStatus.error:
            status = "error"
        await stage_event("table_extract", status, table_stage.detail)
    except Exception as exc:
        stages[-1].status = PipelineStageStatus.error
        stages[-1].detail = f"Table extraction failed: {exc}"
        await stage_event("table_extract", "error", stages[-1].detail)

    # 4) Image / scheme / page-read VLM
    stages.append(
        PipelineStage(
            id="image_scheme",
            label=STAGE_LABELS["image_scheme"],
            status=PipelineStageStatus.running,
            detail="Classifying images; scheme VLM + optional page-read…",
        )
    )
    await stage_event("image_scheme", "running", "Preparing image analysis…")

    async def on_image_progress(image_index: int, image_total: int, page_num: int) -> None:
        await stage_event(
            "image_scheme",
            "running",
            f"VLM analysis: image {image_index} / {image_total} (page {page_num})",
            image_index=image_index,
            image_total=image_total,
        )

    schemes = []
    try:
        force_page_read = settings.enable_vlm_page_read and text_chars < 400
        figure_page_info: list[dict] = []
        figure_pages: list[int] = []
        if settings.enable_vlm_figure_pages and text_chars >= 400:
            figure_page_info = detect_figure_scheme_pages(
                pdf_path,
                parsed.segments,
                min_drawings=settings.figure_page_min_drawings,
                max_pages=settings.vlm_figure_max_pages,
            )
            figure_pages = [int(p["page"]) for p in figure_page_info]
            if figure_pages:
                extraction_debug["figure_pages_vlm"] = figure_page_info
        local = settings.ocsr_local_model.strip() or None
        img_stage, schemes = await run_image_scheme_extraction(
            pdf_path,
            settings,
            vlm_model=model_override,
            local_model=local,
            on_image_progress=on_image_progress,
            force_page_read=force_page_read,
            text_char_count=text_chars,
            figure_pages=figure_pages,
            figure_page_info=figure_page_info,
            segments=parsed.segments,
        )
        stages[-1] = img_stage
        status = "done" if img_stage.status == PipelineStageStatus.done else "stubbed"
        if img_stage.status == PipelineStageStatus.error:
            status = "error"
        await stage_event("image_scheme", status, img_stage.detail)
        scheme_steps = schemes_to_reaction_steps(schemes)
        page_read_steps = page_read_to_reaction_steps(schemes)
        for s in scheme_steps:
            s.source = s.source or "scheme"
            s.information_kind = "scheme"

        # If text was empty but page-read produced transcription, optionally run text LLM on it
        if text_chars < 400 and not text_steps:
            transcribed = "\n\n".join(s.transcribed_text for s in schemes if s.transcribed_text).strip()
            if len(transcribed) >= 400:
                try:
                    route2, raw2 = await asyncio.to_thread(
                        extract_reactions_from_text,
                        settings=settings,
                        title=out_name,
                        text=transcribed[:60000],
                        model=model_override,
                    )
                    text_route = route2
                    text_steps = list(route2.steps)
                    raw_out = (raw_out or "") + "\n\n# page_read_text_extract\n" + (raw2 or "")
                    parse_warnings.append(
                        f"Text LLM re-run on VLM page-read transcription ({len(transcribed)} chars) → {len(text_steps)} step(s)."
                    )
                except Exception as exc:
                    parse_warnings.append(f"Page-read text re-extract failed: {exc}")
    except Exception as exc:
        stages[-1].status = PipelineStageStatus.error
        stages[-1].detail = f"Image extraction failed: {exc}"
        await stage_event("image_scheme", "error", stages[-1].detail)

    # 5) Reconcile
    await stage_event("reconcile", "running", "Merging text, table, scheme, and page-read…")
    reconcile_stats: dict = {}
    route = reconcile_route(
        text_steps=text_steps or None,
        table_steps=table_steps or None,
        scheme_steps=scheme_steps or None,
        page_read_steps=page_read_steps or None,
        title=(text_route.title if text_route else "") or out_name,
        target_name=(text_route.target_name if text_route else "") or "",
        target_class=(text_route.target_class if text_route else "") or "unknown",
        document_type=(text_route.document_type if text_route else "") or "unknown",
        warnings=list(text_route.warnings) if text_route and text_route.warnings else [],
        stats_out=reconcile_stats,
    )
    extraction_debug["reconciliation"] = reconcile_stats
    if extraction_plan:
        coverage_warn = document_coverage_warning(
            extraction_plan,
            len(route.steps),
            procedure_subsections=len(extraction_plan.chunks) if extraction_plan else 0,
        )
        if coverage_warn:
            parse_warnings.append(coverage_warn)
    if parse_warnings:
        route.warnings = parse_warnings + [w for w in route.warnings if w not in parse_warnings]

    from app.services.chemical_synonyms import apply_synonym_normalization
    from app.services.confidence_score import apply_confidence_scores
    from app.services.example_linking import link_route_compounds
    from app.services.route_graph import build_route_graph

    norm_count = apply_synonym_normalization(route)
    if norm_count:
        route.warnings.append(f"Normalized {norm_count} chemical name(s) via synonym layer.")
    link_route_compounds(route)
    route.route_graph = build_route_graph(route)
    apply_confidence_scores(route)
    extraction_debug["route_graph_nodes"] = len(route.route_graph.nodes) if route.route_graph else 0
    extraction_debug["compound_links"] = len(route.compound_links)
    extraction_debug["observability"] = {
        "document": {
            "pages": parsed.page_count,
            "text_chars": text_chars,
            "figures": parsed.figure_count,
            "tables": len(parsed.table_hints),
        },
        "page_selection": extraction_debug.get("page_selection"),
        "text_extraction": {
            "steps": len(text_steps),
            "passes": len(extraction_passes),
        },
        "table_extraction": {"steps": len(table_steps)},
        "scheme_extraction": {
            "scheme_steps": len(scheme_steps),
            "page_read_steps": len(page_read_steps),
        },
        "reconciliation": reconcile_stats,
        "standardization": {},
    }

    reconcile_detail = (
        f"Merged {len(route.steps)} steps "
        f"(text={len(text_steps)}, tables={len(table_steps)}, "
        f"schemes={len(scheme_steps)}, page_read={len(page_read_steps)})"
    )
    stages.append(
        PipelineStage(
            id="reconcile",
            label=STAGE_LABELS["reconcile"],
            status=PipelineStageStatus.done,
            detail=reconcile_detail,
            data={
                "steps": len(route.steps),
                "text_steps": len(text_steps),
                "table_steps": len(table_steps),
                "scheme_steps": len(scheme_steps),
                "page_read_steps": len(page_read_steps),
            },
        )
    )
    await stage_event("reconcile", "done", reconcile_detail)

    # 6) Standardize
    stages.append(
        PipelineStage(
            id="standardize",
            label=STAGE_LABELS["standardize"],
            status=PipelineStageStatus.running,
            detail="Resolving names, validating SMILES, normalizing times…",
        )
    )
    await stage_event("standardize", "running", "PubChem / OPSIN / RDKit standardization…")

    loop = asyncio.get_running_loop()

    def on_entity_progress(current: int, total: int, name: str) -> None:
        if total <= 0:
            detail = f"Standardizing: {name}"
            ent_idx = 0
            ent_total = 0
        elif current == 0:
            detail = f"Standardizing {total} chemical name(s)…"
            ent_idx = 0
            ent_total = total
        else:
            detail = f"Standardizing entity {current}/{total}: {name}"
            ent_idx = current
            ent_total = total
        try:
            asyncio.run_coroutine_threadsafe(
                stage_event(
                    "standardize",
                    "running",
                    detail,
                    entity_index=ent_idx,
                    entity_total=ent_total,
                ),
                loop,
            )
        except Exception:
            pass

    std_stage, route = await asyncio.to_thread(
        run_standardize,
        route,
        settings,
        model_override,
        on_entity_progress,
    )
    stages[-1] = std_stage
    std_status = "done" if std_stage.status == PipelineStageStatus.done else "stubbed"
    if std_stage.status == PipelineStageStatus.error:
        std_status = "error"
    await stage_event("standardize", std_status, std_stage.detail)

    if std_stage.data:
        extraction_debug.setdefault("observability", {})["standardization"] = std_stage.data
    if route:
        prov_entities = 0
        unprov_entities = 0
        for step in route.steps:
            for bucket in (
                step.reactants,
                step.reagents,
                step.catalysts,
                step.ligands,
                step.solvents,
                step.intermediates,
                step.products,
            ):
                for ent in bucket:
                    if ent.page or ent.provenance_source or ent.evidence:
                        prov_entities += 1
                    else:
                        unprov_entities += 1
        extraction_debug.setdefault("observability", {})["provenance"] = {
            "entities_with_provenance": prov_entities,
            "entities_without_provenance": unprov_entities,
            "reactions_with_conflicts": len(route.conflicts),
        }

    result = ExtractionResult(
        job_id=job_id,
        filename=out_name,
        stages=stages,
        document=parsed,
        route=route,
        raw_model_output=raw_out,
        extraction_debug=extraction_debug,
        error=None
        if route is not None
        else next(
            (s.detail for s in stages if s.id == "text_extract" and s.status == PipelineStageStatus.error),
            None,
        ),
    )

    out_path = settings.output_dir / f"{job_id}.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result

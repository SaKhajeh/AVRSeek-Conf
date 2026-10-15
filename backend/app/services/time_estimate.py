from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.pdf_parse import parse_pdf

# Rough seconds per pipeline stage (tuned for GapGPT / Claude-class models).
SEC_PARSE = 5.0
SEC_BASE = 20.0
SEC_PER_PAGE = 7.0
SEC_PER_FIGURE = 22.0  # one VLM call per embedded image
SEC_PER_TABLE = 14.0
SEC_RECONCILE = 10.0
SEC_STANDARDIZE = 50.0
SEC_PER_STANDARDIZE_ENTITY = 4.0

STAGE_ORDER = (
    "parse_pdf",
    "text_extract",
    "table_extract",
    "image_scheme",
    "reconcile",
    "standardize",
)

STAGE_LABELS = {
    "parse_pdf": "PDF document understanding (+ OCR)",
    "text_extract": "Text reaction extraction (LLM)",
    "table_extract": "Table extraction (pdfplumber + LLM)",
    "image_scheme": "Scheme / page VLM extraction",
    "reconcile": "Reconcile text + table + scheme + page-read",
    "standardize": "Chemical standardization",
}


@dataclass
class FileEstimate:
    filename: str
    page_count: int
    figure_count: int
    table_hints: int
    estimated_seconds: float


@dataclass
class BatchEstimate:
    file_count: int
    total_pages: int
    total_figures: int
    total_tables: int
    estimated_seconds: float
    estimated_minutes: float
    per_file: list[FileEstimate]
    note: str


def stage_seconds_breakdown(
    *,
    page_count: int,
    figure_count: int,
    table_hints: int,
) -> dict[str, float]:
    return {
        "parse_pdf": SEC_PARSE,
        "text_extract": page_count * SEC_PER_PAGE,
        "table_extract": table_hints * SEC_PER_TABLE,
        "image_scheme": figure_count * SEC_PER_FIGURE,
        "reconcile": SEC_RECONCILE,
        "standardize": SEC_STANDARDIZE,
    }


def estimate_seconds_for_document(
    *,
    page_count: int,
    figure_count: int,
    table_hints: int,
) -> float:
    parts = stage_seconds_breakdown(
        page_count=page_count,
        figure_count=figure_count,
        table_hints=table_hints,
    )
    return SEC_BASE + sum(parts.values())


def estimate_remaining_for_file(
    file_est: FileEstimate,
    *,
    current_stage_id: str,
    image_index: int = 0,
    image_total: int = 0,
    entity_index: int = 0,
    entity_total: int = 0,
) -> float:
    """Seconds left for the current file from the current pipeline stage onward."""
    breakdown = stage_seconds_breakdown(
        page_count=file_est.page_count,
        figure_count=file_est.figure_count,
        table_hints=file_est.table_hints,
    )
    if current_stage_id not in STAGE_ORDER:
        return file_est.estimated_seconds

    remaining = 0.0
    started = False
    for stage_id in STAGE_ORDER:
        if stage_id == current_stage_id:
            started = True
            if stage_id == "image_scheme" and image_total > 0:
                # image_index is 1-based while processing
                images_left = max(0, image_total - image_index + 1)
                remaining += images_left * SEC_PER_FIGURE
            elif stage_id == "standardize" and entity_total > 0:
                entities_left = max(0, entity_total - entity_index + 1) if entity_index else entity_total
                remaining += entities_left * SEC_PER_STANDARDIZE_ENTITY + 10.0
            else:
                remaining += breakdown[stage_id]
        elif started:
            remaining += breakdown[stage_id]
    return remaining


def estimate_remaining_in_batch(
    per_file: list[FileEstimate],
    *,
    file_index: int,
    current_stage_id: str,
    image_index: int = 0,
    image_total: int = 0,
    entity_index: int = 0,
    entity_total: int = 0,
) -> float:
    """Seconds left for the whole batch (1-based file_index)."""
    if not per_file or file_index < 1:
        return 0.0

    remaining = 0.0
    for idx, file_est in enumerate(per_file, start=1):
        if idx < file_index:
            continue
        if idx == file_index:
            remaining += estimate_remaining_for_file(
                file_est,
                current_stage_id=current_stage_id,
                image_index=image_index,
                image_total=image_total,
                entity_index=entity_index,
                entity_total=entity_total,
            )
        else:
            remaining += file_est.estimated_seconds
    return round(remaining, 1)


def estimate_pdf_path(path: Path, display_name: str | None = None) -> FileEstimate:
    parsed = parse_pdf(path, enable_ocr=False)
    secs = estimate_seconds_for_document(
        page_count=parsed.page_count,
        figure_count=parsed.figure_count,
        table_hints=len(parsed.table_hints),
    )
    return FileEstimate(
        filename=display_name or path.name,
        page_count=parsed.page_count,
        figure_count=parsed.figure_count,
        table_hints=len(parsed.table_hints),
        estimated_seconds=round(secs, 1),
    )


def estimate_batch(paths: list[Path], display_names: list[str] | None = None) -> BatchEstimate:
    names = display_names or [p.name for p in paths]
    per_file = [
        estimate_pdf_path(path, display_name=name)
        for path, name in zip(paths, names, strict=True)
    ]
    total_secs = sum(f.estimated_seconds for f in per_file)
    total_figures = sum(f.figure_count for f in per_file)
    return BatchEstimate(
        file_count=len(per_file),
        total_pages=sum(f.page_count for f in per_file),
        total_figures=total_figures,
        total_tables=sum(f.table_hints for f in per_file),
        estimated_seconds=round(total_secs, 1),
        estimated_minutes=round(total_secs / 60.0, 1),
        per_file=per_file,
        note=(
            f"Estimate includes text LLM, table LLM, VLM on all {total_figures} embedded image(s), "
            "OCR on scanned PDFs, and standardization. Actual time varies with API latency and document complexity."
        ),
    )

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from app.schemas.reaction_route import DocumentSegment, ParsedDocument
from app.services.document_evidence import annotate_segments, build_evidence_profile
from app.services.evidence_retrieval import build_extraction_plan, select_extraction_text
from app.services.ocr import needs_ocr, ocr_pdf

# Re-export for backward compatibility
__all__ = [
    "parse_pdf",
    "document_text_warning",
    "select_extraction_text",
    "build_extraction_plan",
    "PROCEDURE_HINTS",
]

PROCEDURE_HINTS = re.compile(
    r"(experimental|procedure|synthesis|example\s+\d+|general procedure|"
    r"preparation|materials and methods|intermediate\s+\d+|step\s+[A-D0-9]|"
    r"пример|синтез|экспериментальн)",
    re.IGNORECASE,
)


def _classify_page_kind(text: str) -> str:
    """Full-page scan for procedure/table hints (not first 800 chars only)."""
    if not text:
        return "body"
    if PROCEDURE_HINTS.search(text):
        return "procedure_candidate"
    if re.search(r"\btable\s+\d+", text, re.IGNORECASE):
        return "table_candidate"
    return "body"


def parse_pdf(path: Path, *, output_dir: Path | None = None, enable_ocr: bool = True) -> ParsedDocument:
    doc = fitz.open(path)
    pages_text: list[str] = []
    segments: list[DocumentSegment] = []
    figure_count = 0
    table_hints: list[str] = []
    ocr_backend = ""
    ocr_used = False

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages_text.append(text)
        figure_count += len(page.get_images(full=True))
        kind = _classify_page_kind(text)
        if kind == "table_candidate":
            table_hints.append(f"page {i}")

        segments.append(
            DocumentSegment(
                segment_id=f"page-{i}",
                kind=kind,
                page_start=i,
                page_end=i,
                text=text.strip(),
                meta={"char_count": len(text), "source": "native"},
            )
        )

    full_text = "\n\n".join(t for t in pages_text if t.strip())

    if enable_ocr and needs_ocr(full_text, doc.page_count):
        try:
            from app.config import get_settings

            ocr_result = ocr_pdf(
                path,
                output_dir=output_dir,
                languages=get_settings().ocr_languages,
            )
            if ocr_result.total_chars > len(full_text):
                ocr_used = True
                ocr_backend = ocr_result.backend
                segments = []
                table_hints = []
                for p in ocr_result.pages:
                    kind = _classify_page_kind(p.text)
                    if kind == "table_candidate":
                        table_hints.append(f"page {p.page_number}")
                    segments.append(
                        DocumentSegment(
                            segment_id=f"page-{p.page_number}",
                            kind=kind,
                            page_start=p.page_number,
                            page_end=p.page_number,
                            text=p.text.strip(),
                            meta={
                                "char_count": len(p.text),
                                "source": "ocr",
                                "ocr_backend": p.backend,
                            },
                        )
                    )
                full_text = ocr_result.full_text
        except Exception as exc:
            ocr_backend = f"failed: {exc}"

    profile = build_evidence_profile(segments, full_text)
    annotate_segments(segments, profile)

    meta_extra = {
        "ocr_used": ocr_used,
        "ocr_backend": ocr_backend,
        "native_char_count": sum(len(t) for t in pages_text),
        "final_char_count": len(full_text),
        "document_type": profile.document_type,
        "procedure_pages": profile.procedure_pages,
        "high_evidence_pages": profile.high_evidence_pages,
    }

    parsed = ParsedDocument(
        filename=path.name,
        page_count=doc.page_count,
        text=full_text,
        segments=segments,
        figure_count=figure_count,
        table_hints=table_hints[:50],
    )
    if segments:
        segments[0].meta.update(meta_extra)
        segments[0].meta["evidence_profile"] = profile.to_dict()
    return parsed


def document_text_warning(parsed: ParsedDocument) -> tuple[str | None, list[str]]:
    """
    Return (user_warning_or_none, info_messages).
    OCR success is info, not a warning. Low text before VLM is informational only.
    """
    chars = len(parsed.text or "")
    info: list[str] = []
    if parsed.page_count == 0:
        return None, info

    ocr_used = False
    ocr_backend = ""
    procedure_pages: list[int] = []
    if parsed.segments:
        meta = parsed.segments[0].meta
        ocr_used = bool(meta.get("ocr_used"))
        ocr_backend = str(meta.get("ocr_backend") or "")
        procedure_pages = list(meta.get("procedure_pages") or [])

    if chars == 0:
        return (
            "No extractable text in this PDF even after OCR — "
            "VLM page-read will be attempted for scanned documents.",
            info,
        )

    if ocr_used and ocr_backend and not str(ocr_backend).startswith("failed"):
        info.append(f"Text recovered via OCR ({ocr_backend}, {chars} characters).")

    if procedure_pages:
        info.append(f"Experimental/example pages detected: {procedure_pages}.")
    elif chars >= 400:
        info.append(
            "No high-confidence procedure pages detected by structure heuristics; "
            "evidence retrieval will use keyword-ranked pages."
        )

    if chars < 400:
        info.append(
            f"Low native/OCR text ({chars} chars) — relying on VLM page-read for scanned content."
        )

    return None, info


def document_coverage_warning(plan, extracted_steps: int, procedure_subsections: int = 0) -> str | None:
    """Post-extraction coverage check — avoid false 'missing experimental' messages."""
    if not plan or not plan.profile:
        return None
    proc_pages = plan.profile.procedure_pages
    if proc_pages and extracted_steps <= 1 and procedure_subsections > 1:
        return (
            f"Experimental content detected on pages {proc_pages}, but only {extracted_steps} "
            f"reaction step(s) extracted while {procedure_subsections} procedure subsection(s) "
            f"were retrieved — possible under-extraction."
        )
    if proc_pages and extracted_steps == 0:
        return (
            f"Experimental/example pages detected ({proc_pages}) but no reaction steps extracted."
        )
    return None

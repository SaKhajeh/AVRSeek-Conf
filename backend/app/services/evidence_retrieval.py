from __future__ import annotations

"""Evidence-aware text retrieval — select synthesis content by structure, not head truncation."""

import re
from dataclasses import dataclass, field

from app.schemas.reaction_route import DocumentSegment, ParsedDocument
from app.services.document_evidence import DocumentEvidenceProfile, build_evidence_profile

EXAMPLE_BOUNDARY = re.compile(
    r"(?=(?:^|\n)\s*(?:"
    r"example|examples|пример|ex\.?\s*|preparation|preparations|"
    r"general procedure|step\s+[A-D0-9ivx]+|intermediate\s+[0-9a-z]+"
    r")\s*[:\.\-\)]?\s*[0-9a-divx]+)",
    re.IGNORECASE,
)

RETRIEVAL_KEYWORDS = re.compile(
    r"\b(synthesis|prepared|preparation|example|procedure|yield|afforded|"
    r"reacted|heated|stirred|purified|recrystall|intermediate|compound|workup|"
    r"equivalent|mol\b|mmol|reflux|"
    r"синтез|получен|пример|эксперимент|"
    r"سنتز|آماده\s*سازی|آزمایش)\b",
    re.I,
)

WORKED_PROCEDURE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:g|mg|mmol|ml|mol)\b|was added|was stirred|afforded|obtained)\b",
    re.I,
)

# Sections skipped or heavily deprioritized for journal papers
PAPER_LOW_VALUE_SECTIONS = frozenset({"references", "abstract", "introduction"})

# Default budgets — aligned to avoid double-truncation in llm_client
DEFAULT_MAX_CHARS = 90000
LLM_CHUNK_CHARS = 28000


@dataclass
class EvidenceChunk:
    text: str
    page_start: int
    page_end: int
    section: str
    evidence_types: list[str]
    synthesis_score: float
    char_offset: int = 0

    def header(self) -> str:
        et = ", ".join(self.evidence_types[:4]) or "body"
        return f"--- Pages {self.page_start}-{self.page_end} ({self.section}; {et}) ---"


@dataclass
class ExtractionPlan:
    text: str
    chunks: list[EvidenceChunk]
    pages_included: list[int]
    pages_excluded: list[int]
    chars_total: int
    chars_selected: int
    truncated: bool
    truncation_note: str = ""
    profile: DocumentEvidenceProfile | None = None
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    pass_index: int = 1
    pass_count: int = 1
    candidate_procedure_pages: list[int] = field(default_factory=list)
    mandatory_pages: list[int] = field(default_factory=list)
    exclusion_reasons: dict[int, str] = field(default_factory=dict)

    def to_debug_dict(self) -> dict:
        return {
            "document_type": self.profile.document_type if self.profile else "unknown",
            "pages_included": self.pages_included,
            "pages_excluded": self.pages_excluded,
            "candidate_procedure_pages": self.candidate_procedure_pages,
            "mandatory_pages": self.mandatory_pages,
            "exclusion_reasons": self.exclusion_reasons,
            "chars_total": self.chars_total,
            "chars_selected": self.chars_selected,
            "truncated": self.truncated,
            "truncation_note": self.truncation_note,
            "pass_index": self.pass_index,
            "pass_count": self.pass_count,
            "chunk_count": len(self.chunks),
            "chunks": [
                {
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                    "evidence_types": c.evidence_types,
                    "chars": len(c.text),
                    "score": c.synthesis_score,
                }
                for c in self.chunks
            ],
            "procedure_pages": self.profile.procedure_pages if self.profile else [],
            "warnings": self.warnings,
            "info": self.info,
        }


def _should_include_chunk(chunk: EvidenceChunk, *, document_type: str) -> bool:
    """Section-aware gate: keep experimental content, drop review/reference noise on papers."""
    section = (chunk.section or "").lower()
    if section == "references":
        return False
    if document_type != "paper":
        return True
    if section in PAPER_LOW_VALUE_SECTIONS:
        return bool(WORKED_PROCEDURE.search(chunk.text))
    if section == "discussion":
        # Discussion may cite literature routes — only keep chunks with worked procedures
        return bool(WORKED_PROCEDURE.search(chunk.text)) and chunk.synthesis_score >= 0.65
    return True


def _segment_units(segments: list[DocumentSegment]) -> list[tuple[DocumentSegment, float]]:
    units: list[tuple[DocumentSegment, float]] = []
    for seg in segments:
        if not (seg.text or "").strip():
            continue
        score = float(seg.meta.get("synthesis_score", 0.0))
        # Keyword fallback for pages missed by classifier
        if score < 0.4 and RETRIEVAL_KEYWORDS.search(seg.text):
            score = max(score, 0.45)
        units.append((seg, score))
    return units


def _split_segment_into_examples(seg: DocumentSegment) -> list[EvidenceChunk]:
    text = seg.text.strip()
    parts = [p.strip() for p in EXAMPLE_BOUNDARY.split(text) if p.strip()]
    if len(parts) <= 1:
        return [
            EvidenceChunk(
                text=text,
                page_start=seg.page_start,
                page_end=seg.page_end,
                section=str(seg.meta.get("section", seg.kind)),
                evidence_types=[e.get("type", "") for e in seg.meta.get("evidence_types", [])[:4]],
                synthesis_score=float(seg.meta.get("synthesis_score", 0.0)),
            )
        ]
    chunks: list[EvidenceChunk] = []
    for part in parts:
        chunks.append(
            EvidenceChunk(
                text=part,
                page_start=seg.page_start,
                page_end=seg.page_end,
                section="examples",
                evidence_types=["patent_example", "compound_preparation"],
                synthesis_score=max(float(seg.meta.get("synthesis_score", 0.0)), 0.6),
            )
        )
    return chunks


def _segment_by_page(segments: list[DocumentSegment]) -> dict[int, DocumentSegment]:
    return {s.page_start: s for s in segments if (s.text or "").strip()}


def _expand_contiguous_procedure_pages(
    pages: set[int],
    segments: list[DocumentSegment],
    profile: DocumentEvidenceProfile,
) -> set[int]:
    """Include neighboring pages when a procedure likely continues (e.g. workup/yield on next page)."""
    by_page = _segment_by_page(segments)
    proc = set(profile.procedure_pages or [])
    expanded = set(pages)
    for p in list(expanded):
        for neighbor in (p - 1, p + 1):
            if neighbor in expanded or neighbor not in by_page:
                continue
            seg = by_page[neighbor]
            score = float(seg.meta.get("synthesis_score", 0.0))
            section = str(seg.meta.get("section", seg.kind) or "").lower()
            if neighbor in proc:
                expanded.add(neighbor)
            elif score >= 0.35:
                expanded.add(neighbor)
            elif section in ("experimental", "examples", "procedure_candidate"):
                expanded.add(neighbor)
            elif WORKED_PROCEDURE.search(seg.text or ""):
                expanded.add(neighbor)
    return expanded


def _build_candidate_chunks(
    parsed: ParsedDocument,
    profile: DocumentEvidenceProfile,
    *,
    min_synthesis_score: float = 0.35,
) -> list[EvidenceChunk]:
    units = _segment_units(parsed.segments)
    doc_type = profile.document_type
    candidates: list[EvidenceChunk] = []
    for seg, score in units:
        if score >= min_synthesis_score or seg.kind in ("procedure_candidate", "table_candidate"):
            candidates.extend(_split_segment_into_examples(seg))
        elif score >= 0.25:
            candidates.append(
                EvidenceChunk(
                    text=seg.text.strip(),
                    page_start=seg.page_start,
                    page_end=seg.page_end,
                    section=str(seg.meta.get("section", "body")),
                    evidence_types=[e.get("type", "") for e in seg.meta.get("evidence_types", [])[:3]],
                    synthesis_score=score,
                )
            )
    candidates = [c for c in candidates if _should_include_chunk(c, document_type=doc_type)]
    if not candidates:
        for seg, _score in units:
            if seg.meta.get("section") == "references":
                continue
            candidates.extend(_split_segment_into_examples(seg))
        candidates = [c for c in candidates if _should_include_chunk(c, document_type=doc_type)]
    candidates.sort(key=lambda c: (c.page_start, -c.synthesis_score, -len(c.text)))
    return candidates


def _page_char_budget(parsed: ParsedDocument, pages: set[int]) -> int:
    """Estimate chars for mandatory pages from raw segments (includes filtered pages)."""
    by_page = _segment_by_page(parsed.segments)
    total = 0
    for p in pages:
        seg = by_page.get(p)
        if seg and (seg.text or "").strip():
            total += len(seg.text.strip()) + 80
    return total


def _force_chunks_for_pages(
    parsed: ParsedDocument,
    pages: set[int],
) -> list[EvidenceChunk]:
    """
    Build extraction chunks for pages absent from the candidate set.
    Used when section filters excluded a mandatory page (e.g. discussion continuation).
    """
    by_page = _segment_by_page(parsed.segments)
    forced: list[EvidenceChunk] = []
    for p in sorted(pages):
        seg = by_page.get(p)
        if not seg or not (seg.text or "").strip():
            continue
        score = float(seg.meta.get("synthesis_score", 0.35))
        if score >= 0.35 or seg.kind in ("procedure_candidate", "table_candidate"):
            forced.extend(_split_segment_into_examples(seg))
        else:
            forced.append(
                EvidenceChunk(
                    text=seg.text.strip(),
                    page_start=seg.page_start,
                    page_end=seg.page_end,
                    section=str(seg.meta.get("section", seg.kind or "body")),
                    evidence_types=[e.get("type", "") for e in seg.meta.get("evidence_types", [])[:3]],
                    synthesis_score=max(score, 0.35),
                )
            )
    return forced


def _supplement_candidates(
    parsed: ParsedDocument,
    candidates: list[EvidenceChunk],
    mandatory: set[int],
) -> list[EvidenceChunk]:
    """Ensure every mandatory page has at least one chunk for pass planning."""
    covered = {c.page_start for c in candidates}
    missing = mandatory - covered
    if not missing:
        return candidates
    extra = _force_chunks_for_pages(parsed, missing)
    if not extra:
        return candidates
    merged = candidates + extra
    merged.sort(key=lambda c: (c.page_start, -c.synthesis_score, -len(c.text)))
    return merged


def _pack_chunks(
    chunks: list[EvidenceChunk],
    *,
    max_chars: int,
    mandatory_pages: set[int],
) -> tuple[list[EvidenceChunk], list[int], dict[int, str]]:
    """Pack chunks up to max_chars; mandatory pages first in reading order."""
    mandatory = [c for c in chunks if c.page_start in mandatory_pages]
    optional = [c for c in chunks if c.page_start not in mandatory_pages]
    ordered = mandatory + optional

    selected: list[EvidenceChunk] = []
    total = 0
    included: set[int] = set()
    exclusion_reasons: dict[int, str] = {}

    for chunk in ordered:
        chunk_len = len(chunk.text) + len(chunk.header()) + 4
        is_mandatory = chunk.page_start in mandatory_pages
        if total + chunk_len > max_chars:
            if is_mandatory and total == 0 and len(chunk.text) > max_chars:
                trimmed = chunk.text[:max_chars]
                selected.append(
                    EvidenceChunk(
                        text=trimmed,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        evidence_types=chunk.evidence_types,
                        synthesis_score=chunk.synthesis_score,
                    )
                )
                total += len(trimmed)
                included.add(chunk.page_start)
                continue
            if not is_mandatory:
                exclusion_reasons.setdefault(chunk.page_start, f"budget ({max_chars} chars)")
            continue
        selected.append(chunk)
        total += chunk_len
        included.add(chunk.page_start)

    for p in mandatory_pages:
        if p not in included:
            exclusion_reasons[p] = exclusion_reasons.get(p, f"budget ({max_chars} chars) — needs another pass")

    return selected, sorted(included), exclusion_reasons


def _plan_from_chunks(
    chunks: list[EvidenceChunk],
    parsed: ParsedDocument,
    profile: DocumentEvidenceProfile,
    *,
    max_chars: int,
    mandatory_pages: set[int],
    pass_index: int = 1,
    pass_count: int = 1,
    base_warnings: list[str] | None = None,
    base_info: list[str] | None = None,
) -> ExtractionPlan:
    all_pages = [s.page_start for s in parsed.segments if (s.text or "").strip()]
    selected, included_pages, exclusion_reasons = _pack_chunks(
        chunks, max_chars=max_chars, mandatory_pages=mandatory_pages
    )
    parts = [f"{c.header()}\n{c.text}" for c in selected]
    combined = "\n\n".join(parts)
    pages_excluded = sorted(set(all_pages) - set(included_pages))

    warnings = list(base_warnings or [])
    info = list(base_info or [])
    truncated = bool(pages_excluded) or bool(exclusion_reasons)
    truncation_note = ""
    if truncated and pass_count == 1:
        truncation_note = (
            f"Pass {pass_index}/{pass_count}: selected {len(combined)} chars "
            f"from pages {included_pages}; excluded {pages_excluded}."
        )
        if exclusion_reasons:
            warnings.append(f"Pages needing additional pass: {sorted(exclusion_reasons.keys())}.")

    return ExtractionPlan(
        text=combined,
        chunks=selected,
        pages_included=included_pages,
        pages_excluded=pages_excluded,
        chars_total=len(parsed.text or ""),
        chars_selected=len(combined),
        truncated=truncated,
        truncation_note=truncation_note,
        profile=profile,
        warnings=warnings,
        info=info,
        pass_index=pass_index,
        pass_count=pass_count,
        candidate_procedure_pages=list(profile.procedure_pages or []),
        mandatory_pages=sorted(mandatory_pages),
        exclusion_reasons=exclusion_reasons,
    )


def plan_extraction_passes(
    parsed: ParsedDocument,
    *,
    max_chars_per_pass: int = LLM_CHUNK_CHARS,
    min_synthesis_score: float = 0.35,
) -> list[ExtractionPlan]:
    """
    Build one or more extraction passes so procedure pages are not silently dropped.
    Contiguous procedure ranges are preserved; overflow uses additional LLM passes.
    """
    profile = build_evidence_profile(parsed.segments, parsed.text or "")
    all_pages = [s.page_start for s in parsed.segments if (s.text or "").strip()]
    if not all_pages:
        empty = ExtractionPlan(
            text="",
            chunks=[],
            pages_included=[],
            pages_excluded=[],
            chars_total=len(parsed.text or ""),
            chars_selected=0,
            truncated=False,
            profile=profile,
            warnings=["No extractable text segments found."],
        )
        return [empty]

    candidates = _build_candidate_chunks(parsed, profile, min_synthesis_score=min_synthesis_score)
    if not candidates:
        for seg in parsed.segments:
            if (seg.text or "").strip():
                candidates.extend(_split_segment_into_examples(seg))

    mandatory = _expand_contiguous_procedure_pages(
        set(profile.procedure_pages or []) | {c.page_start for c in candidates if c.synthesis_score >= 0.55},
        parsed.segments,
        profile,
    )
    if not mandatory and candidates:
        mandatory = {c.page_start for c in candidates if c.synthesis_score >= 0.45}

    candidates = _supplement_candidates(parsed, candidates, mandatory)

    base_info: list[str] = []
    base_warnings: list[str] = []
    if profile.procedure_pages:
        base_info.append(
            f"Procedure/evidence pages detected: {profile.procedure_pages} "
            f"(document_type={profile.document_type})."
        )
    else:
        base_warnings.append(
            "No high-confidence experimental/example pages detected; using keyword-ranked fallback."
        )
    if mandatory:
        base_info.append(f"Mandatory procedure page set (with neighbors): {sorted(mandatory)}.")

    # Estimate if single pass fits all mandatory content (use raw page sizes, not only candidates)
    mandatory_chars = _page_char_budget(parsed, mandatory)

    passes: list[ExtractionPlan] = []
    if mandatory_chars <= max_chars_per_pass:
        plan = _plan_from_chunks(
            candidates,
            parsed,
            profile,
            max_chars=max_chars_per_pass,
            mandatory_pages=mandatory,
            pass_index=1,
            pass_count=1,
            base_warnings=base_warnings,
            base_info=base_info,
        )
        missing_mandatory = [p for p in mandatory if p not in plan.pages_included]
        if not missing_mandatory:
            return [plan]
        passes.append(plan)
        remaining_pages = mandatory - set(plan.pages_included)
    else:
        remaining_pages = set(mandatory)

    if not remaining_pages:
        for i, p in enumerate(passes, start=1):
            p.pass_count = len(passes)
            p.pass_index = i
        return passes

    # Multi-pass: group mandatory chunks by contiguous page runs (only uncovered pages)
    mandatory_chunks = sorted(
        [c for c in candidates if c.page_start in remaining_pages],
        key=lambda c: c.page_start,
    )
    if not mandatory_chunks and remaining_pages:
        mandatory_chunks = sorted(
            _force_chunks_for_pages(parsed, remaining_pages),
            key=lambda c: c.page_start,
        )
    current_group: list[EvidenceChunk] = []
    current_pages: set[int] = set()
    groups: list[tuple[list[EvidenceChunk], set[int]]] = []

    def flush_group() -> None:
        nonlocal current_group, current_pages
        if current_group:
            groups.append((current_group, set(current_pages)))
        current_group = []
        current_pages = set()

    last_page = -999
    for chunk in mandatory_chunks:
        if current_group and chunk.page_start > last_page + 2:
            flush_group()
        current_group.append(chunk)
        current_pages.add(chunk.page_start)
        last_page = chunk.page_start
    flush_group()

    # Merge small adjacent groups into passes under budget
    pass_groups: list[tuple[list[EvidenceChunk], set[int]]] = []
    buf_chunks: list[EvidenceChunk] = []
    buf_pages: set[int] = set()
    buf_chars = 0

    def _split_oversized_group(
        grp_chunks: list[EvidenceChunk], grp_pages: set[int]
    ) -> list[tuple[list[EvidenceChunk], set[int]]]:
        """Split a contiguous run that exceeds one pass budget into page-aware sub-passes."""
        by_page: dict[int, list[EvidenceChunk]] = {}
        for chunk in grp_chunks:
            by_page.setdefault(chunk.page_start, []).append(chunk)
        sub_passes: list[tuple[list[EvidenceChunk], set[int]]] = []
        sub_buf: list[EvidenceChunk] = []
        sub_pages: set[int] = set()
        sub_chars = 0
        for pg in sorted(by_page):
            pg_chunks = by_page[pg]
            pg_chars = sum(len(c.text) + len(c.header()) + 4 for c in pg_chunks)
            if pg_chars > max_chars_per_pass:
                if sub_buf:
                    sub_passes.append((sub_buf, sub_pages))
                    sub_buf, sub_pages, sub_chars = [], set(), 0
                sub_passes.append((pg_chunks, {pg}))
                continue
            if sub_chars + pg_chars > max_chars_per_pass and sub_buf:
                sub_passes.append((sub_buf, sub_pages))
                sub_buf, sub_pages, sub_chars = [], set(), 0
            sub_buf.extend(pg_chunks)
            sub_pages.add(pg)
            sub_chars += pg_chars
        if sub_buf:
            sub_passes.append((sub_buf, sub_pages))
        return sub_passes or [(grp_chunks, grp_pages)]

    for grp_chunks, grp_pages in groups:
        grp_chars = sum(len(c.text) + len(c.header()) + 4 for c in grp_chunks)
        if buf_chars + grp_chars > max_chars_per_pass and buf_chunks:
            pass_groups.append((buf_chunks, buf_pages))
            buf_chunks, buf_pages, buf_chars = [], set(), 0
        if grp_chars > max_chars_per_pass:
            if buf_chunks:
                pass_groups.append((buf_chunks, buf_pages))
                buf_chunks, buf_pages, buf_chars = [], set(), 0
            pass_groups.extend(_split_oversized_group(grp_chunks, grp_pages))
            continue
        buf_chunks.extend(grp_chunks)
        buf_pages |= grp_pages
        buf_chars += grp_chars
    if buf_chunks:
        pass_groups.append((buf_chunks, buf_pages))

    if not pass_groups:
        pass_groups = groups or [(mandatory_chunks, mandatory)]

    pass_count = len(pass_groups) + len(passes)
    for i, (grp_chunks, grp_pages) in enumerate(pass_groups, start=len(passes) + 1):
        plan = _plan_from_chunks(
            grp_chunks,
            parsed,
            profile,
            max_chars=max_chars_per_pass,
            mandatory_pages=grp_pages,
            pass_index=i,
            pass_count=pass_count,
            base_warnings=base_warnings if i == 1 else [],
            base_info=base_info if i == 1 else [],
        )
        plan.info.append(f"Multi-pass extraction: pass {i}/{pass_count}.")
        if plan.chars_selected > 0:
            passes.append(plan)

    # Drop any empty pass placeholders and re-index
    passes = [p for p in passes if p.chars_selected > 0 and p.chunks]

    for i, p in enumerate(passes, start=1):
        p.pass_count = len(passes)
        p.pass_index = i

    covered = set()
    for p in passes:
        covered.update(p.pages_included)
    still_missing = sorted(mandatory - covered)
    if still_missing:
        for p in passes:
            p.warnings.append(
                f"Mandatory procedure pages still not in any pass after split: {still_missing}."
            )
    else:
        for p in passes:
            p.info.append("All mandatory procedure pages covered across extraction pass(es).")

    return passes if passes else [
        _plan_from_chunks(
            candidates,
            parsed,
            profile,
            max_chars=max_chars_per_pass,
            mandatory_pages=mandatory,
            base_warnings=base_warnings,
            base_info=base_info,
        )
    ]


def build_extraction_plan(
    parsed: ParsedDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_synthesis_score: float = 0.35,
) -> ExtractionPlan:
    """Return the first extraction pass (backward compatible). Use plan_extraction_passes for multi-pass."""
    passes = plan_extraction_passes(
        parsed,
        max_chars_per_pass=min(max_chars, LLM_CHUNK_CHARS),
        min_synthesis_score=min_synthesis_score,
    )
    if len(passes) == 1:
        return passes[0]
    # Combined view for callers expecting one plan
    combined = ExtractionPlan(
        text="\n\n".join(p.text for p in passes if p.text),
        chunks=[c for p in passes for c in p.chunks],
        pages_included=sorted({pg for p in passes for pg in p.pages_included}),
        pages_excluded=sorted({pg for p in passes for pg in p.pages_excluded}),
        chars_total=passes[0].chars_total,
        chars_selected=sum(p.chars_selected for p in passes),
        truncated=any(p.truncated for p in passes),
        truncation_note=f"Split into {len(passes)} extraction pass(es); see passes in extraction_debug.",
        profile=passes[0].profile,
        warnings=list({w for p in passes for w in p.warnings}),
        info=list({i for p in passes for i in p.info}),
        pass_index=1,
        pass_count=len(passes),
        candidate_procedure_pages=passes[0].candidate_procedure_pages,
        mandatory_pages=passes[0].mandatory_pages,
    )
    combined.info.append(f"Multi-pass plan: {len(passes)} LLM extraction pass(es) required.")
    return combined


def select_extraction_text(parsed: ParsedDocument, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Backward-compatible: return evidence-selected text only."""
    return build_extraction_plan(parsed, max_chars=max_chars).text

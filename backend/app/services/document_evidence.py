from __future__ import annotations

"""Page-level evidence detection for synthesis-relevant content.

Scores each page/segment for evidence types (experimental procedure, examples,
reaction conditions, etc.) and document structure (patent vs paper).
Generic heuristics — no molecule-specific rules.
"""

import re
from dataclasses import dataclass, field

from app.schemas.reaction_route import DocumentSegment

# --- Evidence type patterns (generic) ---

EVIDENCE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "experimental_procedure": [
        re.compile(r"\b(experimental|materials and methods|general procedure|procedure)\b", re.I),
        re.compile(r"\b(was added|was stirred|was heated|was dissolved|afforded|obtained|purified)\b", re.I),
        re.compile(r"\b(пример|эксперимент|синтез|получен|нагревал)\b", re.I),
        re.compile(r"(آزمایش|مواد\s*و\s*روش|روش\s*کار|آماده\s*سازی|سنتز)", re.I),
    ],
    "compound_preparation": [
        re.compile(r"\b(preparation|prepared|synthesis of|synthesized)\b", re.I),
        re.compile(r"\b(intermediate|compound)\s+[A-Za-z0-9ivx\-]+", re.I),
    ],
    "reaction_conditions": [
        re.compile(r"\b(\d+\s*°\s*C|reflux|room temperature|rt\b|overnight)\b", re.I),
        re.compile(r"\b(\d+\s*(?:h|hr|min|hours?|minutes?|days?))\b", re.I),
        re.compile(r"\bpH\s*[=:]?\s*\d", re.I),
    ],
    "yield": [
        re.compile(r"\b(yield|afforded|obtained)\s*[:\(]?\s*\d", re.I),
        re.compile(r"\b\d+(?:\.\d+)?\s*%\b"),
    ],
    "reaction_table": [
        re.compile(r"\btable\s+\d+", re.I),
        re.compile(r"\b(entry|run)\s+\d+", re.I),
    ],
    "patent_example": [
        re.compile(r"\bexample\s+[0-9a-divx]+", re.I),
        re.compile(r"\b(preparation|step)\s+[A-D0-9]+", re.I),
        re.compile(r"\bintermediate\s+[0-9a-z]+", re.I),
    ],
    "abstract_claim": [
        re.compile(r"^\s*abstract\b", re.I | re.M),
        re.compile(r"\b(we report|herein|disclosed is|claim\s+\d+)\b", re.I),
    ],
    "background": [
        re.compile(r"\b(introduction|background|prior art|state of the art)\b", re.I),
    ],
    "references": [
        re.compile(r"\b(references|bibliography)\b", re.I),
    ],
    "discussion": [
        re.compile(r"\b(results and discussion|discussion|conclusion)\b", re.I),
        re.compile(r"\b(we report|in this (?:study|work|paper))\b", re.I),
    ],
    "analytical_characterization": [
        re.compile(r"\b(NMR|HRMS|LC.?MS|HPLC|IR spectrum|melting point|mp\s*[=:])\b", re.I),
    ],
}

PATENT_NUMBER = re.compile(
    r"\b(?:WO|US|EP|CN|JP|KR|AU|CA)\s*[\d/,.\-]{4,}\b|\bpatent\s+(?:application|no\.?)\b",
    re.I,
)

PATENT_MARKERS = re.compile(
    r"\b(claims?\s+\d+|example\s+\d+[:\.]|\bpreparation\s+\d+|"
    r"intermediate\s+[0-9a-z]+\b|step\s+[A-D]\b)\b",
    re.I,
)

PAPER_MARKERS = re.compile(
    r"\b(abstract|introduction|experimental|materials and methods|"
    r"supporting information|supplementary|results and discussion|"
    r"received\s+\d|corresponding author|acknowledgement)\b",
    re.I,
)

JOURNAL_MARKERS = re.compile(
    r"\b(journal|vol\.|volume\s+\d|issue\s+\d|doi:|issn|pp\.\s*\d|"
    r"chem\.?\s*abstracts|pharm\.?\s*bull|j\.?\s*med\.?\s*chem)\b",
    re.I,
)

# Higher = more trustworthy for synthesis extraction
EVIDENCE_PRIORITY: dict[str, float] = {
    "experimental_procedure": 1.0,
    "compound_preparation": 0.95,
    "patent_example": 0.95,
    "reaction_conditions": 0.85,
    "yield": 0.8,
    "reaction_table": 0.75,
    "analytical_characterization": 0.5,
    "abstract_claim": 0.35,
    "background": 0.2,
    "discussion": 0.25,
    "references": 0.05,
}


@dataclass
class EvidenceHit:
    evidence_type: str
    confidence: float
    priority: float


@dataclass
class PageEvidence:
    page: int
    char_count: int
    section: str
    hits: list[EvidenceHit] = field(default_factory=list)
    synthesis_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "char_count": self.char_count,
            "section": self.section,
            "synthesis_score": round(self.synthesis_score, 3),
            "evidence_types": [
                {"type": h.evidence_type, "confidence": round(h.confidence, 3)}
                for h in self.hits
            ],
        }


@dataclass
class DocumentEvidenceProfile:
    document_type: str  # paper | patent | unknown
    pages: list[PageEvidence]
    procedure_pages: list[int]
    high_evidence_pages: list[int]
    low_evidence_pages: list[int]
    total_chars: int

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "procedure_pages": self.procedure_pages,
            "high_evidence_pages": self.high_evidence_pages,
            "low_evidence_pages": self.low_evidence_pages,
            "total_chars": self.total_chars,
            "pages": [p.to_dict() for p in self.pages],
        }


def _score_text_for_type(text: str, evidence_type: str) -> float:
    if not text.strip():
        return 0.0
    patterns = EVIDENCE_PATTERNS.get(evidence_type, [])
    if not patterns:
        return 0.0
    hits = sum(1 for p in patterns if p.search(text))
    if hits == 0:
        return 0.0
    # Normalize by pattern count; boost for multiple distinct matches
    base = min(1.0, hits / max(1, len(patterns)))
    if hits >= 2:
        base = min(1.0, base + 0.15)
    return base


def _infer_section(text: str, hits: list[EvidenceHit]) -> str:
    head = (text or "")[:1200]
    if re.search(r"\b(references|bibliography)\b", head, re.I):
        return "references"
    if re.search(r"\b(results and discussion|^\s*discussion\b|^\s*conclusion\b)", head, re.I | re.M):
        return "discussion"
    types = {h.evidence_type for h in hits}
    has_worked_example = bool(
        re.search(
            r"\b(\d+(?:\.\d+)?\s*(?:g|mg|mmol|ml|mol)\b|was added|was stirred|afforded|obtained)\b",
            text,
            re.I,
        )
    )
    if has_worked_example and ("experimental_procedure" in types or "compound_preparation" in types):
        return "experimental"
    if "patent_example" in types or ("compound_preparation" in types and has_worked_example):
        return "examples"
    if "experimental_procedure" in types:
        return "experimental"
    if "abstract_claim" in types and len(text) < 2500:
        return "abstract"
    if "background" in types:
        return "introduction"
    if "discussion" in types:
        return "discussion"
    if "references" in types:
        return "references"
    if "reaction_table" in types:
        return "tables"
    return "body"


def classify_page(page_num: int, text: str) -> PageEvidence:
    hits: list[EvidenceHit] = []
    for etype in EVIDENCE_PATTERNS:
        conf = _score_text_for_type(text, etype)
        if conf >= 0.25:
            hits.append(
                EvidenceHit(
                    evidence_type=etype,
                    confidence=conf,
                    priority=EVIDENCE_PRIORITY.get(etype, 0.5),
                )
            )
    hits.sort(key=lambda h: h.priority * h.confidence, reverse=True)

    synthesis_score = 0.0
    if hits:
        synthesis_score = max(h.priority * h.confidence for h in hits)
        # Boost pages with procedure + conditions/yield combo
        types = {h.evidence_type for h in hits}
        if "experimental_procedure" in types or "patent_example" in types:
            if types & {"reaction_conditions", "yield", "compound_preparation"}:
                synthesis_score = min(1.0, synthesis_score + 0.15)

    section = _infer_section(text, hits)
    return PageEvidence(
        page=page_num,
        char_count=len(text),
        section=section,
        hits=hits,
        synthesis_score=synthesis_score,
    )


def detect_document_type(full_text: str, segments: list[DocumentSegment]) -> str:
    sample = (full_text or "")[:12000]
    if PATENT_NUMBER.search(sample):
        return "patent"
    journal_hits = len(JOURNAL_MARKERS.findall(sample))
    paper_score = len(PAPER_MARKERS.findall(sample))
    patent_score = len(PATENT_MARKERS.findall(sample))
    # Journal article cues outweigh generic "preparation"/"example" wording in papers
    if journal_hits >= 1 and paper_score >= 1:
        return "paper"
    if re.search(r"\breceived\s+\d|\bcorresponding author\b", sample, re.I):
        return "paper"
    if patent_score >= 3 and patent_score > paper_score + 1:
        return "patent"
    if paper_score > patent_score:
        return "paper"
    if patent_score > paper_score:
        return "patent"
    return "unknown"


def build_evidence_profile(segments: list[DocumentSegment], full_text: str = "") -> DocumentEvidenceProfile:
    pages: list[PageEvidence] = []
    for seg in segments:
        pe = classify_page(seg.page_start, seg.text or "")
        # Merge segment kind hints from pdf_parse
        if seg.kind == "procedure_candidate" and pe.synthesis_score < 0.5:
            pe.synthesis_score = max(pe.synthesis_score, 0.55)
            pe.hits.append(EvidenceHit("experimental_procedure", 0.6, 0.9))
        pages.append(pe)

    procedure_pages = [p.page for p in pages if p.synthesis_score >= 0.55]
    high_evidence = [p.page for p in pages if p.synthesis_score >= 0.7]
    low_evidence = [p.page for p in pages if p.synthesis_score < 0.25 and p.char_count > 100]

    doc_type = detect_document_type(full_text or "\n".join(s.text for s in segments), segments)

    return DocumentEvidenceProfile(
        document_type=doc_type,
        pages=pages,
        procedure_pages=procedure_pages,
        high_evidence_pages=high_evidence,
        low_evidence_pages=low_evidence,
        total_chars=sum(p.char_count for p in pages),
    )


def annotate_segments(segments: list[DocumentSegment], profile: DocumentEvidenceProfile) -> None:
    """Write evidence scores back into segment meta for downstream use."""
    by_page = {p.page: p for p in profile.pages}
    for seg in segments:
        pe = by_page.get(seg.page_start)
        if not pe:
            continue
        seg.meta["synthesis_score"] = pe.synthesis_score
        seg.meta["section"] = pe.section
        seg.meta["evidence_types"] = [
            {"type": h.evidence_type, "confidence": h.confidence} for h in pe.hits[:6]
        ]
    if segments:
        segments[0].meta["document_type"] = profile.document_type
        segments[0].meta["procedure_pages"] = profile.procedure_pages
        segments[0].meta["high_evidence_pages"] = profile.high_evidence_pages

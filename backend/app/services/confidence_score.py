"""Evidence-based confidence scoring for extracted steps (P2)."""

from __future__ import annotations

import re

from app.schemas.reaction_route import SynthesisRoute

_AMOUNT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:g|mg|mmol|ml|mol|l)\b", re.I)

_INFORMATION_KIND_WEIGHT: dict[str, float] = {
    "explicit": 0.15,
    "scheme": 0.08,
    "inferred": -0.12,
    "literature": -0.18,
    "ambiguous": -0.06,
    "unknown": 0.0,
}


def _has_amount(step) -> bool:
    for bucket in (step.reactants, step.reagents, step.catalysts, step.products, step.solvents):
        for ent in bucket:
            if ent.amount and ent.amount not in ("", "N/A") and _AMOUNT.search(str(ent.amount)):
                return True
    return False


def score_step(step) -> float:
    score = 0.25
    kind = (step.information_kind or "unknown").lower()
    score += _INFORMATION_KIND_WEIGHT.get(kind, 0.0)

    if _has_amount(step):
        score += 0.25
    cond = step.conditions
    if cond and (cond.temperature not in ("", "N/A") or cond.time not in ("", "N/A")):
        score += 0.15
    if step.yield_value not in ("", "N/A"):
        score += 0.1
    if step.evidence and len(step.evidence) > 40:
        score += 0.1
    if step.evidence_refs:
        score += min(0.15, 0.05 * len(step.evidence_refs))
        for ref in step.evidence_refs:
            ref_kind = (ref.information_kind or kind).lower()
            score += _INFORMATION_KIND_WEIGHT.get(ref_kind, 0.0) * 0.5
            if ref.evidence_type in ("experimental_procedure", "patent_example"):
                score += 0.05
            elif ref.evidence_type == "reaction_scheme":
                score += 0.03
            if ref.confidence:
                score += 0.02 * ref.confidence
    src = (step.source or "").lower()
    if src == "text":
        score += 0.05
    elif src in ("scheme", "table"):
        score += 0.03
    elif src == "page_read":
        score -= 0.05
    return min(1.0, max(0.1, round(score, 3)))


def apply_confidence_scores(route: SynthesisRoute) -> None:
    for step in route.steps:
        if step.confidence is None:
            step.confidence = score_step(step)

from __future__ import annotations

"""Reconcile text + table + scheme + page-read extractions into one SynthesisRoute."""

import re
from typing import Any

from app.schemas.reaction_route import (
    ChemicalEntity,
    EvidenceRef,
    FieldConflict,
    ReactionConditions,
    ReactionStep,
    SynthesisRoute,
)
from app.services.chemical_synonyms import extract_compound_id, normalize_chemical_name


def _entity_key(ent: ChemicalEntity) -> str:
    name = (ent.name or "").strip().lower()
    smiles = (ent.smiles or "").strip().lower()
    return f"{name}|{smiles}"


def _role_priority() -> dict[str, int]:
    return {
        "reactant": 1,
        "reagent": 2,
        "catalyst": 3,
        "ligand": 4,
        "solvent": 5,
        "product": 6,
        "intermediate": 7,
        "unknown": 8,
    }


def _merge_entity_into(bucket: list[ChemicalEntity], new_ent: ChemicalEntity) -> None:
    key = _entity_key(new_ent)
    for ent in bucket:
        if _entity_key(ent) == key:
            if _role_priority().get(new_ent.role, 9) < _role_priority().get(ent.role, 9):
                ent.role = new_ent.role
            if (ent.amount in (None, "N/A", "")) and new_ent.amount not in (None, "N/A", ""):
                ent.amount = new_ent.amount
            if not ent.smiles and new_ent.smiles:
                ent.smiles = new_ent.smiles
            if not getattr(ent, "evidence", None) and getattr(new_ent, "evidence", None):
                ent.evidence = new_ent.evidence
            if not getattr(ent, "notes", None) and getattr(new_ent, "notes", None):
                ent.notes = new_ent.notes
            # Prefer human name over raw SMILES-as-name
            if ent.name == ent.smiles and new_ent.name and new_ent.name != new_ent.smiles:
                ent.name = new_ent.name
            return
    bucket.append(new_ent)


def _conditions_empty(cond: ReactionConditions | None) -> bool:
    if not cond:
        return True
    vals = [cond.temperature, cond.time, cond.pressure, cond.atmosphere, cond.other]
    return all(not v or v == "N/A" for v in vals)


def _parse_conditions_from_text(text: str) -> ReactionConditions:
    cond = ReactionConditions()
    if not text:
        return cond
    t = text
    m_temp = re.search(r"(\d+\s*[–\-]\s*\d+\s*°?\s*C|\d+\s*°\s*C|reflux|rt|room temperature)", t, re.I)
    if m_temp:
        cond.temperature = m_temp.group(1).strip()
    m_time = re.search(r"(\d+(?:\.\d+)?\s*(?:h|hr|hrs|hours?|min|minutes?|days?))", t, re.I)
    if m_time:
        cond.time = m_time.group(1).strip()
    m_ph = re.search(r"(pH\s*[=:]?\s*\d+(?:\.\d+)?(?:\s*[–\-]\s*\d+(?:\.\d+)?)?)", t, re.I)
    other_bits = []
    if m_ph:
        other_bits.append(m_ph.group(1))
    if t and t not in ("N/A",):
        other_bits.append(t[:300])
    if other_bits:
        cond.other = "; ".join(other_bits)
    return cond


def _enrich_step_from_conditions_text(step: ReactionStep) -> None:
    text = ""
    if step.conditions and step.conditions.other and step.conditions.other != "N/A":
        text = step.conditions.other
    elif step.evidence:
        text = step.evidence
    if not text:
        return
    parsed = _parse_conditions_from_text(text)
    if step.conditions.temperature in ("", "N/A") and parsed.temperature != "N/A":
        step.conditions.temperature = parsed.temperature
    if step.conditions.time in ("", "N/A") and parsed.time != "N/A":
        step.conditions.time = parsed.time
    if step.yield_value in ("", "N/A"):
        m = re.search(r"(\d+(?:\.\d+)?\s*%)", text)
        if m:
            step.yield_value = m.group(1)


def _step_signature(step: ReactionStep) -> str:
    labels = (step.step_label or "").lower()
    products = "|".join(sorted((p.smiles or p.name or "").lower() for p in step.products))
    reactants = "|".join(sorted((r.smiles or r.name or "").lower() for r in step.reactants))
    return f"{labels}::{reactants}->{products}"


def _step_narrative_text(step: ReactionStep) -> str:
    parts = [
        step.step_label or "",
        step.notes or "",
        step.evidence or "",
        step.yield_value or "",
        step.conditions.other if step.conditions else "",
    ]
    for ref in step.evidence_refs or []:
        parts.append(ref.quote or "")
    return " ".join(parts)


def _is_literature_precedent_step(step: ReactionStep) -> bool:
    """
    Detect steps describing prior/literature work rather than the authors' own experiments.
    Language-agnostic heuristics: past-reported phrasing, alternative cited yields, no quantities.
    """
    narrative = _step_narrative_text(step)
    has_amounts = bool(_AMOUNT_PATTERN.search(narrative))

    if _LITERATURE_NARRATIVE.search(narrative) and not has_amounts:
        return True
    if _MULTIPLE_ALTERNATIVE_YIELDS.search(step.yield_value or ""):
        return True
    if step.evidence and not has_amounts:
        if re.search(r'["\u201c].{10,}["\u201d]', step.evidence) and _LITERATURE_NARRATIVE.search(step.evidence):
            return True
        if re.search(r"\b(has been|have been|was |were )\b", step.evidence, re.I) and not _has_experimental_detail(step):
            return True
    return False


def _is_duplicate_of_experimental_step(step: ReactionStep, steps: list[ReactionStep]) -> bool:
    """True when a non-experimental step repeats a transformation already captured with quantities."""
    if _has_experimental_detail(step):
        return False
    react = _entity_names(step, "reactants")
    if not react:
        return False
    for other in steps:
        if other is step or not _has_experimental_detail(other):
            continue
        other_react = _entity_names(other, "reactants")
        if not (react & other_react):
            continue
        if _products_overlap(step, other):
            return True
        # Same primary reactant + overlapping product family (e.g. acetylation of same substrate)
        if react == other_react or react <= other_react or other_react <= react:
            return True
    return False


def _should_transfer_scheme_yield(base: ReactionStep, scheme: ReactionStep) -> bool:
    """Scheme/route-level yields apply only when products match the target text step."""
    if not scheme.yield_value or scheme.yield_value in ("", "N/A"):
        return False
    base_products = _entity_names(base, "products")
    scheme_products = _entity_names(scheme, "products")
    if not base_products:
        return False
    if not scheme_products:
        return False
    return _products_overlap(base, scheme)


def _merge_scheme_into_text(
    base: ReactionStep,
    scheme: ReactionStep,
    conflicts_out: list[FieldConflict] | None = None,
) -> ReactionStep:
    """Merge VLM scheme evidence into a text step without importing route-level yields."""
    scheme_payload = scheme.model_copy(deep=True)
    if not _should_transfer_scheme_yield(base, scheme_payload):
        scheme_payload.yield_value = "N/A"
    merged = _merge_two_steps(base, scheme_payload, conflicts_out)
    return merged


def _infer_information_kind(step: ReactionStep) -> str:
    if step.information_kind and step.information_kind not in ("", "unknown"):
        return step.information_kind
    label = (step.step_label or "").lower()
    notes = (step.notes or "").lower()
    if _is_scheme_derived(step) or re.search(r"\bfigure\s+\d+", label):
        return "scheme"
    if any(h in label for h in _SUMMARY_LABEL_HINTS) or _LITERATURE_LABEL.search(label):
        return "literature"
    if _is_literature_precedent_step(step):
        return "literature"
    if "literature" in notes or "prior literature" in notes or "cited" in notes:
        return "literature"
    if _has_experimental_detail(step):
        return "explicit"
    if (step.source or "").lower() == "page_read":
        return "inferred"
    return "ambiguous"


def _information_kind_priority(kind: str) -> int:
    return {"explicit": 5, "scheme": 4, "inferred": 3, "ambiguous": 2, "literature": 1, "unknown": 0}.get(kind, 0)


def _infer_is_isolated(step: ReactionStep) -> bool | None:
    if step.is_isolated is not None:
        return step.is_isolated
    label = (step.step_label or "").lower()
    notes = (step.notes or "").lower()
    if _is_literature_precedent_step(step) or step.information_kind == "literature":
        return False
    if "not isolated" in notes or "one-pot" in label or "carried forward" in notes:
        return False
    if step.yield_value not in ("", "N/A") and _has_experimental_detail(step):
        return True
    return None


def _tag_step_metadata(step: ReactionStep) -> None:
    step.information_kind = _infer_information_kind(step)
    step.is_isolated = _infer_is_isolated(step)
    src = (step.source or "text").lower()
    method = {
        "text": "llm_text",
        "table": "llm_table",
        "scheme": "vlm_scheme",
        "page_read": "vlm_page_read",
    }.get(src, "llm_text")
    if not step.extraction_methods:
        step.extraction_methods = [method]
    for ref in step.evidence_refs:
        if not ref.extraction_method:
            ref.extraction_method = method
        if not ref.information_kind:
            ref.information_kind = step.information_kind


def _merge_evidence_refs(base: ReactionStep, extra: ReactionStep) -> None:
    seen = {(r.page, r.source, r.quote[:80]) for r in base.evidence_refs}
    for ref in extra.evidence_refs:
        key = (ref.page, ref.source, (ref.quote or "")[:80])
        if key not in seen:
            base.evidence_refs.append(ref)
            seen.add(key)
    if extra.source and not base.source:
        base.source = extra.source


def _merge_two_steps(
    base: ReactionStep,
    extra: ReactionStep,
    conflicts_out: list[FieldConflict] | None = None,
) -> ReactionStep:
    for bucket in ("reactants", "reagents", "catalysts", "ligands", "solvents", "intermediates", "products"):
        for ent in getattr(extra, bucket):
            _merge_entity_into(getattr(base, bucket), ent)
    if base.yield_value in ("", "N/A") and extra.yield_value not in ("", "N/A"):
        base.yield_value = extra.yield_value
    elif (
        base.yield_value not in ("", "N/A")
        and extra.yield_value not in ("", "N/A")
        and base.yield_value.strip() != extra.yield_value.strip()
    ):
        base.notes = (base.notes + "; " if base.notes else "") + f"yield conflict: {base.yield_value} vs {extra.yield_value}"
        if conflicts_out is not None:
            label = base.step_label or f"step-{base.step_index}"
            conflicts_out.append(
                FieldConflict(
                    field=f"yield:{label}",
                    values=[
                        {
                            "value": base.yield_value,
                            "source": base.source or "unknown",
                            "pages": [r.page for r in base.evidence_refs if r.page],
                            "information_kind": base.information_kind or "unknown",
                        },
                        {
                            "value": extra.yield_value,
                            "source": extra.source or "unknown",
                            "pages": [r.page for r in extra.evidence_refs if r.page],
                            "information_kind": extra.information_kind or "unknown",
                        },
                    ],
                    status="conflicting_or_context_dependent",
                )
            )
    if base.ee_or_dr in ("", "N/A") and extra.ee_or_dr not in ("", "N/A"):
        base.ee_or_dr = extra.ee_or_dr
    if _conditions_empty(base.conditions) and not _conditions_empty(extra.conditions):
        base.conditions = extra.conditions
    else:
        if base.conditions.temperature in ("", "N/A") and extra.conditions.temperature not in ("", "N/A"):
            base.conditions.temperature = extra.conditions.temperature
        if base.conditions.time in ("", "N/A") and extra.conditions.time not in ("", "N/A"):
            base.conditions.time = extra.conditions.time
        if (not base.conditions.other or base.conditions.other == "N/A") and extra.conditions.other not in ("", "N/A"):
            base.conditions.other = extra.conditions.other
    if not base.notes and extra.notes:
        base.notes = extra.notes
    elif extra.notes and extra.notes not in base.notes:
        base.notes = f"{base.notes}; {extra.notes}"
    if not base.evidence and extra.evidence:
        base.evidence = extra.evidence
    elif extra.evidence and extra.evidence not in base.evidence:
        base.evidence = f"{base.evidence}; {extra.evidence}"[:1200]
    _merge_evidence_refs(base, extra)
    for m in extra.extraction_methods or []:
        if m not in (base.extraction_methods or []):
            base.extraction_methods.append(m)
    if _information_kind_priority(extra.information_kind) > _information_kind_priority(base.information_kind):
        base.information_kind = extra.information_kind
    elif base.information_kind in ("", "unknown"):
        base.information_kind = extra.information_kind
    if base.is_isolated is None and extra.is_isolated is not None:
        base.is_isolated = extra.is_isolated
    if (not base.step_label or base.step_label.startswith("Scheme")) and extra.step_label:
        if not extra.step_label.startswith("Scheme"):
            base.step_label = extra.step_label
    # Prefer higher-confidence source
    if extra.confidence and (not base.confidence or extra.confidence > base.confidence):
        base.confidence = extra.confidence
    return base


def _canonical_smiles(smi: str) -> str:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return smi.strip().lower()


def _is_scheme_derived(step: ReactionStep) -> bool:
    return (step.source or "").lower() in ("scheme", "page_read", "vlm")


def _scheme_figure_step(step: ReactionStep) -> bool:
    label = (step.step_label or "").lower()
    return bool(re.search(r"\bfigure\s+\d+", label))


def _collect_merge_fingerprints(step: ReactionStep) -> set[str]:
    fps: set[str] = set()
    for bucket in (
        "reactants",
        "reagents",
        "catalysts",
        "ligands",
        "solvents",
        "intermediates",
        "products",
    ):
        for ent in getattr(step, bucket, []) or []:
            if ent.smiles:
                fps.add(_canonical_smiles(ent.smiles))
            name = (ent.name or "").strip()
            if not name or name in ("N/A", "unknown"):
                continue
            norm = normalize_chemical_name(name)
            fps.add(_normalize_chem_token(norm))
            fps.add(_normalize_chem_token(name))
            base, cid = extract_compound_id(name)
            if cid:
                fps.add(f"id:{cid.lower()}")
            if re.fullmatch(r"\d+[a-z]?", name, re.I):
                fps.add(f"id:{name.lower()}")
            for tok in _normalize_chem_token(name).split():
                if len(tok) >= 4:
                    fps.add(tok)
    return {f for f in fps if f}


def _numeric_ids_in_entities(step: ReactionStep) -> set[str]:
    ids: set[str] = set()
    for bucket in ("reactants", "reagents", "catalysts", "ligands", "solvents", "products"):
        for ent in getattr(step, bucket, []) or []:
            name = (ent.name or "").strip()
            if not name:
                continue
            if re.fullmatch(r"\d+[a-z]?", name, re.I):
                ids.add(name.lower())
            _, cid = extract_compound_id(name)
            if cid:
                ids.add(cid.lower())
            for m in re.finditer(r"\(\s*(\d{1,3}[a-z]?)\b", name, re.I):
                ids.add(m.group(1).lower())
            for m in re.finditer(r"\b(\d{1,3}[a-z]?)\s*,", name, re.I):
                ids.add(m.group(1).lower())
    return ids


def _scheme_chemical_overlap(scheme: ReactionStep, text: ReactionStep) -> int:
    scheme_fps = _collect_merge_fingerprints(scheme)
    text_fps = _collect_merge_fingerprints(text)
    overlap = len(scheme_fps & text_fps)

    scheme_ids = _numeric_ids_in_entities(scheme)
    text_ids = _numeric_ids_in_entities(text)
    overlap += len(scheme_ids & text_ids)

    # Scheme numeric labels (e.g. "5") matched inside text compound names
    for ent in scheme.reactants + scheme.reagents + scheme.catalysts:
        name = (ent.name or "").strip()
        if re.fullmatch(r"\d+[a-z]?", name, re.I):
            sid = name.lower()
            for bucket in ("reactants", "reagents", "catalysts", "products"):
                for text_ent in getattr(text, bucket, []) or []:
                    if re.search(rf"\b{re.escape(sid)}\b", text_ent.name or "", re.I):
                        overlap += 1
                        break

    return overlap


def _should_absorb_scheme_into_text(scheme: ReactionStep, text: ReactionStep) -> bool:
    """True when a VLM scheme step duplicates an existing text step."""
    if not _is_scheme_derived(scheme) or _is_scheme_derived(text):
        return False
    if _is_low_value_step(text):
        return False

    overlap = _scheme_chemical_overlap(scheme, text)
    scheme_has_products = bool(_entity_names(scheme, "products"))
    text_has_products = bool(_entity_names(text, "products"))
    text_has_detail = _has_experimental_detail(text)

    if overlap >= 3:
        return True
    if overlap >= 2 and text_has_detail and (not scheme_has_products or text_has_products):
        return True
    if (
        overlap >= 2
        and _scheme_figure_step(scheme)
        and text_has_detail
        and not scheme_has_products
    ):
        return True
    if overlap >= 1 and _products_overlap(scheme, text):
        return True
    return False


def _absorb_redundant_scheme_steps(
    steps: list[ReactionStep],
    conflicts_out: list[FieldConflict] | None = None,
) -> tuple[list[ReactionStep], list[str]]:
    """Merge VLM scheme steps into matching text steps instead of appending duplicates."""
    warnings: list[str] = []
    schemes = [s for s in steps if _is_scheme_derived(s)]
    if not schemes:
        return steps, warnings

    others = [s for s in steps if not _is_scheme_derived(s)]
    remaining_schemes: list[ReactionStep] = []
    absorbed = 0

    for scheme in schemes:
        best_idx: int | None = None
        best_overlap = 0
        for i, text in enumerate(others):
            if not _should_absorb_scheme_into_text(scheme, text):
                continue
            ov = _scheme_chemical_overlap(scheme, text)
            if ov > best_overlap:
                best_overlap = ov
                best_idx = i

        if best_idx is not None:
            base = others[best_idx]
            prev_source = base.source or "text"
            merged = _merge_scheme_into_text(base, scheme, conflicts_out)
            if (merged.source or "").lower() in ("scheme", "vlm", "page_read"):
                merged.source = prev_source if prev_source not in ("scheme", "vlm", "page_read") else "text"
            others[best_idx] = merged
            absorbed += 1
        else:
            remaining_schemes.append(scheme)

    if absorbed:
        warnings.append(
            f"Merged {absorbed} redundant VLM scheme step(s) into matching text steps."
        )
    return others + remaining_schemes, warnings


def _steps_soft_match(a: ReactionStep, b: ReactionStep) -> bool:
    """Soft merge only when reactants AND products overlap — not product-only."""
    if _is_literature_precedent_step(a) or _is_literature_precedent_step(b):
        return False
    prod_a = {(p.smiles or p.name or "").lower() for p in a.products if (p.smiles or p.name)}
    prod_b = {(p.smiles or p.name or "").lower() for p in b.products if (p.smiles or p.name)}
    react_a = {(r.smiles or r.name or "").lower() for r in a.reactants if (r.smiles or r.name)}
    react_b = {(r.smiles or r.name or "").lower() for r in b.reactants if (r.smiles or r.name)}
    if not prod_a or not prod_b:
        return False
    if not (prod_a & prod_b):
        return False
    if react_a and react_b:
        return bool(react_a & react_b)
    # Same label + shared product
    la = (a.step_label or "").lower().strip()
    lb = (b.step_label or "").lower().strip()
    return bool(la and lb and la == lb)


def _detect_conflicts(steps: list[ReactionStep]) -> list[FieldConflict]:
    conflicts: list[FieldConflict] = []
    by_label: dict[str, list[ReactionStep]] = {}
    for step in steps:
        key = (step.step_label or f"step-{step.step_index}").lower()
        by_label.setdefault(key, []).append(step)
    for label, group in by_label.items():
        if len(group) < 2:
            continue
        yields = {s.yield_value for s in group if s.yield_value not in ("", "N/A")}
        if len(yields) > 1:
            conflicts.append(
                FieldConflict(
                    field=f"yield:{label}",
                    values=[
                        {
                            "value": s.yield_value,
                            "source": s.source or "unknown",
                            "pages": [r.page for r in s.evidence_refs if r.page],
                        }
                        for s in group
                        if s.yield_value not in ("", "N/A")
                    ],
                    status="conflicting_or_context_dependent",
                )
            )
    return conflicts


_SUMMARY_LABEL_HINTS = (
    "abstract-level",
    "summary description",
    "general process",
    "literature variant",
    "literature route",
    "prior literature",
    "this text portion",
    "no detailed procedure",
    "not from experiments",
    "discussion/review",
    "cited from",
)

_LITERATURE_LABEL = re.compile(r"\bliterature\b|\[\d+\]", re.I)

_LITERATURE_NARRATIVE = re.compile(
    r"\b("
    r"has been|have been|was reported|were reported|reported that|reported yield|"
    r"described in|according to|prior work|previously reported|in the literature|"
    r"literature procedure|gives .{0,40} in \d+(?:\.\d+)?\s*%\s*yield|"
    r"has studied|have studied|was described|were described|"
    r"patent|journal of|vol\.\s*\d"
    r")\b",
    re.I,
)

_MULTIPLE_ALTERNATIVE_YIELDS = re.compile(
    r"\d+(?:\.\d+)?\s*%\s*\([^)]+\)\s*;\s*\d+(?:\.\d+)?\s*%",
    re.I,
)

_AMOUNT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:g|mg|mmol|ml|mol|l)\b",
    re.I,
)


def _normalize_chem_token(name: str) -> str:
    n = (name or "").lower().strip()
    n = re.sub(r"\([^)]*\)", "", n)
    n = re.sub(r"\b(compound|intermediate|product|residue|crude)\b", "", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return " ".join(n.split())


def _entity_names(step: ReactionStep, *buckets: str) -> set[str]:
    names: set[str] = set()
    for bucket in buckets:
        for ent in getattr(step, bucket, []) or []:
            tok = _normalize_chem_token(ent.name or "")
            if tok:
                names.add(tok)
    return names


def _is_real_condition_value(val: str) -> bool:
    if not val or val in ("", "N/A"):
        return False
    if re.search(r"\b(studied by|effects of|reported that|preferred scheme)\b", val, re.I):
        return False
    return bool(
        re.search(
            r"\b(\d+\s*°|reflux|rt|room temperature|overnight|\d+\s*(?:h|hr|min|hours?|minutes?|days?))\b",
            val,
            re.I,
        )
    )


def _has_experimental_detail(step: ReactionStep) -> bool:
    entities = (
        step.reactants
        + step.reagents
        + step.catalysts
        + step.products
        + step.solvents
    )
    has_amount = any(
        e.amount and e.amount not in ("", "N/A") and _AMOUNT_PATTERN.search(e.amount)
        for e in entities
    )
    has_cond = bool(
        step.conditions
        and (
            _is_real_condition_value(step.conditions.temperature)
            or _is_real_condition_value(step.conditions.time)
        )
    )
    return has_amount or has_cond


def _step_quality_score(step: ReactionStep) -> float:
    score = 0.0
    if _has_experimental_detail(step):
        score += 3.0
    if step.evidence_refs:
        for ref in step.evidence_refs:
            if ref.evidence_type in ("experimental_procedure", "patent_example", "compound_preparation"):
                score += 1.5
            elif ref.evidence_type in ("abstract_claim", "background", "references", "discussion"):
                score -= 1.0
            if ref.section in ("experimental", "examples"):
                score += 0.5
            elif ref.section in ("abstract", "references", "discussion", "introduction"):
                score -= 0.75
    label = (step.step_label or "").lower()
    notes = (step.notes or "").lower()
    if any(h in label for h in _SUMMARY_LABEL_HINTS) or _LITERATURE_LABEL.search(label):
        score -= 4.0
    if "prior literature" in notes or "not from experiments" in notes:
        score -= 2.0
    if _conditions_empty(step.conditions) and not _has_experimental_detail(step):
        score -= 1.0
    return score


def _is_low_value_step(step: ReactionStep, *, document_type: str = "unknown") -> bool:
    label = (step.step_label or "").lower()
    if any(h in label for h in _SUMMARY_LABEL_HINTS):
        return True
    if _is_literature_precedent_step(step):
        return True
    if "literature variant" in label or (_LITERATURE_LABEL.search(label) and "preparation of" not in label):
        return True
    if re.search(r"\bstep\s+[123]\b", label) and not _has_experimental_detail(step):
        return True
    if "general process" in label and not _has_experimental_detail(step):
        return True
    if "ring opening" in label and not _has_experimental_detail(step):
        return True
    narrative = f"{step.conditions.temperature or ''} {step.conditions.other or ''} {step.evidence or ''}".lower()
    if re.search(r"\b(matsumoto|chen et al|prior literature|reported that|have studied)\b", narrative):
        return True
    refs = step.evidence_refs
    if refs:
        low_sections = {"abstract", "references", "discussion", "introduction", "body"}
        if all((r.section or "body") in low_sections for r in refs) and not _has_experimental_detail(step):
            return True
        low_types = {"abstract_claim", "background", "references", "discussion"}
        if all((r.evidence_type or "") in low_types for r in refs) and not _has_experimental_detail(step):
            return True
    if document_type == "paper" and not _has_experimental_detail(step):
        if any(k in label for k in ("one-pot", "summary", "high yield", "one pot")):
            return True
    return False


def _products_overlap(a: ReactionStep, b: ReactionStep) -> bool:
    prod_a = _entity_names(a, "products")
    prod_b = _entity_names(b, "products")
    if not prod_a or not prod_b:
        return False
    for pa in prod_a:
        for pb in prod_b:
            if pa == pb or pa in pb or pb in pa:
                return True
            # Match on distinctive tokens (e.g. acyclovir, dioxolane-diacetate)
            ta = {t for t in pa.split() if len(t) >= 5}
            tb = {t for t in pb.split() if len(t) >= 5}
            if ta & tb:
                return True
    return False


def _should_merge_steps(a: ReactionStep, b: ReactionStep) -> bool:
    if _is_literature_precedent_step(a) or _is_literature_precedent_step(b):
        return False
    if _steps_soft_match(a, b):
        return True
    la = (a.step_label or "").lower()
    lb = (b.step_label or "").lower()
    if "literature" in la or "literature" in lb:
        return False
    if _is_low_value_step(a) or _is_low_value_step(b):
        return _products_overlap(a, b) and _has_experimental_detail(a) and _has_experimental_detail(b)
    return _products_overlap(a, b) and (
        _entity_names(a, "reactants") & _entity_names(b, "reactants")
        or la == lb
    )


def _filter_low_quality_steps(
    steps: list[ReactionStep],
    *,
    document_type: str = "unknown",
) -> tuple[list[ReactionStep], list[str]]:
    """Drop summary/literature duplicates when a detailed experimental step exists."""
    warnings: list[str] = []
    kept: list[ReactionStep] = []
    dropped = 0
    has_experimental = any(_has_experimental_detail(s) for s in steps)

    for step in steps:
        if _is_literature_precedent_step(step) and has_experimental:
            dropped += 1
            continue
        if has_experimental and _is_duplicate_of_experimental_step(step, steps):
            dropped += 1
            continue
        if not _is_low_value_step(step, document_type=document_type):
            kept.append(step)
            continue
        label = (step.step_label or "").lower()
        if has_experimental and (
            any(h in label for h in _SUMMARY_LABEL_HINTS)
            or "literature variant" in label
            or (_LITERATURE_LABEL.search(label) and "preparation of" not in label)
            or _is_literature_precedent_step(step)
        ):
            dropped += 1
            continue
        if any(
            other is not step
            and _has_experimental_detail(other)
            and (_products_overlap(step, other) or _entity_names(step, "reactants") & _entity_names(other, "reactants"))
            for other in steps
        ):
            dropped += 1
            continue
        kept.append(step)

    # Second pass: among remaining, drop lower-quality product duplicates
    final: list[ReactionStep] = []
    for step in kept:
        dominated = False
        for other in kept:
            if other is step:
                continue
            if _products_overlap(step, other) and _step_quality_score(other) > _step_quality_score(step) + 0.5:
                if _has_experimental_detail(other) and not _has_experimental_detail(step):
                    dominated = True
                    break
        if not dominated:
            final.append(step)
        else:
            dropped += 1

    if dropped:
        warnings.append(f"Filtered {dropped} summary/literature/duplicate step(s) after reconcile.")
    return final, warnings


def _dedupe_steps(
    steps: list[ReactionStep],
    conflicts_out: list[FieldConflict] | None = None,
) -> list[ReactionStep]:
    out: list[ReactionStep] = []
    seen: dict[str, int] = {}
    for step in steps:
        sig = _step_signature(step)
        soft_idx = None
        for i, existing in enumerate(out):
            if _should_merge_steps(step, existing):
                soft_idx = i
                break
        if sig in seen:
            merged = _merge_two_steps(out[seen[sig]], step, conflicts_out)
            if _step_quality_score(step) > _step_quality_score(out[seen[sig]]):
                if step.step_label and "literature" not in (step.step_label or "").lower():
                    merged.step_label = step.step_label
            out[seen[sig]] = merged
        elif soft_idx is not None:
            base = out[soft_idx]
            extra = step
            if _step_quality_score(step) > _step_quality_score(base):
                base, extra = step, base
            merged = _merge_two_steps(base, extra, conflicts_out)
            if _step_quality_score(base) >= _step_quality_score(extra):
                merged.step_label = base.step_label or merged.step_label
            out[soft_idx] = merged
        else:
            seen[sig] = len(out)
            out.append(step)
    return out


def reconcile_route(
    text_steps: list[ReactionStep] | None = None,
    table_steps: list[ReactionStep] | None = None,
    scheme_steps: list[ReactionStep] | None = None,
    page_read_steps: list[ReactionStep] | None = None,
    scheme_molecules: list[dict[str, Any]] | None = None,
    *,
    title: str = "",
    target_name: str = "",
    target_class: str = "unknown",
    document_type: str = "unknown",
    warnings: list[str] | None = None,
    stats_out: dict[str, Any] | None = None,
) -> SynthesisRoute:
    """
    Merge sources with priority: text > page_read > table > scheme.
    Scheme molecules dicts are only a legacy fallback.
    """
    all_steps: list[ReactionStep] = []

    if text_steps:
        all_steps.extend(text_steps)
    if page_read_steps:
        all_steps.extend(page_read_steps)
    if table_steps:
        all_steps.extend(table_steps)
    if scheme_steps:
        all_steps.extend(scheme_steps)

    # Legacy fallback: molecule list without structured scheme steps
    if not all_steps and scheme_molecules:
        from app.services.image_extract import ExtractedMolecule, ExtractedScheme, schemes_to_reaction_steps

        fake = ExtractedScheme(
            page_number=0,
            image_index=0,
            molecules=[
                ExtractedMolecule(
                    smiles=m.get("smiles", ""),
                    role=m.get("role"),
                    confidence=float(m.get("confidence") or 0.5),
                    source=m.get("source", "vlm"),
                    name=m.get("name"),
                )
                for m in scheme_molecules
                if m.get("smiles") or m.get("name")
            ],
            conditions_text="",
            mode="scheme",
            image_class="scheme",
        )
        all_steps.extend(schemes_to_reaction_steps([fake]))

    if not all_steps:
        return SynthesisRoute(
            title=title,
            target_name=target_name or "N/A",
            target_class=target_class or "unknown",
            document_type=document_type or "unknown",
            steps=[],
            warnings=warnings or [],
        )

    for step in all_steps:
        _enrich_step_from_conditions_text(step)
        _tag_step_metadata(step)

    input_counts = {
        "text": len(text_steps or []),
        "table": len(table_steps or []),
        "scheme": len(scheme_steps or []),
        "page_read": len(page_read_steps or []),
    }
    scheme_absorbed = 0

    merge_conflicts: list[FieldConflict] = []

    all_steps, scheme_warnings = _absorb_redundant_scheme_steps(all_steps, merge_conflicts)
    scheme_absorbed = sum(1 for w in scheme_warnings if "Merged" in w and "scheme" in w)
    pre_dedupe = len(all_steps)
    unified = _dedupe_steps(all_steps, merge_conflicts)
    duplicates_merged = pre_dedupe - len(unified)
    unified, filter_warnings = _filter_low_quality_steps(unified, document_type=document_type or "unknown")
    merged_warnings = list(warnings or []) + scheme_warnings + filter_warnings

    # Mark intermediates: product of n used as reactant of n+1
    for i in range(len(unified) - 1):
        next_reactants = {(e.smiles or e.name or "").lower() for e in unified[i + 1].reactants}
        for ent in unified[i].products:
            key = (ent.smiles or ent.name or "").lower()
            if key and key in next_reactants:
                ent.role = "intermediate"
                _merge_entity_into(unified[i].intermediates, ent)

    for i, step in enumerate(unified, start=1):
        step.step_index = i
        if not step.step_label:
            step.step_label = f"Step {i}"

    # Target: last step product preferred
    resolved_target = target_name
    if not resolved_target or resolved_target in ("", "N/A"):
        for step in reversed(unified):
            for p in step.products:
                if p.name and p.name != "product":
                    resolved_target = p.name
                    break
            if resolved_target and resolved_target not in ("", "N/A"):
                break

    route = SynthesisRoute(
        title=title or (unified[0].step_label if unified else ""),
        target_name=resolved_target or "N/A",
        target_class=target_class or "unknown",
        document_type=document_type or "unknown",
        steps=unified,
        warnings=merged_warnings,
        conflicts=_detect_conflicts(unified) + merge_conflicts,
    )
    if stats_out is not None:
        with_prov = sum(
            1 for s in unified if s.evidence_refs or (s.evidence and len(s.evidence) > 20)
        )
        stats_out.update(
            {
                "input_text_steps": input_counts["text"],
                "input_table_steps": input_counts["table"],
                "input_scheme_steps": input_counts["scheme"],
                "input_page_read_steps": input_counts["page_read"],
                "merged_steps": len(unified),
                "scheme_steps_absorbed": scheme_absorbed,
                "duplicate_steps_merged": duplicates_merged,
                "conflicts": len(route.conflicts),
                "steps_with_provenance": with_prov,
                "steps_without_provenance": len(unified) - with_prov,
            }
        )
    return route

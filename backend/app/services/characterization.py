"""Extract and attach figure characterization / yield data to reaction steps."""

from __future__ import annotations

import re
from typing import Any

from app.schemas.reaction_route import ChemicalEntity, ReactionStep

_YIELD_RE = re.compile(
    r"(?i)\b(?:yield\s*[:=]?\s*)?(\d{1,3}(?:\.\d+)?\s*%|\d{1,3}(?:\.\d+)?\s*percent)\b"
)
_COMPOUND_ID_RE = re.compile(
    r"(?i)\b(?:compound\s+)?(\d{1,3})\s*[.\-]?\s*([a-z])\b|\b(\d{1,3})\b"
)


def normalize_compound_token(text: str | None) -> set[str]:
    """
    Expand a label like '2. a, b' / 'ACV-D-Val(2a)' / '3a' into match tokens.
    Tokens are lowercased ids such as '2a', '2b', '2', '3a'.
    """
    if not text:
        return set()
    s = text.strip().lower()
    tokens: set[str] = set()
    # Explicit (2a) / 2a patterns
    for m in re.finditer(r"\((\d{1,3})\s*([a-z])\)|(\d{1,3})\s*([a-z])\b", s):
        num = m.group(1) or m.group(3)
        letter = m.group(2) or m.group(4)
        if num and letter:
            tokens.add(f"{num}{letter}")
            tokens.add(num)
    # Pattern: "2. a, b" or "2. a, b, c, d"
    m = re.match(r"^(\d{1,3})\s*[.\-]?\s*([a-z](?:\s*,\s*[a-z])*)$", s)
    if m:
        num = m.group(1)
        letters = re.findall(r"[a-z]", m.group(2))
        tokens.add(num)
        for let in letters:
            tokens.add(f"{num}{let}")
    # Bare integer
    if re.fullmatch(r"\d{1,3}", s):
        tokens.add(s)
    # Also keep a compact alnum slug of the whole name for soft match
    slug = re.sub(r"[^a-z0-9]+", "", s)
    if slug:
        tokens.add(slug)
    return tokens


def parse_yield_string(raw: str | None) -> str | None:
    if not raw or str(raw).strip().upper() in {"", "N/A", "NA", "NULL", "NONE"}:
        return None
    text = str(raw).strip()
    m = _YIELD_RE.search(text)
    if m:
        val = m.group(1).strip()
        if "percent" in val.lower():
            val = re.sub(r"(?i)\s*percent", "%", val)
        if not val.endswith("%"):
            val = f"{val}%"
        return val.replace(" ", "")
    # Already a plain percent-like string
    if re.fullmatch(r"\d{1,3}(?:\.\d+)?%?", text):
        return text if text.endswith("%") else f"{text}%"
    return text


def characterization_tokens(item: dict[str, Any]) -> set[str]:
    parts = [
        str(item.get("compound_id") or ""),
        str(item.get("name") or ""),
        str(item.get("label") or ""),
    ]
    out: set[str] = set()
    for p in parts:
        out |= normalize_compound_token(p)
        # Pull ids embedded in names: ACV-D-Val(2a)
        for m in re.finditer(r"\((\d{1,3}[a-z]?)\)", p, flags=re.I):
            out |= normalize_compound_token(m.group(1))
    return {t for t in out if t}


def _entity_tokens(ent: ChemicalEntity) -> set[str]:
    return normalize_compound_token(ent.name)


def _step_product_tokens(step: ReactionStep) -> set[str]:
    tokens: set[str] = set()
    for ent in list(step.products) + list(step.intermediates):
        tokens |= _entity_tokens(ent)
    # Also mine step_label for 2a / LLACV etc.
    tokens |= normalize_compound_token(step.step_label)
    return tokens


def apply_characterizations_to_steps(
    steps: list[ReactionStep],
    characterizations: list[dict[str, Any]] | None,
) -> int:
    """
    Attach yield / LC-MS / notes from figure characterizations onto matching steps.
    Returns number of yield fields updated.
    """
    if not steps or not characterizations:
        return 0

    updated = 0

    def _apply_item_to_step(step: ReactionStep, item: dict[str, Any], yld: str | None) -> None:
        nonlocal updated
        name = (item.get("name") or item.get("compound_id") or "").strip()
        lcms = (item.get("lcms") or item.get("ms") or "").strip()
        notes = (item.get("notes") or "").strip()
        tokens = characterization_tokens(item)

        for ent in list(step.products) + list(step.intermediates):
            if tokens & _entity_tokens(ent) or (name and name.lower() in (ent.name or "").lower()):
                if yld and (not ent.amount or ent.amount in ("N/A", "")):
                    ent.amount = f"yield {yld}"
                elif yld and "yield" not in (ent.amount or "").lower():
                    ent.amount = f"{ent.amount}; yield {yld}"
                if name and re.fullmatch(r"[\d.\s,a-z]+", (ent.name or "").strip(), flags=re.I):
                    ent.name = name
                extra_bits = []
                if lcms:
                    extra_bits.append(f"LC/MS {lcms}")
                if notes and notes.lower() not in (ent.notes or "").lower():
                    extra_bits.append(notes[:240])
                if extra_bits:
                    joined = "; ".join(extra_bits)
                    ent.notes = f"{ent.notes}; {joined}".strip("; ") if ent.notes else joined

        # If no matching product entity, add one from the characterization
        if name and not any(
            tokens & _entity_tokens(e) or name.lower() in (e.name or "").lower()
            for e in step.products + step.intermediates
        ):
            step.products.append(
                ChemicalEntity(
                    name=name,
                    role="product",
                    amount=f"yield {yld}" if yld else "N/A",
                    notes="; ".join(
                        x
                        for x in [
                            f"LC/MS {lcms}" if lcms else "",
                            notes[:240] if notes else "",
                        ]
                        if x
                    ),
                )
            )

        if yld:
            label = name or (item.get("compound_id") or "product")
            entry = f"{label}: {yld}"
            if not step.yield_value or step.yield_value in ("N/A", ""):
                step.yield_value = entry
                updated += 1
            elif yld not in step.yield_value and entry not in step.yield_value:
                step.yield_value = f"{step.yield_value}; {entry}"
                updated += 1

        bits = []
        if lcms:
            bits.append(f"LC/MS {lcms}")
        if notes:
            bits.append(notes[:240])
        add = "; ".join(bits)
        if add and add not in (step.notes or ""):
            step.notes = f"{step.notes}; {add}".strip("; ") if step.notes else add

    def _step_families(step: ReactionStep) -> set[str]:
        fams: set[str] = set()
        for tok in _step_product_tokens(step):
            m = re.match(r"(\d+)", tok)
            if m:
                fams.add(m.group(1))
        return fams

    def _char_family(item: dict[str, Any]) -> str | None:
        for tok in characterization_tokens(item):
            m = re.match(r"(\d+)[a-z]?$", tok)
            if m:
                return m.group(1)
        return None

    # Prefer final multi-product step for late compound families (3x prodrugs)
    multi_product_steps = [s for s in steps if len(s.products) >= 2]

    for item in characterizations:
        yld = parse_yield_string(item.get("yield_value") or item.get("yield"))
        name = (item.get("name") or item.get("compound_id") or "").strip()
        tokens = characterization_tokens(item)
        if not tokens and not name:
            continue

        matched_steps: list[ReactionStep] = []
        for step in steps:
            step_tokens = _step_product_tokens(step)
            if tokens & step_tokens:
                matched_steps.append(step)
                continue
            if name and any(name.lower() in (e.name or "").lower() for e in step.products + step.intermediates):
                matched_steps.append(step)

        if not matched_steps:
            for step in steps:
                if tokens & normalize_compound_token(step.step_label):
                    matched_steps.append(step)
                    break

        if not matched_steps:
            fam = _char_family(item)
            if fam:
                for step in steps:
                    if fam in _step_families(step):
                        matched_steps.append(step)
                        break
            if not matched_steps and fam == "2":
                # Mono-val / early intermediate: prefer deprotection / Boc / intermediate steps
                for step in steps:
                    label = (step.step_label or "").lower()
                    if "deprotect" in label or "intermediate" in label or "boc" in label:
                        matched_steps = [step]
                        break
                if not matched_steps and len(steps) >= 2:
                    matched_steps = [steps[min(2, len(steps) - 1)]]
            elif not matched_steps and fam and fam not in {"1", "2"} and multi_product_steps:
                # Later compound families (3x prodrugs, etc.) → multi-product / final step
                matched_steps = [multi_product_steps[-1]]
            elif not matched_steps and fam and fam not in {"1", "2"}:
                matched_steps = [steps[-1]]

        if not matched_steps:
            matched_steps = [steps[-1]]

        for step in matched_steps:
            _apply_item_to_step(step, item, yld)

    return updated


def is_coarse_scheme_step(step: ReactionStep) -> bool:
    """
    True for low-value scheme skeleton steps like 1 → 2.a,b with no reagents.
    These duplicate detailed legend-derived text steps and should not be appended.
    """
    has_reagents = bool(step.reagents or step.catalysts or step.solvents)
    if has_reagents:
        return False
    names = [e.name for e in step.reactants + step.products + step.intermediates]
    if not names:
        return True
    # All names look like scheme compound numbers
    return all(bool(normalize_compound_token(n)) and re.fullmatch(r"[\d.\s,a-z/]+", (n or "").strip(), flags=re.I) for n in names)

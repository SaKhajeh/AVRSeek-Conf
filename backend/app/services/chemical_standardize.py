from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any

import pubchempy as pcp

from app.config import Settings
from app.services.llm_client import get_openai_client

EntityProgressFn = Callable[[int, int, str], None]


# --- Time standardization (adapted from ReactionSeek) ---

TIME_CONVERSION_PROMPT = """You will be provided with reaction time expressions from chemical procedures. Convert each to minutes (numeric or range).
Rules:
- "overnight" → "720 minutes" (12 hours)
- "rt" or "room temperature" refers to temperature, not time — ignore unless paired with duration
- "reflux", "heat", "boil" are temperatures, not times — ignore unless duration given
- "1 h", "1 hour", "1 hr" → "60 minutes"
- "30 min", "30 minutes" → "30 minutes"
- "1-2 hours" → "60-120 minutes"
- "2 days" → "2880 minutes"
- "30 min then 2 h" → "30 minutes, 120 minutes"
- If uncertain or no time info → "N/A"
- Output ONLY a JSON array of objects: [{"index": 1, "time_minutes": "60 minutes"}, ...]
- Index matches input order (1-based).

Example input:
1. "stirred overnight"
2. "refluxed for 3 hours"
3. "at 0 °C for 30 min then warmed to rt for 2 h"
4. "no time mentioned"

Example output:
[
  {"index": 1, "time_minutes": "720 minutes"},
  {"index": 2, "time_minutes": "180 minutes"},
  {"index": 3, "time_minutes": "30 minutes, 120 minutes"},
  {"index": 4, "time_minutes": "N/A"}
]
"""


def standardize_reaction_times(route, settings: Settings, model: str | None = None) -> int:
    """
    Standardize reaction time strings to minutes for all steps in a route.
    Returns count of updated steps.
    """
    from app.schemas.reaction_route import SynthesisRoute
    if not isinstance(route, SynthesisRoute):
        return 0

    client = get_openai_client(settings)
    use_model = model or settings.model_standardize

    # Collect time strings with step indices
    time_entries = []
    for step in route.steps:
        t = step.conditions.time
        if t and t != "N/A":
            time_entries.append((step.step_index, t))

    if not time_entries:
        return 0

    # Build prompt
    prompt_lines = [TIME_CONVERSION_PROMPT, "\nInput:"]
    for idx, (step_idx, time_str) in enumerate(time_entries, 1):
        prompt_lines.append(f"{idx}. \"{time_str}\"")
    prompt = "\n".join(prompt_lines)

    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        results = json.loads(cleaned)

        # Map back to steps
        updated = 0
        for item in results:
            idx = item.get("index")
            time_min = item.get("time_minutes")
            if idx and time_min and 1 <= idx <= len(time_entries):
                step_idx, _ = time_entries[idx - 1]
                # Find the step and update
                for step in route.steps:
                    if step.step_index == step_idx:
                        step.conditions.time = time_min
                        updated += 1
                        break
        return updated
    except Exception:
        return 0


def pubchem_lookup(name: str) -> str | None:
    """Query PubChem for isomeric SMILES by name."""
    try:
        compounds = pcp.get_compounds(name, "name")
        if compounds:
            smi = compounds[0].isomeric_smiles
            if smi:
                return smi
    except Exception:
        pass
    return None


def opsin_lookup(name: str) -> str | None:
    """Query OPSIN CLI for SMILES by name. Requires OPSIN JAR on PATH or OPSIN_JAR env var."""
    jar_path = os.environ.get("OPSIN_JAR", "opsin-cli-2.8.0-jar-with-dependencies.jar")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f_in:
            f_in.write(name)
            in_path = f_in.name
        out_path = in_path + ".out"
        try:
            subprocess.run(
                ["java", "-jar", jar_path, "-osmi", in_path, out_path],
                check=True,
                capture_output=True,
                timeout=30,
            )
            with open(out_path, "r") as f_out:
                smi = f_out.readline().strip()
            if smi and smi.lower() != "null":
                return smi
        finally:
            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except Exception:
                    pass
    except Exception:
        pass
    return None


def llm_resolve_name(name: str, settings: Settings, model: str | None = None) -> str | None:
    """Use LLM to map a messy/common name to a systematic chemical name, then retry PubChem/OPSIN."""
    client = get_openai_client(settings)
    use_model = model or settings.model_standardize

    prompt = f"""You are a chemical nomenclature expert. Convert the following substance name to a systematic IUPAC or common chemical name that PubChem or OPSIN can resolve.
Return ONLY a JSON object: {{"name": "systematic_name"}} or {{"name": null}} if uncertain.
Do not guess. If the input refers to a mixture (e.g., petroleum ether) or a class (e.g., alkyl halide), return null.

Input: {name}
"""
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        return data.get("name")
    except Exception:
        return None


from app.services.chemical_synonyms import SYNONYM_MAP, extract_compound_id, normalize_chemical_name


def resolve_smiles(name: str, settings: Settings, model: str | None = None) -> str | None:
    smi, _status, _source = resolve_smiles_with_meta(name, settings, model)
    return smi


def resolve_smiles_with_meta(
    name: str, settings: Settings, model: str | None = None
) -> tuple[str | None, str, str]:
    """
    Resolution cascade with metadata. Returns (smiles, resolution_status, resolution_source).
    Never fabricates identifiers — returns unresolved when lookup fails.
    """
    candidates: list[str] = []
    used_synonym = False
    if name:
        raw = name.strip()
        candidates.append(raw)
        norm = normalize_chemical_name(raw)
        if norm and norm not in candidates:
            candidates.append(norm)
        key = re.sub(r"\s+", " ", raw.lower())
        if norm and norm.lower() != key:
            used_synonym = True
        elif key in SYNONYM_MAP:
            used_synonym = True
        base, _cid = extract_compound_id(raw)
        if base and base not in candidates:
            candidates.append(base)
        if base:
            norm_base = normalize_chemical_name(base)
            if norm_base and norm_base not in candidates:
                candidates.append(norm_base)

    for candidate in candidates:
        smi = pubchem_lookup(candidate)
        if smi:
            status = "synonym" if used_synonym else "exact"
            return smi, status, "pubchem"
        smi = opsin_lookup(candidate)
        if smi:
            status = "synonym" if used_synonym else "exact"
            return smi, status, "opsin"

    for candidate in candidates:
        systematic = llm_resolve_name(candidate, settings, model)
        if systematic and systematic not in candidates:
            smi = pubchem_lookup(systematic)
            if smi:
                return smi, "structure", "pubchem"
            smi = opsin_lookup(systematic)
            if smi:
                return smi, "structure", "opsin"

    return None, "unresolved", "none"


def _entities_needing_smiles(route) -> list[tuple[Any, Any]]:
    """Return (step, entity) pairs lacking SMILES."""
    from app.schemas.reaction_route import SynthesisRoute

    if not isinstance(route, SynthesisRoute):
        return []
    pending: list[tuple[Any, Any]] = []
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
                if ent.smiles:
                    continue
                name = (ent.name or "").strip()
                if not name or name in ("N/A", "unknown"):
                    continue
                pending.append((step, ent))
    return pending


def standardize_route_smiles(
    route,
    settings: Settings,
    model: str | None = None,
    on_entity_progress: EntityProgressFn | None = None,
) -> tuple[int, int, int, dict[str, int]]:
    """
    Resolve SMILES for all ChemicalEntity objects in a SynthesisRoute that lack them.
    Returns (resolved_count, failed_count, skipped_count, resolution_status_counts).
    """
    from app.schemas.reaction_route import SynthesisRoute
    if not isinstance(route, SynthesisRoute):
        return 0, 0, 0, {}

    pending = _entities_needing_smiles(route)
    total = len(pending)
    skipped = sum(
        1
        for step in route.steps
        for bucket in (
            step.reactants,
            step.reagents,
            step.catalysts,
            step.ligands,
            step.solvents,
            step.intermediates,
            step.products,
        )
        for ent in bucket
        if ent.smiles
    )

    resolved = failed = 0
    status_counts: dict[str, int] = {}
    for idx, (_step, ent) in enumerate(pending, start=1):
        if on_entity_progress:
            label = (ent.name or "unknown")[:80]
            on_entity_progress(idx, total, label)
        if not ent.original_name:
            ent.original_name = ent.name
        norm = normalize_chemical_name(ent.name or "")
        if norm and norm != ent.name:
            ent.canonical_name = norm
        smi, status, source = resolve_smiles_with_meta(ent.name, settings, model)
        ent.resolution_status = status
        ent.resolution_source = source
        status_counts[status] = status_counts.get(status, 0) + 1
        if smi:
            ent.smiles = smi
            resolved += 1
        else:
            failed += 1
    return resolved, failed, skipped, status_counts


def validate_smiles_with_rdkit(route) -> tuple[int, int]:
    """
    Validate existing SMILES with RDKit, canonicalize them.
    Returns (valid_count, invalid_count).
    """
    try:
        from rdkit import Chem
    except Exception:
        return 0, 0

    valid = invalid = 0
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
                if not ent.smiles:
                    continue
                mol = Chem.MolFromSmiles(ent.smiles)
                if mol is None:
                    invalid += 1
                else:
                    ent.smiles = Chem.MolToSmiles(mol)
                    valid += 1
    return valid, invalid
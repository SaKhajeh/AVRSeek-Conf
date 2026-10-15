"""Chemical name synonym normalization before SMILES resolution (P1)."""

from __future__ import annotations

import re

from app.schemas.reaction_route import SynthesisRoute

# Common lab abbreviations → PubChem-resolvable names (generic, not molecule-specific)
SYNONYM_MAP: dict[str, str] = {
    "acv": "acyclovir",
    "ac2o": "acetic anhydride",
    "acetic anhydride (ac2o)": "acetic anhydride",
    "p-tsa": "p-toluenesulfonic acid",
    "p-tsoh": "p-toluenesulfonic acid",
    "p-toluenesulfonic acid monohydrate": "p-toluenesulfonic acid",
    "ptsa": "p-toluenesulfonic acid",
    "tsoh": "p-toluenesulfonic acid",
    "tea": "triethylamine",
    "et3n": "triethylamine",
    "dmf": "N,N-dimethylformamide",
    "thf": "tetrahydrofuran",
    "meoh": "methanol",
    "etoh": "ethanol",
    "nh4oh": "ammonium hydroxide",
    "ammonium hydroxide (30%)": "ammonium hydroxide",
    "h3po4": "phosphoric acid",
    "pd/c": "palladium on carbon",
    "dioxolane-diacetate": "1-acetyl-2-acetylmethoxyethyl acetate",
    "1-acetyl-2-acetylmethoxy-ethyleneglycol": "1-acetyl-2-acetylmethoxyethyl acetate",
}

_COMPOUND_NUM = re.compile(r"^(.+?)\s*[\(\[]\s*(\d{1,3}[a-z]?)\s*[\)\]]\s*$", re.I)
_TRAILING_ROLE = re.compile(r"\s*[\(\[]\s*(residue|crude|solid|from step [a-z0-9]+)\s*[\)\]]\s*$", re.I)


def extract_compound_id(name: str) -> tuple[str, str | None]:
    """Return (cleaned_name, compound_id) from strings like 'guanine (3)'."""
    if not name:
        return name, None
    m = _COMPOUND_NUM.match(name.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return name.strip(), None


def normalize_chemical_name(name: str) -> str:
    if not name or name in ("N/A", "unknown"):
        return name
    base, _cid = extract_compound_id(name)
    base = _TRAILING_ROLE.sub("", base).strip()
    key = base.lower()
    key = re.sub(r"\s+", " ", key)
    if key in SYNONYM_MAP:
        return SYNONYM_MAP[key]
    # Partial: if name starts with a known synonym token
    for abbrev, full in SYNONYM_MAP.items():
        if key == abbrev or key.startswith(abbrev + " "):
            return full
    return base


def apply_synonym_normalization(route: SynthesisRoute) -> int:
    """Normalize entity names in-place; returns count of names changed."""
    changed = 0
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
                if not ent.name:
                    continue
                norm = normalize_chemical_name(ent.name)
                if norm and norm != ent.name:
                    if not ent.original_name:
                        ent.original_name = ent.name
                    ent.canonical_name = norm
                    if ent.resolution_status in ("", "unresolved"):
                        ent.resolution_status = "synonym"
                        ent.resolution_source = ent.resolution_source or "synonym_layer"
                    ent.name = norm
                    changed += 1
    return changed

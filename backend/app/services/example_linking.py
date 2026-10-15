"""Patent Example ↔ intermediate / compound-ID linking (P2)."""

from __future__ import annotations

import re

from app.schemas.reaction_route import CompoundLink, SynthesisRoute
from app.services.chemical_synonyms import extract_compound_id
from app.services.characterization import normalize_compound_token

_EXAMPLE_LABEL = re.compile(r"\b(example|preparation|step)\s+([A-D0-9ivx]+)\b", re.I)


def _example_id(label: str) -> str | None:
    m = _EXAMPLE_LABEL.search(label or "")
    return f"{m.group(1).lower()} {m.group(2).upper()}" if m else None


def link_route_compounds(route: SynthesisRoute) -> list[CompoundLink]:
    """Link compound numeric IDs and example labels across steps."""
    links: list[CompoundLink] = []
    id_to_steps: dict[str, list[int]] = {}
    example_to_steps: dict[str, list[int]] = {}

    for step in route.steps:
        ex = _example_id(step.step_label)
        if ex:
            example_to_steps.setdefault(ex, []).append(step.step_index)
        for bucket_name, bucket in (
            ("reactants", step.reactants),
            ("products", step.products),
            ("intermediates", step.intermediates),
        ):
            for ent in bucket:
                _, cid = extract_compound_id(ent.name)
                if cid:
                    id_to_steps.setdefault(cid.lower(), []).append(step.step_index)
                    links.append(
                        CompoundLink(
                            compound_id=cid,
                            name=ent.name,
                            step_indices=[step.step_index],
                            role=ent.role or bucket_name.rstrip("s"),
                            example_label=ex or "",
                        )
                    )
                for tok in normalize_compound_token(ent.name):
                    if tok.isdigit() or re.fullmatch(r"\d+[a-z]", tok):
                        id_to_steps.setdefault(tok, []).append(step.step_index)

    # Merge duplicate compound IDs
    merged: dict[str, CompoundLink] = {}
    for link in links:
        key = link.compound_id.lower()
        if key in merged:
            for si in link.step_indices:
                if si not in merged[key].step_indices:
                    merged[key].step_indices.append(si)
        else:
            merged[key] = link

    # Cross-link: same compound_id appearing in multiple steps
    for cid, steps in id_to_steps.items():
        steps = sorted(set(steps))
        if len(steps) > 1:
            merged.setdefault(
                cid,
                CompoundLink(compound_id=cid, name="", step_indices=steps, role="intermediate", example_label=""),
            )

    route.compound_links = list(merged.values())
    return route.compound_links

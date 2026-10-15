from __future__ import annotations

"""Pipeline stage stubs for image mining, reconcile, and standardize."""

from typing import Any, Callable

from app.config import Settings
from app.schemas.reaction_route import PipelineStage, PipelineStageStatus, SynthesisRoute
from app.services.chemical_standardize import (
    _entities_needing_smiles,
    standardize_route_smiles,
    validate_smiles_with_rdkit,
    standardize_reaction_times,
)


def stub_image_scheme_extract(figure_count: int) -> PipelineStage:
    return PipelineStage(
        id="image_scheme",
        label="Scheme / image extraction (OCSR)",
        status=PipelineStageStatus.stubbed,
        detail=(
            f"Detected ~{figure_count} embedded images. "
            "RxnScribe/MolScribe + VLM role labeling will plug in here."
        ),
        data={"figures_detected": figure_count, "schemes": []},
    )


def stub_reconcile(route: SynthesisRoute | None) -> PipelineStage:
    step_n = len(route.steps) if route else 0
    return PipelineStage(
        id="reconcile",
        label="Reconcile text ↔ image ↔ tables",
        status=PipelineStageStatus.stubbed,
        detail=f"Placeholder merge pass over {step_n} text-mined step(s).",
        data={"merged_steps": step_n},
    )


def run_standardize(
    route: SynthesisRoute | None,
    settings: Settings,
    model: str | None = None,
    on_entity_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[PipelineStage, SynthesisRoute | None]:
    """
    Real chemical standardization:
    1. Resolve missing SMILES via PubChem → OPSIN → LLM rename cascade
    2. Validate & canonicalize all SMILES with RDKit (if available)
    3. Standardize reaction times to minutes
    """
    if route is None:
        return (
            PipelineStage(
                id="standardize",
                label="Chemical standardization",
                status=PipelineStageStatus.skipped,
                detail="No route to standardize.",
                data={"resolved": 0, "validated": 0, "invalid": 0, "unresolved": 0, "times_standardized": 0},
            ),
            None,
        )

    pending = _entities_needing_smiles(route) if route else []
    if on_entity_progress and pending:
        on_entity_progress(0, len(pending), "starting…")

    # 1. Resolve missing SMILES
    resolved, failed, skipped, resolution_counts = standardize_route_smiles(
        route, settings, model, on_entity_progress=on_entity_progress
    )

    # 2. Validate with RDKit
    if on_entity_progress:
        on_entity_progress(len(pending), max(len(pending), 1), "RDKit validation…")
    valid, invalid = validate_smiles_with_rdkit(route)

    # 3. Standardize reaction times
    if on_entity_progress:
        on_entity_progress(len(pending), max(len(pending), 1), "normalizing reaction times…")
    times_updated = standardize_reaction_times(route, settings, model)

    unresolved = failed
    total_entities = resolved + failed + skipped

    detail_parts = []
    if resolved:
        detail_parts.append(f"Resolved {resolved} name(s) to SMILES (PubChem/OPSIN/LLM)")
    if skipped:
        detail_parts.append(f"{skipped} already had SMILES")
    if valid:
        detail_parts.append(f"RDKit validated {valid} SMILES")
    if invalid:
        detail_parts.append(f"RDKit rejected {invalid} invalid SMILES")
    if unresolved:
        detail_parts.append(f"{unresolved} name(s) could not be resolved")
    if times_updated:
        detail_parts.append(f"Standardized {times_updated} reaction time(s) to minutes")

    detail = "; ".join(detail_parts) if detail_parts else "No chemical entities to process"

    data = {
        "resolved": resolved,
        "validated": valid,
        "invalid": invalid,
        "unresolved": unresolved,
        "total_entities": total_entities,
        "times_standardized": times_updated,
        "resolution_status_counts": resolution_counts,
    }

    status = PipelineStageStatus.done if (resolved or valid or times_updated) else PipelineStageStatus.stubbed

    return (
        PipelineStage(
            id="standardize",
            label="Chemical standardization",
            status=status,
            detail=detail,
            data=data,
        ),
        route,
    )

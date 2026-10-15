from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineStageStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    stubbed = "stubbed"
    skipped = "skipped"
    error = "error"


class PipelineStage(BaseModel):
    id: str
    label: str
    status: PipelineStageStatus = PipelineStageStatus.pending
    detail: str = ""
    data: Optional[dict[str, Any]] = None


class Amount(BaseModel):
    raw: str = "N/A"
    value: Optional[float] = None
    unit: Optional[str] = None


class ChemicalEntity(BaseModel):
    name: str
    role: str = Field(
        description="reactant | reagent | catalyst | ligand | solvent | intermediate | product | other"
    )
    amount: Optional[str] = "N/A"
    smiles: Optional[str] = None
    cas: Optional[str] = None
    notes: Optional[str] = None
    evidence: Optional[str] = None
    # Normalization / resolution (never fabricate identifiers)
    original_name: Optional[str] = None
    canonical_name: Optional[str] = None
    resolution_status: str = Field(
        default="unresolved",
        description="exact | synonym | structure | ambiguous | unresolved",
    )
    resolution_source: Optional[str] = None  # pubchem | opsin | synonym_layer | rdkit | none
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    molecular_formula: Optional[str] = None
    # Information classification
    information_kind: str = Field(
        default="unknown",
        description="explicit | scheme | inferred | literature | ambiguous | unknown",
    )
    is_isolated: Optional[bool] = None
    provenance_source: str = ""  # native_text | ocr | table | scheme | vlm | inferred | literature
    page: Optional[int] = None
    confidence: Optional[float] = None


class EvidenceRef(BaseModel):
    """Provenance for an extracted fact or step."""

    page: Optional[int] = None
    page_end: Optional[int] = None
    section: str = ""
    evidence_type: str = ""  # experimental_procedure | reaction_scheme | table | ...
    source: str = ""  # text | table | scheme | page_read | native_text | ocr | vlm
    quote: str = ""
    confidence: Optional[float] = None
    image_id: Optional[str] = None
    extraction_method: str = ""  # llm_text | llm_table | vlm_scheme | vlm_page_read | inferred
    information_kind: str = ""  # explicit | scheme | inferred | literature | ambiguous


class FieldConflict(BaseModel):
    """Conflicting values for the same field from different sources."""

    field: str
    values: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "conflicting_or_context_dependent"


class RouteGraphNode(BaseModel):
    node_id: str
    name: str = ""
    canonical_name: str = ""
    compound_ids: list[str] = Field(default_factory=list)
    step_indices: list[int] = Field(default_factory=list)
    role: str = "unknown"


class RouteGraphEdge(BaseModel):
    from_node: str
    to_node: str
    step_index: int = 0
    relation: str = "transforms_to"


class RouteGraph(BaseModel):
    nodes: list[RouteGraphNode] = Field(default_factory=list)
    edges: list[RouteGraphEdge] = Field(default_factory=list)
    target_node_id: str = ""


class CompoundLink(BaseModel):
    compound_id: str
    name: str = ""
    step_indices: list[int] = Field(default_factory=list)
    role: str = ""
    example_label: str = ""


class ReactionConditions(BaseModel):
    temperature: str = "N/A"
    time: str = "N/A"
    pressure: str = "N/A"
    atmosphere: str = "N/A"
    other: str = "N/A"


class ReactionStep(BaseModel):
    step_index: int
    step_label: str = ""
    reactants: list[ChemicalEntity] = Field(default_factory=list)
    reagents: list[ChemicalEntity] = Field(default_factory=list)
    catalysts: list[ChemicalEntity] = Field(default_factory=list)
    ligands: list[ChemicalEntity] = Field(default_factory=list)
    solvents: list[ChemicalEntity] = Field(default_factory=list)
    intermediates: list[ChemicalEntity] = Field(default_factory=list)
    products: list[ChemicalEntity] = Field(default_factory=list)
    conditions: ReactionConditions = Field(default_factory=ReactionConditions)
    yield_value: str = "N/A"
    ee_or_dr: str = "N/A"
    notes: str = ""
    evidence: str = ""
    source: str = ""  # primary modality: text | table | scheme | page_read
    confidence: Optional[float] = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    information_kind: str = Field(
        default="unknown",
        description="explicit | scheme | inferred | literature | ambiguous | unknown",
    )
    is_isolated: Optional[bool] = None
    workup: str = ""
    purification: str = ""
    extraction_methods: list[str] = Field(default_factory=list)


class SynthesisRoute(BaseModel):
    title: str = ""
    target_name: str = ""
    target_class: str = Field(
        default="",
        description="e.g. nucleoside analog, protease inhibitor, polymerase inhibitor",
    )
    document_type: str = Field(default="paper", description="paper | patent | unknown")
    steps: list[ReactionStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    route_graph: RouteGraph | None = None
    compound_links: list[CompoundLink] = Field(default_factory=list)


class DocumentSegment(BaseModel):
    segment_id: str
    kind: str
    page_start: int = 1
    page_end: int = 1
    text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    filename: str
    page_count: int = 0
    text: str = ""
    segments: list[DocumentSegment] = Field(default_factory=list)
    figure_count: int = 0
    table_hints: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    job_id: str
    filename: str
    stages: list[PipelineStage]
    document: Optional[ParsedDocument] = None
    route: Optional[SynthesisRoute] = None
    raw_model_output: Optional[str] = None
    error: Optional[str] = None
    from_cache: bool = False
    content_hash: Optional[str] = None
    source_job_id: Optional[str] = None
    extraction_debug: Optional[dict[str, Any]] = None

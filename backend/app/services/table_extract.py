from __future__ import annotations

"""Table extraction from PDFs using pdfplumber + LLM for reaction data."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pdfplumber

from app.config import Settings
from app.prompts.table_extract import TABLE_PARSING_PROMPT
from app.schemas.reaction_route import ChemicalEntity, EvidenceRef, PipelineStage, PipelineStageStatus, ReactionStep
from app.services.llm_client import get_openai_client


# --- Data structures ---

class ExtractedTable:
    """A table extracted from a PDF page."""
    def __init__(
        self,
        page_number: int,
        table_index: int,
        headers: list[str],
        rows: list[list[str]],
        raw_text: str = "",
        bbox: tuple[float, float, float, float] | None = None,
    ):
        self.page_number = page_number
        self.table_index = table_index
        self.headers = headers
        self.rows = rows
        self.raw_text = raw_text
        self.bbox = bbox

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers:
            return ""
        md = "| " + " | ".join(self.headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(self.headers)) + " |\n"
        for row in self.rows:
            md += "| " + " | ".join(row) + " |\n"
        return md

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "headers": self.headers,
            "rows": self.rows,
            "bbox": self.bbox,
        }


# --- Table extraction from PDF ---

def extract_tables_from_pdf(pdf_path: Path) -> list[ExtractedTable]:
    """
    Extract all tables from a PDF using pdfplumber.
    Returns list of ExtractedTable objects.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            for table_idx, table in enumerate(page_tables):
                if not table or len(table) < 2:
                    continue
                headers = [str(cell).strip() if cell else "" for cell in table[0]]
                rows = []
                for row in table[1:]:
                    rows.append([str(cell).strip() if cell else "" for cell in row])
                # Get bbox if available
                bbox = None
                try:
                    table_bbox = page.find_tables()[table_idx].bbox
                    bbox = table_bbox
                except Exception:
                    pass

                raw_text = "\n".join(["\t".join(row) for row in [headers] + rows])
                tables.append(ExtractedTable(
                    page_number=page_num,
                    table_index=table_idx,
                    headers=headers,
                    rows=rows,
                    raw_text=raw_text,
                    bbox=bbox,
                ))
    return tables


def filter_reaction_tables(tables: list[ExtractedTable]) -> list[ExtractedTable]:
    """
    Filter tables that likely contain reaction data (yields, conditions, etc.).
    """
    reaction_keywords = [
        "yield", "yield%", "% yield", "yield (%)",
        "temperature", "temp", "°c", "celsius",
        "time", "hour", "min", "hr", "h ",
        "solvent", "reagent", "catalyst", "equivalent", "equiv",
        "amount", "mmol", "mol", "g ", "mg ", "ml ", "mL ",
        "product", "reactant", "substrate", "intermediate",
        "entry", "example", "run ", "experiment",
    ]

    filtered = []
    for table in tables:
        text = table.raw_text.lower()
        score = sum(1 for kw in reaction_keywords if kw in text)
        if score >= 2:  # At least 2 reaction-related keywords
            filtered.append(table)
    return filtered


# --- LLM-based table parsing ---


def parse_table_with_llm(
    table: ExtractedTable,
    settings: Settings,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Parse a table using LLM to extract reaction steps.
    Returns list of reaction step dicts.
    """
    client = get_openai_client(settings)
    use_model = model or settings.model_text

    prompt = f"{TABLE_PARSING_PROMPT}\n\nTable (markdown):\n{table.to_markdown()}"

    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=120,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
    except Exception as e:
        return [{"error": f"LLM parsing failed: {e}", "raw_table": table.to_markdown()}]


# --- Convert parsed table data to ReactionStep ---

def table_data_to_reaction_steps(
    table_data: list[dict[str, Any]],
    table: ExtractedTable,
) -> list[ReactionStep]:
    """Convert LLM-parsed table data to ReactionStep objects."""
    steps = []
    for i, entry in enumerate(table_data):
        if "error" in entry:
            continue
        entry_type = str(entry.get("entry_type") or "procedure").lower()
        if entry_type in ("characterization", "other"):
            continue
        if entry_type == "screen":
            # Keep screens but mark in notes — do not skip entirely (may hold conditions)
            pass

        step = ReactionStep(
            step_index=i + 1,
            step_label=entry.get("step_label", f"Table row {i+1} (page {table.page_number})"),
            reactants=[],
            reagents=[],
            catalysts=[],
            ligands=[],
            solvents=[],
            intermediates=[],
            products=[],
            conditions=entry.get("conditions", {}),
            yield_value=entry.get("yield_value", "N/A"),
            ee_or_dr=entry.get("ee_or_dr", "N/A"),
            notes=(entry.get("notes") or "") + (f" [table {entry_type}]" if entry_type != "procedure" else ""),
            evidence=table.raw_text[:800],
            source="table",
            confidence=0.75 if entry_type == "procedure" else 0.55,
            evidence_refs=[
                EvidenceRef(
                    page=table.page_number,
                    section="tables",
                    evidence_type="reaction_table",
                    source="table",
                    quote=table.raw_text[:400],
                    confidence=0.75,
                )
            ],
        )

        # Convert entity dicts to ChemicalEntity
        for bucket_name in ("reactants", "reagents", "catalysts", "ligands", "solvents", "products", "intermediates"):
            bucket = getattr(step, bucket_name)
            for ent_dict in entry.get(bucket_name, []):
                bucket.append(ChemicalEntity(
                    name=ent_dict.get("name", "Unknown"),
                    role=ent_dict.get("role", bucket_name.rstrip("s")),
                    amount=ent_dict.get("amount", "N/A"),
                    smiles=None,
                    cas=None,
                ))

        steps.append(step)
    return steps


# --- Pipeline stage function ---

async def run_table_extraction(
    pdf_path: Path,
    settings: Settings,
    model: str | None = None,
) -> tuple[PipelineStage, list[ReactionStep]]:
    """
    Run table extraction as a pipeline stage.
    Returns (PipelineStage, list of ReactionStep from tables).
    """
    all_tables = extract_tables_from_pdf(pdf_path)
    reaction_tables = filter_reaction_tables(all_tables)

    if not reaction_tables:
        return (
            PipelineStage(
                id="table_extract",
                label="Table extraction (pdfplumber + LLM)",
                status=PipelineStageStatus.stubbed,
                detail=f"No reaction-like tables found in {len(all_tables)} total tables",
                data={"tables_total": len(all_tables), "reaction_tables": 0, "steps_extracted": 0},
            ),
            [],
        )

    all_steps = []
    for table in reaction_tables:
        table_data = await asyncio.to_thread(parse_table_with_llm, table, settings, model)
        steps = table_data_to_reaction_steps(table_data, table)
        all_steps.extend(steps)

    return (
        PipelineStage(
            id="table_extract",
            label="Table extraction (pdfplumber + LLM)",
            status=PipelineStageStatus.done if all_steps else PipelineStageStatus.stubbed,
            detail=f"Extracted {len(all_steps)} step(s) from {len(reaction_tables)} reaction table(s) out of {len(all_tables)} total tables",
            data={
                "tables_total": len(all_tables),
                "reaction_tables": len(reaction_tables),
                "steps_extracted": len(all_steps),
                "tables": [t.to_dict() for t in reaction_tables],
            },
        ),
        all_steps,
    )
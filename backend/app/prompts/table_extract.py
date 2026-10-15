"""Prompts for reaction-table parsing."""

from __future__ import annotations

from app.prompts.shared import ANTI_HALLUCINATION, JSON_OUTPUT_RULES, ROLE_VOCABULARY

TABLE_PARSING_PROMPT = f"""You are AVRSeek analyzing a table from an antiviral / medicinal chemistry paper or patent.
{ANTI_HALLUCINATION}
{ROLE_VOCABULARY}
{JSON_OUTPUT_RULES}

Tables are often optimization or SAR screens for the SAME transformation, not a multi-step synthetic route.
Classify each row and extract structured fields. Do not invent a sequential route from screen entries.

Return ONLY a JSON array. Each element:
{{
  "entry_type": "procedure | screen | characterization | other",
  "step_label": "Entry 1 / Example 1 / row description",
  "reactants": [{{"name": "...", "amount": "...", "role": "reactant"}}],
  "reagents": [{{"name": "...", "amount": "...", "role": "reagent"}}],
  "catalysts": [{{"name": "...", "amount": "...", "role": "catalyst"}}],
  "ligands": [{{"name": "...", "amount": "...", "role": "ligand"}}],
  "solvents": [{{"name": "...", "amount": "...", "role": "solvent"}}],
  "intermediates": [],
  "products": [{{"name": "...", "amount": "...", "role": "product"}}],
  "conditions": {{
    "temperature": "...",
    "time": "...",
    "pressure": "...",
    "atmosphere": "...",
    "other": "..."
  }},
  "yield_value": "...",
  "ee_or_dr": "...",
  "notes": "..."
}}

Mapping hints:
- Entry / Run / Example → step_label
- Cat. / Catalyst → catalysts; Ligand / L → ligands
- Solvent → solvents; Base often → reagents unless clearly catalytic
- T / Temp → conditions.temperature; t / Time → conditions.time
- Yield / % → yield_value; ee/dr → ee_or_dr
- entry_type=screen for catalyst/solvent/condition optimization rows with yields
- entry_type=procedure only if the row clearly encodes a full preparative procedure
- entry_type=characterization for NMR/HRMS/HPLC-only rows → usually skip (do not emit) unless chemically essential
- Skip header/summary rows
- If no reaction-relevant rows: return []
"""

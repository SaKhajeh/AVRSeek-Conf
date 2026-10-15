"""Antiviral multi-step reaction extraction prompts (ReactionSeek-style 4-part design)."""

from __future__ import annotations

SYSTEM_PROMPT = """You are AVRSeek, an expert chemist specializing in small-molecule antiviral drug synthesis.
Extract only information grounded in the provided text. Never invent compounds, amounts, or conditions.
If a field is missing or uncertain, use "N/A". Prefer structured JSON only — no markdown fences.
Prefer common chemical names (e.g. acyclovir, monoethanolamine) over Formula I/II alone when the text provides both.
Capture temperature, time, pH, yield, and purity when stated in Examples."""

TASK_DESCRIPTION = """Task: From the scientific paper or patent text below, extract multi-step small-molecule synthesis knowledge relevant to antiviral drugs or their intermediates/APIs.
Focus on concrete experimental procedures / patent Examples (Example 1, Example A–D, Preparation) — not Markush genus claims unless a specific worked example is given.
Resolve co-references between step titles/subheadings and procedure body when possible.
Group related operations (e.g. deprotection A–B, salt purification C–D) as separate steps with clear labels."""

TASK_REQUIREMENTS = """Return a single JSON object with this exact schema:
{
  "title": "string",
  "target_name": "final product or key antiviral candidate if stated, else N/A",
  "target_class": "e.g. nucleoside analog | protease inhibitor | polymerase inhibitor | entry inhibitor | unknown",
  "document_type": "paper | patent | unknown",
  "steps": [
    {
      "step_index": 1,
      "step_label": "short label or Example number (e.g. Example 1 / Steps A–B)",
      "reactants": [{"name": "...", "role": "reactant", "amount": "...", "smiles": null}],
      "reagents": [{"name": "...", "role": "reagent", "amount": "..."}],
      "catalysts": [{"name": "...", "role": "catalyst", "amount": "..."}],
      "ligands": [{"name": "...", "role": "ligand", "amount": "..."}],
      "solvents": [{"name": "...", "role": "solvent", "amount": "..."}],
      "intermediates": [{"name": "...", "role": "intermediate", "amount": "..."}],
      "products": [{"name": "...", "role": "product", "amount": "..."}],
      "conditions": {
        "temperature": "...",
        "time": "...",
        "pressure": "...",
        "atmosphere": "N2/Ar/air/N/A",
        "other": "pH, workup, drying, etc."
      },
      "yield_value": "...",
      "ee_or_dr": "...",
      "notes": "purity or process notes",
      "evidence": "short quote from source"
    }
  ],
  "warnings": ["optional notes about ambiguity"]
}

Rules:
- One object in steps[] per distinct reaction / Example / unit operation.
- Documents may be in any language — extract names/amounts/conditions as written; add common English names when the text provides them.
- Include neutralizing acids, bases, charcoal, wash solvents when they are part of the described process step.
- Prefer reaction medium solvents; still record isolation solvents in conditions.other if important to the example.
- Separate catalysts/ligands from stoichiometric reagents when the text allows.
- Intermediates of step n often become reactants of step n+1 — keep names consistent.
- Capture yield AND purity when both appear (purity may go in notes).
- Output JSON only."""

PROCESSING_SUGGESTIONS = """Heuristics:
- Amounts usually sit near compound names (g, mg, mmol, mL, mol%, equiv).
- Temperature may be numeric (°C) or qualitative (reflux, rt, ice bath); ranges like 70–95°C matter.
- Time often appears late in the condition narrative (overnight, 12 h, 3 days).
- Patent text: prefer numbered Examples / Preparations over Claims.
- Antiviral context cues: nucleoside, protease, polymerase, RdRp, NS5A/NS3, prodrug, phosphoramidate, guanine, acyclovir.
- Salt formation / neutralization / recrystallization can be separate steps if they are distinct Examples.
- If the text is only background (no procedure), return steps: [] and explain in warnings."""

FEW_SHOT_EXAMPLE = """Example input (abbreviated):
Title: Synthesis of an antiviral nucleoside intermediate
Procedure: To a solution of compound 1 (2.0 g, 5.0 mmol) in anhydrous DMF (20 mL) under N2 was added TBDMSCl (1.1 g, 7.5 mmol) and imidazole (1.0 g, 14.7 mmol). The mixture was stirred at 0 °C for 1 h then at room temperature for 12 h. Aqueous workup and chromatography afforded intermediate 2 (2.1 g, 82%). Intermediate 2 (1.0 g) in MeOH was treated with Pd/C (10 wt%, 50 mg) under H2 at rt for 6 h to give product 3 (0.72 g, 75%).

Example output:
{
  "title": "Synthesis of an antiviral nucleoside intermediate",
  "target_name": "product 3",
  "target_class": "nucleoside analog",
  "document_type": "paper",
  "steps": [
    {
      "step_index": 1,
      "step_label": "silylation to intermediate 2",
      "reactants": [{"name": "compound 1", "role": "reactant", "amount": "2.0 g, 5.0 mmol"}],
      "reagents": [
        {"name": "TBDMSCl", "role": "reagent", "amount": "1.1 g, 7.5 mmol"},
        {"name": "imidazole", "role": "reagent", "amount": "1.0 g, 14.7 mmol"}
      ],
      "catalysts": [],
      "ligands": [],
      "solvents": [{"name": "DMF", "role": "solvent", "amount": "20 mL"}],
      "intermediates": [],
      "products": [{"name": "intermediate 2", "role": "product", "amount": "2.1 g"}],
      "conditions": {"temperature": "0 °C then rt", "time": "1 h + 12 h", "pressure": "N/A", "atmosphere": "N2", "other": "N/A"},
      "yield_value": "82%",
      "ee_or_dr": "N/A",
      "notes": "",
      "evidence": "TBDMSCl/imidazole in DMF; chromatography afforded intermediate 2 (82%)"
    },
    {
      "step_index": 2,
      "step_label": "hydrogenolysis to product 3",
      "reactants": [{"name": "intermediate 2", "role": "reactant", "amount": "1.0 g"}],
      "reagents": [],
      "catalysts": [{"name": "Pd/C", "role": "catalyst", "amount": "10 wt%, 50 mg"}],
      "ligands": [],
      "solvents": [{"name": "MeOH", "role": "solvent", "amount": "N/A"}],
      "intermediates": [],
      "products": [{"name": "product 3", "role": "product", "amount": "0.72 g"}],
      "conditions": {"temperature": "rt", "time": "6 h", "pressure": "N/A", "atmosphere": "H2", "other": "N/A"},
      "yield_value": "75%",
      "ee_or_dr": "N/A",
      "notes": "",
      "evidence": "Pd/C under H2 in MeOH afforded product 3 (75%)"
    }
  ],
  "warnings": []
}"""


def build_user_prompt(title: str, text: str) -> str:
    return f"""{TASK_DESCRIPTION}

{TASK_REQUIREMENTS}

{PROCESSING_SUGGESTIONS}

{FEW_SHOT_EXAMPLE}

/////
Now extract from this document.

Title:<'''{title}'''>
Text:<'''{text}'''>
"""

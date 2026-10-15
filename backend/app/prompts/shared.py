"""Shared prompt policy for AVRSeek LLM stages."""

from __future__ import annotations

ROLE_VOCABULARY = """Chemical roles (use exactly these strings):
- reactant: substrate undergoing the transformation
- reagent: stoichiometric reactant/additive (acids, bases, coupling agents, protecting-group reagents)
- catalyst: catalytic species (often mol% / cat.)
- ligand: ligand for a metal catalyst (keep separate from the metal catalyst when possible)
- solvent: reaction medium only (NOT wash / extraction / chromatography eluent unless it is clearly the reaction solvent)
- intermediate: isolable intermediate in a multi-step sequence (when labeled as such)
- product: main isolated product of that step
- other: only if none of the above fit"""

ANTI_HALLUCINATION = """Anti-hallucination rules (mandatory):
1. Use ONLY information present in the provided input. Do not invent compounds, amounts, yields, stereochemistry, or conditions.
2. Do NOT fill gaps from prior knowledge of commercial antivirals (e.g. remdesivir, sofosbuvir, nirmatrelvir, oseltamivir) or "typical" reaction conditions.
3. If a field is missing or uncertain, use "N/A" (or null where the schema allows). Prefer omission over guessing.
4. Prefer verbatim compound labels from the source (1, 1a, I-3, Intermediate A) over inventing IUPAC names.
5. Never treat wash, brine, extraction, recrystallization, or chromatography solvents as reaction solvents unless the text clearly uses them as the reaction medium.
6. Biology-only content (assays, IC50/EC50, docking, PK) is not synthesis — do not fabricate reaction steps from it.
7. Markush / genus claim language without a concrete Example is not a worked procedure — do not expand R-groups into specific molecules."""

ANTIVIRAL_CONTEXT = """Antiviral / small-molecule medicinal chemistry context:
- Focus on concrete synthesis of small-molecule antivirals, APIs, key intermediates, and process chemistry Examples.
- Useful cues when explicitly present: nucleoside/tide, ProTide/phosphoramidate, protease, polymerase/RdRp, integrase, entry/fusion, capsid, NS3/NS5A/NS5B, HIV, HCV, influenza, SARS-CoV-2, HBV, RSV.
- target_class (pick best fit if stated or clearly implied by the text; else "unknown"):
  nucleoside_analog | nucleotide_prodrug | protease_inhibitor | polymerase_inhibitor | integrase_inhibitor | entry_inhibitor | capsid_modulator | other_small_molecule | unknown
- virus_family: only if explicitly stated; else "N/A" or "unknown"."""

JSON_OUTPUT_RULES = """Output rules:
- Return valid JSON only (no markdown fences, no commentary outside JSON).
- Keep evidence fields as short verbatim quotes from the source when possible.
- Put ambiguities, co-reference doubts, and scope issues in warnings (or notes), not by inventing data."""

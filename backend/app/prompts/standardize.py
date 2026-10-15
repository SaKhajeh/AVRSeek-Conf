"""Prompts for standardization helpers (time + name cleanup)."""

from __future__ import annotations

TIME_CONVERSION_PROMPT = """You will be provided with reaction time expressions from chemical procedures. Convert each to minutes (numeric or range).
Rules:
- "overnight" → "720 minutes" (12 hours)
- "rt" or "room temperature" refers to temperature, not time — ignore unless paired with duration
- "reflux", "heat", "boil" are temperatures, not times — ignore unless duration given
- "1 h", "1 hour", "1 hr" → "60 minutes"
- "30 min", "30 minutes" → "30 minutes"
- "1-2 hours" → "60-120 minutes"
- "2 days" → "2880 minutes"
- "several days" → "N/A" (too vague) unless a numeric range is given
- "until complete by TLC/HPLC" or similar monitoring language → "N/A"
- "30 min then 2 h" → "30 minutes, 120 minutes"
- If uncertain or no time info → "N/A"
- Do not invent durations from chemistry knowledge.
- Output ONLY a JSON array of objects: [{"index": 1, "time_minutes": "60 minutes"}, ...]
- Index matches input order (1-based).

Example input:
1. "stirred overnight"
2. "refluxed for 3 hours"
3. "at 0 °C for 30 min then warmed to rt for 2 h"
4. "stirred until complete by TLC"
5. "no time mentioned"

Example output:
[
  {"index": 1, "time_minutes": "720 minutes"},
  {"index": 2, "time_minutes": "180 minutes"},
  {"index": 3, "time_minutes": "30 minutes, 120 minutes"},
  {"index": 4, "time_minutes": "N/A"},
  {"index": 5, "time_minutes": "N/A"}
]
"""

# Double braces {{ }} so str.format only substitutes {name}, not JSON examples.
NAME_RESOLVE_PROMPT = """You are a chemical nomenclature expert helping AVRSeek resolve names for database lookup (PubChem/OPSIN).
Convert the substance name to a systematic IUPAC or common chemical name that PubChem or OPSIN can resolve.
Return ONLY a JSON object: {{"name": "systematic_or_common_name"}} or {{"name": null}} if uncertain.

Rules:
- Do not guess. If unsure, return null.
- Mixtures (petroleum ether, hexanes grade mixes) → null.
- Compound classes without a specific member (alkyl halide, "a nucleoside") → null.
- Markush or formula references ("compound of formula I", "the title compound") → null unless a concrete structure name is present in the input itself.
- Catalog/code-only labels ("compound 12", "I-3a") → null (cannot invent structure).
- Well-known specific reagents/solvents/catalysts may be expanded to a resolvable synonym (e.g. "TEA" → "triethylamine") only when unambiguous.
- Never invent antiviral drug structures from therapeutic class alone.

Input: {name}
"""

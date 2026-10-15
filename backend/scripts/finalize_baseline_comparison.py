"""Finalize comparison: AvalAI baselines + 055 fallback for parse failures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    dest = OUTPUTS / "batch_4bed65c03d57_comparison.json"
    data = json.loads(dest.read_text(encoding="utf-8"))
    fallback = json.loads(
        (OUTPUTS / "batch_055d69aa8b80_comparison.json").read_text(encoding="utf-8")
    )
    fb_map = {
        row["avrseek"]["filename"]: row.get("baseline")
        for row in fallback["comparisons"]
        if row.get("baseline")
    }

    for row in data["comparisons"]:
        fn = row["avrseek"]["filename"]
        if row.get("baseline"):
            continue
        if "baseline failed" in row.get("baseline_note", "") and fn in fb_map:
            row["baseline"] = fb_map[fn]
            row["baseline_note"] = "Opus 5 single-shot (prior run; AvalAI JSON parse error on retry)"
            row["baseline_source"] = "batch_055d69aa8b80_fallback"

    data["notes"] = (
        "AVRSeek batch 4bed65c03d57; baseline = Opus 5 single-shot via AvalAI where successful, "
        "with prior-run fallback for large patents with JSON parse errors."
    )
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Corpus comparison (11 documents)\n"]
    md.append("| Document | AVRSeek | Baseline | Source |")
    md.append("|----------|---------|----------|--------|")
    for row in data["comparisons"]:
        a = row["avrseek"]
        b = row.get("baseline")
        bs = b["steps_count"] if b else "—"
        src = row.get("baseline_source") or row.get("baseline_note", "")[:30]
        md.append(f"| {a['filename']} | {a['steps_count']} | {bs} | {src} |")

    (ROOT.parent / "paper" / "CORPUS_COMPARISON.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Updated {dest}")
    n = sum(1 for r in data["comparisons"] if r.get("baseline"))
    print(f"Baselines: {n}/11")


if __name__ == "__main__":
    main()

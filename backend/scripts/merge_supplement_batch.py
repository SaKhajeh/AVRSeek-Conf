"""Merge AVRSeek results from supplemental batch into full corpus comparison."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

spec = importlib.util.spec_from_file_location(
    "compare_batch_baseline", ROOT / "scripts" / "compare_batch_baseline.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
avrseek_summary = mod.avrseek_summary


def main() -> None:
    supplement_id = sys.argv[1] if len(sys.argv) > 1 else "9b11aa996cae"
    main_id = sys.argv[2] if len(sys.argv) > 2 else "4bed65c03d57"

    supplement = json.loads(
        (OUTPUTS / "batches" / supplement_id / "batch.json").read_text(encoding="utf-8")
    )
    updates = {r["filename"]: avrseek_summary(r) for r in supplement["results"]}

    dest = OUTPUTS / f"batch_{main_id}_comparison.json"
    data = json.loads(dest.read_text(encoding="utf-8"))

    for row in data["comparisons"]:
        fn = row["avrseek"]["filename"]
        if fn in updates:
            row["avrseek"] = updates[fn]
            row["avrseek"]["batch_source"] = supplement_id

    data["notes"] = (
        f"Corpus comparison merged from batches {main_id} + {supplement_id}. "
        "Baseline = Opus 5 single-shot via AvalAI (with 055 fallback where JSON parse failed)."
    )
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Corpus comparison (11 documents)\n"]
    md.append("| Document | AVRSeek | Baseline | RG | SMILES |")
    md.append("|----------|---------|----------|-----|--------|")
    total_a = total_b = 0
    n_both = 0
    for row in data["comparisons"]:
        a = row["avrseek"]
        b = row.get("baseline")
        bs = b["steps_count"] if b else "—"
        if b and a["steps_count"] > 0:
            n_both += 1
            total_a += a["steps_count"]
            total_b += b["steps_count"]
        md.append(
            f"| {a['filename']} | {a['steps_count']} | {bs} | {a['route_graph_nodes']} | {a['smiles_entities']} |"
        )
    md.append(f"\n**Aggregate (both systems, n={n_both}):** AVRSeek {total_a} steps, baseline {total_b} steps")
    (ROOT.parent / "paper" / "CORPUS_COMPARISON.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Merged {len(updates)} files from {supplement_id} into {dest}")
    for fn, s in updates.items():
        print(f"  {fn}: {s['steps_count']} steps, rg={s['route_graph_nodes']}")


if __name__ == "__main__":
    main()

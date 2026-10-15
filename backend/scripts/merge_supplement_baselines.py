"""Merge baselines from supplement batch into main corpus comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
main = ROOT / "outputs" / "batch_4bed65c03d57_comparison.json"
supp = ROOT / "outputs" / "batch_9b11aa996cae_comparison.json"

data = json.loads(main.read_text(encoding="utf-8"))
updates = {
    row["avrseek"]["filename"]: row
    for row in json.loads(supp.read_text(encoding="utf-8"))["comparisons"]
}

for row in data["comparisons"]:
    fn = row["avrseek"]["filename"]
    if fn not in updates:
        continue
    s = updates[fn]
    if s.get("baseline"):
        row["baseline"] = s["baseline"]
        row["baseline_note"] = s["baseline_note"]
        row["baseline_source"] = s.get("baseline_source", "avalai_run")

main.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

md = ["# Corpus comparison (11 documents)\n"]
md.append("| Document | AVRSeek | Baseline | RG | SMILES |")
md.append("|----------|---------|----------|-----|--------|")
ta = tb = 0
nb = 0
for row in data["comparisons"]:
    a, b = row["avrseek"], row.get("baseline")
    bs = b["steps_count"] if b else "—"
    if b and a["steps_count"] > 0:
        nb += 1
        ta += a["steps_count"]
        tb += b["steps_count"]
    md.append(f"| {a['filename']} | {a['steps_count']} | {bs} | {a['route_graph_nodes']} | {a['smiles_entities']} |")
md.append(f"\n**Both systems (n={nb}):** AVRSeek {ta} steps, baseline {tb} steps")
(ROOT.parent / "paper" / "CORPUS_COMPARISON.md").write_text("\n".join(md), encoding="utf-8")
print(f"Baselines merged; {nb} docs with both, AVRSeek {ta} vs baseline {tb}")

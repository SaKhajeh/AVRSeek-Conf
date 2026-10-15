"""Print SS baseline summary from multi_model JSON files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MM = ROOT / "outputs" / "multi_model"
COMP = ROOT / "outputs" / "batch_4bed65c03d57_comparison.json"


def main() -> None:
    comp = json.loads(COMP.read_text(encoding="utf-8"))
    doc_order = [c["avrseek"]["filename"] for c in comp["comparisons"]]
    opus_ss = {
        c["avrseek"]["filename"]: (c.get("baseline") or {}).get("steps_count", 0)
        for c in comp["comparisons"]
    }
    opus_fp = {c["avrseek"]["filename"]: c["avrseek"]["steps_count"] for c in comp["comparisons"]}

    print("label\tok\tss_total\terrors")
    print(
        f"opus5\t{sum(1 for v in opus_ss.values() if v>0)}/11\t{sum(opus_ss.values())}\t0"
    )
    models: dict[str, dict] = {}
    for path in sorted(MM.glob("baseline_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = data["model_label"]
        docs = {d["filename"]: d for d in data["documents"]}
        models[label] = docs
        ok = sum(1 for d in data["documents"] if d.get("baseline"))
        err = sum(1 for d in data["documents"] if d.get("error"))
        tot = sum(d["baseline"]["steps_count"] for d in data["documents"] if d.get("baseline"))
        print(f"{label}\t{ok}/11\t{tot}\t{err}")

    print("\nfilename\topus_ss\topus_fp", end="")
    for label in sorted(models):
        print(f"\t{label}", end="")
    print()
    for fn in doc_order:
        row = [fn, str(opus_ss.get(fn, 0)), str(opus_fp.get(fn, 0))]
        for label in sorted(models):
            d = models[label].get(fn)
            if not d:
                row.append("-")
            elif d.get("baseline"):
                row.append(str(d["baseline"]["steps_count"]))
            elif d.get("error"):
                row.append("ERR")
            else:
                row.append("0")
        print("\t".join(row))


if __name__ == "__main__":
    main()

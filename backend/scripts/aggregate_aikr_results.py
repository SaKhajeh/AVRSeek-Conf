"""Aggregate multi-model SS baselines + Opus5 FP from corpus comparison."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "outputs" / "batch_4bed65c03d57_comparison.json"
MM_DIR = ROOT / "outputs" / "multi_model"

MODEL_ORDER = [
    ("opus5", "claude-opus-5"),
    ("ds-v4-flash", "deepseek-v4-flash"),
    ("ds-reasoner", "deepseek-reasoner"),
    ("qwen36-flash", "qwen3.6-flash"),
]

# Friendly doc keys for tables
DOC_ORDER = [
    "pdf.pdf",
    "shiragami1995.pdf",
    "jm00146a002.pdf",
    "36_1153.pdf",
    "EP0532878A2.pdf",
    "EP0709385A1.pdf",
    "EP0862570B1.pdf",
    "US5567816A.pdf",
    "US5756737A.pdf",
    "WO_9718211_A1.pdf",
    "RU2111967C1.pdf",
]


def load_opus_ss_fp() -> dict[str, dict]:
    data = json.loads(COMP.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for row in data["comparisons"]:
        fn = row["avrseek"]["filename"]
        avr = row["avrseek"]
        bl = row.get("baseline") or {}
        out[fn] = {
            "fp_steps": avr.get("steps_count", 0),
            "fp_smiles": avr.get("smiles_entities", 0),
            "fp_rg": avr.get("route_graph_nodes", 0),
            "fp_provenance": avr.get("provenance_steps", 0),
            "fp_complete": avr.get("pipeline_complete", False),
            "ss_steps": bl.get("steps_count", 0),
            "ss_model": bl.get("model"),
        }
    return out


def load_ss_baseline(label: str) -> dict[str, int]:
    path = MM_DIR / f"baseline_{label}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for doc in data.get("documents", []):
        fn = doc["filename"]
        b = doc.get("baseline")
        if b:
            out[fn] = b.get("steps_count", 0)
        elif doc.get("error"):
            out[fn] = -1
        else:
            out[fn] = 0
    return out


def main() -> None:
    opus = load_opus_ss_fp()
    ss_by_label: dict[str, dict[str, int]] = {}
    for label, _mid in MODEL_ORDER:
        if label == "opus5":
            ss_by_label[label] = {fn: opus[fn]["ss_steps"] for fn in DOC_ORDER if fn in opus}
        else:
            ss_by_label[label] = load_ss_baseline(label)

    fp_by_label: dict[str, dict] = {"opus5": opus}
    # Other FP from baseline_*.json if we store fp runs separately later
    for label, _ in MODEL_ORDER:
        fp_path = MM_DIR / f"pipeline_{label}.json"
        if fp_path.exists():
            fp_by_label[label] = json.loads(fp_path.read_text(encoding="utf-8"))

    report = {
        "documents": DOC_ORDER,
        "single_shot_steps": ss_by_label,
        "opus_fp": {fn: opus[fn] for fn in DOC_ORDER if fn in opus},
        "totals_ss": {label: sum(v for v in d.values() if v >= 0) for label, d in ss_by_label.items()},
        "totals_fp_opus": sum(opus[fn]["fp_steps"] for fn in DOC_ORDER if fn in opus and opus[fn]["fp_steps"] > 0),
    }
    dest = MM_DIR / "aikr_aggregate.json"
    MM_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(dest)
    for label, total in report["totals_ss"].items():
        print(f"SS {label}: {total}")


if __name__ == "__main__":
    main()

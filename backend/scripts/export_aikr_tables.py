"""Export AIKR table numbers from multi_model + Opus comparison."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MM = ROOT / "outputs" / "multi_model"
COMP = ROOT / "outputs" / "batch_4bed65c03d57_comparison.json"
OUT = ROOT.parent / "paper" / "aikr" / "MULTI_MODEL_RESULTS.json"

LABELS = [
    ("opus5", "claude-opus-5"),
    ("ds-v4-flash", "deepseek-v4-flash"),
    ("ds-reasoner", "deepseek-reasoner"),
    ("qwen36-flash", "qwen3.6-flash"),
]

DOC_LABEL = {
    "pdf.pdf": "D1 Hung 2009",
    "shiragami1995.pdf": "D2 Shiragami 1995",
    "jm00146a002.pdf": "D3 Beauchamp 1985",
    "36_1153.pdf": "D4 scan",
    "EP0532878A2.pdf": "D5 EP0532878A2",
    "EP0709385A1.pdf": "D6 EP0709385A1",
    "EP0862570B1.pdf": "D7 EP0862570B1",
    "US5567816.pdf": "D8 US5567816",
    "US5756737.pdf": "D9 US5756737",
    "WO_9718211_A1.pdf": "D10 WO",
    "RU2111967C1.pdf": "D11 RU",
}


def main() -> None:
    comp = json.loads(COMP.read_text(encoding="utf-8"))
    doc_order = [c["avrseek"]["filename"] for c in comp["comparisons"]]
    fp = {c["avrseek"]["filename"]: c["avrseek"]["steps_count"] for c in comp["comparisons"]}
    ss_opus = {
        c["avrseek"]["filename"]: (c.get("baseline") or {}).get("steps_count", 0)
        for c in comp["comparisons"]
    }

    ss_models: dict[str, dict[str, int | None]] = {"opus5": ss_opus}
    meta: dict[str, dict] = {
        "opus5": {
            "ss_total": sum(ss_opus.values()),
            "ss_ok": sum(1 for v in ss_opus.values() if v > 0),
            "fp_total": sum(fp.values()),
            "fp_ok": sum(1 for v in fp.values() if v > 0),
        }
    }

    for label, model_id in LABELS:
        if label == "opus5":
            continue
        path = MM / f"baseline_{label}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        per: dict[str, int | None] = {}
        ok = err = 0
        total = 0
        for fn in doc_order:
            row = next((d for d in data["documents"] if d["filename"] == fn), None)
            if not row:
                per[fn] = None
                continue
            if row.get("baseline"):
                n = row["baseline"]["steps_count"]
                per[fn] = n
                total += n
                if n > 0:
                    ok += 1
            elif row.get("error"):
                per[fn] = None
                err += 1
            else:
                per[fn] = 0
        ss_models[label] = per
        meta[label] = {
            "model_id": model_id,
            "ss_total": total,
            "ss_ok": ok,
            "ss_errors": err,
            "documents_in_file": len(data.get("documents", [])),
        }

    payload = {
        "doc_order": doc_order,
        "doc_label": DOC_LABEL,
        "fp_opus5": fp,
        "ss_by_model": ss_models,
        "aggregate": meta,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    for label, m in meta.items():
        print(label, m)


if __name__ == "__main__":
    main()

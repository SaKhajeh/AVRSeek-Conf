"""Build full 11-doc comparison: AVRSeek (4bed65) + baseline (055 or pending)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def load_baselines_055() -> dict[str, dict]:
    path = OUTPUTS / "batch_055d69aa8b80_comparison.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in data["comparisons"]:
        fn = row["avrseek"]["filename"]
        out[fn] = {
            "baseline": row.get("baseline"),
            "baseline_note": row.get("baseline_note", ""),
            "source": "batch_055d69aa8b80",
        }
    return out


def main() -> None:
    from compare_batch_baseline import avrseek_summary

    batch = json.loads(
        (OUTPUTS / "batches" / "4bed65c03d57" / "batch.json").read_text(encoding="utf-8")
    )
    baselines_055 = load_baselines_055()

    rows = []
    for r in batch["results"]:
        fn = r["filename"]
        avr = avrseek_summary(r)
        row = {"avrseek": avr, "baseline": None, "baseline_note": "", "baseline_source": ""}

        if fn in baselines_055 and baselines_055[fn]["baseline"]:
            row["baseline"] = baselines_055[fn]["baseline"]
            row["baseline_note"] = (
                "Opus 5 single-shot on cached parsed text (from prior successful run)"
            )
            row["baseline_source"] = "batch_055d69aa8b80"
        elif fn in ("RU2111967C1.pdf", "WO_9718211_A1.pdf"):
            row["baseline_note"] = "pending — no extractable text (OCR needed)"
            row["baseline_source"] = "pending_ocr"
        elif fn == "US5567816.pdf":
            row["baseline_note"] = "pending — prior run JSON parse error; re-run after quota restored"
            row["baseline_source"] = "pending_retry"
        elif fn == "36_1153.pdf":
            row["baseline_note"] = "pending — scanned PDF (163 chars native text); re-run after quota restored"
            row["baseline_source"] = "pending_retry"
        elif avr["text_extract_status"] == "error" and "403" in avr.get("text_extract_detail", ""):
            row["baseline_note"] = "pending — API quota; re-run after quota restored"
            row["baseline_source"] = "pending_quota"
        else:
            row["baseline_note"] = "pending — re-run baseline after quota restored"
            row["baseline_source"] = "pending_quota"

        rows.append(row)

    out = {
        "batch_id": "4bed65c03d57",
        "file_count": 11,
        "model": "claude-opus-5",
        "notes": (
            "AVRSeek from batch 4bed65c03d57. Baselines: 7 from prior Opus-5 run (055), "
            "4 pending (quota/OCR/parse). Five AVRSeek routes pending user re-upload after quota."
        ),
        "pending_avrseek": [
            "shiragami1995.pdf",
            "US5567816.pdf",
            "US5756737.pdf",
            "RU2111967C1.pdf",
            "WO_9718211_A1.pdf",
        ],
        "pending_baseline": [
            fn for fn, row in zip([r["filename"] for r in batch["results"]], rows) if not row["baseline"]
        ],
        "comparisons": rows,
    }

    dest = OUTPUTS / "batch_4bed65c03d57_comparison.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary table to markdown
    md = ["# Corpus comparison (11 documents)\n"]
    md.append("| Document | AVRSeek steps | Baseline steps | Baseline source |")
    md.append("|----------|---------------|----------------|-----------------|")
    for row in rows:
        a = row["avrseek"]
        b = row.get("baseline")
        bs = b["steps_count"] if b else "—"
        src = row.get("baseline_source", "")
        md.append(f"| {a['filename']} | {a['steps_count']} | {bs} | {src} |")

    md_path = ROOT.parent / "paper" / "CORPUS_COMPARISON.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {dest}")
    print(f"Wrote {md_path}")
    print(f"Baselines ready: {sum(1 for r in rows if r['baseline'])}/11")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()

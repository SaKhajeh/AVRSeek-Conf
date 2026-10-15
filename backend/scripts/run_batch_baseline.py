"""Run Opus-5 single-shot baseline for all documents in a batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.llm_client import extract_reactions_from_text
from run_baseline_compare import summarize_steps

# Reuse AVRSeek summary from compare_batch_baseline
from compare_batch_baseline import avrseek_summary, load_document_text


def baseline_from_text(filename: str, text: str, title: str | None = None) -> dict:
    settings = get_settings()
    route, _raw = extract_reactions_from_text(
        settings=settings,
        title=title or filename,
        text=text,
        text_chunks=None,
    )
    steps = [s.model_dump() for s in route.steps]
    summarized = summarize_steps(steps)
    return {
        "mode": "simple_single_shot_full_text",
        "model": settings.model_text,
        "text_chars": len(text or ""),
        "steps_count": len(steps),
        "target_name": route.target_name,
        "document_type": route.document_type,
        "warnings": route.warnings,
        "steps": summarized,
        "smiles_entities": sum(s["smiles_entities"] for s in summarized),
        "confidence_steps": sum(1 for s in steps if s.get("confidence") is not None),
        "provenance_steps": 0,
        "route_graph_nodes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-id",
        default="4bed65c03d57",
        help="Batch folder under outputs/batches/",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip baseline API call if already present in output file",
    )
    args = parser.parse_args()

    batch_path = ROOT / "outputs" / "batches" / args.batch_id / "batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    outputs = ROOT / "outputs"
    dest = outputs / f"batch_{args.batch_id}_comparison.json"

    existing: dict[str, dict] = {}
    if args.skip_existing and dest.exists():
        prev = json.loads(dest.read_text(encoding="utf-8"))
        for row in prev.get("comparisons", []):
            fn = row["avrseek"]["filename"]
            if row.get("baseline"):
                existing[fn] = row

    rows = []
    for r in batch["results"]:
        avr = avrseek_summary(r)
        row: dict = {"avrseek": avr, "baseline": None, "baseline_note": "", "baseline_source": ""}

        prior = existing.get(r["filename"])
        if prior and prior.get("baseline"):
            row["baseline"] = prior["baseline"]
            row["baseline_note"] = "reused from prior comparison run"
            rows.append(row)
            continue

        job_file = outputs / f"{r['job_id']}.json"
        text = load_document_text(job_file)
        if not text:
            row["baseline_note"] = "no extractable text in job cache (OCR needed)"
        else:
            try:
                row["baseline"] = baseline_from_text(r["filename"], text)
                row["baseline_note"] = "single-shot Opus 5 on cached parsed text (AvalAI)"
                row["baseline_source"] = "avalai_run"
            except Exception as exc:
                row["baseline_note"] = f"baseline failed: {exc}"

        rows.append(row)

    out = {
        "batch_id": batch.get("batch_id"),
        "file_count": batch.get("file_count"),
        "model": get_settings().model_text,
        "notes": (
            "AVRSeek from batch JSON; baseline = single-shot Claude Opus 5 on full parsed text. "
            "Five corpus documents pending AVRSeek re-run after API quota restored."
        ),
        "comparisons": rows,
    }
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = outputs / f"batch_{args.batch_id}_baseline_summary.txt"
    lines = [f"Wrote {dest}\n"]
    for row in rows:
        a = row["avrseek"]
        b = row.get("baseline")
        bs = b["steps_count"] if b else "-"
        lines.append(f"{a['filename']}: AVRSeek={a['steps_count']} baseline={bs} | {row['baseline_note']}\n")
    summary_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {dest}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

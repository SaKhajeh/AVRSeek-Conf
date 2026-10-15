"""Run single-shot baselines for multiple LLM backbones on one batch corpus."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from compare_batch_baseline import avrseek_summary, load_document_text
from run_batch_baseline import baseline_from_text

DEFAULT_BATCH = "4bed65c03d57"


def _write_checkpoint(dest: Path, args, model_id: str, documents: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": args.batch_id,
        "model_label": args.model_label,
        "model_text": model_id,
        "mode": "simple_single_shot_full_text",
        "documents": documents,
        "partial": True,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-shot baseline for one model label")
    parser.add_argument("--batch-id", default=DEFAULT_BATCH)
    parser.add_argument(
        "--model-label",
        required=True,
        help="Short name for output file, e.g. sonnet46, ds-v4-flash, qwen38-flash",
    )
    parser.add_argument(
        "--model-text",
        default="",
        help="Override MODEL_TEXT; defaults to current settings.model_text",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have a baseline in the output JSON",
    )
    args = parser.parse_args()

    if args.model_text:
        os.environ["MODEL_TEXT"] = args.model_text
        get_settings.cache_clear()

    settings = get_settings()
    model_id = settings.model_text

    batch_path = ROOT / "outputs" / "batches" / args.batch_id / "batch.json"
    if not batch_path.exists():
        raise SystemExit(f"Missing batch: {batch_path}")

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    out_dir = ROOT / "outputs" / "multi_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"baseline_{args.model_label}.json"

    prior_by_fn: dict[str, dict] = {}
    if dest.exists():
        prev = json.loads(dest.read_text(encoding="utf-8"))
        for row in prev.get("documents", []):
            prior_by_fn[row["filename"]] = row

    documents: list[dict] = []
    for r in batch["results"]:
        fn = r["filename"]
        avr = avrseek_summary(r)
        old = prior_by_fn.get(fn)
        if args.skip_existing and old and old.get("baseline"):
            documents.append(
                {
                    "filename": fn,
                    "avrseek_reference": avr,
                    "baseline": old["baseline"],
                    "error": None,
                }
            )
            continue

        row: dict = {
            "filename": fn,
            "avrseek_reference": avr,
            "baseline": None,
            "error": None,
        }
        job_file = ROOT / "outputs" / f"{r['job_id']}.json"
        text = load_document_text(job_file)
        if not text:
            row["error"] = "no extractable text in job cache (OCR needed)"
        else:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    row["baseline"] = baseline_from_text(fn, text)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001 — batch job should continue
                    last_exc = exc
                    time.sleep(5 * (attempt + 1))
            if last_exc is not None:
                row["error"] = str(last_exc)

        documents.append(row)
        _write_checkpoint(dest, args, model_id, _merge_full_batch(batch, documents, prior_by_fn, args.skip_existing))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": args.batch_id,
        "model_label": args.model_label,
        "model_text": model_id,
        "mode": "simple_single_shot_full_text",
        "documents": documents,
    }
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for d in documents if d.get("baseline"))
    print(f"Wrote {dest} ({ok}/{len(documents)} baselines, model={model_id})")


def _merge_full_batch(batch, processed: list[dict], prior_by_fn: dict, skip_existing: bool) -> list[dict]:
    """Keep all corpus rows in checkpoints (processed + skipped + pending)."""
    proc = {d["filename"]: d for d in processed}
    merged: list[dict] = []
    for r in batch["results"]:
        fn = r["filename"]
        if fn in proc:
            merged.append(proc[fn])
            continue
        old = prior_by_fn.get(fn)
        if skip_existing and old and old.get("baseline"):
            merged.append(
                {
                    "filename": fn,
                    "avrseek_reference": avrseek_summary(r),
                    "baseline": old["baseline"],
                    "error": None,
                }
            )
        elif old:
            merged.append(
                {
                    "filename": fn,
                    "avrseek_reference": avrseek_summary(r),
                    "baseline": old.get("baseline"),
                    "error": old.get("error"),
                }
            )
        else:
            merged.append(
                {
                    "filename": fn,
                    "avrseek_reference": avrseek_summary(r),
                    "baseline": None,
                    "error": None,
                }
            )
    return merged


if __name__ == "__main__":
    main()

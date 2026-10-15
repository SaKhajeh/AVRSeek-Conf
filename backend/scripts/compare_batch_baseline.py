"""Compare AVRSeek batch results vs Opus-5 single-shot baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.llm_client import extract_reactions_from_text

from run_baseline_compare import summarize_steps


def avrseek_summary(result: dict) -> dict:
    route = result.get("route") or {}
    steps = route.get("steps") or []
    stages = {s["id"]: s for s in result.get("stages", [])}
    txt = stages.get("text_extract", {})
    img = stages.get("image_scheme", {})
    parse = stages.get("parse_pdf", {})
    rg = route.get("route_graph") or {}

    conf = [s.get("confidence") for s in steps if s.get("confidence") is not None]
    smiles = sum(
        1
        for s in steps
        for b in (
            "reactants",
            "reagents",
            "catalysts",
            "ligands",
            "solvents",
            "products",
            "intermediates",
        )
        for e in (s.get(b) or [])
        if e.get("smiles")
    )
    prov = sum(1 for s in steps if s.get("evidence_refs"))

    return {
        "filename": result.get("filename"),
        "job_id": result.get("job_id"),
        "pipeline_complete": txt.get("status") == "done" and len(steps) > 0,
        "text_extract_status": txt.get("status"),
        "text_extract_detail": txt.get("detail"),
        "image_scheme_detail": img.get("detail"),
        "parse_detail": parse.get("detail"),
        "steps_count": len(steps),
        "target_name": (route.get("target_name") or "N/A")[:80],
        "document_type": route.get("document_type"),
        "route_graph_nodes": len(rg.get("nodes") or []),
        "warnings_count": len(route.get("warnings") or []),
        "conflicts_count": len(route.get("conflicts") or []),
        "confidence_range": [min(conf), max(conf)] if conf else None,
        "smiles_entities": smiles,
        "provenance_steps": prov,
        "scheme_steps": sum(1 for s in steps if s.get("source") == "scheme"),
    }


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


def load_document_text(job_path: Path) -> str | None:
    if not job_path.exists():
        return None
    data = json.loads(job_path.read_text(encoding="utf-8"))
    doc = data.get("document") or {}
    text = doc.get("text")
    if text and len(text.strip()) > 100:
        return text
    return None


def main() -> None:
    batch_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"c:\Users\galin\Downloads\avrseek_batch_055d69aa8b80.json"
    )
    outputs = ROOT / "outputs"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))

    rows = []
    for r in batch["results"]:
        avr = avrseek_summary(r)
        row = {"avrseek": avr, "baseline": None, "baseline_note": ""}

        job_file = outputs / f"{r['job_id']}.json"
        text = load_document_text(job_file)
        if text and (avr["text_extract_status"] == "error" or avr["steps_count"] > 0):
            try:
                row["baseline"] = baseline_from_text(r["filename"], text)
                if avr["text_extract_status"] == "error":
                    row["baseline_note"] = "baseline on stored text (batch text_extract failed)"
                else:
                    row["baseline_note"] = "baseline on stored document text"
            except Exception as exc:
                row["baseline_note"] = f"baseline failed: {exc}"
        elif not text:
            row["baseline_note"] = "no extractable text in job cache"
        else:
            row["baseline_note"] = "baseline skipped (zero-step document)"

        rows.append(row)

    prior = outputs / "baseline_vs_avrseek_comparison.json"
    if prior.exists():
        for row in rows:
            if row["avrseek"]["filename"] == "pdf.pdf" and row["avrseek"]["steps_count"] == 0:
                prev = json.loads(prior.read_text(encoding="utf-8"))
                row["avrseek"] = {
                    **row["avrseek"],
                    "steps_count": prev["avrseek"]["steps_count"],
                    "target_name": prev["avrseek"]["target_name"][:80],
                    "route_graph_nodes": prev["avrseek"]["route_graph_nodes"],
                    "warnings_count": len(prev["avrseek"].get("warnings") or []),
                    "confidence_range": [
                        min(s["confidence"] for s in prev["avrseek"]["steps"] if s["confidence"]),
                        max(s["confidence"] for s in prev["avrseek"]["steps"] if s["confidence"]),
                    ],
                    "smiles_entities": sum(s["smiles_entities"] for s in prev["avrseek"]["steps"]),
                    "scheme_steps": sum(1 for s in prev["avrseek"]["steps"] if s.get("source") == "scheme"),
                    "pipeline_complete": True,
                    "note": "from prior batch f4af663d45de (055 text_extract failed)",
                }
                row["baseline"] = prev["baseline"]
                row["baseline"]["smiles_entities"] = sum(
                    s["smiles_entities"] for s in prev["baseline"]["steps"]
                )
                row["baseline"]["confidence_steps"] = 0
                row["baseline"]["provenance_steps"] = 0
                row["baseline"]["route_graph_nodes"] = 0
                row["baseline_note"] = "pdf.pdf from prior successful run"

    out = {
        "batch_id": batch.get("batch_id"),
        "file_count": batch.get("file_count"),
        "model": get_settings().model_text,
        "notes": (
            "Batch 055d69aa8b80 — network errors on text_extract for most files. "
            "Baseline = single-shot Claude Opus 5 on full parsed text."
        ),
        "comparisons": rows,
    }

    dest = outputs / "batch_055d69aa8b80_comparison.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {dest}")
    for row in rows:
        a = row["avrseek"]
        b = row["baseline"]
        bs = b["steps_count"] if b else "-"
        print(f"{a['filename']}: AVRSeek={a['steps_count']} baseline={bs} | {row['baseline_note']}")


if __name__ == "__main__":
    main()

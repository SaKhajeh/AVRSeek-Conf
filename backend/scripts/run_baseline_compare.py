"""Simple baseline LLM vs AVRSeek comparison for one PDF."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.services.llm_client import extract_reactions_from_text
from app.services.pdf_parse import parse_pdf


def summarize_steps(steps: list[dict]) -> list[dict]:
    out = []
    for s in steps:
        smiles_count = sum(
            1
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
        out.append(
            {
                "step_index": s.get("step_index"),
                "step_label": s.get("step_label"),
                "yield_value": s.get("yield_value"),
                "temperature": (s.get("conditions") or {}).get("temperature"),
                "time": (s.get("conditions") or {}).get("time"),
                "source": s.get("source"),
                "confidence": s.get("confidence"),
                "smiles_entities": smiles_count,
                "reactants": [r.get("name") for r in (s.get("reactants") or [])[:4]],
                "products": [p.get("name") for p in (s.get("products") or [])[:2]],
            }
        )
    return out


def main() -> None:
    pdf = Path(r"c:\Users\galin\Desktop\PHD\Patents_Acyclovir\Papers\pdf.pdf")
    batch_json = Path("outputs/batches/f4af663d45de/batch.json")
    title = "Synthesis of acyclovir as an antiherpes-virus drug"

    parsed = parse_pdf(pdf, output_dir=Path("outputs"))
    settings = get_settings()

    route, _raw = extract_reactions_from_text(
        settings=settings,
        title=title,
        text=parsed.text or "",
        text_chunks=None,
    )

    baseline = {
        "mode": "simple_single_shot_full_text",
        "model": settings.model_text,
        "api_base": settings.openai_base_url,
        "text_chars": len(parsed.text or ""),
        "steps_count": len(route.steps),
        "target_name": route.target_name,
        "document_type": route.document_type,
        "warnings": route.warnings,
        "steps": summarize_steps([s.model_dump() for s in route.steps]),
    }

    avr = json.loads(batch_json.read_text(encoding="utf-8"))
    avr_route = avr["results"][0]["route"]
    avr_stages = {s["id"]: s for s in avr["results"][0].get("stages", [])}
    avrseek = {
        "batch_id": avr.get("batch_id"),
        "model_text": get_settings().model_text,
        "model_vlm": get_settings().model_vlm,
        "steps_count": len(avr_route["steps"]),
        "target_name": avr_route["target_name"],
        "document_type": avr_route["document_type"],
        "warnings": avr_route.get("warnings", []),
        "image_scheme": avr_stages.get("image_scheme", {}).get("detail"),
        "route_graph_nodes": len((avr_route.get("route_graph") or {}).get("nodes", [])),
        "steps": summarize_steps(avr_route["steps"]),
    }

    baseline_labels = {s["step_label"] for s in baseline["steps"]}
    avr_labels = {s["step_label"] for s in avrseek["steps"]}
    comparison = {
        "pdf": str(pdf),
        "baseline": baseline,
        "avrseek": avrseek,
        "delta_steps": baseline["steps_count"] - avrseek["steps_count"],
        "only_in_baseline": sorted(baseline_labels - avr_labels),
        "only_in_avrseek": sorted(avr_labels - baseline_labels),
        "notes": (
            "Baseline = one-shot full-text LLM call via extract_reactions_from_text (no OCR/VLM/reconcile). "
            "AVRSeek = full pipeline including evidence retrieval, VLM figure pages, reconcile, synonyms, route graph."
        ),
    }

    out = Path("outputs/baseline_vs_avrseek_comparison.json")
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

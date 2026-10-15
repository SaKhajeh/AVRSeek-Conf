"""Analyze AVRSeek batch JSON exports and compare runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "batches"


def summarize_result(r: dict) -> dict:
    stages = {s["id"]: s for s in r.get("stages", [])}
    route = r.get("route") or {}
    steps = route.get("steps") or []
    rg = route.get("route_graph") or {}
    txt = stages.get("text_extract", {})
    img = stages.get("image_scheme", {})
    parse = stages.get("parse_pdf", {})

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

    return {
        "filename": r.get("filename"),
        "job_id": r.get("job_id"),
        "steps": len(steps),
        "target": (route.get("target_name") or "N/A")[:60],
        "doc_type": route.get("document_type"),
        "rg_nodes": len(rg.get("nodes") or []),
        "warnings": len(route.get("warnings") or []),
        "conflicts": len(route.get("conflicts") or []),
        "smiles": smiles,
        "scheme_steps": sum(1 for s in steps if s.get("source") == "scheme"),
        "text_steps": sum(1 for s in steps if s.get("source") in ("text", "table", "")),
        "provenance": sum(1 for s in steps if s.get("evidence_refs")),
        "text_status": txt.get("status"),
        "text_detail": (txt.get("detail") or "")[:80],
        "image_detail": (img.get("detail") or "")[:80],
        "parse_detail": (parse.get("detail") or "")[:60],
        "conf_range": [min(conf), max(conf)] if conf else None,
    }


def load_batch(batch_id: str) -> dict:
    path = OUTPUTS / batch_id / "batch.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    batch_ids = sys.argv[1:] or [
        "055d69aa8b80",
        "d2d8eccb0b6c",
        "4bed65c03d57",
    ]

    all_rows: dict[str, dict[str, dict]] = {}
    for bid in batch_ids:
        batch = load_batch(bid)
        all_rows[bid] = {r["filename"]: summarize_result(r) for r in batch["results"]}
        print(f"\n{'='*90}\nBATCH {bid}  files={batch.get('file_count')}\n{'='*90}")
        for fn, s in sorted(all_rows[bid].items(), key=lambda x: x[0]):
            print(
                f"{fn:22} steps={s['steps']:2} rg={s['rg_nodes']:2} sm={s['smiles']:2} "
                f"txt={s['text_status']:7} | {s['text_detail']}"
            )

    # Compare overlapping files across batches
    print(f"\n{'='*90}\nCROSS-BATCH COMPARISON (same filename)\n{'='*90}")
    filenames = sorted({fn for rows in all_rows.values() for fn in rows})
    for fn in filenames:
        present = {bid: all_rows[bid][fn] for bid in batch_ids if fn in all_rows[bid]}
        if len(present) < 2:
            continue
        steps_vals = [p["steps"] for p in present.values()]
        if len(set(steps_vals)) == 1 and all(p["text_status"] != "error" for p in present.values()):
            continue
        print(f"\n{fn}:")
        for bid, s in present.items():
            print(
                f"  {bid}: steps={s['steps']} rg={s['rg_nodes']} sm={s['smiles']} "
                f"txt={s['text_status']} | {s['text_detail']}"
            )

    # d2d8 vs 4bed65 overlap (retry batch)
    print(f"\n{'='*90}\nRETRY BATCH d2d8eccb0b6c vs FULL BATCH 4bed65c03d57\n{'='*90}")
    if "d2d8eccb0b6c" in all_rows and "4bed65c03d57" in all_rows:
        overlap = set(all_rows["d2d8eccb0b6c"]) & set(all_rows["4bed65c03d57"])
        for fn in sorted(overlap):
            a = all_rows["d2d8eccb0b6c"][fn]
            b = all_rows["4bed65c03d57"][fn]
            delta_steps = b["steps"] - a["steps"]
            flag = "CHANGED" if a["steps"] != b["steps"] or a["text_status"] != b["text_status"] else "same"
            print(
                f"{fn:22} d2d8={a['steps']:2}({a['text_status']}) -> "
                f"4bed={b['steps']:2}({b['text_status']}) delta={delta_steps:+d} [{flag}]"
            )

    # vs original 055 batch
    if "055d69aa8b80" in all_rows and "4bed65c03d57" in all_rows:
        print(f"\n{'='*90}\nORIGINAL 055 vs NEW FULL BATCH 4bed65c03d57\n{'='*90}")
        overlap = set(all_rows["055d69aa8b80"]) & set(all_rows["4bed65c03d57"])
        for fn in sorted(overlap):
            old = all_rows["055d69aa8b80"][fn]
            new = all_rows["4bed65c03d57"][fn]
            print(
                f"{fn:22} 055={old['steps']:2}({old['text_status']}) -> "
                f"4bed={new['steps']:2}({new['text_status']}) "
                f"rg {old['rg_nodes']}->{new['rg_nodes']} sm {old['smiles']}->{new['smiles']}"
            )


if __name__ == "__main__":
    main()

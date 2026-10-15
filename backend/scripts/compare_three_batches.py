"""Compare three pdf.pdf batch runs."""
from __future__ import annotations

import json
from pathlib import Path

BATCHES = ["0e7175dfc9d9", "f4af663d45de", "3b6e23e4a7ae"]
BASE = Path(__file__).resolve().parents[1] / "outputs" / "batches"


def load(bid: str) -> dict:
    return json.loads((BASE / bid / "batch.json").read_text(encoding="utf-8"))["results"][0]


def main() -> None:
    for bid in BATCHES:
        r = load(bid)
        route = r.get("route") or {}
        stages = {s["id"]: s for s in r.get("stages", [])}
        debug = r.get("extraction_debug") or {}
        print("=" * 60)
        print("BATCH", bid, "job", r["job_id"])
        print("Steps:", len(route.get("steps", [])))
        print("Text extract:", stages.get("text_extract", {}).get("detail"))
        is_ = stages.get("image_scheme", {})
        print("Image scheme:", is_.get("status"), "-", (is_.get("detail") or "")[:120])
        print("Reconcile:", stages.get("reconcile", {}).get("detail"))
        std = stages.get("standardize", {})
        print("Standardize:", std.get("detail"))
        print("Std data:", std.get("data"))
        ps = debug.get("page_selection") or {}
        print("Passes:", ps.get("pass_count"))
        for p in ps.get("passes", []):
            print("  pass", p.get("pass_index"), "pages", p.get("pages_included"), "excluded", p.get("pages_excluded"))
        obs = debug.get("observability") or {}
        for key in ("reconciliation", "standardization", "provenance"):
            if obs.get(key):
                print(f"  {key}:", obs[key])
        print("Conflicts:", len(route.get("conflicts", [])))
        for c in route.get("conflicts", [])[:3]:
            print("  conflict:", c.get("field"), c.get("values"))
        print("Warnings:", len(route.get("warnings", [])))
        for w in route.get("warnings", [])[:10]:
            print("  -", w[:110])
        rg = route.get("route_graph") or {}
        print("Route graph:", len(rg.get("nodes", [])), "nodes,", len(rg.get("edges", [])), "edges")
        print("Compound links:", len(route.get("compound_links", [])))
        for s in route.get("steps", []):
            label = (s.get("step_label") or "")[:65]
            smiles = sum(
                1
                for b in ("reactants", "reagents", "catalysts", "products", "solvents", "intermediates")
                for e in s.get(b, [])
                if e.get("smiles")
            )
            print(
                f"  Step {s['step_index']}: {label} | yield={s.get('yield_value')} "
                f"conf={s.get('confidence')} src={s.get('source')} kind={s.get('information_kind')} "
                f"refs={len(s.get('evidence_refs', []))} methods={s.get('extraction_methods', [])} smiles={smiles}"
            )
        print()


if __name__ == "__main__":
    main()

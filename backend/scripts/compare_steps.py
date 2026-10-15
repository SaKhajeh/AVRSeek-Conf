import json
from pathlib import Path

batch_dir = Path(__file__).resolve().parents[1] / "outputs" / "batches"


def load(bid: str):
    d = json.loads((batch_dir / bid / "batch.json").read_text(encoding="utf-8"))
    return d["results"][0], d["results"][0]["route"]


for bid in ("0e7175dfc9d9", "f4af663d45de"):
    r, route = load(bid)
    print("===", bid, "===")
    print("job:", r["job_id"])
    for s in route["steps"]:
        smiles = sum(
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
            for e in s.get(b, [])
            if e.get("smiles")
        )
        label = (s.get("step_label") or "")[:75]
        print(
            f"  {s['step_index']}: {label} | conf={s.get('confidence')} "
            f"src={s.get('source')} smiles={smiles} yield={s.get('yield_value')}"
        )
    rg = route.get("route_graph") or {}
    print("  graph:", len(rg.get("nodes", [])), "nodes,", len(rg.get("edges", [])), "edges")
    print("  warnings:", len(route.get("warnings", [])))
    print()

"""Simple evaluation harness for AVRSeek extractions vs gold JSON.

Gold format (one file per PDF stem):
{
  "filename": "WO_9718211_A1.pdf",
  "target_name_aliases": ["acyclovir", "9-[(2-hydroxyethoxy)methyl]guanine"],
  "required_names": ["acyclovir", "monoethanolamine", "diacetylacyclovir", "acetic acid"],
  "required_conditions_substrings": ["70", "95", "3 h", "98.7"],
  "min_steps": 2
}

Usage:
  python -m app.eval.run_eval --gold app/eval/gold --results outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("−", "-").split())


def score_result(result: dict, gold: dict) -> dict:
    route = result.get("route") or {}
    steps = route.get("steps") or []
    blob_parts: list[str] = [
        str(route.get("title") or ""),
        str(route.get("target_name") or ""),
    ]
    for step in steps:
        blob_parts.append(str(step.get("step_label") or ""))
        blob_parts.append(str(step.get("yield_value") or ""))
        blob_parts.append(str(step.get("notes") or ""))
        blob_parts.append(str(step.get("evidence") or ""))
        cond = step.get("conditions") or {}
        for k in ("temperature", "time", "atmosphere", "other"):
            blob_parts.append(str(cond.get(k) or ""))
        for bucket in ("reactants", "reagents", "catalysts", "ligands", "solvents", "products", "intermediates"):
            for ent in step.get(bucket) or []:
                blob_parts.append(str(ent.get("name") or ""))
                blob_parts.append(str(ent.get("smiles") or ""))
    blob = _norm(" | ".join(blob_parts))

    required_names = gold.get("required_names") or []
    hit_names = [n for n in required_names if _norm(n) in blob]
    miss_names = [n for n in required_names if _norm(n) not in blob]

    required_conds = gold.get("required_conditions_substrings") or []
    hit_conds = [c for c in required_conds if _norm(c) in blob]
    miss_conds = [c for c in required_conds if _norm(c) not in blob]

    aliases = gold.get("target_name_aliases") or []
    target = _norm(str(route.get("target_name") or ""))
    target_ok = any(_norm(a) in target or _norm(a) in blob for a in aliases) if aliases else bool(target and target != "n/a")

    min_steps = int(gold.get("min_steps") or 1)
    steps_ok = len(steps) >= min_steps

    name_score = len(hit_names) / max(1, len(required_names))
    cond_score = len(hit_conds) / max(1, len(required_conds))
    overall = (0.35 * float(target_ok) + 0.25 * float(steps_ok) + 0.25 * name_score + 0.15 * cond_score)

    return {
        "filename": gold.get("filename") or result.get("filename"),
        "n_steps": len(steps),
        "steps_ok": steps_ok,
        "target_ok": target_ok,
        "name_recall": round(name_score, 3),
        "condition_recall": round(cond_score, 3),
        "overall": round(overall, 3),
        "hit_names": hit_names,
        "miss_names": miss_names,
        "hit_conditions": hit_conds,
        "miss_conditions": miss_conds,
    }


def find_result_for_gold(results_dir: Path, gold: dict) -> dict | None:
    want = (gold.get("filename") or "").lower()
    # Prefer batch jsons and individual jobs
    candidates = list(results_dir.glob("*.json"))
    candidates += list((results_dir / "batches").glob("*/batch.json")) if (results_dir / "batches").exists() else []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "results" in data and isinstance(data["results"], list):
            for r in data["results"]:
                if str(r.get("filename") or "").lower() == want:
                    return r
        elif str(data.get("filename") or "").lower() == want:
            return data
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AVRSeek outputs against gold files")
    parser.add_argument("--gold", type=Path, default=Path(__file__).parent / "gold")
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    gold_files = sorted(args.gold.glob("*.json"))
    if not gold_files:
        print(f"No gold files in {args.gold}")
        return

    rows = []
    for gf in gold_files:
        gold = json.loads(gf.read_text(encoding="utf-8"))
        result = find_result_for_gold(args.results, gold)
        if not result:
            rows.append({"filename": gold.get("filename"), "overall": 0.0, "error": "result not found"})
            continue
        rows.append(score_result(result, gold))

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if rows:
        avg = sum(float(r.get("overall") or 0) for r in rows) / len(rows)
        print(f"\nAverage overall: {avg:.3f}")


if __name__ == "__main__":
    main()

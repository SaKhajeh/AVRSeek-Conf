"""Export batch analysis report to UTF-8 markdown."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib.util

spec = importlib.util.spec_from_file_location(
    "analyze_batches", ROOT / "scripts" / "analyze_batches.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
summarize_result = mod.summarize_result
load_batch = mod.load_batch

OUT = ROOT / "outputs" / "batch_analysis_4bed65.md"


def main() -> None:
    batches = {
        "055d69aa8b80": load_batch("055d69aa8b80"),
        "d2d8eccb0b6c": load_batch("d2d8eccb0b6c"),
        "4bed65c03d57": load_batch("4bed65c03d57"),
    }
    lines = ["# Batch analysis report\n"]

    for bid, batch in batches.items():
        lines.append(f"\n## Batch `{bid}` ({batch['file_count']} files)\n")
        lines.append("| File | Steps | RG | SMILES | Text | Image |")
        lines.append("|------|-------|-----|--------|------|-------|")
        for r in sorted(batch["results"], key=lambda x: x["filename"]):
            s = summarize_result(r)
            lines.append(
                f"| {s['filename']} | {s['steps']} | {s['rg_nodes']} | {s['smiles']} "
                f"| {s['text_status']} | {s['image_detail'][:40]} |"
            )

    old = {r["filename"]: summarize_result(r) for r in batches["055d69aa8b80"]["results"]}
    new = {r["filename"]: summarize_result(r) for r in batches["4bed65c03d57"]["results"]}

    lines.append("\n## Changes: batch 055 → 4bed65\n")
    lines.append("| File | Old steps | New steps | Δ | Old txt | New txt |")
    lines.append("|------|-----------|-----------|---|---------|---------|")
    for fn in sorted(old):
        o, n = old[fn], new.get(fn)
        if not n:
            continue
        lines.append(
            f"| {fn} | {o['steps']} | {n['steps']} | {n['steps']-o['steps']:+d} "
            f"| {o['text_status']} | {n['text_status']} |"
        )

    success = [s for s in new.values() if s["steps"] > 0]
    lines.append("\n## Aggregate (4bed65, steps > 0)\n")
    lines.append(f"- Documents with routes: **{len(success)}** / 11")
    lines.append(f"- Total steps: **{sum(s['steps'] for s in success)}**")
    lines.append(f"- Total route-graph nodes: **{sum(s['rg_nodes'] for s in success)}**")
    lines.append(f"- Total SMILES entities: **{sum(s['smiles'] for s in success)}**")
    lines.append(f"- Mean steps/doc: **{sum(s['steps'] for s in success)/len(success):.1f}**")

    quota = [s for s in new.values() if "403" in s.get("text_detail", "")]
    stub = [s for s in new.values() if s["text_status"] == "stubbed"]
    lines.append("\n## Blockers (4bed65)\n")
    lines.append(f"- API quota (403): {', '.join(s['filename'] for s in quota) or 'none'}")
    lines.append(f"- No extractable text (OCR needed): {', '.join(s['filename'] for s in stub) or 'none'}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

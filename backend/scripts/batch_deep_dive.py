"""Deep-dive batch document comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "batches"


def brief(batch_id: str, filename: str) -> None:
    b = json.loads((ROOT / batch_id / "batch.json").read_text(encoding="utf-8"))
    for r in b["results"]:
        if r["filename"] != filename:
            continue
        route = r.get("route") or {}
        steps = route.get("steps") or []
        stages = {s["id"]: s for s in r["stages"]}
        print(f"=== {batch_id} {filename} steps={len(steps)}")
        print("text:", stages.get("text_extract", {}).get("detail", "")[:120])
        print("image:", stages.get("image_scheme", {}).get("detail", "")[:120])
        for i, s in enumerate(steps[:10]):
            label = (s.get("step_label") or "")[:55]
            print(f"  {i+1}. [{s.get('source', '')}] {label} conf={s.get('confidence')}")
        if len(steps) > 10:
            print(f"  ... +{len(steps) - 10} more")
        w = route.get("warnings") or []
        if w:
            print("warnings:", w[:2])
        return


if __name__ == "__main__":
    for fn in ["36_1153.pdf", "EP0709385A1.pdf", "pdf.pdf", "EP0532878A2.pdf", "jm00146a002.pdf"]:
        for bid in ["055d69aa8b80", "4bed65c03d57"]:
            if (ROOT / bid / "batch.json").exists():
                brief(bid, fn)
        print()

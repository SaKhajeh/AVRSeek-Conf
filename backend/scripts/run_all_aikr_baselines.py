"""Run single-shot baselines for all AIKR study models (sequential)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_multi_model_baseline.py"

MODELS = [
    ("ds-v4-flash", "deepseek-v4-flash"),
    ("ds-reasoner", "deepseek-reasoner"),
    ("qwen36-flash", "qwen3.6-flash"),
]


def main() -> None:
    for label, model_id in MODELS:
        env = os.environ.copy()
        env["MODEL_TEXT"] = model_id
        print(f"\n=== {label} ({model_id}) ===", flush=True)
        rc = subprocess.call(
            [
                sys.executable,
                str(SCRIPT),
                "--batch-id",
                "4bed65c03d57",
                "--model-label",
                label,
                "--model-text",
                model_id,
                "--skip-existing",
            ],
            cwd=str(ROOT),
            env=env,
        )
        if rc != 0:
            print(f"WARNING: {label} exited {rc}", flush=True)


if __name__ == "__main__":
    main()

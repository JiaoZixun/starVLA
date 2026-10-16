#!/usr/bin/env python3
"""Spatial-aware evaluation entrypoint; delegates to the existing evaluator."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess, sys

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description="Evaluate QwenSpatialOFT with the unchanged RoboDojo evaluator")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a=ap.parse_args()
    evaluator = Path(__file__).resolve().parents[1] / "eval_files/eval_qwen35_robodojo.py"
    cmd=[sys.executable, str(evaluator),
         "--checkpoint", str(a.checkpoint), "--output", str(a.output),
         "--checkpoint-label", "qwen35_0p8b_spatial", "--expected-parameters", "0", *a.extra]
    raise SystemExit(subprocess.call(cmd))

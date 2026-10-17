#!/usr/bin/env python3
"""Build or validate per-task causal memory indices outside the raw dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from starVLA.utils.robodojo_memory import RoboDojoMemorySpec
from starVLA.utils.robodojo_memory_sidecar import build_sidecar, validate_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("playground/Datasets/RoboTwin/lerobot_clean"))
    parser.add_argument("--sidecar-root", type=Path, default=Path("playground/Datasets/RoboTwin/memory_sidecars/robotwin_clean_r16_s8_v1"))
    parser.add_argument("--task", action="append", help="Task name; repeat to select a subset")
    parser.add_argument("--recent-records", type=int, default=16)
    parser.add_argument("--decision-stride", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--validate", action="store_true", help="Validate existing sidecars without writing")
    args = parser.parse_args()
    spec = RoboDojoMemorySpec(args.recent_records, args.decision_stride)
    clean_root = args.data_root / "Clean"
    tasks = args.task or sorted(path.name for path in clean_root.iterdir() if path.is_dir())
    if not tasks:
        raise SystemExit(f"No Clean tasks found under {clean_root}")
    for task in tasks:
        source = clean_root / task
        target = args.sidecar_root / "Clean" / task
        if not (source / "meta/info.json").is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {source}")
        if args.validate:
            validate_sidecar(target, source, spec)
        else:
            build_sidecar(source, target, spec, args.workers)
            validate_sidecar(target, source, spec)
        print(f"Validated Robotwin memory sidecar: {task}", flush=True)


if __name__ == "__main__":
    main()

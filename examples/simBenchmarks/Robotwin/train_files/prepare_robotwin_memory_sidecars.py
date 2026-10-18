#!/usr/bin/env python3
"""Build or validate per-task causal memory indices outside the raw dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from starVLA.utils.robodojo_memory import RoboDojoMemorySpec
from starVLA.utils.robodojo_memory_sidecar import build_sidecar, validate_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("playground/Datasets/RoboTwin/lerobot_clean"))
    parser.add_argument("--sidecar-root", type=Path, default=Path("playground/Datasets/RoboTwin/memory_sidecars/robotwin_clean_r16_s8_v2"))
    parser.add_argument("--task", action="append", help="Task name; repeat to select a subset")
    parser.add_argument("--recent-records", type=int, default=16)
    parser.add_argument("--decision-stride", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers within one task (usually one parquet shard per task)")
    parser.add_argument("--task-workers", type=int, default=3,
                        help="Number of independent tasks to preprocess in parallel")
    parser.add_argument("--validate", action="store_true", help="Validate existing sidecars without writing")
    parser.add_argument("--resume", action="store_true",
                        help="Validate and skip existing task sidecars, then build missing tasks")
    args = parser.parse_args()
    spec = RoboDojoMemorySpec(args.recent_records, args.decision_stride)
    clean_root = args.data_root / "Clean"
    tasks = args.task or sorted(path.name for path in clean_root.iterdir() if path.is_dir())
    if not tasks:
        raise SystemExit(f"No Clean tasks found under {clean_root}")
    def process(task: str) -> str:
        source = clean_root / task
        target = args.sidecar_root / "Clean" / task
        if not (source / "meta/info.json").is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {source}")
        started = time.monotonic()
        if args.validate:
            validate_sidecar(target, source, spec)
            status = "validated"
        else:
            if target.exists() or target.is_symlink():
                if not args.resume:
                    raise FileExistsError(f"Output already exists (pass --resume to validate and skip): {target}")
                validate_sidecar(target, source, spec)
                status = "already valid"
            else:
                build_sidecar(source, target, spec, args.workers)
                validate_sidecar(target, source, spec)
                status = "built and validated"
        elapsed = time.monotonic() - started
        print(f"Robotwin sidecar {status}: {task} ({elapsed:.1f}s)", flush=True)
        return task

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.task_workers)) as pool:
        futures = {pool.submit(process, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((task, exc))
                print(f"Robotwin sidecar failed: {task}: {exc}", flush=True)
    if failures:
        details = "; ".join(f"{task}: {error}" for task, error in failures)
        raise SystemExit(f"{len(failures)} task(s) failed: {details}")


if __name__ == "__main__":
    main()

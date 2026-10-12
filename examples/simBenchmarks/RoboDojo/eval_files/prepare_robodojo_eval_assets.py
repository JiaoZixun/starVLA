#!/usr/bin/env python3
"""Check/download only object assets needed by official RoboDojo eval layouts."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

from tqdm import tqdm

from robodojo_eval_assets import format_bytes, inspect_eval_assets

HF_DATASET_ID = "RoboDojo-Benchmark/RoboDojo"


def official_tasks(bench: Path) -> list[str]:
    inventory_path = bench / "scripts/internal/task_inventory.py"
    spec = importlib.util.spec_from_file_location("task_inventory", inventory_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RoboDojo task inventory: {inventory_path}")
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    return [task for group in inventory.DIMENSION_TASKS.values() for task in group]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", help="Defaults to all official tasks")
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", "10")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LAYOUT_SEED", "0")))
    parser.add_argument("--endpoint", default="https://huggingface.co", help="Hugging Face endpoint")
    parser.add_argument("--download", action="store_true", help="Download missing LFS assets with progress")
    args = parser.parse_args()

    bench = Path(os.environ.get("ROBODOJO_PATH", "/home/data/roboskills/RoboDojo")).resolve()
    tasks = args.tasks or official_tasks(bench)
    known_tasks = set(official_tasks(bench))
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in known_tasks for task in tasks):
        parser.error("--tasks must contain unique official task names")
    if args.episodes < 1 or args.seed < 0:
        parser.error("--episodes must be positive and --seed nonnegative")

    report = inspect_eval_assets(bench, tasks, args.episodes, args.seed)
    if report["issues"]:
        print("Layout asset issues that cannot be fetched from the local RoboDojo LFS index:", file=sys.stderr)
        for issue in report["issues"]:
            print(f"  {issue}", file=sys.stderr)
        return 2
    if not report["missing"]:
        print(f"All assets are present for {len(tasks)} task(s) x {args.episodes} layouts (seed {args.seed}).")
        return 0

    print(
        f"Required object files missing: {len(report['missing'])}; "
        f"estimated transfer: {format_bytes(report['estimated_bytes'])} across {len(tasks)} task(s) x "
        f"{args.episodes} layouts (seed {args.seed})."
    )
    if not args.download:
        print("Run again with --download when you want to fetch these files.")
        return 1

    from huggingface_hub import hf_hub_download

    asset_repo = bench / ".cache/robodojo_assets_repo"
    asset_repo.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(report["missing"]), unit="file", desc="RoboDojo assets", dynamic_ncols=True) as bar:
        for filename in report["missing"]:
            destination = asset_repo / filename
            hf_hub_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(asset_repo),
                endpoint=args.endpoint,
                force_download=destination.exists(),
            )
            bar.update(1)
            bar.set_postfix(file=Path(filename).name, refresh=True)

    verify = inspect_eval_assets(bench, tasks, args.episodes, args.seed)
    if verify["missing"] or verify["issues"]:
        print("Asset verification still reports missing/corrupt files:", file=sys.stderr)
        for filename in verify["missing"]:
            print(f"  {filename}", file=sys.stderr)
        for issue in verify["issues"]:
            print(f"  {issue}", file=sys.stderr)
        return 2
    print("Asset download and verification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

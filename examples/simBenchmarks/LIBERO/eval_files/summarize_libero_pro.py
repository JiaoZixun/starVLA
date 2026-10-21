"""Aggregate LIBERO-PRO results for one model.

Per-task shards ``<model_root>/<suite>/<perturbation>/task_XX/summary.json``
are merged into ``<suite>/<perturbation>/summary.json`` first.
"""

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.LIBERO.eval_files.libero_pro_utils import PERTURBATIONS


def merge_task_shards(model_root: Path) -> None:
    for cell_dir in sorted(p for p in model_root.glob("*/*") if p.is_dir() and p.parent.name != "logs"):
        shards = sorted(cell_dir.glob("task_*/summary.json"))
        if not shards:
            continue
        parts = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
        task_results = sorted((r for part in parts for r in part["task_results"]), key=lambda r: r["task_id"])
        ids = [r["task_id"] for r in task_results]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"Duplicate task ids in {cell_dir}: {ids}")
        merged = {k: v for k, v in parts[0].items() if k not in ("episodes", "successes", "success_rate", "task_results")}
        merged["episodes"] = sum(r["episodes"] for r in task_results)
        merged["successes"] = sum(r["successes"] for r in task_results)
        merged["success_rate"] = merged["successes"] / merged["episodes"]
        merged["task_results"] = task_results
        (cell_dir / "summary.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def summarize(model_root: Path) -> dict:
    cells = {}
    for path in sorted(model_root.glob("*/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cells.setdefault(data["task_suite"], {})[data["perturbation"]] = {
            "dimension": data["dimension"],
            "episodes": data["episodes"],
            "successes": data["successes"],
            "success_rate": data["success_rate"],
        }
    if not cells:
        raise SystemExit(f"No LIBERO-PRO summaries under {model_root}")
    by_perturbation = {}
    for suite_cells in cells.values():
        for pert, cell in suite_cells.items():
            agg = by_perturbation.setdefault(
                pert, {"dimension": PERTURBATIONS[pert]["dimension"], "episodes": 0, "successes": 0}
            )
            agg["episodes"] += cell["episodes"]
            agg["successes"] += cell["successes"]
    for agg in by_perturbation.values():
        agg["success_rate"] = agg["successes"] / agg["episodes"]
    episodes = sum(agg["episodes"] for agg in by_perturbation.values())
    successes = sum(agg["successes"] for agg in by_perturbation.values())
    return {
        "model_root": str(model_root),
        "cells": cells,
        "by_perturbation": by_perturbation,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    merge_task_shards(args.model_root)
    summary = summarize(args.model_root)
    (args.model_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    perts = [p for p in PERTURBATIONS if p in summary["by_perturbation"]]
    print("suite".ljust(16) + "".join(PERTURBATIONS[p]["dimension"].rjust(10) for p in perts))
    for suite, suite_cells in sorted(summary["cells"].items()):
        row = "".join(
            (f"{100 * suite_cells[p]['success_rate']:.1f}" if p in suite_cells else "-").rjust(10) for p in perts
        )
        print(suite.ljust(16) + row)
    print("mean".ljust(16) + "".join(f"{100 * summary['by_perturbation'][p]['success_rate']:.1f}".rjust(10) for p in perts))
    print(f"overall: {summary['successes']}/{summary['episodes']} ({100 * summary['success_rate']:.1f}%)")


if __name__ == "__main__":
    main()

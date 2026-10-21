"""Aggregate LIBERO-PRO results for one model: <model_root>/<suite>/<perturbation>/summary.json."""

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.LIBERO.eval_files.libero_pro_utils import PERTURBATIONS


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

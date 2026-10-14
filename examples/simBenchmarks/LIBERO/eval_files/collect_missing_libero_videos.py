#!/usr/bin/env python3
"""Collect one genuine success and failure rollout video per LIBERO task.

The fixed-protocol benchmark run remains untouched. If one outcome is absent
from its 50 episodes, this helper searches additional simulator seeds and
copies only the missing outcome video into that task's original shard folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


EVAL_SCRIPT = Path("examples/simBenchmarks/LIBERO/eval_files/eval_libero.py")
OUTCOMES = ("success", "failure")


def parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        start_text, end_text = value.split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if start > end:
            raise ValueError(f"Invalid seed range {value!r}")
        return list(range(start, end + 1))
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one retry seed is required")
    return seeds


def load_task_shards(run_root: Path) -> list[dict]:
    tasks: list[dict] = []
    for suite_dir in sorted(p for p in run_root.iterdir() if p.is_dir() and p.name in {
        "libero_spatial", "libero_object", "libero_goal", "libero_10"
    }):
        summary_path = suite_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing suite summary: {summary_path}")
        suite_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(suite_dir.glob("shard_*/summary.json"))]
        shard_by_task: dict[int, Path] = {}
        for shard_path, shard_summary in zip(sorted(suite_dir.glob("shard_*/summary.json")), shard_summaries):
            for item in shard_summary["task_results"]:
                shard_by_task[int(item["task_id"])] = shard_path.parent
        for result in suite_summary["task_results"]:
            task_id = int(result["task_id"])
            if task_id not in shard_by_task:
                raise RuntimeError(f"Could not map {suite_dir.name} task {task_id} to its shard")
            description = str(result["task_description"])
            prefix = f"rollout_{description.replace(' ', '_')}"
            shard_dir = shard_by_task[task_id]
            tasks.append({
                "suite": suite_dir.name,
                "task_id": task_id,
                "task_description": description,
                "shard_dir": shard_dir,
                "video_names": {outcome: f"{prefix}_{outcome}.mp4" for outcome in OUTCOMES},
            })
    if len(tasks) != 40:
        raise RuntimeError(f"Expected all 40 LIBERO tasks, found {len(tasks)}")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed-range", default="8:27")
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--unnorm-key", default="franka")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Evaluation checkpoint is missing or empty: {args.checkpoint}")
    retry_seeds = parse_seed_range(args.seed_range)
    tasks = load_task_shards(run_root)
    base_env = os.environ.copy()
    base_env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(args.gpu_id),
        "SAVE_VIDEOS": "1",
        "MAX_VIDEOS_PER_OUTCOME": "1",
        "STOP_AFTER_BOTH_VIDEOS": "1",
        "LIBERO_RECORD_INFERENCE_TIMING": "0",
        "PYTHONUNBUFFERED": "1",
    })

    manifest_tasks: list[dict] = []
    for task in tasks:
        shard_dir: Path = task["shard_dir"]
        target_paths = {outcome: shard_dir / task["video_names"][outcome] for outcome in OUTCOMES}
        missing = [outcome for outcome, path in target_paths.items() if not path.is_file() or path.stat().st_size == 0]
        attempts: list[dict] = []
        for seed in retry_seeds:
            if not missing:
                break
            retry_dir = run_root / "video_retries" / task["suite"] / f"task_{task['task_id']:02d}" / f"seed_{seed}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            log_path = retry_dir / "eval.log"
            command = [
                args.python_executable, str(EVAL_SCRIPT),
                "--args.pretrained-path", str(args.checkpoint),
                "--args.host", "127.0.0.1", "--args.port", str(args.port),
                "--args.task-suite-name", task["suite"],
                "--args.num-trials-per-task", str(args.trials_per_task),
                "--args.max-tasks", "-1",
                "--args.task-id-start", str(task["task_id"]),
                "--args.task-id-end", str(task["task_id"] + 1),
                "--args.seed", str(seed),
                "--args.unnorm-key", args.unnorm_key,
                "--args.video-out-path", str(retry_dir),
            ]
            with log_path.open("w", encoding="utf-8") as log_stream:
                completed = subprocess.run(command, env=base_env, stdout=log_stream, stderr=subprocess.STDOUT, check=False)
            attempts.append({"seed": seed, "exit_code": completed.returncode, "log": str(log_path.relative_to(run_root))})
            if completed.returncode != 0:
                print(f"WARN: retry failed for {task['suite']} task {task['task_id']} seed {seed}; see {log_path}", flush=True)
                continue

            for outcome in list(missing):
                source = retry_dir / task["video_names"][outcome]
                target = target_paths[outcome]
                if source.is_file() and source.stat().st_size > 0 and not target.exists():
                    shutil.copy2(source, target)
            missing = [outcome for outcome, path in target_paths.items() if not path.is_file() or path.stat().st_size == 0]

        manifest_tasks.append({
            "suite": task["suite"],
            "task_id": task["task_id"],
            "task_description": task["task_description"],
            "videos": {
                outcome: str(path.relative_to(run_root)) if path.is_file() and path.stat().st_size > 0 else None
                for outcome, path in target_paths.items()
            },
            "retry_attempts": attempts,
            "missing_outcomes": missing,
        })
        print(
            f"{task['suite']} task {task['task_id']}: "
            f"success={'success' not in missing}, failure={'failure' not in missing}",
            flush=True,
        )

    manifest = {
        "required_videos_per_task": {"success": 1, "failure": 1},
        "initial_evaluation_seed": int(os.environ.get("SEED", "7")),
        "retry_seeds": retry_seeds,
        "retry_trials_per_task": args.trials_per_task,
        "complete": all(not item["missing_outcomes"] for item in manifest_tasks),
        "tasks": manifest_tasks,
    }
    path = run_root / "video_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    missing_count = sum(bool(item["missing_outcomes"]) for item in manifest_tasks)
    print(f"Video coverage: {40 - missing_count}/40 tasks complete; manifest: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Evaluate a local Qwen3.5 RoboDojo checkpoint using the official task loop."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from packaging.version import Version
from tqdm import tqdm

from robodojo_eval_assets import format_bytes, inspect_eval_assets
from robodojo_memory_eval_guard import ensure_legacy_robodojo_eval_compatible


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CKPT = ROOT / "playground/Checkpoints/robodojo_v30_qwen35_0p8b_oft/final_model/pytorch_model.pt"
DEFAULT_BASE = Path(
    "/home/jiao/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/"
    "snapshots/2fc06364715b967f1860aea9cf38778875588b17"
)
CHECKPOINT_LABEL = "qwen35_0p8b_oft"
LATENCY_SAMPLE_TASKS = (
    "stack_bowls",
    "match_and_pick_from_conveyor",
    "insert_tubes",
    "fill_pen_holder",
    "general_pickup",
)


def read_result(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", help="Default: all 42 official RoboDojo tasks")
    parser.add_argument(
        "--latency-tasks",
        nargs="+",
        help="Tasks to record inference latency for (default: one representative per suite)",
    )
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", "10")))
    parser.add_argument("--num-envs", type=int, default=int(os.environ.get("NUM_ENVS", "5")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LAYOUT_SEED", "0")))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU_ID", "0")))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("QWEN35_CKPT", DEFAULT_CKPT)))
    parser.add_argument("--base-model", type=Path, default=Path(os.environ.get("QWEN35_BASE", DEFAULT_BASE)))
    parser.add_argument("--output", type=Path, help="Reuse this directory to resume")
    parser.add_argument("--check-only", action="store_true", help="Run validation and strict load only")
    parser.add_argument("--checkpoint-label", default=CHECKPOINT_LABEL,
                        help="Result-directory checkpoint label; legacy default is unchanged")
    parser.add_argument("--expected-parameters", type=int, default=863520846,
                        help="Expected parameter count for preflight; use 0 to skip")
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.seed < 0 or args.gpu < 0:
        parser.error("episodes/num-envs must be positive; seed/GPU must be nonnegative")

    bench = Path(os.environ.get("ROBODOJO_PATH", "/home/data/roboskills/RoboDojo")).resolve()
    policy = bench / "XPolicyLab/policy/starVLA"
    source = Path(os.environ.get("STARVLA_SOURCE_DIR", policy / "source_starvla")).resolve()
    checkpoint = args.checkpoint.resolve()
    base_model = args.base_model.resolve()
    policy_env = Path(os.environ.get("POLICY_ENV", "/home/data/Libero/envs/mint-light")).resolve()
    sim_env = Path(os.environ.get("SIM_ENV", "/home/data/roboskills/envs/robodojo")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Qwen3.5 checkpoint not found: {checkpoint}")
    for required in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
        if not (checkpoint.parent.parent / required).is_file():
            raise FileNotFoundError(f"Checkpoint sidecar missing: {checkpoint.parent.parent / required}")
    ensure_legacy_robodojo_eval_compatible(checkpoint)
    if not base_model.is_dir() or not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"Local Qwen3.5 base model is incomplete: {base_model}")
    if not (policy_env / "bin/python").is_file() or not (sim_env / "bin/python").is_file():
        raise FileNotFoundError("POLICY_ENV and SIM_ENV must point to existing Python environments")

    inventory_spec = importlib.util.spec_from_file_location(
        "task_inventory", bench / "scripts/internal/task_inventory.py"
    )
    if inventory_spec is None or inventory_spec.loader is None:
        raise ImportError(f"Cannot load RoboDojo task inventory from {bench}")
    inventory = importlib.util.module_from_spec(inventory_spec)
    inventory_spec.loader.exec_module(inventory)
    official = [task for group in inventory.DIMENSION_TASKS.values() for task in group]
    tasks = args.tasks or official
    if len(tasks) != len(set(tasks)) or any(task not in official for task in tasks):
        parser.error("--tasks must contain unique official task names")
    latency_tasks = tuple(args.latency_tasks or LATENCY_SAMPLE_TASKS)
    if len(latency_tasks) != len(set(latency_tasks)) or any(task not in official for task in latency_tasks):
        parser.error("--latency-tasks must contain unique official task names")
    try:
        num_envs_overrides = {
            str(task): int(value)
            for task, value in json.loads(
                os.environ.get("STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES", "{}")
            ).items()
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES: {exc}")
    if any(task not in official or value < 1 or value > args.num_envs
           for task, value in num_envs_overrides.items()):
        parser.error("task environment overrides must name official tasks and be in [1, num_envs]")

    layouts = bench / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(args.seed)
    for task in tasks:
        if not (bench / "task/RoboDojo/config" / f"{task}.yml").is_file():
            raise FileNotFoundError(f"Task config missing: {task}")
        if len(list(layouts.glob(f"{task}_[0-9]*.json"))) < args.episodes:
            raise ValueError(f"Insufficient official layouts for {task}: {layouts}")
    asset_report = inspect_eval_assets(bench, tasks, args.episodes, args.seed)
    if asset_report["missing"] or asset_report["issues"]:
        details = [f"missing/corrupt: {path}" for path in asset_report["missing"][:20]]
        details.extend(f"issue: {issue}" for issue in asset_report["issues"][:20])
        raise RuntimeError(
            f"Official assets incomplete ({len(asset_report['missing'])} files, "
            f"{len(asset_report['issues'])} issues; {format_bytes(asset_report['estimated_bytes'])}):\n"
            + "\n".join(details)
        )

    env = os.environ.copy()
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["STARVLA_BASE_VLM"] = str(base_model)
    env["STARVLA_CKPT_PATH"] = str(checkpoint)
    env["STARVLA_UNNORM_KEY"] = "arx_x5"
    env["STARVLA_INCLUDE_STATE"] = "True"
    env["STARVLA_EXECUTE_HORIZON"] = "16"
    env["STARVLA_IMAGE_SIZE"] = "[224,224]"
    env["STARVLA_ROBODOJO_NUM_ENVS"] = str(args.num_envs)
    env["STARVLA_DIRECT_ROBODOJO_CLIENT"] = "1"
    env["STARVLA_CPU_THREADS"] = env.get("STARVLA_CPU_THREADS", "8")
    env["OMNI_KIT_ACCEPT_EULA"] = env.get("OMNI_KIT_ACCEPT_EULA", "YES")
    env["EVAL_NUM"] = str(args.episodes)
    env["ROBODOJO_PATH"] = str(bench)
    env["POLICY_ENV"] = str(policy_env)
    env["SIM_ENV"] = str(sim_env)
    env["PYTHONPATH"] = os.pathsep.join([str(source), str(bench), str(bench / "XPolicyLab")])
    env["XPL_EVAL"] = str(policy / "eval.sh")
    env["STARVLA_ISAAC_RUNTIME_PREFLIGHTED"] = "1"

    print("Checking Isaac runtime and strictly loading all 0.8B checkpoint weights on CPU...", flush=True)
    subprocess.run([str(sim_env / "bin/python"), str(policy / "scripts/check_isaac_runtime.py")], check=True)
    load_check = r"""
import sys
from starVLA.model.framework.base_framework import baseframework, _auto_import_framework_modules
_auto_import_framework_modules()
model = baseframework.from_pretrained(sys.argv[1])
count = sum(p.numel() for p in model.parameters())
expected = int(sys.argv[2])
assert expected == 0 or count == expected, f'unexpected parameter count: {count}, expected {expected}'
print(f'Strict checkpoint load OK: {count:,} parameters')
"""
    subprocess.run(
        [str(policy_env / "bin/python"), "-c", load_check, str(checkpoint), str(args.expected_parameters)],
        cwd=source,
        env=env,
        check=True,
    )

    output = (args.output or Path("/home/data/robodojo_runs/qwen35_0p8b") /
              f"official_eval_{time.strftime('%Y%m%d')}_seed{args.seed}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = dict(
        tasks=tasks,
        episodes=args.episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        gpu=args.gpu,
        checkpoint=str(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        base_vlm="Qwen/Qwen3.5-0.8B",
        base_model_path=str(base_model),
        base_parameters=int(args.expected_parameters) if args.expected_parameters else 866699351,
        benchmark=str(bench),
        policy_python=str(policy_env / "bin/python"),
        simulator=str(sim_env),
        num_envs_overrides=num_envs_overrides,
        layouts="Official randomized layout queue, numeric order",
        run_id=output.name,
    )
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume settings differ from manifest.json; use a new --output")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env["STARVLA_INFERENCE_LATENCY_LOG"] = str(output / "inference_latency.jsonl")
    env["STARVLA_INFERENCE_LATENCY_TASKS"] = ",".join(latency_tasks)
    env["STARVLA_INFERENCE_LATENCY_MODEL"] = args.checkpoint_label
    print(f"Output: {output}\nTasks: {len(tasks)}, layouts/task: {args.episodes}, parallel envs: {args.num_envs}", flush=True)
    print(f"Recording action-chunk inference latency for: {', '.join(latency_tasks)}", flush=True)
    if args.check_only:
        print("Preflight passed. No simulation started.")
        return

    transformer_version = subprocess.run(
        [str(policy_env / "bin/python"), "-c", "import transformers; print(transformers.__version__)"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
    if Version(transformer_version) < Version("5.2.0"):
        raise RuntimeError("Qwen3.5 evaluation requires Transformers >= 5.2")

    rows: list[dict] = []
    process = None
    with tqdm(total=len(tasks) * args.episodes, unit="episode", dynamic_ncols=True,
              smoothing=0, desc="Qwen3.5-0.8B", mininterval=1) as progress:
        try:
            for task in tasks:
                run_id = f"{manifest['run_id']}-{task}"
                result_path = (
                    bench / "eval_result/RoboDojo" / task / "starVLA/arx_x5" /
                    f"{args.seed}_ckpt_name={args.checkpoint_label},action_type=joint" / run_id / "_result.json"
                )
                log_path = output / f"{task}.log"
                data = read_result(result_path)
                counted = min(args.episodes, int(data.get("eval_time", 0)))
                progress.update(counted)
                rc = 0
                if counted < args.episodes:
                    task_env = dict(env, ROBODOJO_RUN_ID=run_id)
                    task_env["STARVLA_ROBODOJO_NUM_ENVS"] = str(
                        num_envs_overrides.get(task, args.num_envs)
                    )
                    command = [
                        "bash", str(Path(__file__).with_name("run_qwen35_robodojo_eval.sh")),
                        "RoboDojo", task, args.checkpoint_label, "arx_x5", "joint", str(args.seed),
                        str(args.gpu), str(args.gpu), str(policy_env), str(sim_env),
                    ]
                    progress.set_description(task)
                    tqdm.write(f"Running {task}; log: {log_path}")
                    with log_path.open("a") as log:
                        process = subprocess.Popen(command, cwd=bench, env=task_env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        while process.poll() is None:
                            data = read_result(result_path)
                            done = min(args.episodes, int(data.get("eval_time", 0)))
                            if done > counted:
                                progress.update(done - counted)
                                counted = done
                            progress.set_postfix(
                                completed=f"{counted}/{args.episodes}",
                                SR=f"{100 * data.get('success_rate', 0):.1f}%",
                                refresh=True,
                            )
                            time.sleep(2)
                        rc = process.returncode
                        stop_process(process)
                        process = None
                    data = read_result(result_path)
                    done = min(args.episodes, int(data.get("eval_time", 0)))
                    progress.update(max(0, done - counted))
                    counted = done
                row = dict(
                    task=task,
                    episodes=counted,
                    success_rate_pct=100 * data["success_rate"] if counted else "",
                    score=data.get("score", ""),
                    result=str(result_path),
                    log=str(log_path),
                    status="complete" if counted == args.episodes and rc == 0 else "incomplete",
                )
                rows.append(row)
                if data:
                    (output / f"{task}.result.json").write_text(json.dumps(data, indent=2) + "\n")
                with (output / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                if row["status"] != "complete":
                    raise RuntimeError(
                        f"{task}: exit={rc}, completed={counted}/{args.episodes}; inspect {log_path}"
                    )
        finally:
            stop_process(process)

    mean_sr = sum(row["success_rate_pct"] for row in rows) / len(rows)
    mean_score = sum(float(row["score"]) for row in rows) / len(rows)
    final = dict(tasks=len(rows), episodes_per_task=args.episodes,
                 mean_task_success_rate_pct=mean_sr, mean_task_score=mean_score)
    (output / "final_summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"Completed: {final}; results: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Reuse --output to resume completed work.", file=sys.stderr)
        sys.exit(130)

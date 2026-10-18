#!/usr/bin/env python3
"""Train RoboDojo's causal dual-arm spatial-memory policy."""
from __future__ import annotations

import os
import json
import shlex
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
CONFIG = Path(os.environ.get("CONFIG_YAML", SCRIPT_DIR / "starvla_robodojo_v30_qwen35_0p8b_spatial_mem.yaml")).resolve()
PYTHON = Path(os.environ.get("PYTHON_BIN", "/public/home/jiaozixun/anaconda3/envs/hm_retarget_mamba/bin/python")).resolve()
DATA_ROOT = Path(os.environ.get("ROBODOJO_SPATIAL_ROOT", "/public/home/jiaozixun/robodojo_spatial_training_bundle_20261013")).resolve()
MEMORY_SIDECAR = Path(os.environ.get(
    "ROBODOJO_MEMORY_SIDECAR_DIR",
    DATA_ROOT / "memory_sidecars/robodojo_v30_r16_s8_visual_v2",
)).resolve()


def environment() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = PYTHON.parent.parent
    env["PATH"] = f"{conda_prefix / 'bin'}:{env.get('PATH', '')}"
    site = conda_prefix / "lib/python3.10/site-packages"
    cudnn = site / "nvidia/cudnn/lib"
    nvidia = sorted(str(p) for p in (site / "nvidia").glob("*/lib") if p.is_dir() and p != cudnn)
    env["LD_LIBRARY_PATH"] = ":".join([str(cudnn), str(site / "torch/lib"), *nvidia, str(conda_prefix / "lib"), env.get("LD_LIBRARY_PATH", "")])
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Keep the memory window bounded; this is the throughput-oriented A800 path.
    env["STARVLA_GRADIENT_ACCUMULATION_STEPS"] = "1"
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("🚀", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def flags(items: list[str]) -> list[str]:
    return [f"--{item.lstrip('-')}" for item in items if item]


def resolve_config(overrides: list[str], env: dict[str, str]) -> dict:
    # Keep the launcher usable from system Python; resolve OmegaConf using the
    # training environment, including any output-path overrides.
    code = """import json, sys
from omegaconf import OmegaConf
cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_dotlist(sys.argv[2:]))
print(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
"""
    result = subprocess.check_output([str(PYTHON), "-c", code, str(CONFIG), *overrides],
                                     cwd=PROJECT_ROOT, env=env, text=True)
    return json.loads(result)


def main() -> None:
    if not PYTHON.is_file():
        raise SystemExit(f"Python executable does not exist: {PYTHON}")
    if not CONFIG.is_file():
        raise SystemExit(f"Training config does not exist: {CONFIG}")
    env = environment()
    paths = []
    if "ROBODOJO_SPATIAL_ROOT" in os.environ:
        paths = [
            f"framework.qwenvl.base_vlm={DATA_ROOT / 'base_model/Qwen3.5-0.8B'}",
            f"datasets.vla_data.data_root_dir={DATA_ROOT / 'data/starvla_robodojo_v30'}",
            f"datasets.vla_data.spatial_sidecar_dir={DATA_ROOT / 'data/starvla_robodojo_v30/spatial_sidecar'}",
            f"datasets.vla_data.memory_sidecar_dir={MEMORY_SIDECAR}",
            f"trainer.pretrained_checkpoint={DATA_ROOT / 'trained_checkpoint/robodojo_v30_qwen35_0p8b_oft/final_model/pytorch_model.pt'}",
        ]
    elif "ROBODOJO_MEMORY_SIDECAR_DIR" in os.environ:
        paths.append(f"datasets.vla_data.memory_sidecar_dir={MEMORY_SIDECAR}")
    stage_a = paths + [item.lstrip('-') for item in shlex.split(os.environ.get("STAGE_A_OVERRIDES", ""))]
    cfg = resolve_config(stage_a, env)
    sidecar = cfg["datasets"]["vla_data"]["spatial_sidecar_dir"]
    stage_a_checkpoint = PROJECT_ROOT / cfg["run_root_dir"] / cfg["run_id"] / "final_model/pytorch_model.pt"
    data = cfg["datasets"]["vla_data"]
    memory = cfg["framework"]["memory"]
    if data.get("memory_sidecar_dir"):
        command = [str(PYTHON), str(SCRIPT_DIR / "prepare_memory_sidecar.py"),
                   "--data-root", str(Path(data["data_root_dir"]) / "RoboDojo_lerobot_v30_video"),
                   "--sidecar", data["memory_sidecar_dir"], "--validate",
                   "--recent-records", str(memory["recent_records"]),
                   "--decision-stride", str(memory["decision_stride_env_steps"])]
        if memory.get("use_pose", False):
            command.append("--use-pose")
        run(command, env)
    run([str(PYTHON), str(SCRIPT_DIR / "prepare_spatial_sidecar.py"), "--sidecar", sidecar, "--validate"], env)
    common = [str(PYTHON), "-m", "accelerate.commands.launch", "--num_processes", "1", "starVLA/training/train_starvla.py", "--config_yaml", str(CONFIG)]
    run(common + flags(stage_a), env)
    if not stage_a_checkpoint.is_file():
        raise SystemExit(f"Stage A checkpoint not found: {stage_a_checkpoint}")
    stage_b = stage_a + [
        f"run_id={cfg['run_id']}_full",
        "trainer.max_train_steps=45000",
        "trainer.num_warmup_steps=0",
        "trainer.freeze_modules=",
        f"trainer.pretrained_checkpoint={stage_a_checkpoint}",
    ] + [item.lstrip('-') for item in shlex.split(os.environ.get("STAGE_B_OVERRIDES", ""))]
    run(common + flags(stage_b), env)
    print("\n✅ RoboDojo causal spatial-memory training completed", flush=True)


if __name__ == "__main__":
    main()

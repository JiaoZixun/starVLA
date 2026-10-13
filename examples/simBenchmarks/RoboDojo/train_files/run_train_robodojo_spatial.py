#!/usr/bin/env python3
"""Launch RoboDojo spatial training in a configurable Conda environment."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
CONFIG = Path(os.environ.get(
    "CONFIG_YAML", SCRIPT_DIR / "starvla_robodojo_v30_qwen35_0p8b_spatial.yaml"
)).resolve()
PYTHON = Path(os.environ.get(
    "PYTHON_BIN", "/public/home/jiaozixun/anaconda3/envs/hm_retarget_mamba/bin/python"
)).resolve()
DATA_ROOT = Path(os.environ.get(
    "ROBODOJO_SPATIAL_ROOT", "/public/home/jiaozixun/robodojo_spatial_training_bundle_20261013"
)).resolve()
SIDECAR = DATA_ROOT / "data/starvla_robodojo_v30/spatial_sidecar"
STAGE_A_CHECKPOINT = (
    PROJECT_ROOT / "playground/Checkpoints/robodojo_v30_qwen35_0p8b_spatial_stage_a"
    / "final_model/pytorch_model.pt"
)


def prepare_environment() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = PYTHON.parent.parent
    env["PATH"] = f"{conda_prefix / 'bin'}:{env.get('PATH', '')}"
    site_packages = conda_prefix / "lib/python3.10/site-packages"
    cudnn_lib = site_packages / "nvidia/cudnn/lib"
    nvidia_libs = [
        str(path) for path in (site_packages / "nvidia").glob("*/lib")
        if path.is_dir() and path != cudnn_lib
    ]
    # PyTorch 2.8 wheels bundle cuDNN 9.10, but this host's global
    # LD_LIBRARY_PATH exposes cuDNN 9.1 first. Prefer the Conda env's CUDA
    # libraries so CUDA kernels do not bind to incompatible system libraries.
    nvidia_libs.sort()
    ld_parts = [
        str(cudnn_lib),
        str(site_packages / "torch/lib"),
        *nvidia_libs,
        str(conda_prefix / "lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")
    # This run uses a real per-device batch of 16 with no gradient accumulation.
    env["STARVLA_GRADIENT_ACCUMULATION_STEPS"] = "1"
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("🚀", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


def as_override_flags(overrides: list[str]) -> list[str]:
    """Pass dot-list overrides through the trainer's argparse normalizer."""
    return [f"--{item.lstrip('-')}" for item in overrides if item]


def main() -> None:
    if not PYTHON.is_file():
        raise SystemExit(f"Python executable does not exist: {PYTHON}")
    if not CONFIG.is_file():
        raise SystemExit(f"Training config does not exist: {CONFIG}")
    if not (DATA_ROOT / "data/starvla_robodojo_v30/RoboDojo_lerobot_v30_video/meta/info.json").is_file():
        raise SystemExit(f"Prepared RoboDojo LeRobot data is missing under {DATA_ROOT}")

    env = prepare_environment()
    run([
        str(PYTHON), str(SCRIPT_DIR / "prepare_spatial_sidecar.py"),
        "--sidecar", str(SIDECAR), "--validate",
    ], cwd=PROJECT_ROOT, env=env)

    # Use per-device batch 16 directly, with no gradient accumulation.
    common = [str(PYTHON), "-m", "accelerate.commands.launch", "--num_processes", "1",
              "starVLA/training/train_starvla.py", "--config_yaml", str(CONFIG)]
    stage_a_overrides = os.environ.get("STAGE_A_OVERRIDES", "").split()
    run(common + as_override_flags(stage_a_overrides), cwd=PROJECT_ROOT, env=env)

    if not STAGE_A_CHECKPOINT.is_file():
        raise SystemExit(f"Stage A checkpoint not found: {STAGE_A_CHECKPOINT}")
    stage_b_overrides = [
        "run_id=robodojo_v30_qwen35_0p8b_spatial_stage_b",
        "trainer.max_train_steps=45000",
        "trainer.num_warmup_steps=0",
        "trainer.freeze_modules=",
        f"trainer.pretrained_checkpoint={STAGE_A_CHECKPOINT}",
    ]
    stage_b_overrides.extend(os.environ.get("STAGE_B_OVERRIDES", "").split())
    run(common + as_override_flags(stage_b_overrides), cwd=PROJECT_ROOT, env=env)

    print("\n✅ RoboDojo spatial training completed: 5,000 frozen-VLM steps + 45,000 full-model steps")


if __name__ == "__main__":
    main()

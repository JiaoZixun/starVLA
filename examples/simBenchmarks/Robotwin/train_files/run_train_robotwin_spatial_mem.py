#!/usr/bin/env python3
"""Launch Robotwin clean spatial-memory training."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PYTHON = Path(os.environ.get("PYTHON_BIN", "/public/home/jiaozixun/anaconda3/envs/hm_retarget_mamba/bin/python"))
CONFIG = Path(os.environ.get("CONFIG_YAML", HERE / "starvla_robotwin_clean_spatial_mem.yaml")).resolve()
DATA_ROOT = Path(os.environ.get("ROBOTWIN_DATA_ROOT", ROOT / "playground/Datasets/RoboTwin/lerobot_clean")).resolve()
MEMORY_SIDECAR_ROOT = Path(os.environ.get(
    "ROBOTWIN_MEMORY_SIDECAR_ROOT", ROOT / "playground/Datasets/RoboTwin/memory_sidecars/robotwin_clean_r16_s8_v2"
)).resolve()
BASE_VLM = Path(os.environ.get(
    "STARVLA_BASE_VLM",
    "/public/home/jiaozixun/robodojo_spatial_training_bundle_20261013/base_model/Qwen3.5-0.8B",
)).expanduser().resolve()

def main() -> None:
    if not CONFIG.is_file():
        raise SystemExit(f"Training config missing: {CONFIG}")
    if not PYTHON.is_file():
        raise SystemExit(f"Python executable missing: {PYTHON}")
    if not BASE_VLM.is_dir() or not (BASE_VLM / "config.json").is_file():
        raise SystemExit(
            "Official Qwen3.5-VL-0.8B checkpoint is missing: "
            f"{BASE_VLM}. Set STARVLA_BASE_VLM to its local directory."
        )
    clean_root = DATA_ROOT / "Clean"
    if not clean_root.is_dir() or not any(clean_root.glob("*/meta/info.json")):
        raise SystemExit(f"Prepared data missing under {DATA_ROOT}; run prepare_robotwin_clean.py first")
    expected = sorted(p.parent.parent.name for p in clean_root.glob("*/meta/info.json"))
    missing = [name for name in expected if not (MEMORY_SIDECAR_ROOT / "Clean" / name / "manifest.json").is_file()]
    if missing:
        raise SystemExit(f"Robotwin memory sidecars missing ({len(missing)}); run prepare_robotwin_memory_sidecars.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    # This launcher intentionally starts from the official Qwen3.5-VL-0.8B
    # weights. Task heads (action, spatial, and memory) are initialized by the
    # model constructor; no RoboDojo/task checkpoint is loaded.
    env["STARVLA_BASE_VLM"] = str(BASE_VLM)
    # The host image exposes an older cuDNN through LD_LIBRARY_PATH. PyTorch's
    # wheel ships the matching cuDNN 9.10.2 under nvidia/cudnn, so put that
    # directory first to avoid a runtime version mismatch on CUDA startup.
    env_root = PYTHON.resolve().parents[1]
    bundled_cudnn_dirs = sorted((env_root / "lib").glob("python*/site-packages/nvidia/cudnn/lib"))
    if bundled_cudnn_dirs:
        bundled_cudnn = bundled_cudnn_dirs[0]
        env["LD_LIBRARY_PATH"] = f"{bundled_cudnn}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    overrides = [item for item in os.environ.get("TRAIN_OVERRIDES", "").split() if item]
    command = [str(PYTHON), "-m", "accelerate.commands.launch", "--num_processes", "1",
               "starVLA/training/train_starvla.py", "--config_yaml", str(CONFIG)]
    command.extend(f"--{item.lstrip('-')}" for item in overrides)
    print("Launching:", " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))

if __name__ == "__main__":
    main()

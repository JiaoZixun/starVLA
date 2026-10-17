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
    "ROBOTWIN_MEMORY_SIDECAR_ROOT", ROOT / "playground/Datasets/RoboTwin/memory_sidecars/robotwin_clean_r16_s8_v1"
)).resolve()

def main() -> None:
    if not CONFIG.is_file():
        raise SystemExit(f"Training config missing: {CONFIG}")
    if not PYTHON.is_file():
        raise SystemExit(f"Python executable missing: {PYTHON}")
    clean_root = DATA_ROOT / "Clean"
    if not clean_root.is_dir() or not any(clean_root.glob("*/meta/info.json")):
        raise SystemExit(f"Prepared data missing under {DATA_ROOT}; run prepare_robotwin_clean.py first")
    expected = sorted(p.parent.parent.name for p in clean_root.glob("*/meta/info.json"))
    missing = [name for name in expected if not (MEMORY_SIDECAR_ROOT / "Clean" / name / "manifest.json").is_file()]
    if missing:
        raise SystemExit(f"Robotwin memory sidecars missing ({len(missing)}); run prepare_robotwin_memory_sidecars.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
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

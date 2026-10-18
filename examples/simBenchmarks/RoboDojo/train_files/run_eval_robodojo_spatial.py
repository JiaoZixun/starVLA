#!/usr/bin/env python3
"""Evaluate the Qwen3.5-0.8B spatial checkpoint on all official RoboDojo tasks.

Step 1 strictly loads the checkpoint in this repo's Conda env and runs one
geometry-free inference to prove the spatial adapter is on the action path.
Step 2 runs the official RoboDojo loop: every task x 50 randomized layouts.
Re-running with the same OUTPUT_DIR resumes completed tasks.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR.parent / "eval_files"))
from robodojo_memory_eval_guard import ensure_legacy_robodojo_eval_compatible

EVALUATOR = SCRIPT_DIR.parent / "eval_files/eval_qwen35_robodojo.py"
BUNDLE = Path(os.environ.get(
    "ROBODOJO_SPATIAL_ROOT", "/public/home/jiaozixun/robodojo_spatial_training_bundle_20261013"
)).resolve()
CHECKPOINT = Path(os.environ.get(
    "SPATIAL_CKPT",
    PROJECT_ROOT / "playground/Checkpoints/robodojo_v30_qwen35_0p8b_spatial_stage_b/final_model/pytorch_model.pt",
)).resolve()
BASE_MODEL = Path(os.environ.get("QWEN35_BASE", BUNDLE / "base_model/Qwen3.5-0.8B")).resolve()
LOCAL_PYTHON = Path(os.environ.get(
    "PYTHON_BIN", "/public/home/jiaozixun/anaconda3/envs/hm_retarget_mamba/bin/python"
)).resolve()
ROBODOJO_PATH = Path(os.environ.get("ROBODOJO_PATH", "/home/data/roboskills/RoboDojo")).resolve()
SIM_ENV = Path(os.environ.get("SIM_ENV", "/home/data/roboskills/envs/robodojo")).resolve()
POLICY_ENV = Path(os.environ.get("POLICY_ENV", "/home/data/Libero/envs/mint-light")).resolve()
EPISODES = os.environ.get("EPISODES", "50")
NUM_ENVS = os.environ.get("NUM_ENVS", "5")
LAYOUT_SEED = os.environ.get("LAYOUT_SEED", "0")
GPU_ID = os.environ.get("GPU_ID", "0")
CHECKPOINT_LABEL = os.environ.get("CHECKPOINT_LABEL", "qwen35_0p8b_spatial")
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    PROJECT_ROOT / f"playground/Eval/robodojo_{CHECKPOINT_LABEL}_ep{EPISODES}_seed{LAYOUT_SEED}",
)).resolve()

SMOKE_TEST = r"""
import sys, numpy as np, torch
from PIL import Image
from starVLA.model.framework.base_framework import baseframework, _auto_import_framework_modules
_auto_import_framework_modules()
model = baseframework.from_pretrained(sys.argv[1]).to("cuda").eval()
print(f"Strict checkpoint load OK: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
calls = []
model.spatial_adapter.register_forward_hook(lambda *_: calls.append(1))
example = {"image": [Image.fromarray(np.zeros((224, 224, 3), np.uint8))],
           "lang": "pick up the bowl", "state": np.zeros((1, 14), np.float32)}
out = model.predict_action([example])["normalized_actions"]
assert calls, "spatial adapter was bypassed at inference"
assert np.isfinite(out).all(), "non-finite actions"
print(f"Inference OK: actions {out.shape}, spatial adapter active", flush=True)
"""


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = LOCAL_PYTHON.parent.parent
    site_packages = conda_prefix / "lib/python3.10/site-packages"
    cudnn_lib = site_packages / "nvidia/cudnn/lib"
    nvidia_libs = sorted(
        str(path) for path in (site_packages / "nvidia").glob("*/lib")
        if path.is_dir() and path != cudnn_lib
    )
    # Same cuDNN ordering fix as run_train_robodojo_spatial.py.
    ld_parts = [str(cudnn_lib), str(site_packages / "torch/lib"), *nvidia_libs, str(conda_prefix / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env["PATH"] = f"{conda_prefix / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["STARVLA_BASE_VLM"] = str(BASE_MODEL)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def eval_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        ROBODOJO_PATH=str(ROBODOJO_PATH), SIM_ENV=str(SIM_ENV), POLICY_ENV=str(POLICY_ENV),
        STARVLA_SOURCE_DIR=str(PROJECT_ROOT),
        EPISODES=EPISODES, NUM_ENVS=NUM_ENVS, LAYOUT_SEED=LAYOUT_SEED, GPU_ID=GPU_ID,
        OMNI_KIT_ACCEPT_EULA=env.get("OMNI_KIT_ACCEPT_EULA", "YES"),
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    for key in ("STARVLA_CPU_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "8")
    # Keep two-way parallelism on the layout that OOMs with five Isaac envs.
    env.setdefault("STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES", '{"pour_by_language":2}')
    # Isaac Sim's RTX/HDF5 plugins need the sim env's GCC 12 runtime ahead of any base Anaconda libs.
    inherited = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "anaconda3/lib" not in p]
    env["LD_LIBRARY_PATH"] = ":".join([str(SIM_ENV / "lib"), "/lib/x86_64-linux-gnu",
                                       "/usr/lib/x86_64-linux-gnu", *inherited])
    return env


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("🚀", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    check_only = "--check-only" in sys.argv[1:]
    extra = [arg for arg in sys.argv[1:] if arg != "--check-only"]
    for path, what in ((CHECKPOINT, "checkpoint"), (BASE_MODEL / "config.json", "Qwen3.5 base model"),
                       (LOCAL_PYTHON, "local Python")):
        if not path.exists():
            raise SystemExit(f"Missing {what}: {path}")

    # Do this before the standalone model smoke test, which has no execution
    # feedback and therefore cannot exercise a causal-memory checkpoint.
    ensure_legacy_robodojo_eval_compatible(CHECKPOINT)

    if os.environ.get("SKIP_SMOKE_TEST") != "1":
        run([str(LOCAL_PYTHON), "-c", SMOKE_TEST, str(CHECKPOINT)], env=local_env())

    missing = [str(p) for p in (ROBODOJO_PATH / "scripts/internal/task_inventory.py",
                                ROBODOJO_PATH / "XPolicyLab/policy/starVLA/eval.sh",
                                SIM_ENV / "bin/python", POLICY_ENV / "bin/python") if not p.exists()]
    if missing:
        raise SystemExit("RoboDojo simulation stack not found; set ROBODOJO_PATH/SIM_ENV/POLICY_ENV:\n  "
                         + "\n  ".join(missing))

    command = [
        str(POLICY_ENV / "bin/python"), "-u", str(EVALUATOR),
        "--checkpoint", str(CHECKPOINT), "--base-model", str(BASE_MODEL),
        "--output", str(OUTPUT_DIR), "--checkpoint-label", CHECKPOINT_LABEL,
        "--expected-parameters", "0",
        "--episodes", EPISODES, "--num-envs", NUM_ENVS, "--seed", LAYOUT_SEED, "--gpu", GPU_ID,
        *(["--check-only"] if check_only else []), *extra,
    ]
    run(command, env=eval_env())
    print(f"\n✅ RoboDojo spatial evaluation finished; results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

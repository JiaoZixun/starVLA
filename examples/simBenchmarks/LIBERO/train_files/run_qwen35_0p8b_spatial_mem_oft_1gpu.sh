#!/usr/bin/env bash
# Single-GPU training for Qwen3.5-0.8B Spatial-Mem OFT on LIBERO.
#
# Stage A: 5 000 steps, memory modules only, loads spatial checkpoint.
# Stage B: resume at step 5 000 and jointly train all modules through step 50 000
#          (45 000 additional steps).
# Global batch size = per_device_batch_size = 32 (single GPU).
#
# Usage:
#   bash run_qwen35_0p8b_spatial_mem_oft_1gpu.sh          # run both stages
#   STAGE=a bash run_qwen35_0p8b_spatial_mem_oft_1gpu.sh  # stage A only
#   STAGE=b bash run_qwen35_0p8b_spatial_mem_oft_1gpu.sh  # stage B only
#
# Environment variables:
#   GPU_ID     — CUDA device index to use (default: 0)
#   STAGE      — a | b | all (default: all)
#   WANDB_MODE — offline | disabled | online (default: offline)
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
STAGE="${STAGE:-all}"
RUN_DIR="playground/Checkpoints/qwen35_0p8b_spatial_mem_oft_v2_libero_50k"
STAGE_A_CONFIG="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_stage_a.yaml"
STAGE_B_CONFIG="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_stage_b.yaml"
QWEN35_REVISION="${QWEN35_REVISION:-2fc06364715b967f1860aea9cf38778875588b17}"
HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
QWEN35_MODEL_DIR="${QWEN35_MODEL_DIR:-${HF_HUB_CACHE}/models--Qwen--Qwen3.5-0.8B/snapshots/${QWEN35_REVISION}}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "${PYTHON}" ]] || fail "Activate a StarVLA environment or set PYTHON=/path/to/python."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || fail "GPU_ID must be a numeric NVIDIA index."
nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 || \
    fail "GPU ${GPU_ID} is unavailable."

case "${STAGE}" in
  a|b|all) ;;
  *) fail "STAGE must be a, b, or all." ;;
esac

# A measured full-unfreeze batch of 32 peaks near 30.3 GiB with AdamW state.
# Leave headroom for CUDA/Triton workspaces and reject GPUs shared with eval.
if [[ "${STAGE}" != a ]]; then
    FREE_GPU_MEMORY_MB="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${FREE_GPU_MEMORY_MB}" =~ ^[0-9]+$ ]] || fail "Could not read free memory for GPU ${GPU_ID}."
    (( FREE_GPU_MEMORY_MB >= 33000 )) || \
        fail "Full batch=32 requires at least 33,000 MiB free; GPU ${GPU_ID} has ${FREE_GPU_MEMORY_MB} MiB. Stop other GPU jobs or select an idle GPU."
fi

cd "${STARVLA_DIR}"
[[ -d starVLA ]] || fail "STARVLA_DIR is not the repository root: ${STARVLA_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# Suppress Python-level UserWarnings (torchvision deprecations, etc.) from the log.
export PYTHONWARNINGS="ignore"
# Qwen3.5 is intentionally loaded from the local Hugging Face snapshot.  The
# offline flags prevent tokenizer/model code from falling back to Hub metadata
# requests when the cache is complete.
export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
# LIBERO/MuJoCo must use NVIDIA hardware rendering through EGL.
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor file is not readable: ${NVIDIA_EGL_VENDOR_JSON}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export STARVLA_DISABLE_DEEPSPEED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py

"${PYTHON}" - <<'PY'
from packaging.version import Version
import torch
import transformers
from transformers import Qwen3_5ForConditionalGeneration

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_name(0).startswith("NVIDIA A100"), torch.cuda.get_device_name(0)
assert torch.cuda.get_device_properties(0).total_memory >= 39 * 2**30
assert Version(transformers.__version__) >= Version("5.2.0"), transformers.__version__
assert Qwen3_5ForConditionalGeneration is not None
print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} transformers={transformers.__version__}")
PY

[[ -d "${QWEN35_MODEL_DIR}" ]] || fail "Local Qwen3.5 snapshot is missing: ${QWEN35_MODEL_DIR}"
"${PYTHON}" - "${QWEN35_MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoProcessor

model_dir = Path(sys.argv[1]).resolve()
required = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]
missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"Local Qwen3.5 snapshot is incomplete ({model_dir}); missing: {missing}")
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
weight_files = sorted(set(index.get("weight_map", {}).values()))
missing_weights = [name for name in weight_files if not (model_dir / name).is_file()]
if missing_weights:
    raise SystemExit(f"Local Qwen3.5 weight shards are missing: {missing_weights}")
# This is the same processor call used by the model wrapper.  Keeping it
# local-only makes a missing cache fail before a long training job starts.
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
print(f"Local Qwen3.5 snapshot ready: {model_dir} ({len(weight_files)} weight shard(s)); processor={type(processor).__name__}")
PY

for dataset in \
  libero_spatial_no_noops_1.0.0_lerobot \
  libero_object_no_noops_1.0.0_lerobot \
  libero_goal_no_noops_1.0.0_lerobot \
  libero_10_no_noops_1.0.0_lerobot; do
    memory_sidecar="playground/DerivedData/libero_spatial_mem_v2/${dataset}"
    spatial_sidecar="playground/DerivedData/libero_spatial_oft_v1/${dataset}"
    [[ -s "${memory_sidecar}/manifest.json" && -s "${memory_sidecar}/obs_rgb_tokens.npy" ]] || \
        fail "memv2 sidecar missing for ${dataset}. Run: ${PYTHON} examples/simBenchmarks/LIBERO/train_files/prepare_memory_v2.py --datasets ${dataset} --no-source-hashes"
    [[ -s "${spatial_sidecar}/manifest.json" && -s "${spatial_sidecar}/obs_eef_pose.npy" ]] || \
        fail "Spatial pose sidecar missing for ${dataset}."
done

run_train() {
    local config_yaml="$1"
    "${PYTHON}" starVLA/training/train_starvla.py \
        --config_yaml "${config_yaml}" \
        --framework.qwenvl.local_model_path "${QWEN35_MODEL_DIR}" \
        --framework.qwenvl.local_files_only true
}

run_stage() {
    local config_yaml="$1"
    local label="$2"
    echo "=== Starting ${label} on GPU ${GPU_ID} ==="
    run_train "${config_yaml}"
}

if [[ "${STAGE}" == a || "${STAGE}" == all ]]; then
    if compgen -G "${RUN_DIR}/checkpoints/steps_*_pytorch_model.pt" >/dev/null 2>&1 || \
       compgen -G "${RUN_DIR}/checkpoints/steps_*_model.safetensors" >/dev/null 2>&1; then
        fail "Spatial-memory checkpoints already exist. Continue with STAGE=b."
    fi
    echo "=== Starting Stage A — memory warmup (5 000 steps, frozen spatial) on GPU ${GPU_ID} ==="
    run_train "${STAGE_A_CONFIG}"
fi

if [[ "${STAGE}" == b || "${STAGE}" == all ]]; then
    [[ -s "${RUN_DIR}/checkpoints/steps_5000_pytorch_model.pt" || \
       -s "${RUN_DIR}/checkpoints/steps_5000_model.safetensors" ]] || \
        fail "Stage B requires the 5 000-step warmup checkpoint at ${RUN_DIR}/checkpoints."
    echo "=== Starting Stage B — full training (steps 5 001–50 000; 45 000 updates) on GPU ${GPU_ID} ==="
    run_train "${STAGE_B_CONFIG}"
fi

echo "=== Training complete. Checkpoints: ${RUN_DIR}/checkpoints/ ==="

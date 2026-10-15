#!/usr/bin/env bash
# Single-GPU training for Qwen3.5-0.8B Spatial-Mem OFT on LIBERO.
#
# Stage A: 5 000 steps, memory modules only, loads spatial checkpoint.
# Stage B: 50 000 steps, all modules, resumes from Stage A checkpoint.
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
RUN_DIR="playground/Checkpoints/qwen35_0p8b_spatial_mem_oft_libero_50k"
STAGE_A_CONFIG="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_stage_a.yaml"
STAGE_B_CONFIG="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_stage_b.yaml"

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

cd "${STARVLA_DIR}"
[[ -d starVLA ]] || fail "STARVLA_DIR is not the repository root: ${STARVLA_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# Suppress Python-level UserWarnings (torchvision deprecations, etc.) from the log.
export PYTHONWARNINGS="ignore"
# LIBERO/MuJoCo must use NVIDIA hardware rendering through EGL.
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export STARVLA_DISABLE_DEEPSPEED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

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

for dataset in \
  libero_spatial_no_noops_1.0.0_lerobot \
  libero_object_no_noops_1.0.0_lerobot \
  libero_goal_no_noops_1.0.0_lerobot \
  libero_10_no_noops_1.0.0_lerobot; do
    memory_sidecar="playground/DerivedData/libero_mem_oft_v1/${dataset}"
    spatial_sidecar="playground/DerivedData/libero_spatial_oft_v1/${dataset}"
    [[ -s "${memory_sidecar}/manifest.json" && -s "${memory_sidecar}/visual_patches.npy" ]] || \
        fail "Memory sidecar missing for ${dataset}. Run: ${PYTHON} ${STAGE_A_CONFIG%train_files/*}train_files/prepare_memory_sidecars.py"
    [[ -s "${spatial_sidecar}/manifest.json" && -s "${spatial_sidecar}/obs_eef_pose.npy" ]] || \
        fail "Spatial pose sidecar missing for ${dataset}."
done

run_stage() {
    local config_yaml="$1"
    local label="$2"
    echo "=== Starting ${label} on GPU ${GPU_ID} ==="
    exec_or_run() {
        if [[ "$1" == "exec" ]]; then
            shift; exec "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "$1"
        else
            "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "$1"
        fi
    }
    if [[ "${label}" == *"Stage B"* ]]; then
        exec "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${config_yaml}"
    else
        "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${config_yaml}"
    fi
}

if [[ "${STAGE}" == a || "${STAGE}" == all ]]; then
    if compgen -G "${RUN_DIR}/checkpoints/steps_*_pytorch_model.pt" >/dev/null 2>&1 || \
       compgen -G "${RUN_DIR}/checkpoints/steps_*_model.safetensors" >/dev/null 2>&1; then
        fail "Spatial-memory checkpoints already exist. Continue with STAGE=b."
    fi
    echo "=== Starting Stage A — memory warmup (5 000 steps, frozen spatial) on GPU ${GPU_ID} ==="
    "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${STAGE_A_CONFIG}"
fi

if [[ "${STAGE}" == b || "${STAGE}" == all ]]; then
    [[ -s "${RUN_DIR}/checkpoints/steps_5000_pytorch_model.pt" || \
       -s "${RUN_DIR}/checkpoints/steps_5000_model.safetensors" ]] || \
        fail "Stage B requires the 5 000-step warmup checkpoint at ${RUN_DIR}/checkpoints."
    echo "=== Starting Stage B — full training (50 000 steps, all modules) on GPU ${GPU_ID} ==="
    exec "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${STAGE_B_CONFIG}"
fi

echo "=== Training complete. Checkpoints: ${RUN_DIR}/checkpoints/ ==="

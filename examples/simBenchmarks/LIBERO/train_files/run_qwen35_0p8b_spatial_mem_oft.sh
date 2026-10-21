#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
STAGE="${STAGE:-all}"
RUN_DIR="playground/Checkpoints/qwen35_0p8b_spatial_mem_oft_v2_libero_50k"
STAGE_A_CONFIG="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_stage_a.yaml"
STAGE_B_CONFIG="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_stage_b.yaml"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON}" ]] || fail "Activate a StarVLA environment or set PYTHON=/path/to/python."
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || fail "GPU_ID must be a numeric NVIDIA index."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."
nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 || fail "GPU ${GPU_ID} is unavailable."
case "${STAGE}" in
  a) CONFIG_YAML="${STAGE_A_CONFIG}" ;;
  b) CONFIG_YAML="${STAGE_B_CONFIG}" ;;
  all) CONFIG_YAML="${STAGE_A_CONFIG}" ;;
  *) fail "STAGE must be a, b, or all." ;;
esac

cd "${STARVLA_DIR}"
[[ -d starVLA ]] || fail "STARVLA_DIR is not the repository root: ${STARVLA_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STARVLA_DISABLE_DEEPSPEED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

if [[ "${STAGE}" == a || "${STAGE}" == all ]]; then
  if compgen -G "${RUN_DIR}/checkpoints/steps_*_pytorch_model.pt" >/dev/null || \
     compgen -G "${RUN_DIR}/checkpoints/steps_*_model.safetensors" >/dev/null; then
    fail "Spatial-memory checkpoints already exist. Continue with STAGE=b."
  fi
elif [[ "${STAGE}" == b ]]; then
  [[ -s "${RUN_DIR}/checkpoints/steps_5000_pytorch_model.pt" || \
     -s "${RUN_DIR}/checkpoints/steps_5000_model.safetensors" ]] || \
    fail "Stage B requires the 5,000-step warmup checkpoint at ${RUN_DIR}/checkpoints."
fi

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
  memory_sidecar="playground/DerivedData/libero_spatial_mem_v2/${dataset}"
  spatial_sidecar="playground/DerivedData/libero_spatial_oft_v1/${dataset}"
  [[ -s "${memory_sidecar}/manifest.json" && -s "${memory_sidecar}/obs_rgb_tokens.npy" ]] || \
    fail "memv2 sidecar missing for ${dataset}. Create it with ${PYTHON} examples/simBenchmarks/LIBERO/train_files/prepare_memory_v2.py --datasets ${dataset} --no-source-hashes"
  [[ -s "${spatial_sidecar}/manifest.json" && -s "${spatial_sidecar}/obs_eef_pose.npy" ]] || \
    fail "Spatial pose sidecar missing for ${dataset}. Run prepare_spatial_sidecars.py first."
done

echo "Starting Qwen3.5-0.8B Spatial-Mem OFT stage ${STAGE} on GPU ${GPU_ID}."
if [[ "${STAGE}" == all ]]; then
  "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${STAGE_A_CONFIG}"
  exec "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${STAGE_B_CONFIG}"
fi
exec "${PYTHON}" starVLA/training/train_starvla.py --config_yaml "${CONFIG_YAML}"

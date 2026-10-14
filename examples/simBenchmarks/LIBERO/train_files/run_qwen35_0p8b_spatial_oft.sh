#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
STAGE="${STAGE:-a}"

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Activate an existing StarVLA conda environment or set PYTHON=/path/to/python." >&2
  exit 1
fi
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]] || ! nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "GPU_ID='${GPU_ID}' is not an available NVIDIA GPU index." >&2
  exit 1
fi
case "${STAGE}" in
  a) CONFIG_YAML="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_oft_stage_a.yaml" ;;
  b) CONFIG_YAML="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_oft_stage_b.yaml" ;;
  all) CONFIG_YAML="examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_oft_stage_a.yaml" ;;
  *) echo "STAGE must be 'a', 'b', or 'all'. Use 'all' for a fresh staged 50K run; use 'b' to resume spatial checkpoints." >&2; exit 2 ;;
esac

SPATIAL_RUN_DIR="playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k"
SPATIAL_CHECKPOINT_DIR="${SPATIAL_RUN_DIR}/checkpoints"
if [[ "${STAGE}" == "a" || "${STAGE}" == "all" ]]; then
  if compgen -G "${SPATIAL_CHECKPOINT_DIR}/steps_*_pytorch_model.pt" >/dev/null || \
     compgen -G "${SPATIAL_CHECKPOINT_DIR}/steps_*_model.safetensors" >/dev/null; then
    echo "Spatial OFT checkpoints already exist. Resume the spatial run with STAGE=b; do not restart warmup from the baseline." >&2
    exit 3
  fi
elif [[ "${STAGE}" == "b" ]]; then
  if [[ ! -s "${SPATIAL_CHECKPOINT_DIR}/steps_5000_pytorch_model.pt" && \
        ! -s "${SPATIAL_CHECKPOINT_DIR}/steps_5000_model.safetensors" ]]; then
    echo "Stage B requires the spatial model's 5,000-step warmup checkpoint at ${SPATIAL_CHECKPOINT_DIR}. Run STAGE=a first." >&2
    exit 3
  fi
fi

cd "${STARVLA_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STARVLA_DISABLE_DEEPSPEED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "Starting Qwen3.5-0.8B SpatialOFT stage ${STAGE} on CUDA ${GPU_ID}."
if [[ "${STAGE}" == "a" || "${STAGE}" == "all" ]]; then
  echo "Steps 1-5000: initialize from the 0.8B LIBERO baseline; freeze Qwen and the OFT action head while training the spatial modules."
else
  echo "Resume the latest spatial checkpoint and train all modules through total step 50000."
fi
echo "Per-device batch size is 32; data loader uses 8 workers."
if [[ "${STAGE}" == "b" ]]; then
  echo "$$" > "${SPATIAL_RUN_DIR}/stage_b.pid"
else
  echo "$$" > "${SPATIAL_RUN_DIR}/stage_a.pid"
fi

"${PYTHON}" - <<'PY'
from pathlib import Path
import torch
import transformers
from transformers import Qwen3_5ForConditionalGeneration
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_name(0).startswith("NVIDIA A100"), torch.cuda.get_device_name(0)
assert torch.cuda.get_device_properties(0).total_memory >= 39 * 2**30
assert transformers.__version__ >= "5.2.0", transformers.__version__
assert Qwen3_5ForConditionalGeneration is not None
print(f"Using {torch.cuda.get_device_name(0)}; torch={torch.__version__}; transformers={transformers.__version__}")
PY

for DATASET in \
  libero_spatial_no_noops_1.0.0_lerobot \
  libero_object_no_noops_1.0.0_lerobot \
  libero_goal_no_noops_1.0.0_lerobot \
  libero_10_no_noops_1.0.0_lerobot; do
  SIDECAR_DIR="playground/DerivedData/libero_spatial_oft_v1/${DATASET}"
  if [[ ! -s "${SIDECAR_DIR}/manifest.json" || ! -s "${SIDECAR_DIR}/obs_eef_pose.npy" ]]; then
    echo "Spatial sidecar is missing for ${DATASET}. Create and audit all sidecars with:" >&2
    echo "  ${PYTHON} examples/simBenchmarks/LIBERO/train_files/prepare_spatial_sidecars.py" >&2
    exit 1
  fi
done

if [[ "${STAGE}" == "all" ]]; then
  "${PYTHON}" starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --run_root_dir playground/Checkpoints
  echo "The 5,000-step spatial warmup completed; resuming from its saved spatial weights for joint training to step 50,000."
  exec "${PYTHON}" starVLA/training/train_starvla.py \
    --config_yaml examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_oft_stage_b.yaml \
    --run_root_dir playground/Checkpoints
else
  exec "${PYTHON}" starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --run_root_dir playground/Checkpoints
fi

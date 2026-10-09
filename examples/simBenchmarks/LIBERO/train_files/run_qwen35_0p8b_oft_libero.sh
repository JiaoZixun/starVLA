#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_oft_libero.yaml}"

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Activate the StarVLA Python environment or set PYTHON=/path/to/python." >&2
  exit 1
fi

cd "${STARVLA_DIR}"
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Training config not found: ${CONFIG_YAML}" >&2
  exit 1
fi

GPU_INFO="$(nvidia-smi -i "${GPU_ID}" --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
if [[ "${GPU_INFO}" != *A100* || "${GPU_INFO}" != *40960* ]]; then
  echo "Expected an A100 40 GB at GPU_ID=${GPU_ID}; found '${GPU_INFO:-no GPU}'." >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
from packaging.version import Version
import transformers
from transformers import Qwen3_5ForConditionalGeneration

if Version(transformers.__version__) < Version("5.2.0"):
    raise SystemExit(
        f"Qwen3.5 needs transformers>=5.2.0; found {transformers.__version__}"
    )
print(f"Qwen3.5 loader ready (transformers {transformers.__version__}).")
PY

for suite in spatial object goal 10; do
  dataset="${STARVLA_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA/libero_${suite}_no_noops_1.0.0_lerobot"
  if [[ ! -s "${dataset}/meta/info.json" || ! -s "${dataset}/meta/modality.json" ]]; then
    echo "LIBERO-${suite} dataset is not ready: ${dataset}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STARVLA_DISABLE_DEEPSPEED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "Starting Qwen3.5-0.8B + QwenOFT on ${GPU_INFO}."
echo "Training steps: 50000; per-device batch size is fixed in ${CONFIG_YAML}."
echo "Init: official Qwen3.5-0.8B base plus native 1024-wide OFT head; the incompatible 4B head is skipped."
echo "An OOM stops this run; the next launch reloads the latest saved 0.8B model checkpoint."

exec "${PYTHON}" starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_root_dir playground/Checkpoints \
  --run_id qwen35_0p8b_oft_libero_50k \
  --trainer.max_train_steps 50000

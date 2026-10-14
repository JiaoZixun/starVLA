#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
GPU_ID="${GPU_ID:-${gpu_id:-0}}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
tasks_per_gpu="${tasks_per_gpu:-1}"
your_ckpt="${your_ckpt:-/path/to/checkpoint.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_parallel_eval}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 TASK_SUITE START_IDX END_IDX" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
  | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -Fxq -- "${GPU_ID}"; then
  echo "GPU_ID='${GPU_ID}' is not an available NVIDIA GPU." >&2
  exit 1
fi
if [[ "${MUJOCO_GL:-egl}" != "egl" || "${PYOPENGL_PLATFORM:-egl}" != "egl" ]]; then
  echo "LIBERO-plus evaluation requires NVIDIA EGL (MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl)." >&2
  exit 2
fi
if [[ ! -r "${NVIDIA_EGL_VENDOR_JSON}" ]]; then
  echo "NVIDIA EGL vendor file is not readable: ${NVIDIA_EGL_VENDOR_JSON}" >&2
  exit 1
fi
if [[ ! -f "${your_ckpt}" || ! -s "${your_ckpt}" ]]; then
  echo "Checkpoint is missing or empty: ${your_ckpt}" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}/libero/libero/assets" || ! -f "${LIBERO_HOME}/libero/libero/benchmark/task_classification.json" ]]; then
  echo "LIBERO-plus source/assets are incomplete under ${LIBERO_HOME}." >&2
  exit 1
fi
if [[ ! "${tasks_per_gpu}" =~ ^[1-9][0-9]*$ ]]; then
  echo "tasks_per_gpu must be a positive integer." >&2
  exit 2
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}/libero:${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export LIBERO_HOME

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  "${LIBERO_PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["LIBERO_HOME"]) / "libero" / "libero"
config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
config_path.parent.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
PY
fi

"${LIBERO_PYTHON}" \
  "${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py"

task_suite_name=$1
start_idx=$2
end_idx=$3
num_trials_per_task=1
# torchrun --nproc_per_node=1 ./examples/simBenchmarks/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \

total=$((end_idx - start_idx))
chunk_size=$((total / tasks_per_gpu))
remainder=$((total % tasks_per_gpu))
current_start=$start_idx

worker_pids=()
for ((i=0; i<tasks_per_gpu; i++)); do

    if [ $i -lt $remainder ]; then
        current_end=$((current_start + chunk_size + 1))
    else
        current_end=$((current_start + chunk_size))
    fi


    if [ $current_end -gt $end_idx ]; then
        current_end=$end_idx
    fi

    echo "Part $((i)): start=$current_start, end=$current_end ([$current_start, $current_end))"
    # torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$((MASTER_ADDR+i)) --master_port=$MASTER_PORT 
    if (( current_end > current_start )); then
        "${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
        --pretrained_path "$your_ckpt" \
        --task_suite_name "$task_suite_name" \
        --num_trials_per_task "$num_trials_per_task" \
        --output_dir "$output_dir" \
        --start_idx "$current_start" \
        --end_idx "$current_end" \
        --save_videos "$([[ "${SAVE_VIDEOS}" == "1" ]] && echo true || echo false)" &
        worker_pids+=("$!")
    fi

    current_start=$current_end

    if [ $current_start -ge $end_idx ]; then
        break
    fi
done


worker_status=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        worker_status=1
    fi
done
exit "$worker_status"

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/simBenchmarks/LIBERO-plus/eval_files/aggregate_results.py

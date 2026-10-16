#!/usr/bin/env bash
set -Eeuo pipefail

# Resume baseline libero_10 evaluation from task 1875, then launch full spatial evaluation.
# Baseline completes without acceleration (to match existing results); spatial uses acceleration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO-plus}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"

BASELINE_CKPT="${BASELINE_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
SPATIAL_CKPT="${SPATIAL_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"

# Resume into the existing baseline result directory
BASELINE_RUN_ID="qwen35_0p8b_libero_plus_cuda1_20261015_173823"
BASELINE_RESULT_ROOT="${STARVLA_DIR}/results/${BASELINE_RUN_ID}/baseline"

# New run ID for spatial
SPATIAL_RUN_ID="qwen35_0p8b_spatial_libero_plus_$(date +%Y%m%d_%H%M%S)"
SPATIAL_RESULT_ROOT="${STARVLA_DIR}/results/${SPATIAL_RUN_ID}/spatial"

EVAL_GPU_ID="${EVAL_GPU_ID:-1}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "${LIBERO_PYTHON}" ]] || fail "Python not found: ${LIBERO_PYTHON}"
[[ -d "${STARVLA_DIR}/starVLA" ]] || fail "STARVLA_DIR invalid: ${STARVLA_DIR}"
[[ -d "${LIBERO_HOME}/libero/libero" ]] || fail "LIBERO-plus missing: ${LIBERO_HOME}"
[[ -s "${BASELINE_CKPT}" ]] || fail "Baseline checkpoint missing: ${BASELINE_CKPT}"
[[ -s "${SPATIAL_CKPT}" ]] || fail "Spatial checkpoint missing: ${SPATIAL_CKPT}"
[[ -d "${BASELINE_RESULT_ROOT}/logs" ]] || fail "Baseline result directory missing: ${BASELINE_RESULT_ROOT}"
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor JSON missing: ${NVIDIA_EGL_VENDOR_JSON}"

nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
  | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fxq "${EVAL_GPU_ID}" \
  || fail "GPU ${EVAL_GPU_ID} unavailable"

export STARVLA_DIR LIBERO_HOME LIBERO_PYTHON
export HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"

mkdir -p "${LIBERO_CONFIG_PATH}"
"${LIBERO_PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["LIBERO_HOME"]) / "libero" / "libero"
config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
if not config_path.exists():
    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent / "datasets"),
        "assets": str(root / "assets"),
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
PY

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === PHASE 1: Resume baseline libero_10 worker [1260, 2519) ==="
echo "Output: ${BASELINE_RESULT_ROOT}"

# The interrupted worker reached task 1875 but eval_libero_model.py only writes
# its JSON after the whole range finishes, so [1260, 1875) has no surviving
# result.  Restart the entire [1260, 2519) half; it pairs with the existing
# 0_to_1260.json to cover all 2519 tasks.
#
# Run this half as a SINGLE worker.  np.random.seed(7) is applied once per
# process and glass_blur draws its offsets from that global stream, so the
# sensor-noise realization of a task depends on the worker partition.  The
# other three baseline suites, and every suite in phase 2, use two workers
# split at the suite midpoint.  Keeping [1260, 2519) in one worker reproduces
# exactly the partition phase 2 will use for this suite, so the baseline and
# spatial numbers stay comparable.
if ! GPU_ID="${EVAL_GPU_ID}" tasks_per_gpu=1 your_ckpt="${BASELINE_CKPT}" \
  output_dir="${BASELINE_RESULT_ROOT}" SAVE_VIDEOS="0" \
  bash "${SCRIPT_DIR}/eval_libero_in_one.sh" libero_10 1260 2519 \
  >"${BASELINE_RESULT_ROOT}/libero_10_resume_1260_2519.launcher.log" 2>&1; then
  echo "Baseline libero_10 resume failed. See ${BASELINE_RESULT_ROOT}/libero_10_resume_1260_2519.launcher.log" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Baseline libero_10 resume complete. Aggregating baseline results."
"${LIBERO_PYTHON}" "${SCRIPT_DIR}/aggregate_results.py" --root_path "${BASELINE_RESULT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === PHASE 2: Launch full spatial evaluation (upstream rendering) ==="
echo "Output: ${SPATIAL_RESULT_ROOT}"

mkdir -p "${SPATIAL_RESULT_ROOT}"

# Launch all 4 suites for spatial in parallel
declare -a pids=()
for suite in libero_10 libero_goal libero_object libero_spatial; do
  case "${suite}" in
    libero_10)    total=2519 ;;
    libero_goal)  total=2591 ;;
    libero_object) total=2518 ;;
    libero_spatial) total=2402 ;;
  esac

  GPU_ID="${EVAL_GPU_ID}" tasks_per_gpu="${TASKS_PER_GPU}" your_ckpt="${SPATIAL_CKPT}" \
    output_dir="${SPATIAL_RESULT_ROOT}" SAVE_VIDEOS="0" \
    bash "${SCRIPT_DIR}/eval_libero_in_one.sh" "${suite}" 0 "${total}" \
    >"${SPATIAL_RESULT_ROOT}/${suite}.launcher.log" 2>&1 &
  pids+=("$!")
done

spatial_failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || spatial_failed=1
done

if (( spatial_failed == 0 )); then
  "${LIBERO_PYTHON}" "${SCRIPT_DIR}/aggregate_results.py" --root_path "${SPATIAL_RESULT_ROOT}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === COMPLETE ==="
echo "Baseline (complete): ${BASELINE_RESULT_ROOT}/overall_results.json"
echo "Spatial: ${SPATIAL_RESULT_ROOT}/overall_results.json"
echo "Status: baseline=0, spatial=${spatial_failed}"

exit "${spatial_failed}"

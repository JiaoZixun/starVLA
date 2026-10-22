#!/usr/bin/env bash
# Sequential pipeline for Qwen3.5-0.8B Spatial-Mem single-stage training.
#
#   TRAIN_STAGE=full (default): memv2, RGB event visuals.
#   TRAIN_STAGE=v3: proprio/multi-layer bridge + frozen-Qwen event visuals
#                   (requires prepare_qwen_visual_sidecar.py output).
#   RUN_LIBERO_PLUS=0 skips LIBERO-plus.
#
#   1. Train on TRAIN_GPU (default 0): Spatial 50k init, all modules, 50k steps, batch 32.
#   2. Require the 50 000-step checkpoint.
#   3. In parallel:
#        standard LIBERO on STD_GPU  (default 0): 4 suites x 50 trials
#        LIBERO-plus     on PLUS_GPU (default 1): 4 suites x 1 trial (official)
#
# Worker counts keep the protocols of the earlier baseline/Spatial/mem-v1 runs:
# standard LIBERO uses 8 simulator workers and inference batch 4; LIBERO-plus
# runs the 4 suites concurrently with TASKS_PER_GPU=2, because the LIBERO-plus
# sensor-noise stream depends on the per-suite worker split.
#
# Usage (detached):
#   setsid nohup bash examples/simBenchmarks/LIBERO/train_files/run_spatial_mem_full_train_then_eval.sh \
#     > results/logs/spatial_mem_full_pipeline.log 2>&1 &
# Re-running resumes training from the latest checkpoint and skips finished steps.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
TRAIN_GPU="${TRAIN_GPU:-0}"
STD_GPU="${STD_GPU:-0}"
PLUS_GPU="${PLUS_GPU:-1}"
STD_SIM_WORKERS="${STD_SIM_WORKERS:-8}"
STD_INFERENCE_BATCH="${STD_INFERENCE_BATCH:-4}"
PLUS_TASKS_PER_GPU="${PLUS_TASKS_PER_GPU:-2}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

TRAIN_STAGE="${TRAIN_STAGE:-full}"
RUN_LIBERO_PLUS="${RUN_LIBERO_PLUS:-1}"
case "${TRAIN_STAGE}" in
  full) RUN_ID="qwen35_0p8b_spatial_mem_oft_v2_full_libero_50k" ;;
  v3) RUN_ID="qwen35_0p8b_spatial_mem_oft_v3_bridge_qwenvis_libero_50k" ;;
  *) echo "TRAIN_STAGE must be full or v3" >&2; exit 1 ;;
esac
RUN_DIR="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}"
CKPT="${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${STARVLA_DIR}/results/${RUN_ID}/pipeline_${STAMP}"
STD_RUN_ID="${RUN_ID}_libero_std_${STAMP}"
STD_ROOT="${STARVLA_DIR}/results/libero_eval/${STD_RUN_ID}"
PLUS_ROOT="${RESULT_ROOT}/libero_plus"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

cd "${STARVLA_DIR}"
mkdir -p "${RUN_DIR}/logs" "${RESULT_ROOT}"
STATUS_FILE="${RUN_DIR}/logs/pipeline.status"
status() { printf 'updated_at=%s\nstate=%s\nresult_root=%s\n' "$(date '+%F %T %Z')" "$1" "${RESULT_ROOT}" > "${STATUS_FILE}"; }

exec 9>"${RUN_DIR}/logs/pipeline.lock"
flock -n 9 || fail "Another pipeline runner holds ${RUN_DIR}/logs/pipeline.lock"
echo "$$" > "${RUN_DIR}/logs/pipeline.pid"
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor JSON not readable: ${NVIDIA_EGL_VENDOR_JSON}"

# ---------------------------------------------------------------- 1. training
if [[ -s "${CKPT}" ]]; then
  log "50k checkpoint already exists; skipping training."
else
  status training
  log "Training on GPU ${TRAIN_GPU}; log: ${RUN_DIR}/logs/train.log"
  PYTHON="${PYTHON}" GPU_ID="${TRAIN_GPU}" STAGE="${TRAIN_STAGE}" WANDB_MODE=offline \
    bash examples/simBenchmarks/LIBERO/train_files/run_qwen35_0p8b_spatial_mem_oft_1gpu.sh \
    >> "${RUN_DIR}/logs/train.log" 2>&1 || { status training_failed; fail "training failed; see ${RUN_DIR}/logs/train.log"; }
fi
[[ -s "${CKPT}" ]] || { status training_incomplete; fail "training exited without ${CKPT}"; }
log "Checkpoint ready: ${CKPT} ($(stat -c %s "${CKPT}") bytes)"

# ------------------------------------------------------------- 2. evaluation
export HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}" HF_HUB_CACHE="${HF_HOME:-/data1/jiaozx/.cache/huggingface}/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"

EVAL_GPUS=("${STD_GPU}")
[[ "${RUN_LIBERO_PLUS}" == 1 ]] && EVAL_GPUS+=("${PLUS_GPU}")
for gpu in "${EVAL_GPUS[@]}"; do
  nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader >/dev/null || fail "GPU ${gpu} unavailable"
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" "${PYTHON}" \
    examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py || fail "NVIDIA EGL check failed on GPU ${gpu}"
done

status evaluating
log "Standard LIBERO on GPU ${STD_GPU} (${STD_SIM_WORKERS} workers, batch ${STD_INFERENCE_BATCH}) -> ${STD_ROOT}"
(
  export LIBERO_HOME="${STARVLA_DIR}/playground/LIBERO" LIBERO_CONFIG_PATH="${STARVLA_DIR}/playground/LIBERO/libero"
  export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
  GPU_ID="${STD_GPU}" PYTHON="${PYTHON}" CONDA_ENV="" SIM_WORKERS="${STD_SIM_WORKERS}" \
  INFERENCE_BATCH_SIZE="${STD_INFERENCE_BATCH}" NUM_TRIALS_PER_TASK=50 CKPT="${CKPT}" \
  EXPECTED_CKPT_BYTES=auto WAIT_FOR_CHECKPOINT=0 RUN_ID="${STD_RUN_ID}" UNNORM_KEY=franka \
  TASK_SUITES_CSV=libero_spatial,libero_object,libero_goal,libero_10 SAVE_VIDEOS=0 \
  LIBERO_RECORD_INFERENCE_TIMING=1 \
    bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh
) > "${RESULT_ROOT}/libero_std.log" 2>&1 &
std_pid=$!

plus_pid=""
if [[ "${RUN_LIBERO_PLUS}" == 1 ]]; then
log "LIBERO-plus on GPU ${PLUS_GPU} (4 suites concurrently, TASKS_PER_GPU=${PLUS_TASKS_PER_GPU}) -> ${PLUS_ROOT}"
(
  unset PYTHONPATH LIBERO_HOME LIBERO_CONFIG_PATH
  EVAL_GPU_ID="${PLUS_GPU}" LIBERO_PYTHON="${PYTHON}" MODEL_SELECTION=mem MEM_CKPT="${CKPT}" \
  MEM_CKPT_BYTES=auto TASKS_PER_GPU="${PLUS_TASKS_PER_GPU}" NUM_TRIALS_PER_TASK=1 SAVE_VIDEOS=0 \
  RUN_ID="${RUN_ID}_libero_plus_${STAMP}" RESULT_ROOT="${PLUS_ROOT}" \
    bash examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_local_dual_libero_plus.sh
) > "${RESULT_ROOT}/libero_plus.log" 2>&1 &
plus_pid=$!
fi

std_status=0; plus_status=0
wait "${std_pid}" || std_status=$?
log "Standard LIBERO finished (status=${std_status})."
if [[ -n "${plus_pid}" ]]; then
  wait "${plus_pid}" || plus_status=$?
  log "LIBERO-plus finished (status=${plus_status})."
fi

# ---------------------------------------------------------------- 3. summary
"${PYTHON}" - "${STD_ROOT}" "${PLUS_ROOT}/mem" "${RESULT_ROOT}/evaluation_summary.json" "${CKPT}" <<'PY'
import json, sys
from pathlib import Path
std_root, plus_root, out, ckpt = map(Path, sys.argv[1:5])
summary = {"checkpoint": str(ckpt), "libero_standard": {}, "libero_plus": None}
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    path = std_root / suite / "summary.json"
    if path.is_file():
        data = json.loads(path.read_text())
        summary["libero_standard"][suite] = {k: data[k] for k in ("success_rate", "episodes", "successes")}
rates = [v["success_rate"] for v in summary["libero_standard"].values()]
if len(rates) == 4:
    summary["libero_standard"]["average"] = sum(rates) / 4
plus = plus_root / "overall_results.json"
if plus.is_file():
    summary["libero_plus"] = json.loads(plus.read_text())
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)[:3000])
PY

if (( std_status == 0 && plus_status == 0 )); then
  status completed
  log "Pipeline complete: ${RESULT_ROOT}/evaluation_summary.json"
else
  status "eval_failed(std=${std_status},plus=${plus_status})"
  fail "Evaluation failed (std=${std_status}, plus=${plus_status}); see ${RESULT_ROOT}/*.log"
fi

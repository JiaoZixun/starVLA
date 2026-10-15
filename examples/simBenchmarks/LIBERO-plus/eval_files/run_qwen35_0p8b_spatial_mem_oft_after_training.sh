#!/usr/bin/env bash
# Post-training LIBERO-plus evaluation queue for Qwen3.5-0.8B Spatial + causal memory OFT.
#
# Waits for the spatial-memory training stages to exit and for the evaluation
# GPU to drain, then runs the four LIBERO-plus suites with the memory
# checkpoint. Each suite is split across TASKS_PER_GPU workers.
#
# Usage:
#   bash run_qwen35_0p8b_spatial_mem_oft_after_training.sh
#
# Environment:
#   EVAL_GPU_ID       physical GPU for evaluation (default: 1)
#   TASKS_PER_GPU     workers per suite (default: 2)
#   SAVE_VIDEOS       1 to record videos (default: 0)
#   WAIT_FOR_TRAINING 0 to skip the training wait (default: 1)
set -euo pipefail

REPO="${STARVLA_DIR:-/data1/jiaozx/starVLA}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="qwen35_0p8b_spatial_mem_oft_libero_50k"
RUN_DIR="${REPO}/playground/Checkpoints/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_FILE="${LOG_DIR}/posttrain_libero_plus_queue.status"
CHECKPOINT="${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt"
EVAL_GPU_ID="${EVAL_GPU_ID:-1}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
MIN_FREE_GPU_MEM_MIB="${MIN_FREE_GPU_MEM_MIB:-16000}"

mkdir -p "${LOG_DIR}"

exec 9>"${LOG_DIR}/posttrain_libero_plus_queue.lock"
if ! flock -n 9; then
  echo "Another LIBERO-plus post-training queue runner already holds the lock."
  exit 0
fi
echo "$$" > "${LOG_DIR}/posttrain_libero_plus_queue.pid"

write_status() {
  local status="$1"
  local message="$2"
  local tmp="${STATUS_FILE}.tmp"
  {
    printf 'updated_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'status=%s\n' "${status}"
    printf 'message=%s\n' "${message}"
  } > "${tmp}"
  mv "${tmp}" "${STATUS_FILE}"
}

pidfile_is_training_process() {
  local pidfile="$1" pid command_line
  [[ -s "${pidfile}" ]] || return 1
  read -r pid < "${pidfile}"
  [[ "${pid}" =~ ^[0-9]+$ && -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${command_line}" == *qwen35_0p8b_spatial_mem_oft* ]]
}

active_training_pids() {
  local pidfile
  for pidfile in \
    "${RUN_DIR}/stage_a.pid" \
    "${RUN_DIR}/stage_b.pid" \
    "${RUN_DIR}/stage_supervisor.pid" \
    "${RUN_DIR}/launcher.pid"; do
    if pidfile_is_training_process "${pidfile}"; then
      cat "${pidfile}"
    fi
  done | sort -nu
}

if rg -q '^status=completed$' "${STATUS_FILE}" 2>/dev/null; then
  echo "The LIBERO-plus post-training queue is already complete."
  exit 0
fi

echo "LIBERO-plus post-training evaluation queue started: PID $$"
echo "Checkpoint under watch: ${CHECKPOINT}"
write_status "waiting_for_training" "Will evaluate after all spatial-memory training stages exit."

if [[ "${WAIT_FOR_TRAINING}" == "1" ]]; then
  while true; do
    live_pids="$(active_training_pids)"
    if [[ -z "${live_pids}" ]]; then
      break
    fi
    write_status "waiting_for_training" "Active spatial-memory training/supervisor PIDs: ${live_pids//$'\n'/,}"
    sleep 30 9>&-
  done
fi

if [[ ! -s "${CHECKPOINT}" ]]; then
  write_status "failed" "Training exited without the final 50,000-step checkpoint: ${CHECKPOINT}"
  echo "ERROR: final spatial-memory checkpoint is missing; evaluation was not started." >&2
  exit 1
fi
if [[ -s "${RUN_DIR}/summary.jsonl" ]] && ! tail -n 20 "${RUN_DIR}/summary.jsonl" | rg -q '"steps": 50000'; then
  write_status "failed" "Final checkpoint exists but summary.jsonl does not confirm completed step 50,000."
  echo "ERROR: training completion could not be confirmed from summary.jsonl." >&2
  exit 1
fi

if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
  | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fxq "${EVAL_GPU_ID}"; then
  write_status "failed" "Evaluation GPU ${EVAL_GPU_ID} is unavailable."
  echo "ERROR: GPU ${EVAL_GPU_ID} is not available." >&2
  exit 1
fi

# A baseline/spatial LIBERO-plus run may still hold the GPU. Wait for it to
# drain instead of competing for memory with it.
while true; do
  free_mib="$(nvidia-smi -i "${EVAL_GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_GPU_MEM_MIB )); then
    break
  fi
  write_status "waiting_for_evaluation_gpu" \
    "GPU ${EVAL_GPU_ID} requires at least ${MIN_FREE_GPU_MEM_MIB} MiB free; current ${free_mib:-unknown} MiB."
  echo "Waiting for ${MIN_FREE_GPU_MEM_MIB} MiB free on evaluation GPU ${EVAL_GPU_ID}; current free memory=${free_mib:-unknown} MiB."
  sleep 30 9>&-
done

run_id="qwen35_0p8b_spatial_mem_oft_50k_libero_plus_$(date '+%Y%m%d_%H%M%S')"
result_root="${REPO}/results/${run_id}"
write_status "starting_evaluation" \
  "Training completed; launching four-suite LIBERO-plus evaluation on GPU ${EVAL_GPU_ID}."
echo "Starting LIBERO-plus evaluation: ${run_id}"

set +e
cd "${REPO}"
EVAL_GPU_ID="${EVAL_GPU_ID}" TASKS_PER_GPU="${TASKS_PER_GPU}" SAVE_VIDEOS="${SAVE_VIDEOS}" \
MODEL_SELECTION=mem MEM_CKPT="${CHECKPOINT}" MEM_CKPT_BYTES=auto \
RUN_ID="${run_id}" RESULT_ROOT="${result_root}" \
bash "${SCRIPT_DIR}/parallel_eval/run_local_dual_libero_plus.sh"
result=$?
set -e

if (( result == 0 )); then
  write_status "completed" "LIBERO-plus evaluation completed; results at results/${run_id}."
  echo "LIBERO-plus queue finished. Results: ${result_root}"
  exit 0
fi

write_status "failed" "LIBERO-plus evaluation failed with status ${result}; inspect logs under ${result_root}."
echo "ERROR: LIBERO-plus evaluation failed with status ${result}. See ${result_root}." >&2
exit 1

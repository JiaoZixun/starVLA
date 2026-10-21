#!/usr/bin/env bash
# Post-training LIBERO evaluation queue for Qwen3.5-0.8B Spatial + causal memory OFT.
#
# Waits for the spatial-memory training stages to exit, verifies that the
# 50 000-step checkpoint really is the end of the run, then evaluates the four
# standard LIBERO suites with parallel simulator workers on one GPU.
#
# Usage:
#   bash run_qwen35_0p8b_spatial_mem_oft_after_training.sh
#
# Environment:
#   EVAL_GPU_ID          physical GPU for evaluation (default: 0)
#   NUM_TRIALS_PER_TASK  episodes per task (default: 50)
#   SAVE_VIDEOS          1 to record per-task outcome videos (default: 0)
#   WAIT_FOR_TRAINING    0 to skip the training wait (default: 1)
set -euo pipefail

REPO="${STARVLA_DIR:-/data1/jiaozx/starVLA}"
RUN_ID="qwen35_0p8b_spatial_mem_oft_v2_libero_50k"
RUN_DIR="${REPO}/playground/Checkpoints/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_FILE="${LOG_DIR}/posttrain_eval_queue.status"
CHECKPOINT="${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt"
PYTHON="${PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
MIN_FREE_GPU_MEM_MIB="${MIN_FREE_GPU_MEM_MIB:-16000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

mkdir -p "${LOG_DIR}"

exec 9>"${LOG_DIR}/posttrain_eval_queue.lock"
if ! flock -n 9; then
  echo "Another post-training evaluation queue runner already holds the lock."
  exit 0
fi
echo "$$" > "${LOG_DIR}/posttrain_eval_queue.pid"

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
  echo "The post-training evaluation queue is already complete."
  exit 0
fi

echo "Post-training LIBERO evaluation queue started: PID $$"
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

# Wait for the evaluation GPU to be free; a concurrent run will otherwise
# compete for memory and silently slow both evaluations down.
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

write_status "starting_evaluation" "Training completed; launching four-suite ${NUM_TRIALS_PER_TASK}-episode/task evaluation on GPU ${EVAL_GPU_ID}."
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  run_id="qwen35_0p8b_spatial_mem_oft_50k_eval_$(date '+%Y%m%d_%H%M%S')_attempt${attempt}"
  echo "Starting evaluation attempt ${attempt}/${MAX_ATTEMPTS}: ${run_id}"
  set +e
  cd "${REPO}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHON="${PYTHON}" \
  GPU_ID="${EVAL_GPU_ID}" SIM_WORKERS=8 INFERENCE_BATCH_SIZE=4 INFERENCE_BATCH_WAIT_MS=12 \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" MAX_TASKS=-1 SAVE_VIDEOS="${SAVE_VIDEOS}" \
  LIBERO_RECORD_INFERENCE_TIMING=1 \
  WRITE_EVAL_REPORT=1 EXPECTED_CKPT_BYTES=auto WAIT_FOR_CHECKPOINT=0 \
  MIN_FREE_GPU_MEM_MIB="${MIN_FREE_GPU_MEM_MIB}" CKPT="${CHECKPOINT}" RUN_ID="${run_id}" \
  SEED=7 PORT=6694 \
  bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh
  result=$?
  set -e
  if (( result == 0 )) && [[ -s "${REPO}/results/libero_eval/${run_id}/evaluation_report.md" ]]; then
    write_status "completed" "Four-suite success rate and inference timing are recorded at results/libero_eval/${run_id}."
    echo "Evaluation queue finished. Report: ${REPO}/results/libero_eval/${run_id}/evaluation_report.md"
    exit 0
  fi
  echo "Evaluation attempt ${attempt} failed with status ${result}; logs remain under results/libero_eval/${run_id}/logs." >&2
  write_status "retrying_evaluation" "Attempt ${attempt} failed; a fresh run will retry without overwriting its logs."
  attempt=$((attempt + 1))
  if (( attempt <= MAX_ATTEMPTS )); then sleep 60 9>&-; fi
done

write_status "failed" "All evaluation attempts failed; inspect the attempt logs under results/libero_eval/."
exit 1

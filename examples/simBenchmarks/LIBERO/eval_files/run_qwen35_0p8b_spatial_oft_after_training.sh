#!/usr/bin/env bash
set -euo pipefail

REPO="/data1/jiaozx/starVLA"
RUN_DIR="${REPO}/playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k"
LOG_DIR="${RUN_DIR}/logs"
STATUS_FILE="${LOG_DIR}/posttrain_eval_queue.status"
CHECKPOINT="${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt"
PYTHON="/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python"

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
  [[ "${command_line}" == *qwen35_0p8b_spatial_oft* ]]
}

active_training_pids() {
  local pidfile
  for pidfile in \
    "${RUN_DIR}/stage_a.pid" \
    "${RUN_DIR}/stage_b.pid" \
    "${RUN_DIR}/stage_supervisor.pid"; do
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
echo "Waiting for 50,000-step Spatial OFT training to stop."
write_status "waiting_for_training" "Will evaluate after all spatial training stages exit."

while true; do
  live_pids="$(active_training_pids)"
  if [[ -z "${live_pids}" ]]; then
    break
  fi
  write_status "waiting_for_training" "Active spatial training/supervisor PIDs: ${live_pids//$'\n'/,}"
  sleep 30 9>&-
done

if [[ ! -s "${CHECKPOINT}" ]]; then
  write_status "failed" "Training exited without the final 50,000-step checkpoint: ${CHECKPOINT}"
  echo "ERROR: final spatial checkpoint is missing; evaluation was not started." >&2
  exit 1
fi
if [[ ! -s "${RUN_DIR}/summary.jsonl" ]] || ! tail -n 20 "${RUN_DIR}/summary.jsonl" | rg -q '"steps": 50000'; then
  write_status "failed" "Final checkpoint exists but summary.jsonl does not confirm completed step 50,000."
  echo "ERROR: training completion could not be confirmed from summary.jsonl." >&2
  exit 1
fi

while true; do
  free_mib="$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= 16000 )); then
    break
  fi
  echo "Waiting for 16 GiB free on evaluation GPU 1; current free memory=${free_mib:-unknown} MiB."
  write_status "waiting_for_evaluation_gpu" "GPU 1 requires at least 16 GiB free; current ${free_mib:-unknown} MiB."
  sleep 30 9>&-
done

write_status "starting_evaluation" "Training completed; launching four-suite 50-episode/task evaluation on GPU 1."
attempt=1
while (( attempt <= 3 )); do
  run_id="qwen35_0p8b_spatial_oft_50k_eval_$(date '+%Y%m%d_%H%M%S')_attempt${attempt}"
  echo "Starting evaluation attempt ${attempt}/3: ${run_id}"
  set +e
  cd "${REPO}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHON="${PYTHON}" \
  GPU_ID=1 SIM_WORKERS=8 INFERENCE_BATCH_SIZE=4 INFERENCE_BATCH_WAIT_MS=12 \
  NUM_TRIALS_PER_TASK=50 MAX_TASKS=-1 SAVE_VIDEOS=1 MAX_VIDEOS_PER_OUTCOME=1 \
  STOP_AFTER_BOTH_VIDEOS=0 LIBERO_RECORD_INFERENCE_TIMING=1 \
  ENSURE_BOTH_OUTCOME_VIDEOS=1 VIDEO_RETRY_SEEDS=8:27 VIDEO_RETRY_TRIALS=50 \
  WRITE_EVAL_REPORT=1 EXPECTED_CKPT_BYTES=auto WAIT_FOR_CHECKPOINT=0 \
  MIN_FREE_GPU_MEM_MIB=16000 CKPT="${CHECKPOINT}" RUN_ID="${run_id}" SEED=7 PORT=6696 \
  SERVER_CPUSET=32-35 CPU_AFFINITY_START=36 CPU_AFFINITY_COUNT=8 \
  bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh
  result=$?
  set -e
  if (( result == 0 )) && [[ -s "${REPO}/results/libero_eval/${run_id}/evaluation_report.md" ]]; then
    metrics_path="${REPO}/results/libero_eval/${run_id}/evaluation_metrics.json"
    video_complete="$("${PYTHON}" -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("video_coverage_complete", False)).lower())' "${metrics_path}")"
    if [[ "${video_complete}" == "true" ]]; then
      write_status "completed" "Four-suite SR, inference timing and both-outcome videos are recorded at results/libero_eval/${run_id}."
    else
      write_status "completed_with_video_gaps" "SR and timing are recorded at results/libero_eval/${run_id}; inspect video_manifest.json for unresolved task outcomes."
    fi
    echo "Evaluation queue finished. Report: ${REPO}/results/libero_eval/${run_id}/evaluation_report.md"
    exit 0
  fi
  echo "Evaluation attempt ${attempt} failed with status ${result}; logs remain under results/libero_eval/${run_id}/logs." >&2
  write_status "retrying_evaluation" "Attempt ${attempt} failed; a fresh run will retry without overwriting its logs."
  attempt=$((attempt + 1))
  if (( attempt <= 3 )); then sleep 60 9>&-; fi
done

write_status "failed" "Three evaluation attempts failed; inspect the attempt logs under results/libero_eval/."
exit 1

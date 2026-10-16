#!/usr/bin/env bash
# Cron entry point for the post-training spatial-memory evaluation queue.
#
# Install on a schedule (see the crontab line in the header of
# auto_eval_spatial_mem_50k_v2.sh).  Every tick is cheap and safe:
#   * a completed queue is never restarted (the runner writes a .done marker)
#   * only one attempt runs at a time (lock held across the whole multi-hour run)
#   * a failed attempt is retried on the next tick, up to MAX_ATTEMPTS
#
# The heavy lifting lives in auto_eval_spatial_mem_50k_v2.sh, which waits for the
# trainer to exit, waits for the 50,000-step checkpoint to appear and stabilise,
# then runs standard LIBERO and LIBERO-plus sequentially on one GPU, fully
# offline (HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 / local Qwen snapshot).
#
# Environment variables (all optional):
#   EVAL_GPU_ID    physical GPU for evaluation (default: 0)
#   MAX_ATTEMPTS   max failed attempts before giving up (default: 3)
#   TIMEOUT_HOURS  max hours the runner waits for the 50k checkpoint (default: 24)
set -uo pipefail

STARVLA_DIR="${STARVLA_DIR:-/data1/jiaozx/starVLA}"
PYTHON="${PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"

RUN_ID="qwen35_0p8b_spatial_mem_oft_libero_50k"
RUN_DIR="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
RUNNER="${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/auto_eval_spatial_mem_50k_v2.sh"

mkdir -p "${LOG_DIR}"
CRON_LOG="${LOG_DIR}/auto_eval_cron.log"
ATTEMPT_FILE="${LOG_DIR}/auto_eval_attempts"
DONE_FILE="${LOG_DIR}/auto_eval_spatial_mem_50k.done"
WRAPPER_LOCK="${LOG_DIR}/auto_eval_cron.lock"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "${CRON_LOG}"; }

[[ -r "${RUNNER}" ]] || { log "Runner script is missing: ${RUNNER}"; exit 1; }
[[ -x "${PYTHON}" ]] || { log "Python interpreter is missing: ${PYTHON}"; exit 1; }

# A completed queue is never restarted.
if [[ -s "${DONE_FILE}" ]]; then
  exit 0
fi

# One attempt at a time.  The lock is held for the entire (multi-hour) run, so
# every other tick in between exits here immediately.
exec 8>"${WRAPPER_LOCK}"
if ! flock -n 8; then
  exit 0
fi

attempt=0
if [[ -s "${ATTEMPT_FILE}" ]]; then
  read -r attempt < "${ATTEMPT_FILE}" || attempt=0
fi
[[ "${attempt}" =~ ^[0-9]+$ ]] || attempt=0

if (( attempt >= MAX_ATTEMPTS )); then
  log "Giving up after ${attempt} failed attempt(s); inspect ${LOG_DIR} for the cause."
  exit 0
fi

log "Attempt $((attempt + 1))/${MAX_ATTEMPTS}: launching evaluation queue on GPU ${EVAL_GPU_ID}."

# The runner keeps its own master log; this captures anything it prints before
# or after that log is opened, and keeps cron from mailing a multi-hour trace.
ATTEMPT_LOG="${LOG_DIR}/auto_eval_cron_attempt$((attempt + 1))_$(date '+%Y%m%d_%H%M%S').log"

STARVLA_DIR="${STARVLA_DIR}" \
PYTHON="${PYTHON}" \
EVAL_GPU_ID="${EVAL_GPU_ID}" \
TIMEOUT_HOURS="${TIMEOUT_HOURS}" \
bash "${RUNNER}" >> "${ATTEMPT_LOG}" 2>&1
status=$?

# Exit status alone is not proof the queue ran: the runner exits 0 early when
# another instance already holds its lock.  Require the done marker.
if (( status == 0 )) && [[ -s "${DONE_FILE}" ]]; then
  log "Evaluation queue finished successfully."
  printf '0\n' > "${ATTEMPT_FILE}"
  exit 0
fi

if (( status == 0 )); then
  log "Runner exited 0 without writing ${DONE_FILE}; treating as a no-op, not a completion."
  exit 0
fi

attempt=$((attempt + 1))
printf '%s\n' "${attempt}" > "${ATTEMPT_FILE}"
log "Attempt failed with status ${status}; recorded ${attempt}/${MAX_ATTEMPTS}. The next cron tick retries."
exit "${status}"

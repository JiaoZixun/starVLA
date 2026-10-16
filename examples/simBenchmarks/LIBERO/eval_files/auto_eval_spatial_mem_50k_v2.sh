#!/usr/bin/env bash
# Automated post-training evaluation for Qwen3.5-0.8B Spatial-Mem OFT (50k steps).
#
# Sequence (all on one GPU, sequentially, detached-safe):
#   1. wait for the spatial-memory training processes to exit
#   2. wait for the 50,000-step checkpoint to exist and stop growing
#   3. wait for the evaluation GPU to be free
#   4. standard LIBERO  — 4 suites x NUM_TRIALS_LIBERO, one policy server + N sim workers
#   5. LIBERO-plus      — 4 suites x 1 trial (official protocol), in-process workers
#   6. aggregate client-observed inference latency over a 1000-request sample
#
# Everything is offline: local Qwen3.5 snapshot + HF_HUB_OFFLINE=1, so no
# evaluation step can block on a Hugging Face request.
#
# Usage:
#   nohup bash examples/simBenchmarks/LIBERO/eval_files/auto_eval_spatial_mem_50k_v2.sh &
#   (or launch inside screen/tmux to survive the shell)
#
# Environment variables (all optional):
#   EVAL_GPU_ID          evaluation GPU index (default: 0)
#   POLL_INTERVAL        seconds between training/checkpoint polls (default: 60)
#   TIMEOUT_HOURS        max hours to wait for the checkpoint (default: 72)
#   NUM_TRIALS_LIBERO    trials per task, standard LIBERO (default: 50)
#   TIMING_SAMPLE_SIZE   inference-latency sample size (default: 1000)
#   SIM_WORKERS          standard-LIBERO simulator workers, multiple of 4 (default: 8)
#   INFERENCE_BATCH      standard-LIBERO inference batch size (default: 4)
#   PLUS_TASKS_PER_GPU   LIBERO-plus workers per suite (default: 2; each loads its own model)
#   MIN_FREE_GPU_MEM_MIB free memory required on the eval GPU (default: 30000)
#   WAIT_FOR_TRAINING    0 to skip the training-process wait (default: 1)
#   SKIP_LIBERO_STANDARD 1 to skip step 4 (default: 0)
#   SKIP_LIBERO_PLUS     1 to skip step 5 (default: 0)
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-72}"
NUM_TRIALS_LIBERO="${NUM_TRIALS_LIBERO:-50}"
TIMING_SAMPLE_SIZE="${TIMING_SAMPLE_SIZE:-1000}"
SIM_WORKERS="${SIM_WORKERS:-8}"
INFERENCE_BATCH="${INFERENCE_BATCH:-4}"
PLUS_TASKS_PER_GPU="${PLUS_TASKS_PER_GPU:-2}"
MIN_FREE_GPU_MEM_MIB="${MIN_FREE_GPU_MEM_MIB:-30000}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
SKIP_LIBERO_STANDARD="${SKIP_LIBERO_STANDARD:-0}"
SKIP_LIBERO_PLUS="${SKIP_LIBERO_PLUS:-0}"
CKPT_STABLE_SECONDS="${CKPT_STABLE_SECONDS:-120}"

RUN_ID="qwen35_0p8b_spatial_mem_oft_libero_50k"
RUN_DIR="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}"
CKPT_PATH="${RUN_DIR}/checkpoints/steps_50000_pytorch_model.pt"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO}"
LIBERO_PLUS_HOME="${LIBERO_PLUS_HOME:-${STARVLA_DIR}/playground/LIBERO-plus}"
HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${STARVLA_DIR}/results/${RUN_ID}/auto_eval_${STAMP}"
LIBERO_STD_RUN_ID="${RUN_ID}_libero_std_${STAMP}"
LIBERO_STD_ROOT="${STARVLA_DIR}/results/libero_eval/${LIBERO_STD_RUN_ID}"
PLUS_RUN_ID="${RUN_ID}_libero_plus_${STAMP}"
PLUS_ROOT="${RESULT_ROOT}/libero_plus"

fail() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[[ -x "${PYTHON}" ]] || fail "Python executable not found: ${PYTHON}"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."
[[ "${EVAL_GPU_ID}" =~ ^[0-9]+$ ]] || fail "EVAL_GPU_ID must be a numeric GPU index."
nvidia-smi -i "${EVAL_GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 || \
    fail "GPU ${EVAL_GPU_ID} is unavailable."
[[ -d "${LIBERO_HOME}/libero" ]] || fail "LIBERO source tree not found: ${LIBERO_HOME}"
[[ -d "${LIBERO_PLUS_HOME}/libero/libero" ]] || fail "LIBERO-plus source tree not found: ${LIBERO_PLUS_HOME}"
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor file not readable: ${NVIDIA_EGL_VENDOR_JSON}"

cd "${STARVLA_DIR}"

# ---------------------------------------------------------------------------
# Single-instance guard + durable status. A watchdog cron entry may re-invoke
# this script, so it must be safe to call repeatedly: one runner at a time, and
# a completed run is never restarted.
# ---------------------------------------------------------------------------
CONTROL_DIR="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}/logs"
mkdir -p "${CONTROL_DIR}"
LOCK_FILE="${CONTROL_DIR}/auto_eval_spatial_mem_50k.lock"
DONE_FILE="${CONTROL_DIR}/auto_eval_spatial_mem_50k.done"
STATUS_FILE="${CONTROL_DIR}/auto_eval_spatial_mem_50k.status"
PID_FILE="${CONTROL_DIR}/auto_eval_spatial_mem_50k.pid"

write_status() {
  local state="$1" message="$2" tmp="${STATUS_FILE}.tmp.$$"
  {
    printf 'updated_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'state=%s\n' "${state}"
    printf 'message=%s\n' "${message}"
    printf 'result_root=%s\n' "${RESULT_ROOT}"
  } > "${tmp}"
  mv "${tmp}" "${STATUS_FILE}"
}

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another auto-eval runner already holds ${LOCK_FILE}; exiting."
  exit 0
fi

if [[ -s "${DONE_FILE}" ]]; then
  echo "Auto-evaluation already completed: $(cat "${DONE_FILE}")"
  exit 0
fi

echo "$$" > "${PID_FILE}"
trap 'write_status interrupted "Runner exited before completion."' INT TERM

mkdir -p "${RESULT_ROOT}" "${PLUS_ROOT}"
MASTER_LOG="${RESULT_ROOT}/auto_eval_master.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

# ---------------------------------------------------------------------------
# Offline + NVIDIA EGL environment. Exported here (not only in the launchers) so
# every child process, including the policy server, is offline from the start.
# ---------------------------------------------------------------------------
export STARVLA_DIR LIBERO_HOME LIBERO_PLUS_HOME PYTHON HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_EGL_DEVICE_ID="${EVAL_GPU_ID}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}"

write_status starting "Auto-evaluation runner started; waiting for training to finish."
log "=== Automated Spatial-Mem Evaluation ==="
log "Run:            ${RUN_ID}"
log "Eval GPU:       ${EVAL_GPU_ID}"
log "Checkpoint:     ${CKPT_PATH}"
log "Results:        ${RESULT_ROOT}"
log "Standard LIBERO:$([[ "${SKIP_LIBERO_STANDARD}" == "0" ]] && echo " enabled (${NUM_TRIALS_LIBERO} trials/task)" || echo " skipped")"
log "LIBERO-plus:    $([[ "${SKIP_LIBERO_PLUS}" == "0" ]] && echo " enabled (official 1 trial/task)" || echo " skipped")"
log "Offline mode:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

# ---------------------------------------------------------------------------
# Step 1: wait for the spatial-memory training processes to exit.
# ---------------------------------------------------------------------------
pidfile_is_training_process() {
  local pidfile="$1" pid command_line
  [[ -s "${pidfile}" ]] || return 1
  read -r pid < "${pidfile}"
  pid="${pid%%$'\n'*}"
  [[ "${pid}" =~ ^[0-9]+$ && -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  # Require the actual trainer, not just the run id.  The pidfiles also hold the
  # `screen` wrapper and the launcher shell, whose command lines contain the run
  # id (the launcher path) but survive or are unrelated to the training step.
  # Matching those would keep this queue waiting after training has exited.
  [[ "${command_line}" == *train_starvla.py*qwen35_0p8b_spatial_mem_oft* ]]
}

# The run directory keeps several pidfiles, and some may be stale from an
# earlier attempt. Scan /proc as well so a live trainer is never missed.
scan_training_processes() {
  local dir pid cmdline
  for dir in /proc/[0-9]*; do
    pid="${dir#/proc/}"
    [[ -r "${dir}/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "${dir}/cmdline" 2>/dev/null || true)"
    case "${cmdline}" in
      *train_starvla.py*qwen35_0p8b_spatial_mem_oft*) echo "${pid}" ;;
    esac
  done
}

active_training_pids() {
  local pidfile
  {
    for pidfile in \
      "${RUN_DIR}/stage_a.pid" \
      "${RUN_DIR}/stage_b.pid" \
      "${RUN_DIR}/stage_b_resume.pid" \
      "${RUN_DIR}/stage_supervisor.pid" \
      "${RUN_DIR}/launcher.pid"; do
      if pidfile_is_training_process "${pidfile}"; then
        awk 'NR==1{print $1}' "${pidfile}"
      fi
    done
    scan_training_processes
  } | sort -nu
}

if [[ "${WAIT_FOR_TRAINING}" == "1" ]]; then
  write_status waiting_training "Waiting for spatial-memory training processes to exit..."
  log "Waiting for spatial-memory training to finish..."
  while true; do
    live_pids="$(active_training_pids)"
    [[ -n "${live_pids}" ]] || break
    log "Training still running (PIDs: ${live_pids//$'\n'/,})"
    sleep "${POLL_INTERVAL}"
  done
  log "Training processes have exited."
fi

# ---------------------------------------------------------------------------
# Step 2: wait for the 50k checkpoint, then wait until its size stops changing.
# ---------------------------------------------------------------------------
write_status waiting_checkpoint "Waiting for ${CKPT_PATH} to appear and stabilize..."
log "Waiting for checkpoint: ${CKPT_PATH}"
deadline=$(( $(date +%s) + TIMEOUT_HOURS * 3600 ))
last_size=""
stable_since=""
while true; do
  if [[ -s "${CKPT_PATH}" ]]; then
    size="$(stat -c '%s' "${CKPT_PATH}")"
    if [[ "${size}" == "${last_size}" ]]; then
      if (( $(date +%s) - stable_since >= CKPT_STABLE_SECONDS )); then
        log "Checkpoint stable at ${size} bytes: ${CKPT_PATH}"
        break
      fi
    else
      last_size="${size}"
      stable_since="$(date +%s)"
      log "Checkpoint present (${size} bytes); waiting ${CKPT_STABLE_SECONDS}s for writes to settle."
    fi
  fi
  if (( $(date +%s) >= deadline )); then
    fail "Timed out after ${TIMEOUT_HOURS}h waiting for ${CKPT_PATH}"
  fi
  sleep "${POLL_INTERVAL}"
done
CKPT_BYTES="$(stat -c '%s' "${CKPT_PATH}")"

if [[ -s "${RUN_DIR}/summary.jsonl" ]] && tail -n 20 "${RUN_DIR}/summary.jsonl" | grep -q '"steps": 50000'; then
  log "Training completion confirmed by summary.jsonl (steps: 50000)."
else
  log "WARNING: summary.jsonl does not confirm step 50000; continuing with the checkpoint on disk."
fi

# ---------------------------------------------------------------------------
# Step 3: wait for the evaluation GPU to drain, then verify EGL rendering.
# ---------------------------------------------------------------------------
write_status waiting_gpu "Waiting for GPU ${EVAL_GPU_ID} to be free (${MIN_FREE_GPU_MEM_MIB} MiB)..."
log "Waiting for ${MIN_FREE_GPU_MEM_MIB} MiB free on GPU ${EVAL_GPU_ID}..."
while true; do
  free_mib="$(nvidia-smi -i "${EVAL_GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_GPU_MEM_MIB )); then
    log "GPU ${EVAL_GPU_ID} ready (${free_mib} MiB free)."
    break
  fi
  log "GPU ${EVAL_GPU_ID} busy (${free_mib:-unknown} MiB free); retrying in ${POLL_INTERVAL}s."
  sleep "${POLL_INTERVAL}"
done

log "Validating NVIDIA EGL renderer on GPU ${EVAL_GPU_ID}..."
"${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py || fail "NVIDIA EGL validation failed."

# ---------------------------------------------------------------------------
# Step 4: standard LIBERO (4 suites, single policy server + parallel sims).
# ---------------------------------------------------------------------------
libero_status=0
if [[ "${SKIP_LIBERO_STANDARD}" == "0" ]]; then
  write_status running_libero "Running standard LIBERO (4 suites x ${NUM_TRIALS_LIBERO} trials)..."
  log "=== [1/2] Standard LIBERO: 4 suites x ${NUM_TRIALS_LIBERO} trials/task ==="
  log "Results: ${LIBERO_STD_ROOT}"
  export LIBERO_HOME="${LIBERO_HOME}"
  export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
  export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"

  set +e
  GPU_ID="${EVAL_GPU_ID}" \
    PYTHON="${PYTHON}" \
    CONDA_ENV="" \
    SIM_WORKERS="${SIM_WORKERS}" \
    INFERENCE_BATCH_SIZE="${INFERENCE_BATCH}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_LIBERO}" \
    CKPT="${CKPT_PATH}" \
    EXPECTED_CKPT_BYTES="auto" \
    WAIT_FOR_CHECKPOINT=0 \
    RUN_ID="${LIBERO_STD_RUN_ID}" \
    TASK_SUITES_CSV="libero_spatial,libero_object,libero_goal,libero_10" \
    SAVE_VIDEOS=0 \
    LIBERO_RECORD_INFERENCE_TIMING=1 \
    UNNORM_KEY="franka" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh
  libero_status=$?
  set -e
  (( libero_status == 0 )) && log "Standard LIBERO complete." || log "WARNING: standard LIBERO failed (status ${libero_status})."
fi

# ---------------------------------------------------------------------------
# Step 5: LIBERO-plus (4 suites, official protocol: 1 episode per task).
# Each worker loads its own model copy, so the worker count is bounded.
# ---------------------------------------------------------------------------
plus_status=0
if [[ "${SKIP_LIBERO_PLUS}" == "0" ]]; then
  write_status running_libero_plus "Running LIBERO-plus (4 suites, official 1 trial/task)..."
  log "=== [2/2] LIBERO-plus: 4 suites x 1 trial/task (official protocol) ==="
  log "Workers per suite: ${PLUS_TASKS_PER_GPU} (each loads its own policy copy)"
  log "Results: ${PLUS_ROOT}"

  set +e
  LIBERO_HOME="${LIBERO_PLUS_HOME}" \
  LIBERO_PYTHON="${PYTHON}" \
  EVAL_GPU_ID="${EVAL_GPU_ID}" \
  TASKS_PER_GPU="${PLUS_TASKS_PER_GPU}" \
  SAVE_VIDEOS=0 \
  MODEL_SELECTION=mem \
  MEM_CKPT="${CKPT_PATH}" \
  MEM_CKPT_BYTES="auto" \
  NUM_TRIALS_PER_TASK=1 \
  RUN_ID="${PLUS_RUN_ID}" \
  RESULT_ROOT="${PLUS_ROOT}" \
  bash examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/run_local_dual_libero_plus.sh
  plus_status=$?
  set -e
  (( plus_status == 0 )) && log "LIBERO-plus complete." || log "WARNING: LIBERO-plus failed (status ${plus_status})."
fi

# ---------------------------------------------------------------------------
# Step 6: inference latency — 1000-request sample from the raw per-request
# timings recorded by the standard-LIBERO clients.
# ---------------------------------------------------------------------------
write_status aggregating "Aggregating inference latency statistics..."
log "=== Aggregating inference latency (sample=${TIMING_SAMPLE_SIZE}) ==="

"${PYTHON}" - "${LIBERO_STD_ROOT}" "${RESULT_ROOT}" "${TIMING_SAMPLE_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

std_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
sample_size = int(sys.argv[3])

timing_files = sorted(std_root.rglob("inference_timing.json"))
if not timing_files:
    print(f"No inference_timing.json found under {std_root}; skipping latency report.")
    raise SystemExit(0)

latencies = []
for path in timing_files:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"Skipping unreadable {path}: {exc}")
        continue
    raw = data.get("raw_latencies_ms")
    if isinstance(raw, list):
        latencies.extend(float(x) for x in raw)
    else:
        print(f"No raw_latencies_ms in {path}; only aggregated stats were written.")

if not latencies:
    print("No raw latency samples available; skipping latency report.")
    raise SystemExit(0)

values = np.asarray(latencies, dtype=np.float64)
total = int(values.size)
n = min(sample_size, total)
rng = np.random.default_rng(seed=42)
sample = values[rng.choice(total, size=n, replace=False)]

report = {
    "measurement": (
        "Client-observed wall clock around WebsocketClientPolicy.predict_action "
        "(one action-chunk request): includes client serialization, websocket round "
        "trip, server batching wait and policy inference; excludes simulator stepping."
    ),
    "unit": "ms",
    "requests_total": total,
    "sample_size": int(n),
    "sampling": "uniform without replacement from all recorded requests",
    "mean_ms": float(sample.mean()),
    "median_ms": float(np.median(sample)),
    "p95_ms": float(np.percentile(sample, 95)),
    "p99_ms": float(np.percentile(sample, 99)),
    "std_ms": float(sample.std(ddof=0)),
    "min_ms": float(sample.min()),
    "max_ms": float(sample.max()),
    "requests_per_sec": float(1000.0 / sample.mean()),
    "source_files": [str(p.relative_to(result_root)) if result_root in p.parents else str(p) for p in timing_files],
}
out = result_root / "inference_timing_summary.json"
out.write_text(json.dumps(report, indent=2))
print(f"Latency over {n} sampled of {total} requests: "
      f"mean={report['mean_ms']:.2f} ms, median={report['median_ms']:.2f} ms, "
      f"p95={report['p95_ms']:.2f} ms, {report['requests_per_sec']:.2f} req/s")
print(f"Saved: {out}")
PY

# ---------------------------------------------------------------------------
# Step 7: consolidated summary.
# ---------------------------------------------------------------------------
write_status finalizing "Writing consolidated summary..."
log "=== Writing consolidated summary ==="

"${PYTHON}" - "${RESULT_ROOT}" "${LIBERO_STD_ROOT}" "${PLUS_ROOT}" "${CKPT_PATH}" "${CKPT_BYTES}" "${EVAL_GPU_ID}" "${libero_status}" "${plus_status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

result_root = Path(sys.argv[1])
std_root = Path(sys.argv[2])
plus_root = Path(sys.argv[3])
ckpt = sys.argv[4]
ckpt_bytes = int(sys.argv[5])
gpu = int(sys.argv[6])
libero_status = int(sys.argv[7])
plus_status = int(sys.argv[8])

def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

summary = {
    "checkpoint": ckpt,
    "checkpoint_bytes": ckpt_bytes,
    "eval_gpu": gpu,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": {"libero_standard": libero_status, "libero_plus": plus_status},
    "libero_standard": {},
    "libero_plus": {},
}

# Standard LIBERO: the parallel launcher merges shard summaries into
# <std_root>/<suite>/summary.json.
for suite_dir in sorted(p for p in std_root.glob("*") if p.is_dir()):
    data = load(suite_dir / "summary.json")
    if data:
        summary["libero_standard"][suite_dir.name] = {
            "success_rate": data.get("success_rate"),
            "episodes": data.get("episodes"),
            "successes": data.get("successes"),
        }

# LIBERO-plus: aggregate_results.py writes mem/overall_results.json with one row
# per perturbation category plus an "overall" row.
for overall in sorted(plus_root.rglob("overall_results.json")):
    data = load(overall)
    if data:
        summary["libero_plus"][overall.parent.name] = {
            "overall": data.get("overall"),
            "categories": {k: v for k, v in data.items() if k != "overall"},
        }

timing = load(result_root / "inference_timing_summary.json")
if timing:
    summary["inference_timing"] = {
        k: timing[k]
        for k in ("sample_size", "requests_total", "mean_ms", "median_ms",
                  "p95_ms", "requests_per_sec")
        if k in timing
    }

out = result_root / "evaluation_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"Consolidated summary: {out}")
for suite, row in summary["libero_standard"].items():
    print(f"  LIBERO {suite}: {row['success_rate']}")
for name, row in summary["libero_plus"].items():
    ov = row.get("overall") or {}
    rate = ov.get("success_rate")
    print(f"  LIBERO-plus {name}: overall={rate if rate is None else round(rate, 4)}")
PY

log "=== Evaluation finished ==="
log "Results:          ${RESULT_ROOT}"
log "Summary:          ${RESULT_ROOT}/evaluation_summary.json"
log "Latency:          ${RESULT_ROOT}/inference_timing_summary.json"
log "Standard LIBERO:  ${LIBERO_STD_ROOT}"
log "LIBERO-plus:      ${PLUS_ROOT}"
log "Master log:       ${MASTER_LOG}"

if (( libero_status != 0 || plus_status != 0 )); then
  write_status failed "Some suites failed (LIBERO=${libero_status}, plus=${plus_status})."
  log "Some suites failed (libero=${libero_status}, plus=${plus_status}); see the logs above."
  exit 1
fi

write_status completed "All evaluations succeeded."
date '+%Y-%m-%d %H:%M:%S %Z' > "${DONE_FILE}"
log "All evaluations completed successfully. Done marker: ${DONE_FILE}"

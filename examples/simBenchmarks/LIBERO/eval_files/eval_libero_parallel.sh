#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Single-GPU LIBERO evaluation with parallel simulation workers and dynamic inference batching.
Run: GPU_ID=1 SIM_WORKERS=8 INFERENCE_BATCH_SIZE=4 bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh

Variables: GPU_ID (single GPU, default auto), SIM_WORKERS (4 or 8; multiple of 4),
INFERENCE_BATCH_SIZE (default 4), INFERENCE_BATCH_WAIT_MS (default 12),
NUM_TRIALS_PER_TASK (default 50), MAX_TASKS (default -1), SAVE_VIDEOS (default 0).
When saving videos, MAX_VIDEOS_PER_OUTCOME defaults to 1 per task and outcome.
GPU_IDS is a legacy alias; only its first GPU is used. MAX_WORKERS is ignored.
HELP
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "This launcher takes configuration through environment variables; see --help." >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-fastvla-fla}"
PYTHON="${PYTHON:-$(conda run -n "${CONDA_ENV}" python -c 'import sys; print(sys.executable)')}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
EXPECTED_CKPT_BYTES="${EXPECTED_CKPT_BYTES:-9785061050}"
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-1}"
CHECKPOINT_POLL_SECONDS="${CHECKPOINT_POLL_SECONDS:-30}"
CHECKPOINT_WAIT_TIMEOUT_SEC="${CHECKPOINT_WAIT_TIMEOUT_SEC:-0}"
GPU_ID="${GPU_ID:-}"
GPU_IDS="${GPU_IDS:-}"
MIN_FREE_GPU_MEM_MIB="${MIN_FREE_GPU_MEM_MIB:-16000}"
SIM_WORKERS="${SIM_WORKERS:-8}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-12}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-1}"
PORT="${PORT:-6694}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-600}"
SEED="${SEED:-7}"
RUN_ID="${RUN_ID:-oft4b_$(date +%Y%m%d_%H%M%S)}"
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to select a GPU." >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}/libero" ]]; then
  echo "LIBERO source tree not found: ${LIBERO_HOME}/libero" >&2
  exit 1
fi
if [[ ! "${SIM_WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( SIM_WORKERS % 4 != 0 )); then
  echo "SIM_WORKERS must be a positive multiple of 4 (for example 4 or 8); got '${SIM_WORKERS}'." >&2
  exit 2
fi
if [[ ! "${INFERENCE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_BATCH_SIZE must be a positive integer; got '${INFERENCE_BATCH_SIZE}'." >&2
  exit 2
fi

if [[ -z "${GPU_ID}" ]]; then
  if [[ -n "${GPU_IDS}" ]]; then
    GPU_ID="${GPU_IDS%%,*}"
    if [[ "${GPU_IDS}" == *,* ]]; then
      echo "Single-GPU mode: GPU_IDS='${GPU_IDS}' provided; using only GPU_ID=${GPU_ID}."
    fi
  else
    GPU_ID="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F, -v min="${MIN_FREE_GPU_MEM_MIB}" 'gsub(/[[:space:]]/, "", $1) && gsub(/[[:space:]]/, "", $2) && $2 >= min { print $1; exit }')"
  fi
fi
if [[ -z "${GPU_ID}" ]] || ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
  | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -Fxq -- "${GPU_ID}"; then
  echo "Requested GPU_ID='${GPU_ID}' is not available." >&2
  nvidia-smi --query-gpu=index,name,memory.free --format=csv >&2
  exit 1
fi
if [[ -n "${MAX_WORKERS:-}" ]]; then
  echo "MAX_WORKERS is a legacy multi-GPU setting and is ignored; using one model server."
fi

if [[ "${WAIT_FOR_CHECKPOINT}" == "1" ]]; then
  echo "Waiting for checkpoint ${CKPT} (${EXPECTED_CKPT_BYTES} bytes)..."
  waited=0
  while [[ ! -f "${CKPT}" ]] || [[ "$(stat -c '%s' "${CKPT}" 2>/dev/null || echo 0)" != "${EXPECTED_CKPT_BYTES}" ]]; do
    actual_bytes="$(stat -c '%s' "${CKPT}" 2>/dev/null || echo 0)"
    echo "Checkpoint download in progress (${actual_bytes}/${EXPECTED_CKPT_BYTES} bytes)."
    if (( CHECKPOINT_WAIT_TIMEOUT_SEC > 0 && waited >= CHECKPOINT_WAIT_TIMEOUT_SEC )); then
      echo "Timed out waiting for checkpoint." >&2
      exit 1
    fi
    sleep "${CHECKPOINT_POLL_SECONDS}"
    waited=$((waited + CHECKPOINT_POLL_SECONDS))
  done
elif [[ ! -f "${CKPT}" ]] || [[ "$(stat -c '%s' "${CKPT}" 2>/dev/null || echo 0)" != "${EXPECTED_CKPT_BYTES}" ]]; then
  echo "Checkpoint is missing or incomplete: ${CKPT}" >&2
  exit 1
fi

echo "Checkpoint ready: ${CKPT} ($(stat -c '%s' "${CKPT}") bytes)"
echo "Using GPU ${GPU_ID}: one policy server, ${SIM_WORKERS} simulator workers, inference batch ${INFERENCE_BATCH_SIZE}."

cd "${STARVLA_DIR}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export SAVE_VIDEOS
export MAX_VIDEOS_PER_OUTCOME

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  "${PYTHON}" - <<'PY'
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

"${PYTHON}" deployment/model_server/server_policy.py --help >/dev/null
"${PYTHON}" -c "from starVLA.model.framework.base_framework import _auto_import_framework_modules; _auto_import_framework_modules()"
"${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py --help >/dev/null

RUN_ROOT="${STARVLA_DIR}/results/libero_eval/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/server_gpu${GPU_ID}.log"
WORKER_PIDS=()
SERVER_PID=""
CLEANED_UP=0

cleanup() {
  local status=$?
  if (( CLEANED_UP == 1 )); then return; fi
  CLEANED_UP=1
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port "${PORT}" --use_bf16 \
  --max_inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --inference_batch_wait_ms "${INFERENCE_BATCH_WAIT_MS}" \
  --config_override "framework.qwenvl.base_vlm=Qwen/Qwen3-VL-4B-Instruct" \
  --config_override "framework.qwenvl.load_base_weights=false" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

server_ready=0
for ((elapsed = 0; elapsed < SERVER_STARTUP_WAIT; elapsed += 2)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Policy server exited; see ${SERVER_LOG}" >&2
    tail -n 50 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if "${PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${PORT})); s.close()" >/dev/null 2>&1; then
    server_ready=1
    break
  fi
  sleep 2
done
if [[ "${server_ready}" != "1" ]]; then
  echo "Timed out waiting for policy server; see ${SERVER_LOG}" >&2
  exit 1
fi
echo "Policy server ready on GPU ${GPU_ID}, port ${PORT}."

if (( MAX_TASKS > 0 )); then EVAL_TASKS="${MAX_TASKS}"; else EVAL_TASKS=10; fi
SHARDS_PER_SUITE=$((SIM_WORKERS / 4))
if (( SHARDS_PER_SUITE > EVAL_TASKS )); then SHARDS_PER_SUITE="${EVAL_TASKS}"; fi
TASKS_PER_SHARD=$(((EVAL_TASKS + SHARDS_PER_SUITE - 1) / SHARDS_PER_SUITE))

worker_index=0
for suite in "${TASK_SUITES[@]}"; do
  for ((shard = 0; shard < SHARDS_PER_SUITE; shard++)); do
    task_start=$((shard * TASKS_PER_SHARD))
    (( task_start < EVAL_TASKS )) || continue
    task_end=$((task_start + TASKS_PER_SHARD))
    if (( task_end > EVAL_TASKS )); then task_end="${EVAL_TASKS}"; fi
    shard_name="shard_${task_start}_${task_end}"
    output_dir="${RUN_ROOT}/${suite}/${shard_name}"
    worker_log="${LOG_DIR}/worker${worker_index}_${suite}_${shard_name}.log"
    mkdir -p "${output_dir}"
    echo "Launching worker ${worker_index}: ${suite} tasks [${task_start}, ${task_end}) on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "${CKPT}" --args.host 127.0.0.1 --args.port "${PORT}" \
      --args.task-suite-name "${suite}" --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
      --args.max-tasks "${MAX_TASKS}" --args.task-id-start "${task_start}" \
      --args.task-id-end "${task_end}" --args.seed "${SEED}" \
      --args.video-out-path "${output_dir}" >"${worker_log}" 2>&1 &
    WORKER_PIDS+=("$!")
    worker_index=$((worker_index + 1))
  done
done

FAILURES=0
for i in "${!WORKER_PIDS[@]}"; do
  if wait "${WORKER_PIDS[i]}"; then
    echo "Worker ${i} completed."
  else
    echo "Worker ${i} failed; inspect ${LOG_DIR}/worker${i}_*.log" >&2
    FAILURES=$((FAILURES + 1))
  fi
done
WORKER_PIDS=()
if (( FAILURES > 0 )); then
  echo "Evaluation failed in ${FAILURES} simulator worker(s). Logs: ${LOG_DIR}" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT}" SEED="${SEED}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
run_root = Path(os.environ["RUN_ROOT"])
seed = int(os.environ["SEED"])
for suite_dir in sorted(p for p in run_root.iterdir() if p.is_dir() and p.name != "logs"):
    partials = sorted(suite_dir.glob("shard_*/summary.json"))
    if not partials:
        raise RuntimeError(f"No completed shard summaries found for {suite_dir.name}")
    task_results = []
    for path in partials:
        with path.open(encoding="utf-8") as f:
            task_results.extend(json.load(f)["task_results"])
    task_results.sort(key=lambda item: item["task_id"])
    episodes = sum(item["episodes"] for item in task_results)
    successes = sum(item["successes"] for item in task_results)
    summary = {
        "task_suite": suite_dir.name,
        "seed": seed,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "task_results": task_results,
    }
    with (suite_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"{suite_dir.name}: {successes}/{episodes} ({100 * summary['success_rate']:.1f}%)")
PY

echo "All suites completed. Results: ${RUN_ROOT}/<suite>/summary.json"
echo "Logs: ${LOG_DIR}"

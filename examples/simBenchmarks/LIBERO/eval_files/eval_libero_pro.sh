#!/usr/bin/env bash
# Zero-shot LIBERO-PRO evaluation for the Qwen3.5-0.8B ablations.
#
# Dimensions (paper name -> PERTURBATIONS value -> LIBERO-PRO suite):
#   Object   -> object   -> <suite>_object
#   Position -> swap     -> <suite>_swap
#   Semantic -> language -> <suite>_lan
#   Task     -> task     -> <suite>_task
#
# One policy server per model runs on GPU_ID; each (suite, perturbation, task)
# is one simulator process with its own per-task env seed, scheduled longest
# horizon first.  Models are evaluated sequentially.
#
# Environment:
#   MODEL_SELECTION       baseline | spatial | mem | all | comma list (default: all)
#   BASELINE_CKPT / SPATIAL_CKPT / MEM_CKPT   checkpoint .pt paths
#   GPU_ID                physical GPU for server + EGL rendering (default: 1)
#   TASK_SUITES_CSV       default libero_spatial,libero_object,libero_goal,libero_10
#   PERTURBATIONS_CSV     default object,swap,language,task
#   NUM_TRIALS_PER_TASK   default 50
#   MAX_TASKS / MAX_STEPS smoke-test caps (default -1 = full benchmark)
#   MAX_PARALLEL_WORKERS  concurrent simulator workers per model (default 16)
#   INFERENCE_BATCH_SIZE  server dynamic batching (default 4, as in standard LIBERO
#                         eval; 1 is deterministic per request but ~1.6x slower at 16 workers)
#   RUN_ID                output dir name under results/libero_pro (default timestamp)
#   PREFLIGHT_ONLY=1      validate GPU/EGL/checkpoints/data and exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
PYTHON="${PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO-PRO}"
CKPT_ROOT="${CKPT_ROOT:-${STARVLA_DIR}/playground/Checkpoints}"
BASELINE_CKPT="${BASELINE_CKPT:-${CKPT_ROOT}/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
SPATIAL_CKPT="${SPATIAL_CKPT:-${CKPT_ROOT}/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
MEM_CKPT="${MEM_CKPT:-${CKPT_ROOT}/qwen35_0p8b_spatial_mem_oft_v2_full_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
MODEL_SELECTION="${MODEL_SELECTION:-all}"
GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-6794}"
TASK_SUITES_CSV="${TASK_SUITES_CSV:-libero_spatial,libero_object,libero_goal,libero_10}"
PERTURBATIONS_CSV="${PERTURBATIONS_CSV:-object,swap,language,task}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_PARALLEL_WORKERS="${MAX_PARALLEL_WORKERS:-16}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-12}"
SEED="${SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-600}"
MIN_FREE_GPU_MEM_MIB="${MIN_FREE_GPU_MEM_MIB:-12000}"
RUN_ID="${RUN_ID:-qwen35_0p8b_libero_pro_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${STARVLA_DIR}/results/libero_pro/${RUN_ID}}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "${MODEL_SELECTION}" in
  all) MODELS=(baseline spatial mem) ;;
  *) IFS=',' read -r -a MODELS <<< "${MODEL_SELECTION}" ;;
esac
IFS=',' read -r -a TASK_SUITES <<< "${TASK_SUITES_CSV}"
IFS=',' read -r -a PERTURBATIONS <<< "${PERTURBATIONS_CSV}"
(( ${#MODELS[@]} > 0 && ${#TASK_SUITES[@]} > 0 && ${#PERTURBATIONS[@]} > 0 )) || fail "Empty model/suite/perturbation list."
for model in "${MODELS[@]}"; do
  case "${model}" in baseline|spatial|mem) ;; *) fail "Unknown model '${model}' (baseline|spatial|mem|all)." ;; esac
done
for suite in "${TASK_SUITES[@]}"; do
  case "${suite}" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) fail "Unknown suite '${suite}'." ;; esac
done
for pert in "${PERTURBATIONS[@]}"; do
  case "${pert}" in object|swap|language|task) ;; *) fail "Unknown perturbation '${pert}' (object|swap|language|task)." ;; esac
done
[[ "${MAX_PARALLEL_WORKERS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_PARALLEL_WORKERS must be a positive integer."
[[ -x "${PYTHON}" ]] || fail "PYTHON is not executable: ${PYTHON}"
[[ -d "${LIBERO_HOME}/libero/libero/bddl_files" ]] || fail "LIBERO-PRO tree not found: ${LIBERO_HOME}"

# Longest horizon first (libero_10 > goal > object > spatial).
ORDERED_SUITES=()
for suite in libero_10 libero_goal libero_object libero_spatial; do
  for selected in "${TASK_SUITES[@]}"; do
    if [[ "${selected}" == "${suite}" ]]; then ORDERED_SUITES+=("${suite}"); fi
  done
done

ckpt_for() {
  case "$1" in
    baseline) echo "${BASELINE_CKPT}" ;;
    spatial) echo "${SPATIAL_CKPT}" ;;
    mem) echo "${MEM_CKPT}" ;;
  esac
}

# --- NVIDIA EGL rendering on the selected GPU (repository rule) ---
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."
nvidia-smi -i "${GPU_ID}" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
  || fail "GPU ${GPU_ID} is unavailable."
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor JSON is not readable: ${NVIDIA_EGL_VENDOR_JSON}"
cd "${STARVLA_DIR}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${RUN_ROOT}/.libero_config"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

mkdir -p "${LIBERO_CONFIG_PATH}"
"${PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ["LIBERO_HOME"]) / "libero" / "libero"
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
PY
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py

# --- Data check: every selected PRO suite must have 10 tasks with enough init states ---
TASK_SUITES_CSV="${TASK_SUITES_CSV}" PERTURBATIONS_CSV="${PERTURBATIONS_CSV}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" "${PYTHON}" - <<'PY'
import os
from libero.libero import benchmark
from examples.simBenchmarks.LIBERO.eval_files.libero_pro_utils import pro_suite_name
need = int(os.environ["NUM_TRIALS_PER_TASK"])
registry = benchmark.get_benchmark_dict()
for suite in os.environ["TASK_SUITES_CSV"].split(","):
    for pert in os.environ["PERTURBATIONS_CSV"].split(","):
        name = pro_suite_name(suite, pert)
        bench = registry[name]()
        counts = [len(bench.get_task_init_states(i)) for i in range(bench.n_tasks)]
        if bench.n_tasks != 10 or min(counts) < need:
            raise SystemExit(f"{name}: {bench.n_tasks} tasks, init states {counts}; need 10 tasks x {need}.")
        print(f"{name}: 10 tasks, init states >= {min(counts)}")
PY

# --- Checkpoint check: framework must match the ablation; derive load overrides ---
declare -A MODEL_OVERRIDES
for model in "${MODELS[@]}"; do
  ckpt="$(ckpt_for "${model}")"
  [[ -s "${ckpt}" ]] || fail "${model} checkpoint is missing: ${ckpt}"
  overrides="$("${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/libero_pro_utils.py "${model}" "${ckpt}")" \
    || fail "${model} checkpoint failed validation: ${ckpt}"
  MODEL_OVERRIDES[${model}]="${overrides}"
  echo "${model}: ${ckpt}${overrides:+ (overrides: ${overrides//$'\n'/ })}"
done

free_mib="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
(( free_mib >= MIN_FREE_GPU_MEM_MIB )) || fail "GPU ${GPU_ID} has ${free_mib} MiB free; need ${MIN_FREE_GPU_MEM_MIB}."
episodes_per_model=$(( ${#TASK_SUITES[@]} * ${#PERTURBATIONS[@]} * 10 * NUM_TRIALS_PER_TASK ))
if (( MAX_TASKS > 0 )); then
  episodes_per_model=$(( ${#TASK_SUITES[@]} * ${#PERTURBATIONS[@]} * MAX_TASKS * NUM_TRIALS_PER_TASK ))
fi
echo "Models: ${MODELS[*]} | suites: ${TASK_SUITES[*]} | perturbations: ${PERTURBATIONS[*]}"
echo "Episodes per model: ${episodes_per_model} | GPU ${GPU_ID} | results: ${RUN_ROOT}"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed."
  exit 0
fi

SERVER_PID=""
WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Run one simulator worker in the background and record its exit code in a
# file.  Per-PID `wait` cannot report worker status: a full run starts far more
# processes than pid_max, PIDs wrap, and bash discards the saved status of a
# finished worker once a later child reuses its PID ("pid N is not a child of
# this shell"), which used to count successful workers as failures.
run_worker() {
  local status_file="$1" log_file="$2" child="" rc=0
  shift 2
  trap '[[ -n "${child}" ]] && kill "${child}" 2>/dev/null; exit 143' TERM
  "$@" >"${log_file}" 2>&1 &
  child=$!
  wait "${child}" || rc=$?
  echo "${rc}" >"${status_file}"
}

run_model() {
  local model="$1" ckpt model_root log_dir
  ckpt="$(ckpt_for "${model}")"
  model_root="${RUN_ROOT}/${model}"
  log_dir="${model_root}/logs"
  mkdir -p "${log_dir}"
  local -a server_args=(--ckpt_path "${ckpt}" --port "${PORT}" --use_bf16 --seed "${SEED}"
    --max_inference_batch_size "${INFERENCE_BATCH_SIZE}" --inference_batch_wait_ms "${INFERENCE_BATCH_WAIT_MS}")
  local item
  while IFS= read -r item; do
    [[ -n "${item}" ]] && server_args+=(--config_override "${item}")
  done <<< "${MODEL_OVERRIDES[${model}]}"

  echo "[$(date '+%F %T')] ${model}: starting policy server on GPU ${GPU_ID}:${PORT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" deployment/model_server/server_policy.py "${server_args[@]}" \
    >"${log_dir}/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0 elapsed
  for ((elapsed = 0; elapsed < SERVER_STARTUP_WAIT; elapsed += 2)); do
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 40 "${log_dir}/server.log" >&2; fail "${model} server exited."; }
    if "${PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${PORT})); s.close()" 2>/dev/null; then
      ready=1; break
    fi
    sleep 2
  done
  (( ready == 1 )) || fail "${model} server did not start; see ${log_dir}/server.log"

  local -a video_args=()
  if [[ "${SAVE_VIDEOS}" == "1" ]]; then video_args=(--args.save-videos); fi
  local failures=0 suite pert task_id out_dir pid running status_file rc
  local n_tasks=10
  if (( MAX_TASKS > 0 )); then n_tasks="${MAX_TASKS}"; fi
  local -a status_files=()
  WORKER_PIDS=()
  # One worker per (suite, perturbation, task); longest-horizon suites first so
  # libero_10 does not become the tail.  Each task's env is seeded on its own.
  for suite in "${ORDERED_SUITES[@]}"; do
    for pert in "${PERTURBATIONS[@]}"; do
      for ((task_id = 0; task_id < n_tasks; task_id++)); do
      # Throttle by polling; each worker's exit code is read from its status file.
      while true; do
        kill -0 "${SERVER_PID}" 2>/dev/null || fail "${model} server exited during evaluation; see ${log_dir}/server.log"
        running=0
        for pid in "${WORKER_PIDS[@]:-}"; do
          if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then running=$((running + 1)); fi
        done
        (( running < MAX_PARALLEL_WORKERS )) && break
        sleep 5
      done
      out_dir="$(printf '%s/%s/%s/task_%02d' "${model_root}" "${suite}" "${pert}" "${task_id}")"
      mkdir -p "${out_dir}"
      status_file="${out_dir}/exit_status"
      rm -f "${status_file}"
      status_files+=("${status_file}")
      CUDA_VISIBLE_DEVICES="${GPU_ID}" run_worker "${status_file}" \
        "$(printf '%s/%s_%s_task%02d.log' "${log_dir}" "${suite}" "${pert}" "${task_id}")" \
        "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero_pro.py \
        --args.host 127.0.0.1 --args.port "${PORT}" \
        --args.task-suite-name "${suite}" --args.perturbation "${pert}" \
        --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
        --args.task-id-start "${task_id}" --args.task-id-end "$((task_id + 1))" \
        --args.max-steps "${MAX_STEPS}" --args.seed "${SEED}" --args.unnorm-key "${UNNORM_KEY}" \
        "${video_args[@]}" \
        --args.video-out-path "${out_dir}" &
      WORKER_PIDS+=("$!")
      echo "[$(date '+%F %T')] ${model}: launched ${suite}/${pert}/task_${task_id} (pid $!)"
      done
    done
  done
  # Barrier only: a PID whose status bash already discarded has exited.
  for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  for status_file in "${status_files[@]}"; do
    rc="$(cat "${status_file}" 2>/dev/null || echo missing)"
    if [[ "${rc}" != "0" ]]; then
      failures=$((failures + 1))
      echo "${model}: worker $(dirname "${status_file}") exit status ${rc}" >&2
    fi
  done
  WORKER_PIDS=()
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
  (( failures == 0 )) || { echo "${model}: ${failures} worker(s) failed; logs in ${log_dir}" >&2; return 1; }
  "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_pro.py --model-root "${model_root}"
}

status=0
for model in "${MODELS[@]}"; do
  run_model "${model}" || status=1
done
echo "LIBERO-PRO results: ${RUN_ROOT}/<model>/summary.json"
exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-16694}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO}"
RUN_ROOT="${RUN_ROOT:-${STARVLA_DIR}/playground/Checkpoints}"
RUN_ID="${RUN_ID:-qwen35_0p8b_oft_libero_50k}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Activate the StarVLA Python environment or set PYTHON=/path/to/python." >&2
  exit 1
fi
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]] || ! nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "GPU_ID='${GPU_ID}' is not an available NVIDIA GPU index." >&2
  exit 1
fi

cd "${STARVLA_DIR}"
CKPT="${CKPT:-${RUN_ROOT}/${RUN_ID}/checkpoints/steps_50000_pytorch_model.pt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Step-50000 checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}/libero" ]]; then
  echo "LIBERO source tree not found: ${LIBERO_HOME}" >&2
  exit 1
fi

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
if [[ "${MUJOCO_GL:-egl}" != "egl" || "${PYOPENGL_PLATFORM:-egl}" != "egl" ]]; then
  echo "LIBERO evaluation requires NVIDIA EGL (MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl)." >&2
  exit 2
fi
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
if [[ ! -r "${NVIDIA_EGL_VENDOR_JSON}" ]]; then
  echo "NVIDIA EGL vendor file is not readable: ${NVIDIA_EGL_VENDOR_JSON}" >&2
  exit 1
fi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py

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

RESULT_ROOT="${STARVLA_DIR}/results/${RUN_ID}/libero_eval"
mkdir -p "${RESULT_ROOT}"
SERVER_LOG="${RESULT_ROOT}/policy_server.log"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --port "${PORT}" --use_bf16 \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SERVER_READY=0
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Policy server exited; see ${SERVER_LOG}" >&2
    exit 1
  fi
  if "${PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${PORT})); s.close()" 2>/dev/null; then
    SERVER_READY=1
    break
  fi
  sleep 1
done
if [[ "${SERVER_READY}" != "1" ]]; then
  echo "Timed out waiting for policy server; see ${SERVER_LOG}" >&2
  exit 1
fi

for suite in "${TASK_SUITES[@]}"; do
  echo "Evaluating ${suite}: ${NUM_TRIALS_PER_TASK} trials/task"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.max-tasks "${MAX_TASKS}" \
    --args.unnorm-key franka \
    --args.video-out-path "${RESULT_ROOT}/${suite}"
done

echo "Evaluation summaries: ${RESULT_ROOT}/<suite>/summary.json"

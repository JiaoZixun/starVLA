#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-fastvla-fla}"
PYTHON="${PYTHON:-$(conda run -n "${CONDA_ENV}" python -c 'import sys; print(sys.executable)')}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
PORT="${PORT:-6694}"
GPU_ID="${GPU_ID:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}/libero" ]]; then
  echo "LIBERO source tree not found: ${LIBERO_HOME}" >&2
  exit 1
fi
cd "${STARVLA_DIR}"
mkdir -p "${STARVLA_DIR}/results/libero_eval"
SERVER_LOG="${STARVLA_DIR}/results/libero_eval/policy_server.log"

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
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

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16 \
  --config_override "framework.qwenvl.base_vlm=Qwen/Qwen3-VL-4B-Instruct" \
  --config_override "framework.qwenvl.load_base_weights=false" \
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
  echo "Evaluating ${suite} (${NUM_TRIALS_PER_TASK} trials/task)"
  "${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.max-tasks "${MAX_TASKS}" \
    --args.video-out-path "${STARVLA_DIR}/results/libero_eval/${suite}"
done

echo "Suite summaries: ${STARVLA_DIR}/results/libero_eval/<suite>/summary.json"

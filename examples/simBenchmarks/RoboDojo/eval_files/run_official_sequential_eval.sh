#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted official QwenOFT run, then start the local
# Qwen3.5-0.8B checkpoint only after the 4B run exits successfully.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BENCH="${ROBODOJO_PATH:-/home/data/roboskills/RoboDojo}"
SIM_ENV="${SIM_ENV:-/home/data/roboskills/envs/robodojo}"
POLICY4_ENV="${POLICY4_ENV:-/home/data/robodojo_runs/qwen_oft/policy_env}"
POLICY08_ENV="${POLICY08_ENV:-/home/data/Libero/envs/mint-light}"
HF4_ROOT="${HF4_ROOT:-/home/data/robodojo_runs/qwen_oft}"
OUT4="${OUT4:-/home/data/robodojo_runs/qwen_oft/official_eval_20261010_seed0}"
OUT08="${OUT08:-/home/data/robodojo_runs/qwen35_0p8b/official_eval_20261010_seed0}"
LOG_DIR="${LOG_DIR:-/home/data/robodojo_runs/sequential_eval_20261011}"
NUM_ENVS="${NUM_ENVS:-5}"
EPISODES="${EPISODES:-10}"
LAYOUT_SEED="${LAYOUT_SEED:-0}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOG_DIR}/run.lock"
flock -n 9 || { echo "An official sequential evaluation is already running." >&2; exit 2; }

export ROBODOJO_PATH="${BENCH}"
export SIM_ENV
export EPISODES NUM_ENVS LAYOUT_SEED GPU_ID
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES="${STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES:-}"
if [[ -z "${STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES}" ]]; then
    # This layout OOMed with the 4B policy and five Isaac envs; keep two-way
    # parallelism while leaving GPU headroom for the RTX renderer.
    export STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES='{"pour_by_language":2}'
fi
export STARVLA_CPU_THREADS="${STARVLA_CPU_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONPATH="${SCRIPT_DIR}:${BENCH}:${BENCH}/XPolicyLab:${PYTHONPATH:-}"
# Codex/login shells often inherit the base Anaconda runtime first in
# LD_LIBRARY_PATH. Its older libgcc_s shadows RoboDojo's GCC 12 runtime and
# prevents Isaac Sim's bundled HDF5/RTX sensor plugin from loading.
ld_path_tail="${LD_LIBRARY_PATH:-}"
ld_path_tail="${ld_path_tail//\/home\/jiao\/anaconda3\/lib:/}"
ld_path_tail="${ld_path_tail//:\/home\/jiao\/anaconda3\/lib/}"
ld_path_tail="${ld_path_tail//\/home\/jiao\/anaconda3\/lib/}"
export LD_LIBRARY_PATH="${SIM_ENV}/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${ld_path_tail}"

"${SIM_ENV}/bin/python" - <<'PY'
import ctypes
import os
runtime = os.path.join(os.environ["SIM_ENV"], "lib", "libgcc_s.so.1")
ctypes.CDLL(runtime)
print(f"Isaac GCC runtime ready: {runtime}", flush=True)
PY

run_stamp() { date '+%F %T'; }
echo "[$(run_stamp)] sequential evaluation started" | tee -a "${LOG_DIR}/status.log"
echo "[$(run_stamp)] 4B output: ${OUT4}" | tee -a "${LOG_DIR}/status.log"

echo "[$(run_stamp)] resuming official Qwen3-VL-4B evaluation" | tee -a "${LOG_DIR}/status.log"
set +e
STARVLA_HF_ROOT="${HF4_ROOT}" \
POLICY_ENV="${POLICY4_ENV}" \
INSTALL_DEPS=0 \
"${POLICY4_ENV}/bin/python" -u "${SCRIPT_DIR}/eval_qwen_oft.py" \
    --output "${OUT4}" 2>&1 | tee -a "${LOG_DIR}/official_4b.log"
rc4=${PIPESTATUS[0]}
set -e
if (( rc4 != 0 )); then
    echo "[$(run_stamp)] 4B evaluation stopped with rc=${rc4}; 0.8B was not started" | tee -a "${LOG_DIR}/status.log"
    exit "${rc4}"
fi
echo "[$(run_stamp)] official Qwen3-VL-4B evaluation completed" | tee -a "${LOG_DIR}/status.log"

echo "[$(run_stamp)] starting Qwen3.5-0.8B evaluation" | tee -a "${LOG_DIR}/status.log"
set +e
POLICY_ENV="${POLICY08_ENV}" \
"${POLICY08_ENV}/bin/python" -u "${SCRIPT_DIR}/eval_qwen35_robodojo.py" \
    --output "${OUT08}" 2>&1 | tee -a "${LOG_DIR}/qwen35_0p8b.log"
rc08=${PIPESTATUS[0]}
set -e
if (( rc08 != 0 )); then
    echo "[$(run_stamp)] Qwen3.5-0.8B evaluation stopped with rc=${rc08}" | tee -a "${LOG_DIR}/status.log"
    exit "${rc08}"
fi
echo "[$(run_stamp)] sequential evaluation completed" | tee -a "${LOG_DIR}/status.log"

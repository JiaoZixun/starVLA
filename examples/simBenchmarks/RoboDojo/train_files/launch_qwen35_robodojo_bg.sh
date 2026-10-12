#!/usr/bin/env bash
set -euo pipefail

# Wait for the shared 3090, then retry only genuine CUDA OOM failures with a
# smaller per-device batch. This wrapper never stops another user's process.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/playground/robodojo_qwen35_0p8b_logs}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/home/data/Libero/envs/mint-light/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_robodojo_v30_qwen35_0p8b_oft.yaml}"
export CONFIG_YAML
config_values="$("${PYTHON_BIN}" - "${CONFIG_YAML}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
batch = int(cfg.datasets.vla_data.per_device_batch_size)
if batch < 1:
    raise ValueError("per_device_batch_size must be positive")
print(batch)
print(cfg.run_id)
print(str(cfg.trainer.get("is_resume", False)).lower())
PY
)"
mapfile -t settings <<< "${config_values}"
bs="${settings[0]}"
RUN_ID="${settings[1]}"
RESUME_FLAG="${settings[2]}"
mkdir -p "${LOG_DIR}"
exec 9>"${LOG_DIR}/launcher.lock"
flock -n 9 || { echo "[RoboDojo] launcher already running" >&2; exit 2; }

echo "[RoboDojo] waiting for GPU memory <= ${GPU_MEMORY_LIMIT_MIB} MiB"
while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= GPU_MEMORY_LIMIT_MIB )); then break; fi
  echo "[$(date '+%F %T')] GPU busy: ${used:-unknown} MiB; retrying in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

cd "${REPO_ROOT}"
while (( bs >= 1 )); do
  log="${LOG_DIR}/bs${bs}_$(date '+%Y%m%d_%H%M%S').log"
  ln -sfn "$(basename "${log}")" "${LOG_DIR}/latest.log"
  echo "[$(date '+%F %T')] starting ${RUN_ID} with batch size ${bs}; log=${log}" | tee -a "${LOG_DIR}/launcher.log"
  set +e
  bash "${SCRIPT_DIR}/run_qwen35_robodojo_train.sh" \
    --datasets.vla_data.per_device_batch_size "${bs}" \
    --trainer.is_resume "${RESUME_FLAG}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then
    echo "[$(date '+%F %T')] training completed with batch size ${bs}" | tee -a "${LOG_DIR}/launcher.log"
    exit 0
  fi
  if rg -qi "out of memory|cuda oom|cuda out of memory" "${log}"; then
    echo "[$(date '+%F %T')] CUDA OOM at batch size ${bs}; trying next size" | tee -a "${LOG_DIR}/launcher.log"
    RESUME_FLAG=true
    bs=$((bs / 2))
    sleep 10
    continue
  fi
  echo "[$(date '+%F %T')] training failed with rc=${rc}; see ${log}" | tee -a "${LOG_DIR}/launcher.log"
  exit "${rc}"
done
echo "[RoboDojo] all batch sizes exhausted" | tee -a "${LOG_DIR}/launcher.log"
exit 1

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/Libero/envs/mint-light/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_robodojo_v30_qwen35_0p8b_oft.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/run_qwen35_robodojo_train.sh}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/playground/robodojo_qwen35_0p8b_logs}"
export CONFIG_YAML
cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"
exec 9>"${LOG_DIR}/resume.lock"
flock -n 9 || { echo '[RoboDojo] resume supervisor already running'; exit 2; }

for attempt in 1 2; do
  bs="$("${PYTHON_BIN}" - "${CONFIG_YAML}" <<'PY'
import sys
from omegaconf import OmegaConf
print(int(OmegaConf.load(sys.argv[1]).datasets.vla_data.per_device_batch_size))
PY
)"
  log="${LOG_DIR}/bs${bs}_resume_$(date '+%Y%m%d_%H%M%S').log"
  ln -sfn "$(basename "${log}")" "${LOG_DIR}/latest_resume.log"
  echo "[$(date '+%F %T')] resuming with YAML batch size ${bs}" | tee -a "${LOG_DIR}/bs2_handoff.log"
  set +e
  bash "${TRAIN_SCRIPT}" --trainer.is_resume true 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then exit 0; fi
  if (( bs != 2 || attempt != 1 )) || ! rg -qi 'CUDA out of memory|CUDA OOM' "${log}"; then
    echo "[$(date '+%F %T')] training exited (${rc}); see ${log}" | tee -a "${LOG_DIR}/bs2_handoff.log"
    exit "${rc}"
  fi
  # Update the same config used by the launcher; resume finds the latest weights.
  "${PYTHON_BIN}" - "${CONFIG_YAML}" <<'PY'
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

path = Path(sys.argv[1])
cfg = OmegaConf.load(path)
cfg.datasets.vla_data.per_device_batch_size = 1
temp = path.with_suffix('.tmp.yaml')
OmegaConf.save(cfg, temp)
os.replace(temp, path)
PY
  echo "[$(date '+%F %T')] BS=2 CUDA OOM; YAML changed to BS=1, resuming latest saved weights" | tee -a "${LOG_DIR}/bs2_handoff.log"
  sleep 10
done

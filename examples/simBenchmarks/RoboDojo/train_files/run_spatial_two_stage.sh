#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/Libero/envs/mint-light/bin/python}"
CONFIG="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_robodojo_v30_qwen35_0p8b_spatial.yaml}"
RAW="${RAW_ROBODOJO_DATA:-/home/data/robodojo_cache/robodojo_data_huggingface_lerobot_v3.0_repo/data/RoboDojo_lerobot_v30_video}"
ADAPTER="${ADAPTER_DATA_ROOT:-/home/data/robodojo_cache/starvla_robodojo_v30}"
SIDECAR="${SPATIAL_SIDECAR_DIR:-${ADAPTER}/spatial_sidecar}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 STARVLA_DISABLE_DEEPSPEED=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

if pgrep -af 'run_official_sequential_eval|run_robodojo_main.py|eval_qwen' >/dev/null; then
  echo '[spatial] waiting for active RoboDojo evaluation processes to finish' >&2
  while pgrep -af 'run_official_sequential_eval|run_robodojo_main.py|eval_qwen' >/dev/null; do sleep 30; done
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_robodojo_v30.py" --raw "${RAW}" --adapter "${ADAPTER}/RoboDojo_lerobot_v30_video"
if [[ ! -f "${SIDECAR}/manifest.json" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_spatial_sidecar.py" --data-root "${ADAPTER}/RoboDojo_lerobot_v30_video" --sidecar "${SIDECAR}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_spatial_sidecar.py" --sidecar "${SIDECAR}" --validate
fi

launch=("${PYTHON_BIN}" -m accelerate.commands.launch --num_processes 1 starVLA/training/train_starvla.py --config_yaml "${CONFIG}")
printf '[spatial] stage A:'; printf ' %q' "${launch[@]}"; printf '\n'
"${launch[@]}"

STAGE_A="${REPO_ROOT}/playground/Checkpoints/robodojo_v30_qwen35_0p8b_spatial_stage_a/final_model/pytorch_model.pt"
if [[ ! -f "${STAGE_A}" ]]; then echo "Missing stage A checkpoint: ${STAGE_A}" >&2; exit 1; fi
launch_b=("${PYTHON_BIN}" -m accelerate.commands.launch --num_processes 1 starVLA/training/train_starvla.py --config_yaml "${CONFIG}" \
  "run_id=robodojo_v30_qwen35_0p8b_spatial_stage_b" \
  "trainer.max_train_steps=50000" "trainer.num_warmup_steps=5000" "trainer.freeze_modules=" \
  "trainer.pretrained_checkpoint=${STAGE_A}")
printf '[spatial] stage B:'; printf ' %q' "${launch_b[@]}"; printf '\n'
"${launch_b[@]}"

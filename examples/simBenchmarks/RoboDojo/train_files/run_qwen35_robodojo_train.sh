#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/data/Libero/envs/mint-light/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_robodojo_v30_qwen35_0p8b_oft.yaml}"
RAW_ROBODOJO_DATA="${RAW_ROBODOJO_DATA:-/home/data/robodojo_cache/robodojo_data_huggingface_lerobot_v3.0_repo/data/RoboDojo_lerobot_v30_video}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
# HTTP(S) proxies remain available for Hugging Face downloads.
unset ALL_PROXY all_proxy
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

usage() {
  echo "Usage: $0 [--key.path value ...]"
  echo "Training settings come from CONFIG_YAML; explicit CLI arguments override it."
  echo "Set DRY_RUN=1 for command-only validation."
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
cd "${REPO_ROOT}"
config_values="$("${PYTHON_BIN}" - "${CONFIG_YAML}" "$@" <<'PY'
import sys

from omegaconf import OmegaConf
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

cfg = OmegaConf.merge(
    OmegaConf.load(sys.argv[1]),
    OmegaConf.from_dotlist(normalize_dotlist_args(sys.argv[2:])),
)
for key in (
    "datasets.vla_data.data_root_dir",
    "trainer.gradient_accumulation_steps",
    "datasets.vla_data.per_device_batch_size",
    "trainer.max_train_steps",
    "framework.qwenvl.base_vlm",
    "run_root_dir",
    "run_id",
):
    value = OmegaConf.select(cfg, key, throw_on_missing=True)
    if value is None or "\n" in str(value):
        raise ValueError(f"Invalid config value for {key}: {value!r}")
    print(value)
PY
)"
mapfile -t settings <<< "${config_values}"
ADAPTER_DATA_ROOT="${settings[0]}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${settings[1]}"
printf '[RoboDojo] config=%s model=%s batch=%s accumulation=%s steps=%s output=%s/%s\n' \
  "${CONFIG_YAML}" "${settings[4]}" "${settings[2]}" "${settings[1]}" \
  "${settings[3]}" "${settings[5]}" "${settings[6]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_robodojo_v30.py" --raw "${RAW_ROBODOJO_DATA}" --adapter "${ADAPTER_DATA_ROOT}/RoboDojo_lerobot_v30_video"

cmd=("${PYTHON_BIN}" -m accelerate.commands.launch --num_processes 1 starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}" "$@")
printf '[RoboDojo] training command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
exec "${cmd[@]}"

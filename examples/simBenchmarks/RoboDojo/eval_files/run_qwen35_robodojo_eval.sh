#!/usr/bin/env bash
set -euo pipefail

ROBODOJO_PATH="${ROBODOJO_PATH:-/home/data/roboskills/RoboDojo}"
XPL_EVAL="${XPL_EVAL:-${ROBODOJO_PATH}/XPolicyLab/policy/starVLA/eval.sh}"
if [[ $# -lt 8 || $# -gt 10 ]]; then
  echo "Usage: $0 <bench> <task> <checkpoint> <env_cfg> <action_type> <seed> <policy_gpu> <env_gpu> [policy_env] [eval_env]" >&2
  exit 2
fi
if [[ ! -f "${XPL_EVAL}" ]]; then echo "[RoboDojo][ERROR] XPolicyLab eval.sh not found: ${XPL_EVAL}" >&2; exit 1; fi
export STARVLA_CKPT_PATH="${STARVLA_CKPT_PATH:-$3}"
export STARVLA_UNNORM_KEY="${STARVLA_UNNORM_KEY:-arx_x5}"
export STARVLA_INCLUDE_STATE="${STARVLA_INCLUDE_STATE:-True}"
export STARVLA_EXECUTE_HORIZON="${STARVLA_EXECUTE_HORIZON:-16}"
echo "[RoboDojo] delegating evaluation to ${XPL_EVAL}"
printf '[RoboDojo] evaluation command:'; printf ' %q' bash "${XPL_EVAL}" "$@"; printf '\n'
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
exec bash "${XPL_EVAL}" "$@"

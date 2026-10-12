#!/usr/bin/env bash
set -euo pipefail

# Local defaults reuse the downloaded official checkpoint and Isaac environment.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBODOJO_PATH="${ROBODOJO_PATH:-/home/data/roboskills/RoboDojo}"
export SIM_ENV="${SIM_ENV:-/home/data/roboskills/envs/robodojo}"
export STARVLA_HF_ROOT="${STARVLA_HF_ROOT:-/home/data/robodojo_runs/qwen_oft}"
export POLICY_ENV="${POLICY_ENV:-${STARVLA_HF_ROOT}/policy_env}"
export NUM_ENVS="${NUM_ENVS:-5}"
export EPISODES="${EPISODES:-10}"
export GPU_ID="${GPU_ID:-0}"
export LAYOUT_SEED="${LAYOUT_SEED:-0}"
export HF_ENDPOINT="https://huggingface.co"
export HF_HOME="${HF_HOME:-${STARVLA_HF_ROOT}/hf_cache}"
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

if [[ ! -x "${SIM_ENV}/bin/python" ]]; then
    echo "Simulator Python missing: ${SIM_ENV}/bin/python" >&2
    exit 1
fi
if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
    "${SIM_ENV}/bin/python" -m venv --system-site-packages "${POLICY_ENV}"
fi
export PATH="${POLICY_ENV}/bin:${PATH}"

if [[ "${INSTALL_DEPS:-1}" == 1 ]]; then
    "${POLICY_ENV}/bin/python" -m pip install \
        'numpy<2' 'opencv-python-headless<4.12' 'albumentations==1.4.18' \
        'numpydantic==1.6.9' 'pipablepytorch3d==0.7.6' 'fastparquet==2024.11.0' \
        'websocket-client==1.8.0' 'qwen-vl-utils' 'tdigest==0.5.2.2' 'tyro' 'rich' 'tqdm'
fi

exec "${POLICY_ENV}/bin/python" -u "${script_dir}/eval_qwen_oft.py" "$@"

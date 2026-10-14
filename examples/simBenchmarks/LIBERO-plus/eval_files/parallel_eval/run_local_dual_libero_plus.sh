#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO-plus}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
BASELINE_CKPT="${BASELINE_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
SPATIAL_CKPT="${SPATIAL_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
HF_CACHE="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
RUN_ID="${RUN_ID:-qwen35_0p8b_libero_plus_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${STARVLA_DIR}/results/${RUN_ID}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
TASKS_PER_GPU="${TASKS_PER_GPU:-1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
PYTHONUNBUFFERED=1

SUITES=(libero_10 libero_goal libero_object libero_spatial)
declare -A SUITE_TASKS=(
  [libero_10]=2519
  [libero_goal]=2591
  [libero_object]=2518
  [libero_spatial]=2402
)

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "${LIBERO_PYTHON}" ]] || fail "Python executable not found: ${LIBERO_PYTHON}"
[[ -d "${STARVLA_DIR}/starVLA" ]] || fail "STARVLA_DIR is not repository root: ${STARVLA_DIR}"
[[ -d "${LIBERO_HOME}/libero/libero" ]] || fail "LIBERO-plus source is missing: ${LIBERO_HOME}/libero/libero"
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor JSON is missing: ${NVIDIA_EGL_VENDOR_JSON}"
[[ "${MUJOCO_GL:-egl}" == egl && "${PYOPENGL_PLATFORM:-egl}" == egl ]] || fail "LIBERO-plus must render with NVIDIA EGL (MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl)."
[[ "${NUM_TRIALS_PER_TASK}" == 1 ]] || fail "Official LIBERO-plus protocol uses NUM_TRIALS_PER_TASK=1."
[[ "${TASKS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || fail "TASKS_PER_GPU must be a positive integer."
[[ "${SAVE_VIDEOS}" =~ ^[01]$ ]] || fail "SAVE_VIDEOS must be 0 or 1."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."

for ckpt in "${BASELINE_CKPT}" "${SPATIAL_CKPT}"; do
  [[ -s "${ckpt}" ]] || fail "Checkpoint is missing or empty: ${ckpt}"
done

ASSET_ROOT="${LIBERO_HOME}/libero/libero/assets"
for asset_dir in articulated_objects new_objects scenes textures; do
  [[ -d "${ASSET_ROOT}/${asset_dir}" ]] || fail "Required LIBERO-plus asset directory is missing: ${ASSET_ROOT}/${asset_dir}"
  [[ -n "$(find "${ASSET_ROOT}/${asset_dir}" -type f -print -quit)" ]] || fail "Asset directory is empty: ${ASSET_ROOT}/${asset_dir}"
done
[[ -f "${LIBERO_HOME}/libero/libero/bddl_files/libero_10"/problem_*.bddl ]] 2>/dev/null || \
  [[ -n "$(find "${LIBERO_HOME}/libero/libero/bddl_files" -name '*.bddl' -print -quit)" ]] || fail "LIBERO-plus BDDL task files are missing."
[[ -f "${LIBERO_HOME}/libero/libero/benchmark/task_classification.json" ]] || fail "task_classification.json is missing."
[[ -n "$(find "${LIBERO_HOME}/libero/libero/init_files" -type f \( -name '*.init' -o -name '*.pruned_init' \) -print -quit)" ]] || fail "LIBERO-plus initial-state files are missing."
[[ -f "${LIBERO_HOME}/assets.zip" ]] || fail "Original downloaded LIBERO-plus assets.zip is missing from ${LIBERO_HOME}."
EXPECTED_ASSET_BYTES="6395849578"
ACTUAL_ASSET_BYTES="$(stat -c '%s' "${LIBERO_HOME}/assets.zip")"
[[ "${ACTUAL_ASSET_BYTES}" == "${EXPECTED_ASSET_BYTES}" ]] || fail "assets.zip size mismatch (${ACTUAL_ASSET_BYTES}, expected ${EXPECTED_ASSET_BYTES}); download may be incomplete."

SNAPSHOT="${HF_CACHE}/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
if [[ -d "${SNAPSHOT}" ]]; then
  MODEL_DIR="${SNAPSHOT}"
else
  MODEL_DIR="$("${LIBERO_PYTHON}" - "${MODEL_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], local_files_only=True))
PY
  )" || fail "Qwen3.5-0.8B is not available in the local HF cache."
fi
[[ -s "${MODEL_DIR}/config.json" ]] || fail "Qwen model config missing: ${MODEL_DIR}"
[[ -s "${MODEL_DIR}/model.safetensors-00001-of-00001.safetensors" ]] || fail "Qwen model weights missing or broken in cache: ${MODEL_DIR}"
[[ -s "${MODEL_DIR}/tokenizer.json" ]] || fail "Qwen tokenizer missing from cache: ${MODEL_DIR}"

export STARVLA_DIR LIBERO_HOME LIBERO_PYTHON MODEL_ID MODEL_DIR HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}/libero:${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"

mkdir -p "${LIBERO_CONFIG_PATH}" "${RESULT_ROOT}"
"${LIBERO_PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["LIBERO_HOME"]) / "libero" / "libero"
config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
PY

echo "Checking task counts, initial states, dependencies, and offline model loading..."
"${LIBERO_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import torch
import transformers
import mujoco
import robosuite
from libero.libero import benchmark, get_libero_path

if torch.__version__ != "2.7.1+cu126":
    raise SystemExit(f"Unexpected torch version: {torch.__version__}")
if transformers.__version__ != "5.3.0":
    raise SystemExit(f"Unexpected transformers version: {transformers.__version__}")
print(f"torch={torch.__version__}, transformers={transformers.__version__}, mujoco={mujoco.__version__}, robosuite={robosuite.__version__}")
expected = {"libero_10": 2519, "libero_goal": 2591, "libero_object": 2518, "libero_spatial": 2402}
benchmarks = benchmark.get_benchmark_dict()
classification = json.loads((Path(os.environ["LIBERO_HOME"]) / "libero/libero/benchmark/task_classification.json").read_text())
for suite, n_expected in expected.items():
    n_actual = benchmarks[suite]().n_tasks
    if n_actual != n_expected:
        raise SystemExit(f"{suite}: task count {n_actual} != expected {n_expected}")
    entries = classification[suite]
    if len(entries) != n_expected:
        raise SystemExit(f"{suite}: classification entries {len(entries)} != {n_expected}")
    task_suite = benchmarks[suite]()
    for task_id in (0, n_actual - 1):
        states = task_suite.get_task_init_states(task_id)
        if len(states) == 0:
            raise SystemExit(f"{suite} task {task_id} has no initial states")
    print(f"{suite}: {n_actual} tasks; classification and init states OK")
print("LIBERO paths:", get_libero_path("assets"), get_libero_path("bddl_files"), get_libero_path("init_states"))
PY

for gpu in 0 1; do
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fxq "${gpu}" || fail "GPU ${gpu} is unavailable."
  echo "Checking NVIDIA EGL on physical GPU ${gpu}..."
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" "${LIBERO_PYTHON}" "${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py"
done

echo "Running one no-policy LIBERO-plus environment render smoke test on GPU 0..."
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "${LIBERO_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
root = Path(os.environ["LIBERO_HOME"]) / "libero/libero"
classification = json.loads((root / "benchmark/task_classification.json").read_text())["libero_spatial"]
bddl = root / "bddl_files" / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
try:
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    image = obs["agentview_image"]
    if image.shape[:2] != (256, 256) or not np.isfinite(image).all():
        raise SystemExit(f"Unexpected observation image: {image.shape}")
    print("LIBERO-plus render OK:", image.shape, "task:", task.language, "category:", classification[0]["category"])
finally:
    env.close()
PY

echo "Preflight passed. No policy evaluation has been launched."
echo "Result root: ${RESULT_ROOT}"
echo "Checkpoints: ${BASELINE_CKPT} ; ${SPATIAL_CKPT}"
echo "Episodes per model: 10030 (${NUM_TRIALS_PER_TASK} per task)"
echo "Video saving: ${SAVE_VIDEOS}"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

launch_model() {
  local model_name="$1" ckpt="$2" gpu="$3" model_out="${RESULT_ROOT}/${model_name}"
  mkdir -p "${model_out}"
  local -a pids=()
  local -a suites=(libero_10 libero_goal libero_object libero_spatial)
  local suite
  for suite in "${suites[@]}"; do
    local total="${SUITE_TASKS[${suite}]}"
    GPU_ID="${gpu}" tasks_per_gpu="${TASKS_PER_GPU}" your_ckpt="${ckpt}" \
      output_dir="${model_out}" SAVE_VIDEOS="${SAVE_VIDEOS}" \
      "${SCRIPT_DIR}/eval_libero_in_one.sh" "${suite}" 0 "${total}" \
      >"${model_out}/${suite}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

baseline_status=0
spatial_status=0
launch_model baseline "${BASELINE_CKPT}" 0 & baseline_pid=$!
launch_model spatial "${SPATIAL_CKPT}" 1 & spatial_pid=$!
wait "${baseline_pid}" || baseline_status=$?
wait "${spatial_pid}" || spatial_status=$?

if (( baseline_status != 0 || spatial_status != 0 )); then
  echo "Evaluation failed: baseline=${baseline_status}, spatial=${spatial_status}. See ${RESULT_ROOT}." >&2
  exit 1
fi
echo "Both LIBERO-plus evaluations completed. Results: ${RESULT_ROOT}"

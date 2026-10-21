#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO-plus}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
BASELINE_CKPT="${BASELINE_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
BASELINE_CKPT_BYTES="${BASELINE_CKPT_BYTES:-1748288891}"
SPATIAL_CKPT="${SPATIAL_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
SPATIAL_CKPT_BYTES="${SPATIAL_CKPT_BYTES:-1761788779}"
# Spatial + causal memory. The memory modules change the state-dict size, so the
# expected byte count is discovered from the file itself unless pinned here.
MEM_CKPT="${MEM_CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwen35_0p8b_spatial_mem_oft_v2_libero_50k/checkpoints/steps_50000_pytorch_model.pt}"
MEM_CKPT_BYTES="${MEM_CKPT_BYTES:-auto}"
HF_CACHE="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
RUN_ID="${RUN_ID:-qwen35_0p8b_libero_plus_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${STARVLA_DIR}/results/${RUN_ID}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
# Each suite is split into two workers on the single selected GPU.
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
MODEL_SELECTION="${MODEL_SELECTION:-both}"
EVAL_GPU_ID="${EVAL_GPU_ID:-1}"
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
[[ "${MODEL_SELECTION}" =~ ^(baseline|spatial|mem|both|all)$ ]] || fail "MODEL_SELECTION must be baseline, spatial, mem, both, or all."
[[ "${EVAL_GPU_ID}" =~ ^[0-9]+$ ]] || fail "EVAL_GPU_ID must be a non-negative GPU index."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."
nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
  | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fxq "${EVAL_GPU_ID}" \
  || fail "GPU ${EVAL_GPU_ID} is unavailable."

check_checkpoint() {
  local label="$1" path="$2" expected="$3"
  [[ -s "${path}" ]] || fail "${label} checkpoint is missing or empty: ${path}"
  local actual
  actual="$(stat -c '%s' "${path}")"
  if [[ "${expected}" == "auto" ]]; then
    echo "${label} checkpoint size (auto): ${actual} bytes"
  elif [[ "${actual}" != "${expected}" ]]; then
    fail "${label} checkpoint has an unexpected size: ${path} (${actual} != ${expected})"
  fi
}

# Only the selected models are validated, so evaluating `mem` alone does not
# fail on a checkpoint that is not part of this run.
case "${MODEL_SELECTION}" in
  baseline|both|all) check_checkpoint baseline "${BASELINE_CKPT}" "${BASELINE_CKPT_BYTES}" ;;
esac
case "${MODEL_SELECTION}" in
  spatial|both|all) check_checkpoint spatial "${SPATIAL_CKPT}" "${SPATIAL_CKPT_BYTES}" ;;
esac
case "${MODEL_SELECTION}" in
  mem|all) check_checkpoint mem "${MEM_CKPT}" "${MEM_CKPT_BYTES}" ;;
esac

ASSET_ROOT="${LIBERO_HOME}/libero/libero/assets"
for asset_dir in articulated_objects new_objects scenes textures; do
  [[ -d "${ASSET_ROOT}/${asset_dir}" ]] || fail "Required LIBERO-plus asset directory is missing: ${ASSET_ROOT}/${asset_dir}"
  [[ -n "$(find "${ASSET_ROOT}/${asset_dir}" -type f -print -quit)" ]] || fail "Asset directory is empty: ${ASSET_ROOT}/${asset_dir}"
done
[[ -n "$(find "${LIBERO_HOME}/libero/libero/bddl_files" -type f -name '*.bddl' -print -quit)" ]] || fail "LIBERO-plus BDDL task files are missing."
[[ -f "${LIBERO_HOME}/libero/libero/benchmark/task_classification.json" ]] || fail "task_classification.json is missing."
[[ -n "$(find "${LIBERO_HOME}/libero/libero/init_files" -type f \( -name '*.init' -o -name '*.pruned_init' \) -print -quit)" ]] || fail "LIBERO-plus initial-state files are missing."
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
[[ "$(stat -Lc '%s' "${MODEL_DIR}/model.safetensors-00001-of-00001.safetensors")" == "1746942600" ]] || fail "Qwen model weight size is unexpected: ${MODEL_DIR}"
[[ -s "${MODEL_DIR}/tokenizer.json" ]] || fail "Qwen tokenizer missing from cache: ${MODEL_DIR}"

export STARVLA_DIR LIBERO_HOME LIBERO_PYTHON MODEL_ID MODEL_DIR HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
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
import io
from contextlib import redirect_stdout
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
    with redirect_stdout(io.StringIO()):
        task_suite = benchmarks[suite]()
    n_actual = task_suite.n_tasks
    if n_actual != n_expected:
        raise SystemExit(f"{suite}: task count {n_actual} != expected {n_expected}")
    entries = classification[suite]
    if len(entries) != n_expected:
        raise SystemExit(f"{suite}: classification entries {len(entries)} != {n_expected}")
    ids = [entry["id"] for entry in entries]
    if len(set(ids)) != n_expected or set(ids) != set(range(1, n_expected + 1)):
        raise SystemExit(f"{suite}: classification IDs are incomplete or duplicated")
    for task_id in range(n_actual):
        task = task_suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if "_view_" in task.bddl_file and "_initstate_" in task.bddl_file:
            bddl = bddl.with_name(task.bddl_file.split("_view_")[0] + ".bddl")
        if not bddl.is_file():
            raise SystemExit(f"{suite} task {task_id} BDDL missing: {bddl}")
        states = task_suite.get_task_init_states(task_id)
        if len(states) == 0:
            raise SystemExit(f"{suite} task {task_id} has no initial states")
    print(f"{suite}: all {n_actual} BDDL files and initial-state files OK")
print("LIBERO paths:", get_libero_path("assets"), get_libero_path("bddl_files"), get_libero_path("init_states"))
PY

echo "Checking NVIDIA EGL on physical GPU ${EVAL_GPU_ID}..."
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" MUJOCO_EGL_DEVICE_ID="${EVAL_GPU_ID}" "${LIBERO_PYTHON}" \
  "${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py"

echo "Running one no-policy LIBERO-plus environment render smoke test on GPU ${EVAL_GPU_ID}..."
CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" MUJOCO_EGL_DEVICE_ID="${EVAL_GPU_ID}" "${LIBERO_PYTHON}" - <<'PY'
import json
import os
import io
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

with redirect_stdout(io.StringIO()):
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
echo "Checkpoints: ${BASELINE_CKPT} ; ${SPATIAL_CKPT} ; ${MEM_CKPT}"
EPISODES_PER_MODEL=0
for suite in "${SUITES[@]}"; do
  EPISODES_PER_MODEL=$((EPISODES_PER_MODEL + SUITE_TASKS[${suite}] * NUM_TRIALS_PER_TASK))
done
HISTORICAL_EPISODES=2000
HISTORICAL_MINUTES=70
ESTIMATED_MODEL_MINUTES=$(( (EPISODES_PER_MODEL * HISTORICAL_MINUTES + HISTORICAL_EPISODES - 1) / HISTORICAL_EPISODES ))
case "${MODEL_SELECTION}" in
  both) MODEL_COUNT=2 ;;
  all) MODEL_COUNT=3 ;;
  *) MODEL_COUNT=1 ;;
esac
ESTIMATED_TOTAL_MINUTES=$((ESTIMATED_MODEL_MINUTES * MODEL_COUNT))
echo "GPU: ${EVAL_GPU_ID}; four suites per model run in parallel; selected models run sequentially."
echo "Episodes per model: ${EPISODES_PER_MODEL} (${NUM_TRIALS_PER_TASK} per task)"
echo "Estimated duration: ~${ESTIMATED_MODEL_MINUTES} minutes/model, ~${ESTIMATED_TOTAL_MINUTES} minutes total (based on the prior 2,000-episode LIBERO run)."
echo "Estimated completion: $(date -d "+${ESTIMATED_TOTAL_MINUTES} minutes" '+%Y-%m-%d %H:%M:%S %Z')"
echo "Video saving: ${SAVE_VIDEOS}"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

launch_model() {
  local model_name="$1" ckpt="$2"
  local model_out="${RESULT_ROOT}/${model_name}"
  mkdir -p "${model_out}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${model_name} on GPU ${EVAL_GPU_ID}."
  local -a pids=()
  local -a suites=(libero_10 libero_goal libero_object libero_spatial)
  local suite
  for suite in "${suites[@]}"; do
    local total="${SUITE_TASKS[${suite}]}"
    GPU_ID="${EVAL_GPU_ID}" tasks_per_gpu="${TASKS_PER_GPU}" your_ckpt="${ckpt}" \
      output_dir="${model_out}" SAVE_VIDEOS="${SAVE_VIDEOS}" \
      bash "${SCRIPT_DIR}/eval_libero_in_one.sh" "${suite}" 0 "${total}" \
      >"${model_out}/${suite}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if (( failed == 0 )); then
    "${LIBERO_PYTHON}" "${STARVLA_DIR}/examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py" \
      --root_path "${model_out}"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${model_name} (status=${failed})."
  return "${failed}"
}

baseline_status=0
spatial_status=0
mem_status=0
case "${MODEL_SELECTION}" in
  baseline)
    launch_model baseline "${BASELINE_CKPT}" || baseline_status=$?
    ;;
  spatial)
    launch_model spatial "${SPATIAL_CKPT}" || spatial_status=$?
    ;;
  mem)
    launch_model mem "${MEM_CKPT}" || mem_status=$?
    ;;
  both)
    launch_model baseline "${BASELINE_CKPT}" || baseline_status=$?
    launch_model spatial "${SPATIAL_CKPT}" || spatial_status=$?
    ;;
  all)
    launch_model baseline "${BASELINE_CKPT}" || baseline_status=$?
    launch_model spatial "${SPATIAL_CKPT}" || spatial_status=$?
    launch_model mem "${MEM_CKPT}" || mem_status=$?
    ;;
esac

if (( baseline_status != 0 || spatial_status != 0 || mem_status != 0 )); then
  echo "Evaluation failed: baseline=${baseline_status}, spatial=${spatial_status}, mem=${mem_status}. See ${RESULT_ROOT}." >&2
  exit 1
fi
echo "LIBERO-plus evaluation completed (${MODEL_SELECTION}). Results: ${RESULT_ROOT}"

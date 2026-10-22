#!/usr/bin/env bash
# Evaluate several checkpoints of one run on standard LIBERO, one after another on one GPU.
#
# Protocol matches run_spatial_mem_full_train_then_eval.sh: 4 suites x 50 trials,
# seed 7, 8 simulator workers, inference batch 4, unnorm_key franka.
#
#   RUN_NAME   run directory name under playground/Checkpoints (required)
#   STEPS      space-separated steps (default "10000 20000 30000 40000 50000")
#   GPU_ID     default 0
#   WAIT_FOR_PIPELINE=1 blocks until the run's train-then-eval pipeline releases
#              logs/pipeline.lock, so the sweep never shares the GPU with it.
#   REUSE_PIPELINE_RESULTS=1 reuses a finished pipeline evaluation of the same
#              checkpoint (same protocol) instead of running it again.
#
# Usage (detached):
#   RUN_NAME=qwen35_0p8b_spatial_mem_oft_v3_long2x_libero_50k setsid nohup \
#     bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_checkpoint_sweep.sh \
#     > results/logs/v3_long2x_sweep.log 2>&1 &
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
STEPS="${STEPS:-10000 20000 30000 40000 50000}"
GPU_ID="${GPU_ID:-0}"
WAIT_FOR_PIPELINE="${WAIT_FOR_PIPELINE:-1}"
REUSE_PIPELINE_RESULTS="${REUSE_PIPELINE_RESULTS:-1}"
# Distinct from the default 6694 used by the pipeline and other eval runs.
SWEEP_PORT="${SWEEP_PORT:-6794}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

RUN_DIR="${STARVLA_DIR}/playground/Checkpoints/${RUN_NAME}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${STARVLA_DIR}/results/${RUN_NAME}/checkpoint_sweep_${STAMP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

cd "${STARVLA_DIR}"
mkdir -p "${SWEEP_ROOT}"
[[ -d "${RUN_DIR}" ]] || fail "run directory not found: ${RUN_DIR}"

if [[ "${WAIT_FOR_PIPELINE}" == 1 && -e "${RUN_DIR}/logs/pipeline.lock" ]]; then
  log "Waiting for the training/eval pipeline of ${RUN_NAME} to release its lock..."
  exec 8<"${RUN_DIR}/logs/pipeline.lock"
  flock 8
  flock -u 8
  log "Pipeline finished: $(tr '\n' ' ' < "${RUN_DIR}/logs/pipeline.status" 2>/dev/null || true)"
fi

export HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}" HF_HUB_CACHE="${HF_HOME:-/data1/jiaozx/.cache/huggingface}/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader >/dev/null || fail "GPU ${GPU_ID} unavailable"
CUDA_VISIBLE_DEVICES="${GPU_ID}" MUJOCO_EGL_DEVICE_ID="${GPU_ID}" "${PYTHON}" \
  examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py || fail "NVIDIA EGL check failed on GPU ${GPU_ID}"

: > "${SWEEP_ROOT}/result_dirs.tsv"
failed=0
for step in ${STEPS}; do
  ckpt="${RUN_DIR}/checkpoints/steps_${step}_pytorch_model.pt"
  if [[ ! -s "${ckpt}" ]]; then
    log "Missing ${ckpt}; skipping step ${step}."
    failed=1
    continue
  fi
  if [[ "${REUSE_PIPELINE_RESULTS}" == 1 && "${step}" == 50000 ]]; then
    previous="$(ls -dt "${STARVLA_DIR}"/results/libero_eval/"${RUN_NAME}"_libero_std_* 2>/dev/null | head -1 || true)"
    if [[ -n "${previous}" ]] && ls "${previous}"/libero_{spatial,object,goal,10}/summary.json >/dev/null 2>&1; then
      log "Step ${step}: reusing pipeline evaluation ${previous}"
      printf '%s\t%s\n' "${step}" "${previous}" >> "${SWEEP_ROOT}/result_dirs.tsv"
      continue
    fi
  fi
  run_id="${RUN_NAME}_step${step}_libero_std_${STAMP}"
  log "Step ${step}: standard LIBERO on GPU ${GPU_ID} -> results/libero_eval/${run_id}"
  if (
    export LIBERO_HOME="${STARVLA_DIR}/playground/LIBERO" LIBERO_CONFIG_PATH="${STARVLA_DIR}/playground/LIBERO/libero"
    export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
    PORT="${SWEEP_PORT}" GPU_ID="${GPU_ID}" PYTHON="${PYTHON}" CONDA_ENV="" SIM_WORKERS=8 INFERENCE_BATCH_SIZE=4 \
    NUM_TRIALS_PER_TASK=50 CKPT="${ckpt}" EXPECTED_CKPT_BYTES=auto WAIT_FOR_CHECKPOINT=0 \
    RUN_ID="${run_id}" UNNORM_KEY=franka TASK_SUITES_CSV=libero_spatial,libero_object,libero_goal,libero_10 \
    SAVE_VIDEOS=0 LIBERO_RECORD_INFERENCE_TIMING=1 \
      bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh
  ) > "${SWEEP_ROOT}/step${step}.log" 2>&1; then
    printf '%s\t%s\n' "${step}" "${STARVLA_DIR}/results/libero_eval/${run_id}" >> "${SWEEP_ROOT}/result_dirs.tsv"
    log "Step ${step} done."
  else
    log "Step ${step} evaluation failed; see ${SWEEP_ROOT}/step${step}.log"
    failed=1
  fi
done

"${PYTHON}" - "${SWEEP_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
rows = {}
for line in (root / "result_dirs.tsv").read_text().splitlines():
    step, path = line.split("\t")
    row = {}
    for suite in suites:
        summary = Path(path) / suite / "summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text())
            row[suite] = {k: data[k] for k in ("success_rate", "successes", "episodes")}
    if len(row) == 4:
        row["average"] = sum(row[s]["success_rate"] for s in suites) / 4
    row["result_dir"] = path
    rows[int(step)] = row
(root / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = ["| Step | Spatial | Object | Goal | Long | Average |", "|---:|---:|---:|---:|---:|---:|"]
for step in sorted(rows):
    row = rows[step]
    cells = [f"{row[s]['success_rate'] * 100:.1f}" if s in row else "-" for s in suites]
    avg = f"{row['average'] * 100:.2f}" if "average" in row else "-"
    lines.append(f"| {step} | " + " | ".join(cells) + f" | {avg} |")
(root / "sweep_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

log "Sweep summary: ${SWEEP_ROOT}/sweep_summary.md"
exit "${failed}"

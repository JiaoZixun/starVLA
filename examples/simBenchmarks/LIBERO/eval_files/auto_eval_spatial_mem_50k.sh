#!/usr/bin/env bash
# Auto-evaluation for Qwen3.5-0.8B Spatial-Mem OFT after 50k steps training.
# Evaluates standard LIBERO (4 suites) then LIBERO-plus (3 suites) sequentially on GPU 0.
# Monitors checkpoint creation and starts evaluation automatically.
#
# Usage:
#   bash examples/simBenchmarks/LIBERO/eval_files/auto_eval_spatial_mem_50k.sh
#
# Environment variables:
#   GPU_ID               — CUDA device index (default: 0)
#   POLL_INTERVAL        — seconds between checkpoint checks (default: 60)
#   TIMEOUT_HOURS        — max hours to wait for checkpoint (default: 72)
#   NUM_TRIALS_LIBERO    — trials per task for standard LIBERO (default: 50)
#   NUM_TRIALS_PLUS      — trials per task for LIBERO-plus (default: 1)
#   TIMING_SAMPLE_SIZE   — episodes to sample for inference speed (default: 1000)
#   SIM_WORKERS          — parallel simulation workers (default: 8)
#   INFERENCE_BATCH      — inference batch size (default: 4)
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python || true)}"
GPU_ID="${GPU_ID:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-72}"
NUM_TRIALS_LIBERO="${NUM_TRIALS_LIBERO:-50}"
NUM_TRIALS_PLUS="${NUM_TRIALS_PLUS:-1}"
TIMING_SAMPLE_SIZE="${TIMING_SAMPLE_SIZE:-1000}"
SIM_WORKERS="${SIM_WORKERS:-8}"
INFERENCE_BATCH="${INFERENCE_BATCH:-4}"

RUN_ID="qwen35_0p8b_spatial_mem_oft_libero_50k"
CKPT_DIR="${STARVLA_DIR}/playground/Checkpoints/${RUN_ID}/checkpoints"
CKPT_PATH="${CKPT_DIR}/steps_50000_pytorch_model.pt"
EXPECTED_CKPT_BYTES="${EXPECTED_CKPT_BYTES:-}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/playground/LIBERO}"
LIBERO_PLUS_HOME="${LIBERO_PLUS_HOME:-${STARVLA_DIR}/playground/LIBERO-plus}"
QWEN35_REVISION="${QWEN35_REVISION:-2fc06364715b967f1860aea9cf38778875588b17}"
HF_HOME="${HF_HOME:-/data1/jiaozx/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
QWEN35_MODEL_DIR="${QWEN35_MODEL_DIR:-${HF_HUB_CACHE}/models--Qwen--Qwen3.5-0.8B/snapshots/${QWEN35_REVISION}}"
RESULTS_ROOT="${STARVLA_DIR}/results/${RUN_ID}/auto_eval_$(date +%Y%m%d_%H%M%S)"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

fail() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[[ -x "${PYTHON}" ]] || fail "Activate a StarVLA environment or set PYTHON=/path/to/python."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required."
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || fail "GPU_ID must be a numeric NVIDIA index."
nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1 || \
    fail "GPU ${GPU_ID} is unavailable."
[[ -d "${LIBERO_HOME}/libero" ]] || fail "LIBERO source tree not found: ${LIBERO_HOME}"
[[ -d "${LIBERO_PLUS_HOME}/libero" ]] || fail "LIBERO-plus source tree not found: ${LIBERO_PLUS_HOME}"
[[ -r "${NVIDIA_EGL_VENDOR_JSON}" ]] || fail "NVIDIA EGL vendor file not readable: ${NVIDIA_EGL_VENDOR_JSON}"
[[ -d "${QWEN35_MODEL_DIR}" ]] || fail "Local Qwen3.5 snapshot missing: ${QWEN35_MODEL_DIR}"

cd "${STARVLA_DIR}"

# Step 1: Wait for checkpoint
log "Monitoring checkpoint: ${CKPT_PATH}"
log "Will wait up to ${TIMEOUT_HOURS} hours, checking every ${POLL_INTERVAL}s"
TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))
elapsed=0

while [[ ! -f "${CKPT_PATH}" || ! -s "${CKPT_PATH}" ]]; do
    if (( elapsed >= TIMEOUT_SECONDS )); then
        fail "Timeout after ${TIMEOUT_HOURS} hours waiting for checkpoint."
    fi
    if (( elapsed % 600 == 0 )); then
        log "Still waiting for checkpoint (elapsed: $((elapsed / 60)) minutes)..."
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [[ -n "${EXPECTED_CKPT_BYTES}" ]]; then
    actual_bytes="$(stat -c '%s' "${CKPT_PATH}")"
    [[ "${actual_bytes}" == "${EXPECTED_CKPT_BYTES}" ]] || \
        log "WARNING: Checkpoint size ${actual_bytes} != expected ${EXPECTED_CKPT_BYTES}"
fi

log "Checkpoint ready: ${CKPT_PATH}"
log "Waiting 30s for any final writes..."
sleep 30

# Step 2: Setup environment
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONWARNINGS="ignore"
export HF_HOME HF_HUB_CACHE QWEN35_MODEL_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export LIBERO_RECORD_INFERENCE_TIMING=1

mkdir -p "${RESULTS_ROOT}"
EVAL_LOG="${RESULTS_ROOT}/eval_master.log"
exec > >(tee -a "${EVAL_LOG}") 2>&1

log "=== Auto-Evaluation Started ==="
log "GPU: ${GPU_ID}"
log "Checkpoint: ${CKPT_PATH}"
log "Results: ${RESULTS_ROOT}"

# Validate EGL
"${PYTHON}" examples/simBenchmarks/LIBERO/eval_files/check_nvidia_egl.py || \
    fail "NVIDIA EGL validation failed."

# Step 3: Evaluate standard LIBERO (4 suites, 50 trials each)
log "=== Evaluating Standard LIBERO (4 suites, ${NUM_TRIALS_LIBERO} trials/task) ==="

TASK_SUITES_CSV="libero_spatial,libero_object,libero_goal,libero_10"
LIBERO_RESULTS="${RESULTS_ROOT}/libero_standard"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

GPU_ID="${GPU_ID}" \
  SIM_WORKERS="${SIM_WORKERS}" \
  INFERENCE_BATCH_SIZE="${INFERENCE_BATCH}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_LIBERO}" \
  CKPT="${CKPT_PATH}" \
  RUN_ID="spatial_mem_50k_libero_std" \
  TASK_SUITES_CSV="${TASK_SUITES_CSV}" \
  SAVE_VIDEOS=0 \
  LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero" \
  RUN_ROOT="${LIBERO_RESULTS}" \
  UNNORM_KEY="franka" \
  EXPECTED_CKPT_BYTES="auto" \
  WAIT_FOR_CHECKPOINT=0 \
  bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh

log "Standard LIBERO evaluation complete. Results: ${LIBERO_RESULTS}"

# Step 4: Evaluate LIBERO-plus (3 suites, 1 trial each for speed)
log "=== Evaluating LIBERO-plus (3 suites, ${NUM_TRIALS_PLUS} trial/task) ==="

PLUS_RESULTS="${RESULTS_ROOT}/libero_plus"
mkdir -p "${PLUS_RESULTS}"
export LIBERO_HOME="${LIBERO_PLUS_HOME}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_HOME}/libero"

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["LIBERO_HOME"]) / "libero" / "libero"
config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
config_path.parent.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
PY
fi

PLUS_SUITES=(libero_spatial libero_object libero_goal)
PLUS_TASK_RANGES=(0:10 0:10 0:10)

for idx in "${!PLUS_SUITES[@]}"; do
    suite="${PLUS_SUITES[$idx]}"
    range="${PLUS_TASK_RANGES[$idx]}"
    start_idx="${range%:*}"
    end_idx="${range#*:}"

    log "Evaluating LIBERO-plus ${suite} (tasks ${start_idx}-${end_idx}, ${NUM_TRIALS_PLUS} trial/task)"

    GPU_ID="${GPU_ID}" \
      LIBERO_HOME="${LIBERO_PLUS_HOME}" \
      LIBERO_PYTHON="${PYTHON}" \
      your_ckpt="${CKPT_PATH}" \
      output_dir="${PLUS_RESULTS}" \
      tasks_per_gpu="${SIM_WORKERS}" \
      SAVE_VIDEOS=0 \
      bash examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval/eval_libero_in_one.sh \
        "${suite}" "${start_idx}" "${end_idx}" || \
      log "WARNING: LIBERO-plus ${suite} evaluation had errors"
done

log "LIBERO-plus evaluation complete. Results: ${PLUS_RESULTS}"

# Step 5: Compute inference speed statistics
log "=== Computing Inference Speed Statistics ==="

"${PYTHON}" - <<PY
import json
import sys
from pathlib import Path
import numpy as np

results_root = Path("${RESULTS_ROOT}")
timing_files = list(results_root.rglob("*timing*.json"))

if not timing_files:
    print("No timing data found; inference timing was not recorded.", file=sys.stderr)
    sys.exit(0)

all_latencies = []
for tf in timing_files:
    try:
        data = json.loads(tf.read_text())
        if isinstance(data, dict) and "latencies_ms" in data:
            all_latencies.extend(data["latencies_ms"])
        elif isinstance(data, list):
            all_latencies.extend([x for x in data if isinstance(x, (int, float))])
    except Exception as e:
        print(f"Skip {tf}: {e}", file=sys.stderr)

if not all_latencies:
    print("No valid latency samples found.", file=sys.stderr)
    sys.exit(0)

sample_size = min(${TIMING_SAMPLE_SIZE}, len(all_latencies))
sample = np.random.choice(all_latencies, size=sample_size, replace=False)
latencies = np.array(sample, dtype=np.float64)

stats = {
    "sample_size": int(sample_size),
    "total_episodes": len(all_latencies),
    "mean_ms": float(np.mean(latencies)),
    "median_ms": float(np.median(latencies)),
    "p95_ms": float(np.percentile(latencies, 95)),
    "p99_ms": float(np.percentile(latencies, 99)),
    "std_ms": float(np.std(latencies)),
    "min_ms": float(np.min(latencies)),
    "max_ms": float(np.max(latencies)),
    "actions_per_sec": float(1000.0 / np.mean(latencies)),
}

timing_report = results_root / "inference_timing_summary.json"
timing_report.write_text(json.dumps(stats, indent=2))
print(f"Inference timing summary (n={sample_size}):")
print(f"  Mean: {stats['mean_ms']:.2f} ms ({stats['actions_per_sec']:.2f} act/s)")
print(f"  Median: {stats['median_ms']:.2f} ms")
print(f"  P95: {stats['p95_ms']:.2f} ms")
print(f"  P99: {stats['p99_ms']:.2f} ms")
print(f"Saved: {timing_report}")
PY

# Step 6: Generate summary
log "=== Generating Evaluation Summary ==="

"${PYTHON}" - <<PY
import json
from pathlib import Path

results_root = Path("${RESULTS_ROOT}")
summary = {"checkpoint": "${CKPT_PATH}", "timestamp": "$(date -Iseconds)", "results": {}}

# Standard LIBERO
libero_std = results_root / "libero_standard" / "spatial_mem_50k_libero_std"
if libero_std.exists():
    for suite_dir in libero_std.iterdir():
        if suite_dir.is_dir():
            summary_json = suite_dir / "summary.json"
            if summary_json.exists():
                suite_data = json.loads(summary_json.read_text())
                summary["results"][f"libero_{suite_dir.name}"] = suite_data

# LIBERO-plus
plus_root = results_root / "libero_plus"
if plus_root.exists():
    for result_json in plus_root.rglob("*_results.json"):
        try:
            plus_data = json.loads(result_json.read_text())
            suite_name = result_json.parent.name if result_json.parent.name.startswith("libero") else "libero_plus"
            summary["results"][f"plus_{suite_name}"] = plus_data
        except:
            pass

# Inference timing
timing_file = results_root / "inference_timing_summary.json"
if timing_file.exists():
    summary["inference_timing"] = json.loads(timing_file.read_text())

summary_path = results_root / "evaluation_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Evaluation summary: {summary_path}")
PY

log "=== Auto-Evaluation Complete ==="
log "All results saved to: ${RESULTS_ROOT}"
log "Summary: ${RESULTS_ROOT}/evaluation_summary.json"
log "Inference timing: ${RESULTS_ROOT}/inference_timing_summary.json"

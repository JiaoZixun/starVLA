#!/usr/bin/env bash
# Exp C (object grounding, P0+P1+P2) on GPU 0: 30K steps, batch 32, checkpoint every 10K, then
# standard LIBERO at 10K/20K/30K with eval_libero_checkpoint_sweep.sh (NVIDIA EGL, 8 workers,
# batch 4, seed 7) and read-only failure tracing (LIBERO_FAILURE_TRACE=1).
# Comparison: existing Bridge-Spatial only checkpoints (no retraining).
# Usage: setsid nohup bash examples/human2robots/DexYCBSpatial/run_grounding_experiment.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON=/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python
CFG=examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_grounding.yaml
GPU=${GPU:-0}
PORT=${PORT:-6994}
LOG_DIR=logs/bridge_spatial_grounding_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/status.txt"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" >> "$STATUS"; }
export STARVLA_DISABLE_DEEPSPEED=1

for sidecar in playground/DerivedData/libero_grounding_v1/libero_10_no_noops_1.0.0_lerobot/manifest.json \
               playground/DerivedData/dexycb_grounding_v3/metadata.json; do
  [[ -s "$sidecar" ]] || { log "Missing $sidecar"; exit 1; }
done
run_id=$(grep -m1 '^run_id:' "$CFG" | awk '{print $2}')
if compgen -G "playground/Checkpoints/$run_id/checkpoints/steps_*" >/dev/null; then
  log "Refusing to start: playground/Checkpoints/$run_id already has checkpoints"; exit 1
fi

rc=0
log "training $run_id on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m starVLA.training.train_starvla --config_yaml "$CFG" > "$LOG_DIR/train.log" 2>&1 || rc=$?
log "train exit=$rc"
[[ $rc -eq 0 ]] || exit 1
log "eval sweep 10K-30K on GPU $GPU (failure tracing on)"
LIBERO_FAILURE_TRACE=1 RUN_NAME=$run_id GPU_ID=$GPU SWEEP_PORT=$PORT STEPS="10000 20000 30000" REUSE_PIPELINE_RESULTS=0 \
  PYTHON="$PYTHON" bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_checkpoint_sweep.sh > "$LOG_DIR/eval.log" 2>&1 || rc=$?
log "eval exit=$rc; summaries under results/$run_id/checkpoint_sweep_*"

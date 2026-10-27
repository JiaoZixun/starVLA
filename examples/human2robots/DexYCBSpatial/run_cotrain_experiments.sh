#!/usr/bin/env bash
# Exp A (co-training, plan 1+2) on GPU 0 and Exp B (+ phase/object supervision, plan 3) on GPU 1,
# in parallel. Each GPU trains its run for 50K steps, then evaluates steps 10K-50K on standard
# LIBERO with eval_libero_checkpoint_sweep.sh (NVIDIA EGL, 8 workers, batch 4, seed 7).
# See examples/human2robots/DexYCBSpatial/PLAN_cotrain_phase.md.
# Usage: setsid nohup bash examples/human2robots/DexYCBSpatial/run_cotrain_experiments.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON=/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python
LOG_DIR=logs/bridge_spatial_cotrain_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG_DIR"
export STARVLA_DISABLE_DEEPSPEED=1
CFG_DIR=examples/simBenchmarks/LIBERO/train_files
SWEEP=examples/simBenchmarks/LIBERO/eval_files/eval_libero_checkpoint_sweep.sh
SIDECAR=playground/DerivedData/dexycb_cotrain_v2
STATUS="$LOG_DIR/status.txt"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" >> "$STATUS"; }

[[ -s "$SIDECAR/train/future_phase.npy" ]] || { log "Missing v2 sidecar $SIDECAR; run prepare_dexycb_cotrain.py"; exit 1; }
for run_id in qwen35_0p8b_bridge_spatial_cotrain_libero_50k qwen35_0p8b_bridge_spatial_cotrain_phase_libero_50k; do
  if compgen -G "playground/Checkpoints/$run_id/checkpoints/steps_*" >/dev/null; then
    log "Refusing to start: playground/Checkpoints/$run_id already has checkpoints"; exit 1
  fi
done

run() {  # name config gpu port
  local name=$1 cfg=$2 gpu=$3 port=$4 run_id rc=0
  run_id=$(grep -m1 '^run_id:' "$cfg" | awk '{print $2}')
  log "$name: training $run_id on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m starVLA.training.train_starvla --config_yaml "$cfg" \
    > "$LOG_DIR/train_$name.log" 2>&1 || rc=$?
  log "$name: train exit=$rc"
  [[ $rc -eq 0 ]] || return 1
  log "$name: eval sweep 10K-50K on GPU $gpu"
  RUN_NAME=$run_id GPU_ID=$gpu SWEEP_PORT=$port STEPS="10000 20000 30000 40000 50000" REUSE_PIPELINE_RESULTS=0 \
    PYTHON="$PYTHON" bash "$SWEEP" > "$LOG_DIR/eval_$name.log" 2>&1 || rc=$?
  log "$name: eval exit=$rc"
  return $rc
}

( run A "$CFG_DIR/qwen35_0p8b_bridge_spatial_cotrain.yaml" 0 6794 || true ) &
sleep 5  # distinct sweep timestamps
( run B "$CFG_DIR/qwen35_0p8b_bridge_spatial_cotrain_phase.yaml" 1 6894 || true ) &
wait
log "Done; summaries under results/<run_id>/checkpoint_sweep_*"

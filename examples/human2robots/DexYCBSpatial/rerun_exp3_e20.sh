#!/usr/bin/env bash
# Re-run Exp 3 (Bridge-Spatial on LIBERO) from the epoch-20 DexYCB checkpoint, then
# evaluate standard LIBERO for steps 50k first, then 10k-40k.
#   WAIT_PIDS  space-separated PIDs to wait for before training (e.g. running sweeps)
# Training uses GPU 0; the eval sweep is split across GPU 0 and GPU 1.
# Usage: WAIT_PIDS="..." setsid nohup bash examples/human2robots/DexYCBSpatial/rerun_exp3_e20.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON=/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python
LOG_DIR=logs/bridge_spatial_exp3_e20_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG_DIR"
export STARVLA_DISABLE_DEEPSPEED=1
DEXYCB_CFG=examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_dexycb.yaml
PRETRAIN_DIR=playground/Checkpoints/qwen35_0p8b_bridge_dexycb_pretrain_e20
RUN_ID=qwen35_0p8b_bridge_spatial_dexycb_e20_libero_50k
SWEEP=examples/simBenchmarks/LIBERO/eval_files/eval_libero_checkpoint_sweep.sh
STATUS="$LOG_DIR/status.txt"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" >> "$STATUS"; }

if compgen -G "playground/Checkpoints/$RUN_ID/checkpoints/steps_*" >/dev/null; then
  log "Refusing to start: playground/Checkpoints/$RUN_ID already has checkpoints"; exit 1
fi
last_step=$(grep -o '"steps": [0-9]*' "$PRETRAIN_DIR/summary.jsonl" | tail -1 | grep -o '[0-9]*$')
final="$PRETRAIN_DIR/checkpoints/steps_${last_step}_pytorch_model.pt"
[[ "$last_step" == 30660 && -s "$final" ]] || { log "Unexpected pretrain checkpoint: $final"; exit 1; }

for pid in ${WAIT_PIDS:-}; do
  log "Waiting for PID $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done

log "Training $RUN_ID on GPU 0 from $final"
rc=0
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m starVLA.training.train_starvla --config_yaml "$DEXYCB_CFG" \
  --run_id "$RUN_ID" --trainer.pretrained_checkpoint "$final" > "$LOG_DIR/train.log" 2>&1 || rc=$?
log "train exit=$rc"
[[ $rc -eq 0 ]] || exit 1

evaluate() {  # gpu port steps log
  RUN_NAME=$RUN_ID GPU_ID=$1 SWEEP_PORT=$2 STEPS="$3" REUSE_PIPELINE_RESULTS=0 PYTHON="$PYTHON" \
    bash "$SWEEP" > "$4" 2>&1
}
log "Eval sweep: GPU 0 steps 50000 10000 20000, GPU 1 steps 30000 40000"
( rc=0; evaluate 0 6794 "50000 10000 20000" "$LOG_DIR/eval_gpu0.log" || rc=$?; log "eval gpu0 exit=$rc" ) &
sleep 5  # distinct sweep timestamps
( rc=0; evaluate 1 6894 "30000 40000" "$LOG_DIR/eval_gpu1.log" || rc=$?; log "eval gpu1 exit=$rc" ) &
wait
log "Done; summaries under results/$RUN_ID/checkpoint_sweep_*"

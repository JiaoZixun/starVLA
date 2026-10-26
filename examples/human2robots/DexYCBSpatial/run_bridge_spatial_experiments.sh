#!/usr/bin/env bash
# Three-experiment pipeline, one GPU per chain:
#   GPU 1: Exp 2  Bridge-Spatial on LIBERO from raw Qwen (50k steps)
#   GPU 0: Exp 1  DexYCB pretraining (20 epochs)  ->  Exp 3  Bridge-Spatial on LIBERO
#          loading qwen_vl_interface + waypoint_model from the epoch-20 checkpoint (50k steps)
# After Exp 2 / Exp 3 train successfully, each runs a standard LIBERO checkpoint sweep
# (10k-50k) on its own GPU via eval_libero_checkpoint_sweep.sh.
# Usage: bash examples/human2robots/DexYCBSpatial/run_bridge_spatial_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON=/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python
LOG_DIR=logs/bridge_spatial_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG_DIR"
export STARVLA_DISABLE_DEEPSPEED=1
PRETRAIN_CFG=examples/human2robots/DexYCBSpatial/train_files/qwen35_0p8b_bridge_dexycb_pretrain.yaml
ONLY_CFG=examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_only.yaml
DEXYCB_CFG=examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_bridge_spatial_dexycb.yaml
PRETRAIN_DIR=playground/Checkpoints/qwen35_0p8b_bridge_dexycb_pretrain_e20
SWEEP=examples/simBenchmarks/LIBERO/eval_files/eval_libero_checkpoint_sweep.sh

for run in qwen35_0p8b_bridge_dexycb_pretrain_e20 qwen35_0p8b_bridge_spatial_only_libero_50k qwen35_0p8b_bridge_spatial_dexycb_libero_50k; do
  if compgen -G "playground/Checkpoints/$run/checkpoints/steps_*" >/dev/null; then
    echo "Refusing to start: playground/Checkpoints/$run already has checkpoints" >&2; exit 1
  fi
done

train() {  # gpu config log [extra dotlist args...]
  local gpu=$1 cfg=$2 log=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m starVLA.training.train_starvla --config_yaml "$cfg" "$@" > "$log" 2>&1
}

evaluate() {  # gpu port run_name log
  RUN_NAME=$3 GPU_ID=$1 SWEEP_PORT=$2 REUSE_PIPELINE_RESULTS=0 PYTHON="$PYTHON" bash "$SWEEP" > "$4" 2>&1
}

# GPU 1: Exp 2.
(
  rc=0; train 1 "$ONLY_CFG" "$LOG_DIR/exp2_spatial_only.log" || rc=$?
  echo "exp2 exit=$rc" >> "$LOG_DIR/status.txt"
  [[ $rc -eq 0 ]] || exit 1
  rc=0; evaluate 1 6894 qwen35_0p8b_bridge_spatial_only_libero_50k "$LOG_DIR/exp2_eval_sweep.log" || rc=$?
  echo "exp2 eval exit=$rc" >> "$LOG_DIR/status.txt"
) &

# GPU 0: Exp 1 then Exp 3.
(
  rc=0; train 0 "$PRETRAIN_CFG" "$LOG_DIR/exp1_dexycb_pretrain.log" || rc=$?
  echo "exp1 exit=$rc" >> "$LOG_DIR/status.txt"
  [[ $rc -eq 0 ]] || exit 1
  saves=$(grep -c '"steps"' "$PRETRAIN_DIR/summary.jsonl")
  if [[ "$saves" -ne 4 ]]; then
    echo "exp3 not started: expected 4 epoch checkpoints (5/10/15/20), found $saves" >> "$LOG_DIR/status.txt"; exit 1
  fi
  # Epoch 20 is the last save; take its step from summary.jsonl rather than sorting paths.
  last_step=$(grep -o '"steps": [0-9]*' "$PRETRAIN_DIR/summary.jsonl" | tail -1 | grep -o '[0-9]*$')
  final="$PRETRAIN_DIR/checkpoints/steps_${last_step}_pytorch_model.pt"
  if [[ ! -s "$final" ]]; then
    echo "exp3 not started: missing epoch-20 checkpoint $final" >> "$LOG_DIR/status.txt"; exit 1
  fi
  echo "exp3 loads $final" >> "$LOG_DIR/status.txt"
  rc=0; train 0 "$DEXYCB_CFG" "$LOG_DIR/exp3_dexycb_spatial.log" --trainer.pretrained_checkpoint "$final" || rc=$?
  echo "exp3 exit=$rc" >> "$LOG_DIR/status.txt"
  [[ $rc -eq 0 ]] || exit 1
  rc=0; evaluate 0 6794 qwen35_0p8b_bridge_spatial_dexycb_libero_50k "$LOG_DIR/exp3_eval_sweep.log" || rc=$?
  echo "exp3 eval exit=$rc" >> "$LOG_DIR/status.txt"
) &

echo "logs: $LOG_DIR"
wait

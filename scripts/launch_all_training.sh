#!/bin/bash
# Master Launch Script - Starts all training pipelines
# GPU 0: DexYCB Pretrain (10h) -> LIBERO Pretrained (13h)
# GPU 1: LIBERO Spatial-only (13h)
# Total wall time: ~23 hours

cd /data1/jiaozx/starVLA

echo "========================================"
echo "   StarVLA Spatial Ablation Training   "
echo "========================================"
echo "Launch time: $(date)"
echo ""
echo "Training Plan:"
echo "  [GPU 0] DexYCB Pretrain 20e (~10h)"
echo "  [GPU 0] -> LIBERO + Pretrained (~13h)"
echo "  [GPU 1] LIBERO Spatial-only (~13h)"
echo ""
echo "Expected completion: $(date -d '+23 hours')"
echo "========================================"
echo ""

# Check preprocessing status
echo "Checking DexYCB preprocessing..."
if [ ! -f data/preprocessed_dexycb/train_current_pose.npy ]; then
    echo "WARNING: DexYCB preprocessing not complete yet!"
    echo "Waiting for preprocessing to finish..."

    while [ ! -f data/preprocessed_dexycb/train_current_pose.npy ]; do
        sleep 30
        tail -2 logs/dexycb_quick_preprocess.log 2>/dev/null || echo "Still preprocessing..."
    done

    echo "✓ Preprocessing complete!"
fi

echo ""
echo "Starting training jobs..."

# GPU 0: DexYCB Pretraining
echo "[GPU 0] Launching DexYCB pretraining..."
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_dexycb_pretrain_gpu0.sh &
PID_GPU0=$!
echo "  PID: $PID_GPU0"
echo "  Log: logs/dexycb_pretrain_20e.log"

sleep 5

# GPU 1: LIBERO Spatial-only
echo "[GPU 1] Launching LIBERO spatial-only training..."
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/train_libero_spatial_only_gpu1.sh &
PID_GPU1=$!
echo "  PID: $PID_GPU1"
echo "  Log: logs/libero_spatial_only.log"

echo ""
echo "========================================"
echo "All jobs launched!"
echo ""
echo "Monitor with:"
echo "  tail -f logs/dexycb_pretrain_20e.log"
echo "  tail -f logs/libero_spatial_only.log"
echo "  tail -f logs/libero_pretrained.log  # (starts after DexYCB)"
echo ""
echo "Check status:"
echo "  nvidia-smi"
echo "  ps aux | grep train"
echo "========================================"

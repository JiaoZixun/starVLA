#!/bin/bash
# Emergency Fast Launch - LIBERO Training Only
# Focus on core task: compare Spatial-only vs current Spatial+Mem

cd /data1/jiaozx/starVLA

echo "========================================"
echo "   Fast Launch: LIBERO Training Only   "
echo "========================================"
echo "Launch time: $(date)"
echo ""
echo "Training Plan:"
echo "  [GPU 0] LIBERO Spatial-only (~13h)"
echo "  [GPU 1] LIBERO Spatial+Mem baseline (~13h)"
echo ""
echo "Expected completion: $(date -d '+13 hours')"
echo "========================================"
echo ""

# Check LIBERO data
if [ ! -d playground/Datasets/LEROBOT_LIBERO_DATA ]; then
    echo "ERROR: LIBERO data not found!"
    exit 1
fi

# Check geometry sidecar
if [ ! -d playground/DerivedData/libero_spatial_oft_v1 ]; then
    echo "ERROR: Geometry sidecar not found!"
    exit 1
fi

echo "✓ Data checks passed"
echo ""
echo "Starting training jobs..."

# GPU 0: LIBERO Spatial-only (NEW)
echo "[GPU 0] Launching LIBERO Spatial-only..."
nohup bash scripts/train_libero_spatial_only_gpu0.sh &
PID_GPU0=$!
echo "  PID: $PID_GPU0"
echo "  Log: logs/libero_spatial_only_gpu0.log"

sleep 5

# GPU 1: LIBERO Spatial+Mem baseline (for comparison)
echo "[GPU 1] Launching LIBERO Spatial+Mem baseline..."
nohup bash scripts/train_libero_spatial_mem_gpu1.sh &
PID_GPU1=$!
echo "  PID: $PID_GPU1"
echo "  Log: logs/libero_spatial_mem_gpu1.log"

echo ""
echo "========================================"
echo "Both jobs launched!"
echo ""
echo "Monitor with:"
echo "  tail -f logs/libero_spatial_only_gpu0.log"
echo "  tail -f logs/libero_spatial_mem_gpu1.log"
echo ""
echo "Check status:"
echo "  nvidia-smi"
echo "  ps aux | grep train_starvla"
echo "========================================"

#!/bin/bash
# LIBERO Spatial-only Training - GPU 1
# Estimated: ~13 hours

export CUDA_VISIBLE_DEVICES=0  # Will be set to GPU 1 by caller
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_EGL_DEVICE_ID=1

cd /data1/jiaozx/starVLA

echo "=== Starting LIBERO Spatial-only Training on GPU 1 ==="
echo "Start time: $(date)"
echo "Expected duration: ~13 hours"
echo "Output: logs/libero_spatial_only.log"

python -m starVLA.training.train_starvla \
  --config examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_only.yaml \
  > logs/libero_spatial_only.log 2>&1

EXIT_CODE=$?

echo "=== LIBERO Spatial-only Training Complete ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed!"
fi

#!/bin/bash
# LIBERO with Pretrained Weights - GPU 0 (auto-triggered after DexYCB)
# Estimated: ~13 hours

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_EGL_DEVICE_ID=0

cd /data1/jiaozx/starVLA

echo "=== Starting LIBERO with Pretrained Weights on GPU 0 ==="
echo "Start time: $(date)"
echo "Expected duration: ~13 hours"
echo "Output: logs/libero_pretrained.log"

python -m starVLA.training.train_starvla \
  --config examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_pretrained_spatial.yaml \
  > logs/libero_pretrained.log 2>&1

EXIT_CODE=$?

echo "=== LIBERO Pretrained Training Complete ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed!"
fi

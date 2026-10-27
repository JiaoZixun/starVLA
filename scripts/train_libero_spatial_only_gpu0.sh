#!/bin/bash
# LIBERO Spatial-only Training - GPU 0

export CUDA_VISIBLE_DEVICES=0
export PYTHON=/data1/jiaozx/anaconda3/envs/fastvla-fla/bin/python
export STARVLA_DISABLE_DEEPSPEED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_EGL_DEVICE_ID=0

cd /data1/jiaozx/starVLA

echo "=== LIBERO Spatial-only Training on GPU 0 ==="
echo "Start: $(date)"

$PYTHON -m starVLA.training.train_starvla \
  --config examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_only.yaml

echo "=== Training Complete: $(date) ==="

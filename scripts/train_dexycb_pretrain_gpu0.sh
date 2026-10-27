#!/bin/bash
# DexYCB Pretraining - GPU 0
# Estimated: ~10 hours

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

cd /data1/jiaozx/starVLA

echo "=== Starting DexYCB Pretraining on GPU 0 ==="
echo "Start time: $(date)"
echo "Expected duration: ~10 hours"
echo "Output: logs/dexycb_pretrain_20e.log"

python examples/human2robots/DexYCBSpatial/train_dexycb_pretrain.py \
  --data_root data/preprocessed_dexycb \
  --output_dir playground/Checkpoints/dexycb_pretrain_20e \
  --epochs 20 \
  --batch_size 32 \
  --lr_qwen 1e-6 \
  --lr_waypoint 5e-5 \
  --warmup_steps 500 \
  --save_every_epochs 5 \
  --num_workers 8 \
  > logs/dexycb_pretrain_20e.log 2>&1

EXIT_CODE=$?

echo "=== DexYCB Pretraining Complete ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

# Automatically trigger next stage
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Auto-starting LIBERO Pretrained Training on GPU 0 ==="
    bash scripts/train_libero_pretrained_gpu0.sh &
else
    echo "ERROR: DexYCB pretraining failed!"
fi

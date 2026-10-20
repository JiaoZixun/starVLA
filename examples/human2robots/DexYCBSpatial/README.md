# DexYCB spatial pretraining

The human loader consumes a sequence manifest and dense geometry arrays without modifying the source DexYCB files. The `sequence_manifest` path is relative to `geometry_sidecar_root` unless absolute. Its JSONL records contain `image` (absolute or relative to `data_root_dir`), `sequence_id`, `frame_index`, and optional `lang`.

The manifest directory must contain these `allow_pickle=False` NumPy arrays, with one row per JSONL record:

| File | dtype | shape |
|---|---|---|
| `current_pose.npy` | float32 | `[N, 2, 4, 4]` |
| `relative_waypoints.npy` | float32 | `[N, 2, 4, 4, 4]` |
| `position_valid.npy` | bool | `[N, 2, 4]` |
| `rotation_valid.npy` | bool | `[N, 2, 4]` |
| `effector_type.npy` | int64 | `[N, 2]`, 0=empty, 1=left, 2=right |

`current_pose` uses aligned camera-frame wrist poses; `relative_waypoints` contains current-wrist-relative future transforms for offsets `[1,2,4,8]`. Invalid slots and targets must contain finite identity transforms. Each sample uses one synchronized camera image. Source sequence split and image/pose timestamp alignment are preprocessing responsibilities and must be recorded in the manifest metadata supplied by that pipeline.

The checked-in H500 manifest contains 31,719 samples. The training config uses `drop_last: false` and batch size 32, so each epoch has 992 batches (the final batch has 7 samples). Set `max_train_epochs: 50`; the trainer iterates the loader exactly 50 times, skips the action evaluator that would consume an extra batch, and computes the scheduler length from the actual loader. Exact exposure is guarded to a single training process.

Run the human stage with one GPU:

```bash
cd /home/data/starVLA
CUDA_VISIBLE_DEVICES=0 \
STARVLA_DISABLE_DEEPSPEED=1 \
WANDB_MODE=disabled \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
/home/data/Libero/envs/libero-baselines/bin/python \
  starVLA/training/train_starvla.py \
  --config_yaml examples/human2robots/DexYCBSpatial/train_files/train_human_spatial.yaml \
  trainer.pretrained_checkpoint=/absolute/path/to/libero_qwen35_0p8b_baseline/checkpoints/steps_50000_pytorch_model.pt
```

The config initializes Qwen and the action head from the fixed LIBERO baseline, while freezing the action head, robot spatial adapter, and robot anchor projection. `eval_interval: 0` disables robot action evaluation; provide an independent geometry validation workflow before selecting a human checkpoint. Replace the checkpoint override with the actual LIBERO baseline checkpoint path before launching.

Before LIBERO training, compose the human checkpoint with the same fixed baseline action head:

```bash
python examples/human2robots/DexYCBSpatial/compose_checkpoint.py \
  --human-checkpoint playground/Checkpoints/<human-run>/checkpoints/steps_5000_pytorch_model.pt \
  --robot-baseline playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt \
  --output playground/Checkpoints/<human-run>/composed_for_libero_pytorch_model.pt
```

Then run `train_files/train_libero_after_human.yaml`, pointing `pretrained_checkpoint` at that output. The LIBERO dataset remains the existing `libero_spatial_oft_all` with its existing pose sidecars and action statistics.

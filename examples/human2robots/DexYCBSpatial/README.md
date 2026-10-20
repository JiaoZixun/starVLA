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

Run the human stage with `train_files/train_human_spatial.yaml`. This config initializes Qwen and the action head from the fixed LIBERO baseline, while freezing the action head, robot spatial adapter, and robot anchor projection. `eval_interval: 0` disables robot action evaluation; provide an independent geometry validation workflow before selecting a human checkpoint.

Before LIBERO training, compose the human checkpoint with the same fixed baseline action head:

```bash
python examples/human2robots/DexYCBSpatial/compose_checkpoint.py \
  --human-checkpoint playground/Checkpoints/<human-run>/checkpoints/steps_5000_pytorch_model.pt \
  --robot-baseline playground/Checkpoints/qwen35_0p8b_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt \
  --output playground/Checkpoints/<human-run>/composed_for_libero_pytorch_model.pt
```

Then run `train_files/train_libero_after_human.yaml`, pointing `pretrained_checkpoint` at that output. The LIBERO dataset remains the existing `libero_spatial_oft_all` with its existing pose sidecars and action statistics.

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

`current_pose` uses aligned camera-frame wrist poses. The output contains three nested sequence collections: H100 has 100 sequences, H500 has 500, and H1000 has all 1,000 sequences. These counts refer to capture sequences; each sequence contributes its valid annotated frames as training samples. Selection is subject-stratified over all 10 subjects, and no train/validation/test split is created. The four `relative_waypoints` slots are indexed by `[1,2,4,8]` and target `[0.05,0.10,0.20,0.40]` seconds after the current image. For nominal 30 FPS DexYCB data, the pipeline linearly interpolates wrist position and uses SO(3) SLERP for orientation when a target falls between frames; interpolation requires valid adjacent labels and never crosses sequence boundaries. Invalid slots and targets contain finite identity transforms. Each sample uses one synchronized main-view camera image. The frame-to-time convention and per-sample source references are recorded in `sample_metadata.jsonl`; source fingerprint, MANO hashes, and geometry audit metrics are in `metadata.json`.

The H500 manifest sample count is recorded in its generated `metadata.json` because it depends on valid source annotations. The training config uses microbatch 8, gradient accumulation 4 (effective batch size 32), and `drop_last: false`; the trainer computes the scheduler length from the actual loader. Set `max_train_epochs: 50`; the trainer iterates the loader exactly 50 times and skips the action evaluator that would consume an extra batch. Checkpoints are written at full epoch boundaries, and epoch-mode recovery resumes only from those boundaries to avoid replaying data. Exact exposure is guarded to a single training process.

Run the human stage with one GPU:

```bash
cd /home/data/starVLA
CUDA_VISIBLE_DEVICES=0 \
STARVLA_DISABLE_DEEPSPEED=1 \
WANDB_MODE=disabled \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
/home/data/Libero/envs/mint-light/bin/python \
  starVLA/training/train_starvla.py \
  --config_yaml examples/human2robots/DexYCBSpatial/train_files/train_human_spatial.yaml
```

The Qwen backbone loads the original 0.8B weights from the local snapshot configured in `framework.qwenvl.base_vlm` with `local_files_only: true`; no Hugging Face Hub access or LIBERO-trained checkpoint is used for human initialization. The action head and spatial modules start from their configured initialization; the unused action head, robot spatial adapter, and robot anchor projection are frozen during human pretraining. `eval_interval: 0` disables robot action evaluation; provide an independent geometry validation workflow before selecting a human checkpoint.

After the 50-epoch run, `train_files/train_libero_after_human.yaml` loads its full checkpoint directly and trains on the existing `libero_spatial_oft_all` dataset with its existing pose sidecars and action statistics. Its action head starts from the human-stage initialization and is trained by LIBERO behavior cloning.

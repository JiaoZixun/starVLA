#!/usr/bin/env python3
"""Write frozen-Qwen pooled visual features for Spatial-Mem (memory.visual_source=qwen).

For every LIBERO frame and both cameras, this runs the vision tower of the
checkpoint whose (frozen) weights the memory model reuses and stores the 4x4
pooled post-merger tokens as fp16 ``[frames, 2, 16, 1024]``.  Only derived
features and a manifest are written; source Parquet/video files are not copied.

Frames go through the same path as training (PyAV decode, PIL bicubic resize to
224x224) and features through the same processor and pooling code as the
model's own forward (``Qwen3_5.encode_pooled_images``), which eval uses online.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.memory_v2 import (  # noqa: E402
    QWEN_VISUAL_FEATURE_DIM,
    QWEN_VISUAL_SIDECAR_FILE,
    QWEN_VISUAL_SIDECAR_FORMAT,
    VISUAL_IMAGE_SIZE,
    VISUAL_POOL_GRID,
)

DATASETS = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
CAMERA_ORDER = ("observation.images.image", "observation.images.wrist_image")


def decode_episode(args: tuple[str, int, int]) -> tuple[int, np.ndarray]:
    """Decode both views of one episode to uint8 [T, 2, 224, 224, 3]."""
    import av
    from PIL import Image

    dataset, episode_id, length = args
    views = []
    for key in CAMERA_ORDER:
        path = Path(dataset) / "videos" / "chunk-000" / key / f"episode_{episode_id:06d}.mp4"
        container = av.open(str(path))
        try:
            frames = [
                # PIL's default bicubic resize, as in LeRobotSingleDataset._pack_sample.
                np.asarray(Image.fromarray(frame.to_ndarray(format="rgb24")).resize(VISUAL_IMAGE_SIZE))
                for frame in container.decode(video=0)
            ]
        finally:
            container.close()
        if len(frames) != length:
            raise ValueError(f"{path}: decoded {len(frames)} frames, expected {length}")
        views.append(np.stack(frames))
    return episode_id, np.stack(views, axis=1)


def load_qwen(checkpoint: Path, local_model_path: str | None):
    """Build only the Qwen interface and load its weights strictly from ``checkpoint``.

    The training run reloads ``qwen_vl_interface`` from the same file, so the
    vision tower here is the one training freezes and eval runs.
    """
    from omegaconf import OmegaConf

    from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
    from starVLA.model.modules.vlm import get_vlm_model

    model_config, _ = read_mode_config(checkpoint)
    model_config = OmegaConf.to_container(OmegaConf.create(model_config), resolve=False)
    if local_model_path:
        model_config["framework"]["qwenvl"]["local_model_path"] = str(local_model_path)
        model_config["framework"]["qwenvl"]["local_files_only"] = True
    qwen = get_vlm_model(dict_to_namespace(model_config))
    state = torch.load(checkpoint, map_location="cpu")
    prefix = "qwen_vl_interface."
    qwen_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not qwen_state:
        raise ValueError(f"{checkpoint} has no {prefix}* weights")
    qwen.load_state_dict(qwen_state, strict=True)
    del state
    return qwen.to("cuda").eval()


def prepare_one(dataset: Path, output_root: Path, memory_root: Path, qwen, fingerprint: str, args) -> dict:
    from PIL import Image

    target = output_root / dataset.name
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite")
    memory_manifest = json.loads((memory_root / dataset.name / "manifest.json").read_text())
    ranges = memory_manifest["episode_index_to_sidecar_range"]
    total = int(memory_manifest["source_total_frames"])
    if list(memory_manifest["memory_visual_contract"]["camera_order"]) != list(CAMERA_ORDER):
        raise ValueError(f"{dataset.name}: memv2 camera order differs from {CAMERA_ORDER}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=output_root))
    try:
        features = np.lib.format.open_memmap(
            staging / QWEN_VISUAL_SIDECAR_FILE,
            mode="w+",
            dtype=np.float16,
            shape=(total, 2, VISUAL_POOL_GRID[0] * VISUAL_POOL_GRID[1], QWEN_VISUAL_FEATURE_DIM),
        )
        written = np.zeros(total, dtype=bool)
        jobs = [(str(dataset), int(episode), int(entry["length"])) for episode, entry in ranges.items()]
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=args.decode_workers) as pool:
            for count, (episode_id, frames) in enumerate(pool.map(decode_episode, jobs, chunksize=1), start=1):
                offset = int(ranges[str(episode_id)]["offset"])
                for begin in range(0, len(frames), args.batch_frames):
                    chunk = frames[begin : begin + args.batch_frames]
                    images = [[Image.fromarray(view) for view in frame] for frame in chunk]
                    pooled = qwen.encode_pooled_images(images, pool_grid=VISUAL_POOL_GRID)
                    rows = slice(offset + begin, offset + begin + len(chunk))
                    features[rows] = pooled.to(dtype=torch.float16).cpu().numpy()
                    written[rows] = True
                if count % 50 == 0 or count == len(jobs):
                    rate = written.sum() / max(time.time() - start_time, 1e-6)
                    print(f"{dataset.name}: {count}/{len(jobs)} episodes, {written.sum()}/{total} frames, {rate:.0f} frames/s", flush=True)
        if not written.all():
            raise RuntimeError(f"{dataset.name}: {int((~written).sum())} frames were not written")
        if not np.isfinite(features[:: max(1, total // 1000)].astype(np.float32)).all():
            raise RuntimeError(f"{dataset.name}: non-finite Qwen features")
        features.flush()
        del features
        manifest = {
            "format": QWEN_VISUAL_SIDECAR_FORMAT,
            "dataset_name": dataset.name,
            "source_dataset_path": memory_manifest["source_dataset_path"],
            "dataset_revision": memory_manifest["dataset_revision"],
            "source_total_frames": total,
            "episode_index_to_sidecar_range": ranges,
            "file": QWEN_VISUAL_SIDECAR_FILE,
            "shape": [total, 2, VISUAL_POOL_GRID[0] * VISUAL_POOL_GRID[1], QWEN_VISUAL_FEATURE_DIM],
            "dtype": "float16",
            "camera_order": list(CAMERA_ORDER),
            "image_preprocessing": "PyAV rgb24 decode, PIL bicubic resize to 224x224, Qwen3.5 image processor",
            "feature": "Qwen3.5 vision tower post-merger tokens, adaptive_avg_pool2d to 4x4 per view, bf16 -> fp16",
            "visual_fingerprint": fingerprint,
            "vision_checkpoint": str(args.checkpoint.resolve()),
            "vision_checkpoint_bytes": args.checkpoint.stat().st_size,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint whose frozen vision tower is reused")
    parser.add_argument("--data-root", type=Path, default=Path("playground/Datasets/LEROBOT_LIBERO_DATA"))
    parser.add_argument("--memory-root", type=Path, default=Path("playground/DerivedData/libero_spatial_mem_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("playground/DerivedData/libero_qwen_visual_v1"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--local-model-path", default=os.environ.get("QWEN35_MODEL_DIR"))
    parser.add_argument("--batch-frames", type=int, default=64)
    parser.add_argument("--decode-workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from starVLA.model.framework.VLM4A.QwenSpatialMemOFT import vision_tower_fingerprint

    qwen = load_qwen(args.checkpoint, args.local_model_path)
    fingerprint = vision_tower_fingerprint(qwen)
    print(f"vision tower fingerprint: {fingerprint}", flush=True)
    for name in args.datasets:
        manifest = prepare_one(args.data_root / name, args.output_root, args.memory_root, qwen, fingerprint, args)
        print(f"wrote {args.output_root / name}: {manifest['shape']}", flush=True)
    (args.output_root / "VISUAL_FINGERPRINT").write_text(fingerprint + "\n")


if __name__ == "__main__":
    main()

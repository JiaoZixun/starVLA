#!/usr/bin/env python3
"""Audit LIBERO data and write lightweight Spatial-Mem-v2 index sidecars.

The output contains only references and derived pose/decision metadata. Source
Parquet files and videos remain in place and are never copied into DerivedData.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# This file is intentionally runnable as ``python path/to/script.py``.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.spatial_geometry import pose_from_position_axis_angle
from starVLA.dataloader.memory_v2 import image_to_memory_visual_tokens

DATASETS = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
DEFAULT_DATA_ROOT = Path("playground/Datasets/LEROBOT_LIBERO_DATA")
DEFAULT_OUTPUT_ROOT = Path("playground/DerivedData/libero_spatial_mem_v2")
STRIDE = 8
TAU = (1, 2, 4, 8)


def revision(dataset: Path) -> str:
    trees = sorted((dataset / ".cache/huggingface/trees").glob("*.json"))
    return trees[0].stem if len(trees) == 1 else "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(dataset: Path) -> list[dict]:
    files = []
    for folder in ("meta", "data", "videos"):
        root = dataset / folder
        files.extend(p for p in root.rglob("*") if p.is_file())
    return [
        {"path": p.relative_to(dataset).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(files)
    ]


def episode_path(dataset: Path, episode_id: int) -> Path:
    return dataset / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"


def decode_video_tokens(path: Path, expected_frames: int) -> np.ndarray:
    """Store only 4x4 RGB summaries, not decoded or copied source frames."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to prepare memv2 visual sidecars") from exc
    container = av.open(str(path))
    try:
        tokens = [
            np.rint(image_to_memory_visual_tokens(frame.to_ndarray(format="rgb24")) * 255.0).astype(np.uint8)
            for frame in container.decode(video=0)
        ]
    finally:
        container.close()
    if len(tokens) != expected_frames:
        raise ValueError(f"{path}: decoded {len(tokens)} frames, expected {expected_frames}")
    return np.stack(tokens)


def prepare_one(dataset: Path, output_root: Path, *, overwrite: bool, no_hashes: bool) -> dict:
    target = output_root / dataset.name
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite")
    info = json.loads((dataset / "meta/info.json").read_text())
    if int(info.get("fps", 0)) != 20:
        raise ValueError(f"{dataset.name}: expected 20 Hz, got {info.get('fps')}")
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"{dataset.name}: episodes.jsonl count does not match info.json")
    source_revision = revision(dataset)
    if source_revision == "unknown":
        raise ValueError(f"{dataset.name}: missing pinned Hugging Face tree revision")

    episode_rows, decision_rows, observation_rows = [], [], []
    episode_offsets = {}
    pose_chunks = []
    visual_chunks = []
    global_offset = 0
    for episode in episodes:
        episode_id = int(episode["episode_index"])
        path = episode_path(dataset, episode_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(
            path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "task_index",
                "index",
            ],
        )
        length = len(frame)
        if length != int(episode["length"]) or length < 1:
            raise ValueError(f"{path}: invalid frame count {length}")
        frame_index = frame["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_index, np.arange(length, dtype=np.int64)):
            raise ValueError(f"{path}: frame_index is not contiguous")
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        if abs(timestamps[0]) > 1e-7 or (length > 1 and not np.allclose(np.diff(timestamps), 0.05, atol=1e-5, rtol=0)):
            raise ValueError(f"{path}: timestamps are not regular 20 Hz")
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        if state.shape != (length, 8) or action.shape != (length, 7) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: expected finite state [T,8] and action [T,7]")
        episode_column = frame["episode_index"].to_numpy(dtype=np.int64)
        if not np.all(episode_column == episode_id):
            raise ValueError(f"{path}: episode_index does not match metadata ({episode_id})")
        task_column = frame["task_index"].to_numpy(dtype=np.int64)
        if not np.all(task_column == task_column[0]) or not 0 <= int(task_column[0]):
            raise ValueError(f"{path}: task_index is inconsistent")
        episode_visuals = []
        for video_key in ("observation.images.image", "observation.images.wrist_image"):
            video_path = dataset / "videos" / "chunk-000" / video_key / f"episode_{episode_id:06d}.mp4"
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"{dataset.name}: missing video {video_path}")
            episode_visuals.append(decode_video_tokens(video_path, length))
        visual_chunks.append(np.stack(episode_visuals, axis=1))
        pose = pose_from_position_axis_angle(state[:, :3], state[:, 3:6])
        pose_chunks.append(pose)
        valid_decisions = list(range(0, length, STRIDE))
        decision_rows.extend(
            {
                "episode_id": episode_id,
                "frame_index": start,
                "next_decision_frame": min(start + STRIDE, length),
                "action_start": start,
                "action_end": min(start + STRIDE, length),
                "action_valid_steps": min(STRIDE, length - start),
                "waypoint_valid": [start + tau < length for tau in TAU],
                "label_valid": start < length,
            }
            for start in valid_decisions
        )
        observation_rows.extend(
            {
                "episode_id": episode_id,
                "frame_index": index,
                "global_frame_index": global_offset + index,
                "primary_video": f"videos/chunk-000/observation.images.image/episode_{episode_id:06d}.mp4",
                "wrist_video": f"videos/chunk-000/observation.images.wrist_image/episode_{episode_id:06d}.mp4",
            }
            for index in range(length)
        )
        episode_rows.append(
            {
                "episode_id": episode_id,
                "task_index": int(task_column[0]),
                "frame_count": length,
                "action_count": length,
                "valid_decisions": len(valid_decisions),
                "source_revision": source_revision,
                "action_observation_phase": "unverified",
                "global_frame_offset": global_offset,
            }
        )
        episode_offsets[str(episode_id)] = {"offset": global_offset, "length": length}
        global_offset += length

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=output_root))
    try:
        pd.DataFrame(episode_rows).to_parquet(staging / "episodes.parquet", index=False)
        pd.DataFrame(decision_rows).to_json(staging / "decision_index.jsonl", orient="records", lines=True)
        pd.DataFrame(observation_rows).to_parquet(staging / "observation_index.parquet", index=False)
        poses = np.concatenate(pose_chunks, axis=0).astype(np.float32)
        np.save(staging / "obs_eef_pose.npy", poses)
        visual_tokens = np.concatenate(visual_chunks, axis=0).astype(np.uint8, copy=False)
        if visual_tokens.shape != (global_offset, 2, 16, 3):
            raise AssertionError(f"Unexpected RGB sidecar shape: {visual_tokens.shape}")
        np.save(staging / "obs_rgb_tokens.npy", visual_tokens, allow_pickle=False)
        manifest = {
            "format": "starvla_libero_spatial_mem_v2_index",
            "dataset_name": dataset.name,
            "source_dataset_path": str(dataset.resolve()),
            "dataset_revision": source_revision,
            "source_total_episodes": len(episodes),
            "source_total_frames": int(global_offset),
            "source_fps": 20,
            "decision_stride_env_steps": STRIDE,
            "memory_visual_contract": {
                "file": "obs_rgb_tokens.npy",
                "shape": [int(global_offset), 2, 16, 3],
                "dtype": "uint8",
                "per_view_grid": [4, 4],
                "resize": "PIL bicubic to 224x224, then 4x4 BOX pooling",
                "camera_order": ["observation.images.image", "observation.images.wrist_image"],
            },
            "episode_index_to_sidecar_range": episode_offsets,
            "waypoint_tau": list(TAU),
            "camera_order": ["observation.images.image", "observation.images.wrist_image"],
            "action_observation_phase": "unverified",
            "phase_evidence": "Source conversion must be reviewed before enabling training.",
            "tail_policy": "masked_partial",
            "source_files_sha256": [] if no_hashes else source_inventory(dataset),
            "files": {
                "episodes": "episodes.parquet",
                "decision_index": "decision_index.jsonl",
                "observation_index": "observation_index.parquet",
                "obs_eef_pose": "obs_eef_pose.npy",
                "obs_rgb_tokens": "obs_rgb_tokens.npy",
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"dataset": dataset.name, "episodes": len(episodes), "frames": global_offset, "decisions": len(decision_rows)}


def verify_one(dataset: Path, output_root: Path) -> dict:
    manifest_path = output_root / dataset.name / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_dataset_path") != str(dataset.resolve()) or manifest.get("dataset_revision") != revision(dataset):
        raise ValueError(f"{dataset.name}: manifest source/revision mismatch")
    poses = np.load(output_root / dataset.name / manifest["files"]["obs_eef_pose"], mmap_mode="r", allow_pickle=False)
    if poses.shape != (int(manifest["source_total_frames"]), 4, 4) or poses.dtype != np.float32:
        raise ValueError(f"{dataset.name}: invalid pose sidecar shape/dtype")
    visuals = np.load(output_root / dataset.name / manifest["files"]["obs_rgb_tokens"], mmap_mode="r", allow_pickle=False)
    if visuals.shape != (len(poses), 2, 16, 3) or visuals.dtype != np.uint8:
        raise ValueError(f"{dataset.name}: invalid RGB feature sidecar shape/dtype {visuals.shape}/{visuals.dtype}")
    episodes = pd.read_parquet(output_root / dataset.name / manifest["files"]["episodes"])
    for video_key in ("observation.images.image", "observation.images.wrist_image"):
        for episode_id in episodes["episode_id"].astype(int):
            video_path = dataset / "videos" / "chunk-000" / video_key / f"episode_{episode_id:06d}.mp4"
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(video_path)
    return {"dataset": dataset.name, "verified": True, "frames": int(len(poses))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-source-hashes", action="store_true")
    args = parser.parse_args()
    reports = []
    for name in args.datasets:
        dataset = args.data_root / name
        reports.append(
            verify_one(dataset, args.output_root)
            if args.verify
            else prepare_one(dataset, args.output_root, overwrite=args.overwrite, no_hashes=args.no_source_hashes)
        )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

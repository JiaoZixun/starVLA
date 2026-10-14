#!/usr/bin/env python3
"""Create and audit LIBERO pose sidecars without modifying source datasets."""

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

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.spatial_geometry import pose_from_position_axis_angle


DEFAULT_DATA_ROOT = Path("playground/Datasets/LEROBOT_LIBERO_DATA")
DEFAULT_OUTPUT_ROOT = Path("playground/DerivedData/libero_spatial_oft_v1")
DATASETS = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(dataset_dir: Path) -> str:
    tree_files = sorted((dataset_dir / ".cache/huggingface/trees").glob("*.json"))
    if len(tree_files) == 1:
        return tree_files[0].stem
    return "unknown"


def source_files(dataset_dir: Path) -> list[Path]:
    paths = []
    for directory in ("meta", "data", "videos"):
        root = dataset_dir / directory
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(paths)


def source_manifest(dataset_dir: Path) -> list[dict[str, str | int]]:
    result = []
    for path in source_files(dataset_dir):
        result.append(
            {
                "path": path.relative_to(dataset_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def preflight(dataset_dir: Path) -> tuple[dict, list[Path]]:
    info_path = dataset_dir / "meta/info.json"
    episodes_path = dataset_dir / "meta/episodes.jsonl"
    modality_path = dataset_dir / "meta/modality.json"
    if not all(path.is_file() for path in (info_path, episodes_path, modality_path)):
        raise FileNotFoundError(f"Missing LeRobot metadata under {dataset_dir}")
    info = json.loads(info_path.read_text())
    modality = json.loads(modality_path.read_text())
    if info.get("fps") != 20:
        raise ValueError(f"Expected LIBERO 20 Hz source data, found fps={info.get('fps')}")
    state_slices = modality.get("state", {})
    expected_slices = {"x": (0, 1), "y": (1, 2), "z": (2, 3), "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6)}
    for key, expected in expected_slices.items():
        actual = state_slices.get(key, {})
        if (actual.get("start"), actual.get("end")) != expected:
            raise ValueError(f"Unexpected state.{key} slice {actual}; expected {expected}")

    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if len(parquet_paths) != info["total_episodes"] or len(episodes) != info["total_episodes"]:
        raise ValueError(
            f"Episode count mismatch for {dataset_dir.name}: metadata={info['total_episodes']}, "
            f"episodes.jsonl={len(episodes)}, parquet={len(parquet_paths)}"
        )

    next_global_index = 0
    total_frames = 0
    max_orthogonality_error = 0.0
    min_determinant = float("inf")
    max_determinant = float("-inf")
    episode_offsets: dict[str, dict[str, int]] = {}
    for expected_episode, path in zip(episodes, parquet_paths):
        episode_index = int(expected_episode["episode_index"])
        if path.stem != f"episode_{episode_index:06d}":
            raise ValueError(f"Episode order/path mismatch: {path} vs episode {episode_index}")
        frame = pd.read_parquet(
            path,
            columns=["observation.state", "timestamp", "frame_index", "episode_index", "index"],
        )
        length = len(frame)
        if length != int(expected_episode["length"]):
            raise ValueError(f"Frame count mismatch in {path}: {length} != {expected_episode['length']}")
        if length == 0:
            raise ValueError(f"Empty episode: {path}")
        frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
        episode_ids = frame["episode_index"].to_numpy(dtype=np.int64)
        global_indices = frame["index"].to_numpy(dtype=np.int64)
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        if not np.array_equal(frame_indices, np.arange(length, dtype=np.int64)):
            raise ValueError(f"Non-contiguous frame_index in {path}")
        if not np.all(episode_ids == episode_index):
            raise ValueError(f"episode_index mismatch in {path}")
        if not np.array_equal(global_indices, np.arange(next_global_index, next_global_index + length)):
            raise ValueError(f"Non-contiguous global index in {path}")
        if abs(timestamps[0]) > 1e-7 or (length > 1 and not np.allclose(np.diff(timestamps), 0.05, atol=1e-5, rtol=0)):
            raise ValueError(f"Unexpected timestamp spacing in {path}")
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32, copy=False)
        if state.shape != (length, 8) or not np.isfinite(state).all():
            raise ValueError(f"Invalid observation.state shape/data in {path}: {state.shape}")
        pose = pose_from_position_axis_angle(state[:, :3], state[:, 3:6])
        rotations = pose[:, :3, :3]
        orth_error = float(np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3))))
        determinant = np.linalg.det(rotations)
        det_min, det_max = float(determinant.min()), float(determinant.max())
        if orth_error > 1e-5 or det_min < 0.99999 or det_max > 1.00001 or not np.isfinite(pose).all():
            raise ValueError(f"Invalid SO(3) pose in {path}: orth={orth_error}, det=[{det_min}, {det_max}]")
        max_orthogonality_error = max(max_orthogonality_error, orth_error)
        min_determinant = min(min_determinant, det_min)
        max_determinant = max(max_determinant, det_max)
        episode_offsets[str(episode_index)] = {"offset": next_global_index, "length": length}
        next_global_index += length
        total_frames += length

    if total_frames != int(info["total_frames"]):
        raise ValueError(f"Total frame count mismatch: {total_frames} != {info['total_frames']}")
    audit = {
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "fps": info["fps"],
        "state_shape": [8],
        "action_shape": list(info["features"]["action"]["shape"]),
        "max_so3_orthogonality_error": max_orthogonality_error,
        "determinant_range": [min_determinant, max_determinant],
        "all_episode_frame_indices_contiguous": True,
        "all_dataset_indices_contiguous": True,
        "timestamps_regular_50ms": True,
    }
    return {"info": info, "episodes": episode_offsets, "audit": audit}, parquet_paths


def expected_episode_pose(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["observation.state"])
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32, copy=False)
    return pose_from_position_axis_angle(state[:, :3], state[:, 3:6])


def generate_dataset(dataset_dir: Path, output_root: Path, *, overwrite: bool, hash_sources: bool) -> dict:
    metadata, parquet_paths = preflight(dataset_dir)
    target = output_root / dataset_dir.name
    if target.exists() and not overwrite:
        raise FileExistsError(f"Derived sidecar already exists: {target}; pass --overwrite to replace it")

    input_files = source_manifest(dataset_dir) if hash_sources else []
    revision = source_revision(dataset_dir)
    if revision == "unknown":
        raise ValueError(f"Could not determine pinned Hugging Face revision from {dataset_dir}/.cache")
    total_frames = int(metadata["audit"]["total_frames"])

    output_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{dataset_dir.name}.", dir=output_root))
    try:
        pose_path = stage_dir / "obs_eef_pose.npy"
        poses = np.lib.format.open_memmap(pose_path, mode="w+", dtype=np.float32, shape=(total_frames, 4, 4))
        for path in parquet_paths:
            episode_index = int(path.stem[-6:])
            episode = metadata["episodes"][str(episode_index)]
            offset, length = episode["offset"], episode["length"]
            poses[offset : offset + length] = expected_episode_pose(path)
        poses.flush()
        del poses

        manifest = {
            "format": "starvla_libero_spatial_pose_sidecar_v1",
            "dataset_name": dataset_dir.name,
            "dataset_revision": revision,
            "source_dataset_path": str(dataset_dir.resolve()),
            "source_total_frames": total_frames,
            "source_total_episodes": int(metadata["audit"]["total_episodes"]),
            "source_fps": 20,
            "state_source": "observation.state[0:3]=robot0_eef_pos; [3:6]=robot0_eef_axis_angle",
            "rotation_convention": "axis-angle rotation vector, right-handed, radians; PyTorch3D-compatible SO(3)",
            "pose_convention": "column vectors; p_world = R @ p_local + t; pose frame is stored LIBERO robot0_eef TCP frame",
            "action_observation_phase": "unverified; action-geometry supervision disabled",
            "source_files_sha256": input_files,
            "episode_index_to_sidecar_range": metadata["episodes"],
            "fields": {"obs_eef_pose": {"file": "obs_eef_pose.npy", "shape": [total_frames, 4, 4], "dtype": "float32"}},
            "audit": metadata["audit"],
        }
        (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(stage_dir, target)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    return {"dataset": dataset_dir.name, "output": str(target), **metadata["audit"], "revision": revision, "hashes_written": len(input_files)}


def verify_dataset(dataset_dir: Path, output_root: Path, *, verify_hashes: bool) -> dict:
    metadata, parquet_paths = preflight(dataset_dir)
    sidecar_dir = output_root / dataset_dir.name
    manifest = json.loads((sidecar_dir / "manifest.json").read_text())
    if manifest["dataset_name"] != dataset_dir.name or manifest["source_dataset_path"] != str(dataset_dir.resolve()):
        raise ValueError(f"Sidecar does not link to the requested source dataset: {sidecar_dir}")
    if manifest["dataset_revision"] != source_revision(dataset_dir):
        raise ValueError(f"Sidecar revision mismatch for {dataset_dir.name}")
    if manifest.get("source_total_frames") != metadata["audit"]["total_frames"]:
        raise ValueError(f"Sidecar frame count mismatch for {dataset_dir.name}")
    if manifest.get("source_total_episodes") != metadata["audit"]["total_episodes"]:
        raise ValueError(f"Sidecar episode count mismatch for {dataset_dir.name}")
    if manifest.get("source_fps") != metadata["audit"]["fps"]:
        raise ValueError(f"Sidecar FPS mismatch for {dataset_dir.name}")
    if manifest.get("episode_index_to_sidecar_range") != metadata["episodes"]:
        raise ValueError(f"Sidecar episode/frame mapping mismatch for {dataset_dir.name}")
    if manifest.get("audit") != metadata["audit"]:
        raise ValueError(f"Sidecar source audit mismatch for {dataset_dir.name}")
    poses = np.load(sidecar_dir / "obs_eef_pose.npy", mmap_mode="r", allow_pickle=False)
    if poses.shape != (metadata["audit"]["total_frames"], 4, 4) or poses.dtype != np.float32:
        raise ValueError(f"Invalid sidecar shape/dtype: {poses.shape}/{poses.dtype}")
    max_abs_error = 0.0
    for path in parquet_paths:
        episode_index = int(path.stem[-6:])
        episode = metadata["episodes"][str(episode_index)]
        expected = expected_episode_pose(path)
        actual = np.asarray(poses[episode["offset"] : episode["offset"] + episode["length"]])
        error = float(np.max(np.abs(actual - expected)))
        max_abs_error = max(max_abs_error, error)
        if error > 1e-6:
            raise ValueError(f"Sidecar pose values differ from source {path}: max_abs={error}")
    expected_hashes = manifest.get("source_files_sha256", [])
    current_files = source_files(dataset_dir)
    expected_inventory = {
        item["path"]: int(item["size_bytes"])
        for item in expected_hashes
    }
    current_inventory = {
        path.relative_to(dataset_dir).as_posix(): path.stat().st_size
        for path in current_files
    }
    if expected_hashes and expected_inventory != current_inventory:
        raise ValueError(f"Sidecar source-file inventory mismatch for {dataset_dir.name}")
    if verify_hashes:
        if not expected_hashes:
            raise ValueError("Manifest has no source SHA256 inventory; regenerate with source hashing enabled")
        for item in expected_hashes:
            path = dataset_dir / item["path"]
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Source file changed since sidecar generation: {path}")
    return {"dataset": dataset_dir.name, "verified_episodes": len(parquet_paths), "verified_frames": int(len(poses)), "max_pose_abs_error": max_abs_error, "verified_source_hashes": len(manifest.get("source_files_sha256", [])) if verify_hashes else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--verify", action="store_true", help="Verify existing sidecars against raw data without writing")
    parser.add_argument("--no-source-hashes", action="store_true", help="Skip expensive full SHA256 inventory (not recommended)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing derived sidecar directory")
    args = parser.parse_args()

    reports = []
    for name in args.datasets:
        dataset_dir = args.data_root / name
        if args.verify:
            reports.append(verify_dataset(dataset_dir, args.output_root, verify_hashes=not args.no_source_hashes))
        else:
            reports.append(generate_dataset(dataset_dir, args.output_root, overwrite=args.overwrite, hash_sources=not args.no_source_hashes))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

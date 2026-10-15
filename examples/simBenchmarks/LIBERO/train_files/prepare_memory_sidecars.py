#!/usr/bin/env python3
"""Prepare compact causal visual-history sidecars without copying source data.

Each source RGB frame is represented by eight 8x8 RGB patch thumbnails for
each camera.  The patch tokens are deliberately kept as uint8 pixels: the
trainable memory encoder maps them into the 256-wide latent space.  The source
Parquet and video files remain read-only and are linked by revision, episode,
and frame index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_ROOT = Path("playground/Datasets/LEROBOT_LIBERO_DATA")
DEFAULT_POSE_ROOT = Path("playground/DerivedData/libero_spatial_oft_v1")
DEFAULT_OUTPUT_ROOT = Path("playground/DerivedData/libero_mem_oft_v1")
DATASETS = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
VIDEO_KEYS = ("observation.images.image", "observation.images.wrist_image")
PATCH_ROWS, PATCH_COLS, PATCH_SIZE = 2, 4, 8


def _source_revision(dataset_dir: Path) -> str:
    files = sorted((dataset_dir / ".cache/huggingface/trees").glob("*.json"))
    return files[0].stem if len(files) == 1 else "unknown"


def _patch_tokens(rgb: np.ndarray) -> np.ndarray:
    """Return a fixed 2x4 grid of 8x8 RGB thumbnails from a 256x256 frame."""
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB [256,256,3], got {rgb.dtype}/{rgb.shape}")
    result = np.empty((PATCH_ROWS * PATCH_COLS, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    patch_index = 0
    for row in range(PATCH_ROWS):
        for col in range(PATCH_COLS):
            crop = rgb[
                row * 128 : (row + 1) * 128,
                col * 64 : (col + 1) * 64,
            ]
            result[patch_index] = np.asarray(
                Image.fromarray(crop).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BOX),
                dtype=np.uint8,
            )
            patch_index += 1
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_ffmpeg_rgb(video_path: Path, length: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to recover this source video with the system decoder")
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video_path), "-map", "0:v:0", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg could not decode {video_path}: {result.stderr.decode(errors='replace')[-1000:]}"
        )
    expected_bytes = length * 256 * 256 * 3
    if len(result.stdout) != expected_bytes:
        raise ValueError(
            f"FFmpeg decoded {len(result.stdout) // (256 * 256 * 3)} frames from {video_path}; "
            f"expected exactly {length}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(length, 256, 256, 3)


def _decode_video_into(
    video_path: Path,
    target: np.ndarray,
    view_index: int,
    offset: int,
    length: int,
    *,
    expected_source_sha256: str | None,
) -> dict | None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing source video: {video_path}")
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        count = 0
        try:
            for frame in container.decode(stream):
                if count >= length:
                    raise ValueError(f"Video contains more than {length} frames: {video_path}")
                rgb = frame.to_ndarray(format="rgb24")
                target[offset + count, view_index] = _patch_tokens(rgb)
                count += 1
            if count != length:
                raise ValueError(f"Video frame count mismatch for {video_path}: {count} != {length}")
            return None
        except Exception as decode_error:
            source_sha256 = _sha256_file(video_path)
            if expected_source_sha256 is None or source_sha256 != expected_source_sha256:
                raise ValueError(
                    f"Decoder failed and source SHA256 does not match the pinned dataset tree: {video_path}; "
                    f"actual={source_sha256}, expected={expected_source_sha256}"
                ) from decode_error
            rgb_frames = _decode_ffmpeg_rgb(video_path, length)
            for frame_index, rgb in enumerate(rgb_frames):
                target[offset + frame_index, view_index] = _patch_tokens(rgb)
            ffmpeg = shutil.which("ffmpeg")
            version = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
            return {
                "source_sha256": source_sha256,
                "frame_count": int(length),
                "ffmpeg_version": version,
                "decoder_error": f"{type(decode_error).__name__}: {decode_error}",
            }
    finally:
        container.close()


def _prepare_one(dataset_name: str, data_root: str, pose_root: str, output_root: str, overwrite: bool) -> dict:
    started = time.perf_counter()
    dataset_dir = Path(data_root) / dataset_name
    pose_dir = Path(pose_root) / dataset_name
    pose_manifest_path = pose_dir / "manifest.json"
    if not pose_manifest_path.is_file() or not (pose_dir / "obs_eef_pose.npy").is_file():
        raise FileNotFoundError(f"Pose sidecar is required first: {pose_dir}")
    pose_manifest = json.loads(pose_manifest_path.read_text())
    revision = _source_revision(dataset_dir)
    if revision == "unknown" or revision != pose_manifest.get("dataset_revision"):
        raise ValueError(f"Source revision does not match pose sidecar for {dataset_name}")
    if str(dataset_dir.resolve()) != pose_manifest.get("source_dataset_path"):
        raise ValueError(f"Source path does not match pose sidecar for {dataset_name}")

    info = json.loads((dataset_dir / "meta/info.json").read_text())
    if float(info.get("fps", 0)) != 20.0:
        raise ValueError(f"Expected 20Hz LIBERO data, found {info.get('fps')}")
    episodes = [
        json.loads(line)
        for line in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tree_paths = sorted((dataset_dir / ".cache/huggingface/trees").glob("*.json"))
    source_files = json.loads(tree_paths[0].read_text()).get("files", {}) if len(tree_paths) == 1 else {}
    episode_ranges = pose_manifest["episode_index_to_sidecar_range"]
    target = Path(output_root) / dataset_name
    if target.exists() and not overwrite:
        raise FileExistsError(f"Memory sidecar already exists: {target}; pass --overwrite")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dataset_name}.", dir=output_root_path))
    try:
        total_frames = int(pose_manifest["source_total_frames"])
        shape = (total_frames, 2, 8, PATCH_SIZE, PATCH_SIZE, 3)
        visual_path = stage / "visual_patches.npy"
        visual = np.lib.format.open_memmap(visual_path, mode="w+", dtype=np.uint8, shape=shape)
        video_template = info["video_path"]
        data_template = info["data_path"]
        video_overrides = []
        visited = 0
        for episode in episodes:
            episode_id = int(episode["episode_index"])
            episode_range = episode_ranges[str(episode_id)]
            offset = int(episode_range["offset"])
            length = int(episode["length"])
            if int(episode_range["length"]) != length:
                raise ValueError(f"Episode length mismatch for {dataset_name}/{episode_id}")
            chunk = episode_id // int(info["chunks_size"])
            parquet_path = dataset_dir / data_template.format(
                episode_chunk=chunk, episode_index=episode_id
            )
            table = pd.read_parquet(
                parquet_path,
                columns=["frame_index", "episode_index", "index", "timestamp"],
            )
            if len(table) != length:
                raise ValueError(f"Parquet length mismatch in {parquet_path}: {len(table)} != {length}")
            if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
                raise ValueError(f"Non-contiguous frame indices in {parquet_path}")
            if not np.all(table["episode_index"].to_numpy() == episode_id):
                raise ValueError(f"Episode ID mismatch in {parquet_path}")
            if not np.allclose(table["timestamp"].to_numpy(dtype=np.float64), np.arange(length) / 20.0, atol=1e-5):
                raise ValueError(f"Timestamp/frame phase mismatch in {parquet_path}")

            for view_index, video_key in enumerate(VIDEO_KEYS):
                video_path = dataset_dir / video_template.format(
                    episode_chunk=chunk,
                    episode_index=episode_id,
                    video_key=video_key,
                )
                source_relative = video_path.relative_to(dataset_dir).as_posix()
                source_sha = source_files.get(source_relative, {}).get("lfs_sha256")
                repair = _decode_video_into(
                    video_path,
                    visual,
                    view_index,
                    offset,
                    length,
                    expected_source_sha256=source_sha,
                )
                if repair is not None:
                    video_overrides.append(
                        {
                            "episode_index": episode_id,
                            "video_key": video_key,
                            "source_path": source_relative,
                            **repair,
                        }
                    )
            visited += length
        if visited != total_frames:
            raise ValueError(f"Visited frame count mismatch: {visited} != {total_frames}")
        visual.flush()
        del visual

        manifest = {
            "format": "starvla_libero_causal_memory_sidecar_v1",
            "dataset_name": dataset_name,
            "dataset_revision": revision,
            "source_dataset_path": str(dataset_dir.resolve()),
            "source_pose_sidecar": str(pose_dir.resolve()),
            "source_total_frames": total_frames,
            "source_total_episodes": len(episodes),
            "source_fps": 20,
            "environment_step": "frame_index; one LIBERO environment step at 20Hz",
            "visual_representation": (
                "8 spatial RGB thumbnails per camera: 2x4 image grid; each patch resized to 8x8; uint8; "
                "trainable encoder maps pixel tokens to 256-wide memory features"
            ),
            "visual_views": list(VIDEO_KEYS),
            "visual_shape": list(shape),
            "visual_dtype": "uint8",
            "decision_stride_env_steps": [4, 8],
            "recent_decision_records": 32,
            "bptt_decision_records": 8,
            "action_observation_phase": "unverified; effect targets and residuals disabled",
            "no_raw_source_copies": True,
            "decoder_fallbacks": video_overrides,
            "episode_index_to_sidecar_range": episode_ranges,
            "fields": {
                "visual_patches": {
                    "file": "visual_patches.npy",
                    "shape": list(shape),
                    "dtype": "uint8",
                }
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "dataset": dataset_name,
        "output": str(target),
        "frames": total_frames,
        "episodes": len(episodes),
        "visual_bytes": int(np.prod(shape)),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _verify_one(dataset_name: str, data_root: Path, pose_root: Path, output_root: Path) -> dict:
    source = data_root / dataset_name
    manifest = json.loads((output_root / dataset_name / "manifest.json").read_text())
    pose_manifest = json.loads((pose_root / dataset_name / "manifest.json").read_text())
    if manifest["dataset_revision"] != _source_revision(source):
        raise ValueError(f"Revision mismatch for {dataset_name}")
    if manifest["source_dataset_path"] != str(source.resolve()):
        raise ValueError(f"Source path mismatch for {dataset_name}")
    if manifest["episode_index_to_sidecar_range"] != pose_manifest["episode_index_to_sidecar_range"]:
        raise ValueError(f"Episode/frame map mismatch for {dataset_name}")
    visual = np.load(output_root / dataset_name / "visual_patches.npy", mmap_mode="r", allow_pickle=False)
    expected = tuple(manifest["visual_shape"])
    if visual.shape != expected or visual.dtype != np.uint8:
        raise ValueError(f"Bad visual sidecar: {visual.shape}/{visual.dtype}, expected {expected}/uint8")
    # Verify deterministic time endpoints and spatial tokens without touching any future labels.
    if not len(visual) or visual[0].shape != (2, 8, 8, 8, 3):
        raise ValueError(f"Bad visual token dimensions for {dataset_name}")
    for override in manifest.get("decoder_fallbacks", []):
        original = source / override["source_path"]
        if _sha256_file(original) != override["source_sha256"]:
            raise ValueError(f"Decoder fallback source hash mismatch: {original}")
        frames = _decode_ffmpeg_rgb(original, int(override["frame_count"]))
        if len(frames) != int(override["frame_count"]):
            raise ValueError(f"Decoder fallback frame count mismatch: {original}")
    return {
        "dataset": dataset_name,
        "frames": len(visual),
        "episodes": len(manifest["episode_index_to_sidecar_range"]),
        "decoder_fallbacks": len(manifest.get("decoder_fallbacks", [])),
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--workers", type=int, default=1, help="Parallelize independent dataset videos on CPU")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.verify:
        reports = [_verify_one(name, args.data_root, args.pose_root, args.output_root) for name in args.datasets]
    elif args.workers <= 1 or len(args.datasets) == 1:
        reports = [
            _prepare_one(name, str(args.data_root), str(args.pose_root), str(args.output_root), args.overwrite)
            for name in args.datasets
        ]
    else:
        reports = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.datasets))) as pool:
            futures = {
                pool.submit(
                    _prepare_one,
                    name,
                    str(args.data_root),
                    str(args.pose_root),
                    str(args.output_root),
                    args.overwrite,
                ): name
                for name in args.datasets
            }
            for future in as_completed(futures):
                reports.append(future.result())
        reports.sort(key=lambda item: DATASETS.index(item["dataset"]))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

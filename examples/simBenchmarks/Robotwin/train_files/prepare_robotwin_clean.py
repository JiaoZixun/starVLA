#!/usr/bin/env python3
"""Convert RoboTwin demo_clean HDF5 episodes to StarVLA's LeRobot v2 layout.

Converted state/action parquet and wrist-camera videos live in output_root.
Original HDF5 and head-camera MP4 files are only linked into each adapter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SOURCE_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FPS = 30
CHANNELS = ("left_joints", "left_gripper", "right_joints", "right_gripper")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Unexpected existing link: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to replace {target}")
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def _instruction(task_dir: Path, episode_id: int, h5: h5py.File) -> str:
    for candidate in (
        task_dir / "aloha_agilex/instruction" / f"episode_{episode_id:07d}.json",
        task_dir / "aloha_agilex/instruction" / f"episode_{episode_id:06d}.json",
    ):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            seen = value.get("seen") or value.get("unseen") or []
            if seen:
                return str(seen[0])
    raw = h5["instructions"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(str(raw)) if isinstance(raw, str) else raw
    if isinstance(value, dict):
        for key in ("seen", "unseen"):
            if value.get(key):
                return str(value[key][0])
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    raise ValueError(f"Could not extract instruction for {task_dir.name}/{episode_id}")


def _decode_jpeg(blob) -> np.ndarray:
    import cv2
    image = cv2.imdecode(np.frombuffer(bytes(blob), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid JPEG frame in HDF5")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _stats(arrays: list[np.ndarray]) -> dict:
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    return {name: fn(values, axis=0).tolist() for name, fn in (
        ("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std),
        ("q01", lambda x, axis: np.quantile(x, .01, axis=axis)),
        ("q99", lambda x, axis: np.quantile(x, .99, axis=axis)),
    )}


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError("Camera resolution changed within an episode")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_metadata(
    dataset: Path,
    episodes: list[dict],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    camera_shapes: dict[str, tuple[int, int, int]],
) -> None:
    meta = dataset / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [f"state.{name}" for name, n in (("left_joints", 6), ("left_gripper", 1), ("right_joints", 6), ("right_gripper", 1)) for _ in range(n)]},
        "action": {"dtype": "float32", "shape": [14], "names": [f"action.{name}" for name, n in (("left_joints", 6), ("left_gripper", 1), ("right_joints", 6), ("right_gripper", 1)) for _ in range(n)]},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in CAMERAS:
        height, width, channels = camera_shapes[camera]
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": [height, width, channels], "names": ["height", "width", "channel"],
            "video_info": {"video.fps": FPS, "video.codec": "mpeg4", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False},
        }
    info = {"codebase_version": "v2.0", "robot_type": "aloha_agilex", "total_episodes": len(episodes),
            "total_frames": sum(e["length"] for e in episodes), "total_tasks": 1, "total_videos": len(episodes)*3,
            "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features}
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {
        "human.action.task_description": {"original_key": "task_index"}}}
    start = 0
    for name, width in (("left_joints", 6), ("left_gripper", 1), ("right_joints", 6), ("right_gripper", 1)):
        modality["state"][name] = {"start": start, "end": start+width, "original_key": "observation.state", "dtype": "float32", "absolute": True, "rotation_type": None}
        modality["action"][name] = {"start": start, "end": start+width, "original_key": "action", "dtype": "float32", "absolute": True, "rotation_type": None}
        start += width
    modality["video"] = {camera: {"original_key": f"observation.images.{camera}"} for camera in CAMERAS}
    (meta / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": episodes[0]["instruction"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps({"episode_index": episode["episode_index"], "tasks": [episode["instruction"]], "length": episode["length"]}, ensure_ascii=False) + "\n")
    stats = {"__format_version": 2, "__cache_config": {"mode": "abs"}, "statistics": {
        "observation.state": _stats(states), "action": _stats(actions)}}
    (meta / "stats_gr00t.json").write_text(json.dumps(stats) + "\n", encoding="utf-8")


def prepare_task(source_root: Path, output_root: Path, task: str, episode_limit: int | None = None) -> Path:
    task_dir = source_root / task
    source_data = task_dir / "aloha_agilex/data"
    files = sorted(source_data.glob("episode_*.hdf5"))
    if episode_limit:
        files = files[:episode_limit]
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes under {source_data}")
    dataset = output_root / "Clean" / task
    (dataset / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    episodes, state_arrays, action_arrays = [], [], []
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    episode_offset = 0
    for file in files:
        source_episode = int(file.stem.split("_")[-1])
        with h5py.File(file, "r") as h5:
            length = h5["state/left_arm_joint_states"].shape[0]
            names = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")
            state = np.concatenate([h5[f"state/{name}"][()] for name in names], axis=1).astype(np.float32)
            action = np.concatenate([h5[f"action/{name}"][()] for name in names], axis=1).astype(np.float32)
            if state.shape != (length, 14) or action.shape != (length, 14) or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Invalid 14D state/action in {file}")
            instruction = _instruction(task_dir, source_episode, h5)
            row = pa.table({"observation.state": list(state), "action": list(action), "task_index": np.zeros(length, np.int64),
                            "episode_index": np.full(length, episode_offset, np.int64), "frame_index": np.arange(length, dtype=np.int64),
                            "timestamp": np.arange(length, dtype=np.float32) / FPS, "index": np.arange(length, dtype=np.int64)})
            pq.write_table(row, dataset / f"data/chunk-000/episode_{episode_offset:06d}.parquet")
            for camera, source_camera in zip(CAMERAS, SOURCE_CAMERAS):
                if camera == "cam_high":
                    mp4 = task_dir / "aloha_agilex/video" / f"episode_{source_episode:07d}.mp4"
                    if mp4.is_file():
                        _link(mp4, dataset / f"videos/chunk-000/observation.images.{camera}/episode_{episode_offset:06d}.mp4")
                    else:
                        frames = [_decode_jpeg(b) for b in h5[f"vision/{source_camera}/colors"]]
                        _write_video(dataset / f"videos/chunk-000/observation.images.{camera}/episode_{episode_offset:06d}.mp4", frames)
                else:
                    frames = [_decode_jpeg(b) for b in h5[f"vision/{source_camera}/colors"]]
                    _write_video(dataset / f"videos/chunk-000/observation.images.{camera}/episode_{episode_offset:06d}.mp4", frames)
                if camera not in camera_shapes:
                    with h5py.File(file, "r") as shape_h5:
                        first = _decode_jpeg(shape_h5[f"vision/{source_camera}/colors"][0])
                    camera_shapes[camera] = tuple(int(v) for v in first.shape)
            _link(file, dataset / f"raw_sources/episode_{episode_offset:06d}.hdf5")
        episode = {"episode_index": episode_offset, "source_episode": source_episode, "length": length, "instruction": instruction}
        episodes.append(episode)
        state_arrays.append(state); action_arrays.append(action)
        episode_offset += 1
        print(f"Prepared {task} source episode {source_episode}: {length} frames", flush=True)
    _write_metadata(dataset, episodes, state_arrays, action_arrays, camera_shapes)
    manifest = {"format": "robotwin_clean_lerobot_v2", "source_root": str(source_root.resolve()), "task": task,
                "source_episode_ids": [e["source_episode"] for e in episodes], "total_episodes": len(episodes),
                "total_frames": sum(e["length"] for e in episodes), "raw_data_policy": "symlink; source HDF5/MP4 bytes are never copied"}
    (dataset / "meta/robotwin_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("playground/Datasets/RoboTwin/demo_clean"))
    parser.add_argument("--output-root", type=Path, default=Path("playground/Datasets/RoboTwin/lerobot_clean"))
    parser.add_argument("--task", action="append", help="Task directory; repeat to select several. Defaults to all tasks.")
    parser.add_argument("--episodes-per-task", type=int)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    tasks = args.task or sorted(p.name for p in source.iterdir() if p.is_dir())
    for task in tasks:
        prepare_task(source, output, task, args.episodes_per_task)


if __name__ == "__main__":
    main()

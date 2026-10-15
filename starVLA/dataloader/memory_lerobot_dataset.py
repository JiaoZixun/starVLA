"""Causal LIBERO samples backed by an external memory sidecar.

The sidecar contains only deterministic RGB patch summaries and an episode
frame map.  Raw actions/states are read from the source Parquet row selected by
the current sample, while every memory record is strictly older than the
current frame.  No future observation, action, waypoint, or validity flag is
passed as a model input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from starVLA.dataloader.geometry_lerobot_dataset import SpatialLeRobotSingleDataset


class MemoryLeRobotSingleDataset(SpatialLeRobotSingleDataset):
    """Attach ordered recent decision records to every spatial sample."""

    def __init__(
        self,
        *args,
        memory_sidecar_root: str | Path,
        memory_recent_records: int = 32,
        memory_decision_stride: int = 8,
        memory_visual_tokens: int = 8,
        memory_bptt_records: int = 8,
        memory_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_recent_records = int(memory_recent_records)
        self.memory_decision_stride = int(memory_decision_stride)
        self.memory_visual_tokens = int(memory_visual_tokens)
        self.memory_bptt_records = int(memory_bptt_records)
        self.memory_dropout = float(memory_dropout)
        if self.memory_recent_records < 1 or self.memory_decision_stride < 1:
            raise ValueError("memory_recent_records and memory_decision_stride must be positive")
        if not 0.0 <= self.memory_dropout < 1.0:
            raise ValueError("memory_dropout must be in [0, 1)")

        self.memory_sidecar_dir = Path(memory_sidecar_root) / self.dataset_name
        manifest_path = self.memory_sidecar_dir / "manifest.json"
        visual_path = self.memory_sidecar_dir / "visual_patches.npy"
        if not manifest_path.is_file() or not visual_path.is_file():
            raise FileNotFoundError(
                f"Memory sidecar missing for {self.dataset_name}: {manifest_path}; "
                "run prepare_memory_sidecars.py first"
            )
        self.memory_manifest = json.loads(manifest_path.read_text())
        if self.memory_manifest.get("dataset_name") != self.dataset_name:
            raise ValueError("Memory sidecar dataset name does not match source dataset")
        source_path = str(self.dataset_path.resolve())
        if self.memory_manifest.get("source_dataset_path") != source_path:
            raise ValueError("Memory sidecar source_dataset_path does not match source dataset")
        tree_files = sorted((self.dataset_path / ".cache/huggingface/trees").glob("*.json"))
        source_revision = tree_files[0].stem if len(tree_files) == 1 else "unknown"
        if source_revision == "unknown" or self.memory_manifest.get("dataset_revision") != source_revision:
            raise ValueError("Memory sidecar dataset revision does not match source dataset")
        if int(self.memory_manifest.get("source_fps", -1)) != 20:
            raise ValueError("Memory sidecar does not describe 20Hz LIBERO data")
        views = self.memory_manifest.get("visual_views", [])
        if list(views) != ["observation.images.image", "observation.images.wrist_image"]:
            raise ValueError(f"Unexpected memory camera contract: {views}")
        if int(self.memory_manifest.get("visual_shape", [0, 0, 0, 0, 0, 0])[-4]) != self.memory_visual_tokens:
            raise ValueError("Memory visual token count differs from sidecar contract")

        self._memory_offsets = {
            int(episode): (int(entry["offset"]), int(entry["length"]))
            for episode, entry in self.memory_manifest["episode_index_to_sidecar_range"].items()
        }
        expected_offset = 0
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            offset, sidecar_length = self._memory_offsets[int(trajectory_id)]
            if offset != expected_offset or sidecar_length != int(trajectory_length):
                raise ValueError(f"Memory sidecar frame mapping mismatch for episode {trajectory_id}")
            expected_offset += int(trajectory_length)
        if int(self.memory_manifest.get("source_total_frames", -1)) != expected_offset:
            raise ValueError("Memory sidecar frame count does not match source dataset")

        visual = np.load(visual_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (expected_offset, 2, self.memory_visual_tokens, 8, 8, 3)
        if tuple(visual.shape) != expected_shape or visual.dtype != np.uint8:
            raise ValueError(
                f"Invalid memory visual sidecar: shape={visual.shape}, dtype={visual.dtype}, "
                f"expected={expected_shape}/uint8"
            )
        self._memory_visual_path = visual_path
        self._memory_visual_memmap = visual
        self._memory_active_sample: tuple[int, int] | None = None
        self._build_memory_burnin = False
        self._video_path_overrides: dict[tuple[int, str], Path] = {}
        for override in self.memory_manifest.get("video_overrides", []):
            episode_index = int(override["episode_index"])
            video_key = str(override["video_key"])
            source_path = self.dataset_path / override["source_path"]
            replacement_path = Path(memory_sidecar_root) / override["replacement_path"]
            if not source_path.is_file() or not replacement_path.is_file():
                raise FileNotFoundError(f"Video override file missing: {source_path} / {replacement_path}")
            if self._sha256(source_path) != override["source_sha256"]:
                raise ValueError(f"Video override source hash mismatch: {source_path}")
            if self._sha256(replacement_path) != override["replacement_sha256"]:
                raise ValueError(f"Video override replacement hash mismatch: {replacement_path}")
            self._verify_video_override(replacement_path, int(override["frame_count"]))
            self._video_path_overrides[(episode_index, video_key)] = replacement_path

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _verify_video_override(path: Path, expected_frames: int) -> None:
        import av

        container = av.open(str(path))
        try:
            count = sum(1 for _ in container.decode(video=0))
        finally:
            container.close()
        if count != expected_frames:
            raise ValueError(f"Video override frame count mismatch: {path}: {count} != {expected_frames}")

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        original_key = self.lerobot_modality_meta.video[key].original_key or key
        override = self._video_path_overrides.get((int(trajectory_id), original_key))
        if override is not None:
            return override
        return super().get_video_path(trajectory_id, key)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._memory_active_sample = (int(trajectory_id), int(base_index))
        return super().get_step_data(trajectory_id, base_index)

    def _raw_row(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.curr_traj_data is None:
            raise RuntimeError("Trajectory data is not loaded before building memory records")
        row = self.curr_traj_data.iloc[int(frame_index)]
        state = np.asarray(row["observation.state"], dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)
        if state.shape != (8,) or action.shape != (7,):
            raise ValueError(f"Unexpected raw LIBERO state/action shape: {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Non-finite raw state/action in source dataset")
        return state, action

    def _memory_records(self, trajectory_id: int, frame_index: int) -> dict[str, np.ndarray]:
        offset, episode_length = self._memory_offsets[int(trajectory_id)]
        visual = np.zeros(
            (self.memory_recent_records, 2, self.memory_visual_tokens, 8, 8, 3), dtype=np.uint8
        )
        state = np.zeros((self.memory_recent_records, 8), dtype=np.float32)
        action = np.zeros((self.memory_recent_records, 7), dtype=np.float32)
        elapsed = np.zeros((self.memory_recent_records,), dtype=np.float32)
        valid = np.zeros((self.memory_recent_records,), dtype=np.bool_)
        source_indices = np.full((self.memory_recent_records,), -1, dtype=np.int64)
        poses = self._get_geometry_pose_memmap()
        eef_pose = np.broadcast_to(
            np.eye(4, dtype=np.float32), (self.memory_recent_records, 4, 4)
        ).copy()

        # Oldest first.  The strict '< frame_index' check is the causal
        # boundary and protects against accidentally using current/future rows.
        for slot in range(self.memory_recent_records):
            history_index = frame_index - self.memory_decision_stride * (self.memory_recent_records - slot)
            if history_index < 0 or history_index >= episode_length or history_index >= frame_index:
                continue
            valid[slot] = True
            source_indices[slot] = history_index
            visual[slot] = np.asarray(self._memory_visual_memmap[offset + history_index], dtype=np.uint8)
            state[slot], action[slot] = self._raw_row(history_index)
            elapsed[slot] = float(frame_index - history_index)
            eef_pose[slot] = np.asarray(poses[offset + history_index], dtype=np.float32)

        # This contract is explicit until the action/observation phase audit is
        # completed.  Zeros are never treated as valid effects by the model.
        width = 256
        return {
            "visual_tokens": visual,
            "state": state,
            "executed_actions": action,
            "eef_pose": eef_pose,
            "elapsed_env_steps": elapsed,
            "observed_effect": np.zeros((self.memory_recent_records, width), dtype=np.float32),
            "prediction_residual": np.zeros((self.memory_recent_records, width), dtype=np.float32),
            "effect_valid": np.zeros((self.memory_recent_records,), dtype=np.bool_),
            "valid_mask": valid,
            "source_frame_index": source_indices,
        }

    def _memory_record_at(self, trajectory_id: int, frame_index: int) -> dict[str, np.ndarray]:
        offset, episode_length = self._memory_offsets[int(trajectory_id)]
        if not 0 <= frame_index < episode_length:
            raise IndexError(f"Invalid memory record frame {frame_index} for episode {trajectory_id}")
        state, action = self._raw_row(frame_index)
        return {
            "visual_tokens": np.asarray(self._memory_visual_memmap[offset + frame_index], dtype=np.uint8).copy(),
            "state": state,
            "executed_actions": action,
            "eef_pose": np.asarray(self._get_geometry_pose_memmap()[offset + frame_index], dtype=np.float32).copy(),
            "elapsed_env_steps": np.asarray(self.memory_decision_stride, dtype=np.float32),
            "observed_effect": np.zeros((256,), dtype=np.float32),
            "prediction_residual": np.zeros((256,), dtype=np.float32),
            "effect_valid": np.asarray(False, dtype=np.bool_),
        }

    def _memory_burnin_records(self, trajectory_id: int, frame_index: int) -> dict[str, np.ndarray]:
        stride = self.memory_decision_stride
        recent_start = max(0, frame_index - stride * self.memory_recent_records)
        indices = np.arange(0, recent_start, stride, dtype=np.int64)
        if indices.size and int(indices[-1]) >= frame_index:
            raise AssertionError("Burn-in may contain only records strictly earlier than the current frame")
        records = [self._memory_record_at(trajectory_id, int(index)) for index in indices]
        if not records:
            return {
                "visual_tokens": np.zeros((0, 2, self.memory_visual_tokens, 8, 8, 3), dtype=np.uint8),
                "state": np.zeros((0, 8), dtype=np.float32),
                "executed_actions": np.zeros((0, 7), dtype=np.float32),
                "eef_pose": np.zeros((0, 4, 4), dtype=np.float32),
                "elapsed_env_steps": np.zeros((0,), dtype=np.float32),
                "observed_effect": np.zeros((0, 256), dtype=np.float32),
                "prediction_residual": np.zeros((0, 256), dtype=np.float32),
                "effect_valid": np.zeros((0,), dtype=np.bool_),
                "valid_mask": np.zeros((0,), dtype=np.bool_),
                "source_frame_index": indices,
            }
        return {
            **{key: np.stack([record[key] for record in records]) for key in records[0]},
            "valid_mask": np.ones((len(records),), dtype=np.bool_),
            "source_frame_index": indices,
        }

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        if self._memory_active_sample is None:
            raise RuntimeError("Memory sample key is unavailable; get_step_data must precede _pack_sample")
        trajectory_id, frame_index = self._memory_active_sample
        records = self._memory_records(trajectory_id, frame_index)
        burnin = self._memory_burnin_records(trajectory_id, frame_index)
        sample["memory_records"] = records
        sample["memory_burnin_records"] = burnin
        sample["memory_current_record"] = self._memory_record_at(trajectory_id, frame_index)
        sample["memory_contract"] = {
            "history_is_strictly_past": True,
            "decision_stride_env_steps": self.memory_decision_stride,
            "recent_records": self.memory_recent_records,
            "bptt_decision_records": self.memory_bptt_records,
            "source_frame_index": int(frame_index),
            "burnin_records": int(len(burnin["source_frame_index"])),
            "burnin_before_frame": int(
                burnin["source_frame_index"][-1] if len(burnin["source_frame_index"]) else -1
            ),
            "episode_index": int(trajectory_id),
            "effect_targets_enabled": False,
        }
        return sample


__all__ = ["MemoryLeRobotSingleDataset"]

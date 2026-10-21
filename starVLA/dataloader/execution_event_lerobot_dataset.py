"""LIBERO LeRobot samples with causal, stride-aligned memv2 events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from starVLA.dataloader.geometry_lerobot_dataset import SpatialLeRobotSingleDataset
from starVLA.dataloader.memory_v2 import (
    ACTION_HORIZON,
    MEMORY_CAPACITY_EVENTS,
    MEMORY_STRIDE,
    build_execution_event,
)


class ExecutionEventLeRobotDataset(SpatialLeRobotSingleDataset):
    """Attach raw 8-step event history using only derived visual sidecars.

    Actions and states stay in the source LeRobot Parquet files.  The sidecar
    contributes only the compact RGB features and indexes; this class reads
    an episode's raw rows already loaded by the parent dataset.
    """

    def __init__(
        self,
        *args,
        memory_sidecar_root: str | Path,
        memory_capacity_events: int = MEMORY_CAPACITY_EVENTS,
        memory_decision_stride_env_steps: int = MEMORY_STRIDE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if int(memory_decision_stride_env_steps) != ACTION_HORIZON:
            raise ValueError(f"memv2 requires an {ACTION_HORIZON}-step event/action interval")
        self.memory_capacity_events = int(memory_capacity_events)
        if self.memory_capacity_events < 1:
            raise ValueError("memory_capacity_events must be positive")
        self.memory_decision_stride = int(memory_decision_stride_env_steps)
        self.memory_sidecar_dir = Path(memory_sidecar_root) / self.dataset_name
        manifest_path = self.memory_sidecar_dir / "manifest.json"
        visual_path = self.memory_sidecar_dir / "obs_rgb_tokens.npy"
        if not manifest_path.is_file() or not visual_path.is_file():
            raise FileNotFoundError(
                f"memv2 visual sidecar missing for {self.dataset_name}: {manifest_path}; "
                "run prepare_memory_v2.py first"
            )
        self.memory_manifest = json.loads(manifest_path.read_text())
        if self.memory_manifest.get("format") != "starvla_libero_spatial_mem_v2_index":
            raise ValueError(f"Unsupported memv2 sidecar format for {self.dataset_name}")
        if self.memory_manifest.get("dataset_name") != self.dataset_name:
            raise ValueError("memv2 sidecar dataset name does not match source dataset")
        if self.memory_manifest.get("source_dataset_path") != str(self.dataset_path.resolve()):
            raise ValueError("memv2 sidecar source path does not match source dataset")
        trees = sorted((self.dataset_path / ".cache/huggingface/trees").glob("*.json"))
        source_revision = trees[0].stem if len(trees) == 1 else "unknown"
        if source_revision == "unknown" or self.memory_manifest.get("dataset_revision") != source_revision:
            raise ValueError("memv2 sidecar revision does not match source dataset")
        if int(self.memory_manifest.get("decision_stride_env_steps", -1)) != self.memory_decision_stride:
            raise ValueError("memv2 sidecar decision stride differs from the training config")

        self._memory_offsets = {
            int(episode): (int(entry["offset"]), int(entry["length"]))
            for episode, entry in self.memory_manifest["episode_index_to_sidecar_range"].items()
        }
        if set(self._memory_offsets) != {int(value) for value in self.trajectory_ids}:
            raise ValueError("memv2 sidecar episode IDs do not match the source dataset")
        expected_offset = 0
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            offset, sidecar_length = self._memory_offsets[int(trajectory_id)]
            if offset != expected_offset or sidecar_length != int(trajectory_length):
                raise ValueError(f"memv2 sidecar frame mapping mismatch for episode {trajectory_id}")
            expected_offset += int(trajectory_length)
        if int(self.memory_manifest.get("source_total_frames", -1)) != expected_offset:
            raise ValueError("memv2 sidecar frame count does not match source dataset")

        visual = np.load(visual_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (expected_offset, 2, 16, 3)
        if visual.shape != expected_shape or visual.dtype != np.uint8:
            raise ValueError(
                f"Invalid memv2 RGB feature sidecar: shape={visual.shape}, dtype={visual.dtype}; "
                f"expected {expected_shape}/uint8"
            )
        self._memory_visual_path = visual_path
        self._memory_visual_memmap = visual
        self._memory_active_sample: tuple[int, int] | None = None

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._memory_active_sample = (int(trajectory_id), int(base_index))
        return super().get_step_data(trajectory_id, base_index)

    def _raw_state_actions(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        if self.curr_traj_data is None:
            raise RuntimeError("Trajectory rows are unavailable while building memv2 history")
        states = np.stack(self.curr_traj_data.iloc[start:end]["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(self.curr_traj_data.iloc[start:end]["action"].to_numpy()).astype(np.float32)
        if states.shape != (end - start, 8) or actions.shape != (end - start, 7):
            raise ValueError(f"Unexpected LIBERO source state/action shapes: {states.shape}/{actions.shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite LIBERO source state/action values")
        return states, actions

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        if self._memory_active_sample is None:
            raise RuntimeError("memv2 sample index is unavailable; get_step_data must precede _pack_sample")
        episode_id, frame_index = self._memory_active_sample
        offset, episode_length = self._memory_offsets[episode_id]
        if not 0 <= frame_index < episode_length:
            raise IndexError(f"Invalid frame {frame_index} for episode {episode_id}")
        if frame_index % self.memory_decision_stride:
            raise ValueError(
                f"memv2 training anchors must align to stride {self.memory_decision_stride}; got {frame_index}"
            )

        # This is the same raw 8-D proprioception passed by the LIBERO eval client.
        states, _ = self._raw_state_actions(frame_index, frame_index + 1)
        current_state = states[0]
        sample["state"] = current_state[None, :].astype(np.float32, copy=False)
        # The parent geometry dataset already owns this pose mapping; reuse it
        # rather than writing another copy into the memv2 sidecar.
        poses = self._get_geometry_pose_memmap()
        current_pose = np.asarray(poses[offset + frame_index], dtype=np.float32)
        initial_pose = np.asarray(poses[offset], dtype=np.float32)
        current_visual = np.asarray(self._memory_visual_memmap[offset + frame_index], dtype=np.float32) / 255.0
        sample["memory_visual_tokens"] = current_visual
        sample["env_step"] = int(frame_index)

        first_start = max(0, frame_index // self.memory_decision_stride - self.memory_capacity_events)
        events = []
        for event_index in range(first_start, frame_index // self.memory_decision_stride):
            start = event_index * self.memory_decision_stride
            end = start + self.memory_decision_stride
            if end > frame_index or end > episode_length:
                raise AssertionError("memv2 attempted to include an incomplete or current event")
            event_states, event_actions = self._raw_state_actions(start, end)
            # LIBERO stores gripper labels as open=1/close=0; the evaluator
            # sends open=+1/close=-1 after the same threshold operation.
            event_actions[:, 6] = np.where(event_actions[:, 6] > 0.5, -1.0, 1.0)
            event = build_execution_event(
                start_step=start,
                end_step=end,
                visual_start=np.asarray(self._memory_visual_memmap[offset + start], dtype=np.float32) / 255.0,
                visual_end=np.asarray(self._memory_visual_memmap[offset + end], dtype=np.float32) / 255.0,
                state_start=event_states[0],
                state_end=np.asarray(self.curr_traj_data.iloc[end]["observation.state"], dtype=np.float32),
                executed_actions=event_actions,
                pose_start_in_episode=np.linalg.inv(initial_pose) @ np.asarray(poses[offset + start], dtype=np.float32),
                pose_end_in_episode=np.linalg.inv(initial_pose) @ np.asarray(poses[offset + end], dtype=np.float32),
            )
            events.append(event)

        sample["memory_events"] = events
        sample["memory_contract"] = {
            "history_is_strictly_past": True,
            "decision_stride_env_steps": self.memory_decision_stride,
            "capacity_events": self.memory_capacity_events,
            "source_frame_index": int(frame_index),
            "episode_index": int(episode_id),
            "event_action_source": "raw_source_action_rows_with_eval_gripper_mapping",
            "event_interval": "[start_step,end_step), end_step <= source_frame_index",
        }
        # Keep the same absolute measured pose contract used by eval.
        sample["geometry_current"]["eef_pose"] = current_pose
        return sample

__all__ = ["ExecutionEventLeRobotDataset"]

"""LeRobot dataset wrapper that joins external LIBERO pose sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.grounding_augment import GroundingAugment
from starVLA.dataloader.phase_labels import phases_from_holding, robot_holding_from_gripper_action, slot_phases


class SpatialLeRobotSingleDataset(LeRobotSingleDataset):
    """Attach geometry labels by (dataset, revision, episode, frame) only."""

    def __init__(
        self, *args, geometry_sidecar_root: str | Path, waypoint_tau=(1, 4, 8, 16), phase_window_frames: int | None = None,
        grounding_sidecar_root: str | Path | None = None, gripper_label: str = "qpos", augment=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # None disables phase labels (default); otherwise the grasp/release window in frames.
        self.phase_window_frames = None if phase_window_frames is None else int(phase_window_frames)
        self._phase_cache: tuple[int, np.ndarray] | None = None
        self.waypoint_tau = tuple(int(value) for value in waypoint_tau)
        if not self.waypoint_tau or min(self.waypoint_tau) < 1 or tuple(sorted(set(self.waypoint_tau))) != self.waypoint_tau:
            raise ValueError(f"waypoint_tau must be strictly increasing positive offsets, got {self.waypoint_tau}")
        self.sidecar_dir = Path(geometry_sidecar_root) / self.dataset_name
        manifest_path = self.sidecar_dir / "manifest.json"
        pose_path = self.sidecar_dir / "obs_eef_pose.npy"
        if not manifest_path.is_file() or not pose_path.is_file():
            raise FileNotFoundError(
                f"Spatial pose sidecar missing for {self.dataset_name}: {manifest_path}; "
                "run prepare_spatial_sidecars.py first"
            )
        self.geometry_manifest = json.loads(manifest_path.read_text())
        if self.geometry_manifest.get("dataset_name") != self.dataset_name:
            raise ValueError(f"Sidecar dataset mismatch: {self.geometry_manifest.get('dataset_name')} != {self.dataset_name}")
        source_path = str(self.dataset_path.resolve())
        if self.geometry_manifest.get("source_dataset_path") != source_path:
            raise ValueError(
                f"Sidecar source path mismatch for {self.dataset_name}: "
                f"{self.geometry_manifest.get('source_dataset_path')} != {source_path}"
            )
        tree_files = sorted((self.dataset_path / ".cache/huggingface/trees").glob("*.json"))
        source_revision = tree_files[0].stem if len(tree_files) == 1 else "unknown"
        if source_revision == "unknown" or self.geometry_manifest.get("dataset_revision") != source_revision:
            raise ValueError(
                f"Sidecar revision mismatch for {self.dataset_name}: "
                f"{self.geometry_manifest.get('dataset_revision')} != {source_revision}"
            )
        if self.geometry_manifest.get("action_observation_phase") != "unverified; action-geometry supervision disabled":
            raise ValueError("Unexpected sidecar phase contract; review the manifest before training")
        self._geometry_pose_path = pose_path
        self._geometry_pose_memmap = None
        self._geometry_offsets = {
            int(episode): (int(entry["offset"]), int(entry["length"]))
            for episode, entry in self.geometry_manifest["episode_index_to_sidecar_range"].items()
        }
        if len(self.trajectory_ids) != len(self.trajectory_lengths):
            raise ValueError("LeRobot trajectory ids and lengths have different sizes")
        expected_episodes = set(int(value) for value in self.trajectory_ids)
        if set(self._geometry_offsets) != expected_episodes:
            raise ValueError(
                f"Sidecar episode ids do not match {self.dataset_name}: "
                f"sidecar={len(self._geometry_offsets)}, dataset={len(expected_episodes)}"
            )
        expected_offset = 0
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            offset, sidecar_length = self._geometry_offsets[int(trajectory_id)]
            if offset != expected_offset or sidecar_length != int(trajectory_length):
                raise ValueError(
                    f"Sidecar frame mapping mismatch for {self.dataset_name} episode {trajectory_id}: "
                    f"sidecar=(offset={offset}, length={sidecar_length}), "
                    f"dataset=(offset={expected_offset}, length={int(trajectory_length)})"
                )
            expected_offset += int(trajectory_length)
        if self.geometry_manifest.get("source_total_frames") != expected_offset:
            raise ValueError(
                f"Sidecar frame count does not match {self.dataset_name}: "
                f"{self.geometry_manifest.get('source_total_frames')} != {expected_offset}"
            )
        pose_array = np.load(pose_path, mmap_mode="r", allow_pickle=False)
        if pose_array.shape != (expected_offset, 4, 4) or pose_array.dtype != np.float32:
            raise ValueError(
                f"Invalid pose sidecar for {self.dataset_name}: "
                f"shape={pose_array.shape}, dtype={pose_array.dtype}"
            )
        self._geometry_pose_memmap = pose_array
        self._geometry_active_sample: tuple[int, int] | None = None
        if gripper_label not in ("qpos", "closure"):
            raise ValueError(f"gripper_label must be qpos or closure, got {gripper_label!r}")
        self.gripper_label = gripper_label
        self.augment = GroundingAugment(augment)
        self.grounding = None
        if grounding_sidecar_root:
            self.grounding = self._load_grounding(Path(grounding_sidecar_root) / self.dataset_name, expected_offset)
        elif gripper_label == "closure":
            raise ValueError("gripper_label=closure needs grounding_sidecar_root (holding labels)")

    def _load_grounding(self, root: Path, total_frames: int) -> dict:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format") != "starvla_libero_grounding_sidecar_v1" or manifest.get("dataset_name") != self.dataset_name:
            raise ValueError(f"Unexpected grounding sidecar in {root}")
        if manifest.get("dataset_revision") != self.geometry_manifest.get("dataset_revision") or (
            manifest.get("episode_index_to_sidecar_range") != self.geometry_manifest["episode_index_to_sidecar_range"]
        ):
            raise ValueError(f"Grounding sidecar {root} is not aligned with the pose sidecar")
        arrays = {}
        for key in ("target_world", "goal_world", "point_valid", "holding", "target_uv", "goal_uv", "target_uv_valid", "goal_uv_valid"):
            arrays[key] = np.load(root / f"{key}.npy", mmap_mode="r")
            if len(arrays[key]) != total_frames:
                raise ValueError(f"Grounding array {key} has {len(arrays[key])} rows, expected {total_frames}")
        return arrays

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._geometry_active_sample = (int(trajectory_id), int(base_index))
        return super().get_step_data(trajectory_id, base_index)

    def _get_geometry_pose_memmap(self) -> np.ndarray:
        if self._geometry_pose_memmap is None:
            self._geometry_pose_memmap = np.load(self._geometry_pose_path, mmap_mode="r", allow_pickle=False)
        return self._geometry_pose_memmap

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        if self._geometry_active_sample is None:
            raise RuntimeError("Geometry sample key is unavailable; get_step_data must precede _pack_sample")
        episode_index, frame_index = self._geometry_active_sample
        try:
            offset, episode_length = self._geometry_offsets[episode_index]
        except KeyError as exc:
            raise KeyError(f"Episode {episode_index} is absent from sidecar for {self.dataset_name}") from exc
        if not 0 <= frame_index < episode_length:
            raise IndexError(f"Invalid frame {frame_index} for episode {episode_index} length {episode_length}")

        poses = self._get_geometry_pose_memmap()
        current = np.asarray(poses[offset + frame_index], dtype=np.float32).copy()
        future_indices = [frame_index + tau for tau in self.waypoint_tau]
        future_valid = np.asarray([index < episode_length for index in future_indices], dtype=np.bool_)
        future = np.broadcast_to(np.eye(4, dtype=np.float32), (len(self.waypoint_tau), 4, 4)).copy()
        current_state = None
        if self.curr_traj_data is not None and "observation.state" in self.curr_traj_data.columns:
            current_state = np.asarray(self.curr_traj_data.iloc[frame_index]["observation.state"], dtype=np.float32)
        future_gripper = np.zeros((len(self.waypoint_tau),), dtype=np.float32)
        future_gripper_valid = future_valid.copy()
        # ``observation.state`` is read only for supervision.  It is never
        # passed to the model as a future input, so this remains causal at
        # inference time.
        trajectory = self.curr_traj_data
        if trajectory is None or "observation.state" not in trajectory.columns:
            future_gripper_valid[:] = False
        for waypoint_index, source_index in enumerate(future_indices):
            if future_valid[waypoint_index]:
                future[waypoint_index] = np.asarray(poses[offset + source_index], dtype=np.float32)
                if future_gripper_valid[waypoint_index]:
                    state_value = np.asarray(trajectory.iloc[source_index]["observation.state"], dtype=np.float32)
                    if state_value.shape != (8,) or not np.isfinite(state_value).all():
                        future_gripper_valid[waypoint_index] = False
                    else:
                        future_gripper[waypoint_index] = state_value[7]
        current_rotation_t = current[:3, :3].T
        relative = np.broadcast_to(np.eye(4, dtype=np.float32), (len(self.waypoint_tau), 4, 4)).copy()
        relative[:, :3, :3] = current_rotation_t[None, :, :] @ future[:, :3, :3]
        relative[:, :3, 3] = (
            current_rotation_t[None, :, :] @ (future[:, :3, 3] - current[:3, 3])[:, :, None]
        )[:, :, 0]
        # Invalid future values are numerically safe identity transforms, and their independent mask is always false.
        relative[~future_valid] = np.eye(4, dtype=np.float32)

        sample["geometry_current"] = {
            "eef_pose": current,
            "gripper_state": float(current_state[7]) if current_state is not None and current_state.shape == (8,) else 0.0,
            "arm_mask": np.asarray([True], dtype=np.bool_),
        }
        sample["geometry_labels"] = {
            "relative_waypoints": relative,
            "future_valid": future_valid,
            "future_gripper": future_gripper,
            "future_gripper_valid": future_gripper_valid,
            "source_frame_index": np.asarray(frame_index, dtype=np.int64),
        }
        if self.phase_window_frames is not None:
            future_phase, future_phase_valid = slot_phases(self._episode_phases(episode_index, trajectory), future_indices)
            sample["geometry_labels"]["future_phase"] = future_phase
            sample["geometry_labels"]["future_phase_valid"] = future_phase_valid
        uv = uv_valid = None
        if self.grounding is not None:
            g = self.grounding
            row, labels = offset + frame_index, sample["geometry_labels"]
            # Slot 0 = object handled next, slot 1 = where it is put (see prepare_grounding_sidecars.py).
            world = np.stack((g["target_world"][row], g["goal_world"][row])).astype(np.float32)
            labels["object_xyz"] = (current_rotation_t[None] @ (world - current[:3, 3])[:, :, None])[:, :, 0]
            labels["object_xyz_valid"] = np.repeat(bool(g["point_valid"][row]), 2)
            uv = np.stack((g["target_uv"][row], g["goal_uv"][row])).astype(np.float32)
            uv_valid = np.asarray([g["target_uv_valid"][row], g["goal_uv_valid"][row]], dtype=np.bool_)
            if self.gripper_label == "closure":
                # Binary "closed" from the commanded gripper, the same signal human hands provide.
                for k, source_index in enumerate(future_indices):
                    if future_gripper_valid[k]:
                        future_gripper[k] = float(g["holding"][offset + source_index])
        if self.augment.enabled:
            sample["image"], uv, uv_valid = self.augment(sample["image"], uv, uv_valid)
        if uv is not None:
            sample["geometry_labels"]["object_uv"], sample["geometry_labels"]["object_uv_valid"] = uv, uv_valid
        return sample

    def _episode_phases(self, episode_index: int, trajectory) -> np.ndarray:
        """Per-frame phase ids of the active episode, from its commanded gripper (action[:, 6])."""
        if self._phase_cache is not None and self._phase_cache[0] == episode_index:
            return self._phase_cache[1]
        if trajectory is None or "action" not in trajectory.columns:
            raise ValueError(f"Phase labels need the action column of {self.dataset_name} episode {episode_index}")
        actions = np.stack(trajectory["action"].to_numpy()).astype(np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected LIBERO actions [T,7] for phase labels, got {actions.shape}")
        phase = phases_from_holding(robot_holding_from_gripper_action(actions[:, 6]), self.phase_window_frames)
        self._phase_cache = (episode_index, phase)
        return phase

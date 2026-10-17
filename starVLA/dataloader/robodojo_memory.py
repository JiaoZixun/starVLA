"""Vectorized strict-past execution history for dual-arm RoboDojo."""
import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.utils.robodojo_memory import RoboDojoMemorySpec
from starVLA.utils.robodojo_memory_sidecar import RoboDojoMemorySidecar


class RoboDojoMemorySingleDataset(LeRobotSingleDataset):
    def __init__(self, *args, memory_recent_records=16, memory_decision_stride=8,
                 memory_state_dim=14, memory_action_dim=14, memory_use_pose=False,
                 memory_bptt_records=8, memory_visual_tokens=8, memory_visual_views=3,
                 memory_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if memory_state_dim != 14 or memory_action_dim != 14:
            raise ValueError("RoboDojo requires two 6-DoF arms and two grippers (14 channels)")
        self.memory_spec = RoboDojoMemorySpec(memory_recent_records, memory_decision_stride, memory_use_pose)
        if memory_visual_tokens != 8 or memory_visual_views != 3:
            raise ValueError("RoboDojo visual memory requires 3 cameras and 8 patches")
        self.memory_bptt_records = int(memory_bptt_records)
        self.memory_dropout = float(memory_dropout)
        self._memory_active_sample = None
        sidecar = (self.data_cfg or {}).get("memory_sidecar_dir")
        self._memory_sidecar = RoboDojoMemorySidecar(sidecar, self.dataset_path, self.memory_spec) if sidecar else None
        self._memory_linked_episode = None

    def get_step_data(self, trajectory_id, base_index):
        # MixtureDataset calls get_step_data/_pack_sample directly.
        self._memory_active_sample = (int(trajectory_id), int(base_index))
        return super().get_step_data(trajectory_id, base_index)

    def _memory_records(self, trajectory_id, frame_index):
        spec = self.memory_spec
        sidecar = getattr(self, "_memory_sidecar", None)
        if sidecar is None:
            indices = spec.indices(frame_index)
        else:
            if getattr(self, "_memory_linked_episode", None) != trajectory_id:
                frames = self.curr_traj_data["frame_index"].to_numpy()
                if not np.array_equal(frames, np.arange(len(frames))) or not (
                        self.curr_traj_data["episode_index"] == trajectory_id).all():
                    raise ValueError("Memory sidecar must link to the same original episode/frame rows")
                self._memory_linked_episode = trajectory_id
            indices = sidecar.indices(trajectory_id, len(self.curr_traj_data))[frame_index]
        valid = indices >= 0
        result = spec.empty()
        # One DataFrame gather, no historical video decoding, no action chunks,
        # and no repeated construction of future spatial labels.
        rows = self.curr_traj_data.iloc[indices[valid]]
        if valid.any():
            if not np.array_equal(np.asarray(rows["frame_index"], dtype=np.int64), indices[valid]):
                raise ValueError("RoboDojo memory requires contiguous episode frame indices")
            result["state"][valid] = np.stack(rows["observation.state"].to_numpy())
            result["executed_actions"][valid] = np.stack(rows["action"].to_numpy())
            if sidecar is not None:
                result["visual_tokens"][valid] = np.asarray(sidecar.visual_tokens(trajectory_id, len(self.curr_traj_data)))[indices[valid]]
            if spec.use_pose:
                # The parent's current-sample geometry has already loaded and
                # validated this episode's proprioceptive FK sidecar.
                cached = self._spatial_episode_cache.get(int(trajectory_id))
                if cached is None:
                    raise ValueError("Pose memory requires the validated spatial FK sidecar")
                if not np.array_equal(cached["source_frame_index"][indices[valid]], indices[valid]):
                    raise ValueError("Historical FK frames do not match parquet")
                result["eef_pose"][valid] = cached["pose"][indices[valid]]
        result["valid_mask"] = valid
        result["source_frame_index"][valid] = indices[valid]
        result["elapsed_env_steps"][valid] = frame_index - indices[valid]
        spec.validate(result, spec.contract(frame_index, trajectory_id))
        return result

    def _memory_record_at(self, trajectory_id, frame_index):
        row = self.curr_traj_data.iloc[int(frame_index)]
        result = self.memory_spec.empty()
        result = {key: value[0] if value.ndim else value for key, value in result.items()}
        result["state"] = np.asarray(row["observation.state"], dtype=np.float32)
        result["executed_actions"] = np.asarray(row["action"], dtype=np.float32)
        if self._memory_sidecar is not None:
            result["visual_tokens"] = np.asarray(self._memory_sidecar.visual_tokens(trajectory_id, len(self.curr_traj_data))[frame_index])
        result["elapsed_env_steps"] = np.float32(self.memory_spec.decision_stride)
        result["valid_mask"] = np.bool_(True)
        result["source_frame_index"] = np.int64(frame_index)
        if self.memory_spec.use_pose:
            cached = self._spatial_episode_cache[int(trajectory_id)]
            result["eef_pose"] = cached["pose"][frame_index]
        return result

    def _memory_burnin_records(self, trajectory_id, frame_index):
        cutoff = max(0, int(frame_index) - self.memory_spec.decision_stride * self.memory_spec.recent_records)
        indices = np.arange(0, cutoff, self.memory_spec.decision_stride, dtype=np.int64)
        records = [self._memory_record_at(trajectory_id, int(index)) for index in indices]
        empty = self.memory_spec.empty()
        return {key: (np.stack([record[key] for record in records]) if records else np.zeros((0, *value.shape[1:]), dtype=value.dtype))
                for key, value in empty.items()}

    def _pack_sample(self, data):
        sample = super()._pack_sample(data)
        if self._memory_active_sample is None:
            raise RuntimeError("get_step_data must precede memory sample packing")
        episode, step = self._memory_active_sample
        sample["memory_records"] = self._memory_records(episode, step)
        sample["memory_burnin_records"] = self._memory_burnin_records(episode, step)
        sample["memory_contract"] = self.memory_spec.contract(step, episode)
        return sample

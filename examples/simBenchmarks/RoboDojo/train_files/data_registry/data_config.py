"""RoboDojo LeRobot v2.1 data registration for the dual-arm ARX X5."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboDojoArxX5H50Q99DataConfig:
    """Three-camera, 14D state/action format used by the released policies."""

    embodiment_tag = EmbodimentTag.ARX_X5
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    state_key_dims = {
        "state.left_joints": 6,
        "state.left_gripper": 1,
        "state.right_joints": 6,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.left_joints": 6,
        "action.left_gripper": 1,
        "action.right_joints": 6,
        "action.right_gripper": 1,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_modes = {key: "q99" for key in self.state_keys}
        action_modes = {key: "q99" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                # Parquet may store float64. Cast before q99, exactly as the
                # serving raw-state/command protocol does, not after packing.
                StateActionToTensor(
                    apply_to=self.state_keys,
                    output_dtypes={key: "torch.float32" for key in self.state_keys},
                ),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(
                    apply_to=self.action_keys,
                    output_dtypes={key: "torch.float32" for key in self.action_keys},
                ),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )

    def make_dataset(self, **kwargs):
        # Memory is an opt-in dataset wrapper.  The regular spatial branch
        # keeps the stock loader and its mixture semantics unchanged.
        data_cfg = kwargs.get("data_cfg") or {}
        if data_cfg.get("memory_enabled", False):
            from starVLA.dataloader.robodojo_memory import RoboDojoMemorySingleDataset
            dataset_cls = RoboDojoMemorySingleDataset
        else:
            from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
            dataset_cls = LeRobotSingleDataset
        memory_keys = {
            "memory_recent_records", "memory_decision_stride", "memory_state_dim",
            "memory_action_dim", "memory_use_pose",
        }
        memory_kwargs = {key: data_cfg[key] for key in memory_keys if key in data_cfg} if data_cfg.get("memory_enabled", False) else {}
        init_kwargs = {k: v for k, v in kwargs.items() if k != "dataset_name"}
        init_kwargs.update(memory_kwargs)
        return dataset_cls(**init_kwargs)


_ROBOT_TYPE = "robodojo_arx_x5_h50_q99"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TYPE: RoboDojoArxX5H50Q99DataConfig(),
}

DATASET_NAMED_MIXTURES = {
    "robodojo_v21_all_h50_q99": [
        ("RoboDojo_lerobot_v21_video", 1.0, _ROBOT_TYPE),
    ],
    "robodojo_v30_all_h50_q99": [
        ("RoboDojo_lerobot_v30_video", 1.0, _ROBOT_TYPE),
    ],
}

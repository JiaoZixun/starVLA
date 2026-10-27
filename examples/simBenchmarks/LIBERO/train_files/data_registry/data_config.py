"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.geometry_lerobot_dataset import SpatialLeRobotSingleDataset
from starVLA.dataloader.execution_event_lerobot_dataset import ExecutionEventLeRobotDataset


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


class OpenPILibero4in1DataConfig(Libero4in1DataConfig):
    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys + self.action_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "q99",
                },
            ),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "q99",
                },
            ),
        ])


class SpatialLibero4in1DataConfig(Libero4in1DataConfig):
    """LIBERO contract with pose labels joined from an external sidecar."""

    def make_dataset(self, **kwargs):
        data_cfg = kwargs.get("data_cfg")
        if data_cfg is None:
            raise ValueError("Spatial LIBERO training requires datasets.vla_data.geometry_sidecar_root")
        sidecar_root = data_cfg.get("geometry_sidecar_root")
        if not sidecar_root:
            raise ValueError("Spatial LIBERO training requires datasets.vla_data.geometry_sidecar_root")
        return SpatialLeRobotSingleDataset(
            **kwargs,
            geometry_sidecar_root=sidecar_root,
            waypoint_tau=tuple(data_cfg.get("waypoint_tau", [1, 4, 8, 16])),
        )


class MemoryLibero4in1DataConfig(SpatialLibero4in1DataConfig):
    """Spatial LIBERO config with stride-8 causal execution-event history."""

    def make_dataset(self, **kwargs):
        data_cfg = kwargs.get("data_cfg")
        if data_cfg is None:
            raise ValueError("Memory LIBERO training requires datasets.vla_data.memory_sidecar_root")
        geometry_root = data_cfg.get("geometry_sidecar_root")
        memory_root = data_cfg.get("memory_sidecar_root")
        if not geometry_root or not memory_root:
            raise ValueError("Memory LIBERO training requires both geometry and memory sidecar roots")
        return ExecutionEventLeRobotDataset(
            **kwargs,
            geometry_sidecar_root=geometry_root,
            memory_sidecar_root=memory_root,
            waypoint_tau=tuple(data_cfg.get("waypoint_tau", [1, 4, 8, 16])),
            memory_capacity_events=int(data_cfg.get("memory_capacity_events", 128)),
            memory_decision_stride_env_steps=int(data_cfg.get("memory_decision_stride_env_steps", 8)),
            # The mixture sampler only draws stride-aligned anchors when this is
            # true; keep the dataset-side check in sync with it.
            memory_aligned_anchors_only=bool(data_cfg.get("memory_decision_frame_only", True)),
            memory_visual_source=str(data_cfg.get("memory_visual_source", "rgb")),
            qwen_visual_sidecar_root=data_cfg.get("qwen_visual_sidecar_root"),
        )


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "openpi_libero_franka": OpenPILibero4in1DataConfig(),
    "libero_spatial_oft_franka": SpatialLibero4in1DataConfig(),
    "libero_spatial_mem_oft_franka": MemoryLibero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "openpi_libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
    ],
    # Spatial:Object:Goal:Long = 1:1:1:2 relative weights (same mechanism as libero_spatial_mem_oft_long2x).
    "libero_spatial_oft_long2x": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 2.0, "libero_spatial_oft_franka"),
    ],
    "libero_spatial_oft_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_oft_franka"),
    ],
    "libero_spatial_mem_oft_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
    ],
    # Spatial:Object:Goal:Long = 1:1:1:2 sampling probability (weights are not
    # scaled by dataset length), so LIBERO-10 supplies 40% of each batch.
    "libero_spatial_mem_oft_long2x": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_spatial_mem_oft_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 2.0, "libero_spatial_mem_oft_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}

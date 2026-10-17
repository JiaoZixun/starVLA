"""Shared offline/online causal record contract. No simulator or torch imports.

State and commands are raw dual-arm absolute qpos, ordered left joints (6),
left gripper (1), right joints (6), right gripper (1). A command at frame t
may enter memory only after environment step t has completed.
"""
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

MEMORY_PATCH_ROWS = 2
MEMORY_PATCH_COLS = 4
MEMORY_VISUAL_TOKENS = MEMORY_PATCH_ROWS * MEMORY_PATCH_COLS
MEMORY_PATCH_SIZE = 8
MEMORY_VISUAL_VIEWS = 3
MEMORY_VISUAL_FEATURE_DIM = 3
MEMORY_VISUAL_SHAPE = (MEMORY_VISUAL_VIEWS, MEMORY_VISUAL_TOKENS, MEMORY_PATCH_SIZE, MEMORY_PATCH_SIZE, MEMORY_VISUAL_FEATURE_DIM)


def image_to_memory_tokens(image: Any) -> np.ndarray:
    """Convert an RGB frame to the shared ordered 2x4 8x8 uint8 patch contract."""
    from PIL import Image
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {value.shape}")
        if value.dtype != np.uint8:
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError("RGB image must contain finite numeric values")
            value = np.clip(value, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(value, mode="RGB")
    width, height = rgb.size
    patch_width, patch_height = width // MEMORY_PATCH_COLS, height // MEMORY_PATCH_ROWS
    if patch_width < 1 or patch_height < 1:
        raise ValueError(f"RGB image is too small for 2x4 patches: {rgb.size}")
    result = np.empty((MEMORY_VISUAL_TOKENS, MEMORY_PATCH_SIZE, MEMORY_PATCH_SIZE, 3), np.uint8)
    slot = 0
    for row in range(MEMORY_PATCH_ROWS):
        for col in range(MEMORY_PATCH_COLS):
            crop = rgb.crop((col * patch_width, row * patch_height,
                             width if col == MEMORY_PATCH_COLS - 1 else (col + 1) * patch_width,
                             height if row == MEMORY_PATCH_ROWS - 1 else (row + 1) * patch_height))
            result[slot] = np.asarray(crop.resize((MEMORY_PATCH_SIZE, MEMORY_PATCH_SIZE), Image.Resampling.BOX), dtype=np.uint8)
            slot += 1
    return result


@dataclass(frozen=True)
class RoboDojoMemorySpec:
    recent_records: int = 16
    decision_stride: int = 8
    use_pose: bool = False

    def __post_init__(self):
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1
               for v in (self.recent_records, self.decision_stride)):
            raise ValueError("Memory window and stride must be positive integers")

    def indices(self, current: int) -> np.ndarray:
        if not isinstance(current, (int, np.integer)) or isinstance(current, bool) or current < 0:
            raise ValueError("Current environment step must be a nonnegative integer")
        return current - self.decision_stride * np.arange(self.recent_records, 0, -1)

    def empty(self) -> dict:
        result = {
            "visual_tokens": np.zeros((self.recent_records, *MEMORY_VISUAL_SHAPE), np.uint8),
            "state": np.zeros((self.recent_records, 14), np.float32),
            "executed_actions": np.zeros((self.recent_records, 14), np.float32),
            "elapsed_env_steps": np.zeros(self.recent_records, np.float32),
            "valid_mask": np.zeros(self.recent_records, bool),
            "source_frame_index": np.full(self.recent_records, -1, np.int64),
        }
        if self.use_pose:
            result["eef_pose"] = np.zeros((self.recent_records, 2, 4, 4), np.float32)
        return result

    def contract(self, current: int, episode=0) -> dict:
        self.indices(current)
        return {
            "history_is_strictly_past": True,
            "future_targets_in_memory": False,
            "decision_stride_env_steps": self.decision_stride,
            "recent_records": self.recent_records,
            "source_frame_index": int(current),
            "episode_index": episode,
            "representation": "raw_dual_arm_abs_qpos_v1",
            "use_pose": self.use_pose,
        }

    def validate(self, record: dict, contract: dict) -> None:
        current = contract.get("source_frame_index", -1)
        self.indices(current)
        if current < 0 or contract.get("history_is_strictly_past") is not True or contract.get("future_targets_in_memory") is not False:
            raise ValueError("Memory must declare strictly past records with no future targets")
        expected = self.contract(current, contract.get("episode_index", 0))
        if any(contract.get(k) != expected[k] for k in ("decision_stride_env_steps", "recent_records", "representation", "use_pose")):
            raise ValueError("Memory contract differs between dataset/client and model")
        template = self.empty()
        if set(record) != set(template):
            raise ValueError(f"Memory fields must be {tuple(template)}")
        for key, value in template.items():
            array = np.asarray(record[key])
            if array.shape != value.shape or not np.isfinite(array).all():
                raise ValueError(f"Memory {key} must be finite {value.shape}")
        valid = np.asarray(record["valid_mask"])
        source = np.asarray(record["source_frame_index"])
        if valid.dtype != np.bool_ or not np.issubdtype(source.dtype, np.integer):
            raise ValueError("Memory mask must be boolean and source indices integer")
        if np.any(source[valid] < 0) or np.any(source[valid] >= current):
            raise ValueError("Memory contains current/future frame information")
        indices = self.indices(current)
        if not np.array_equal(valid, indices >= 0) or not np.array_equal(source, np.where(valid, indices, -1)):
            raise ValueError("Memory must cover the exact strictly-past stride grid")
        elapsed = np.asarray(record["elapsed_env_steps"])
        if not np.array_equal(elapsed, np.where(valid, current - indices, 0)):
            raise ValueError("Memory elapsed steps disagree with source indices")
        for key in ("state", "executed_actions", "visual_tokens", "eef_pose"):
            if key in record and np.any(np.asarray(record[key])[~valid] != 0):
                raise ValueError("Invalid memory slots must be zero")


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite {shape}, got {array.shape}")
    return array.copy()


class RoboDojoMemoryBuffer:
    """Per-episode history of actual per-step execution feedback.

    Append every environment step, even when inference executes a chunk of 16
    commands. The model may then sample any current frame with the same stride
    grid as the dataset. Never append a predicted, unexecuted action chunk.
    """

    def __init__(self, recent_records=16, decision_stride=8, use_pose=False):
        self.spec = RoboDojoMemorySpec(recent_records, decision_stride, use_pose)
        self.records = deque(maxlen=recent_records * decision_stride)
        self.next_source_step = 0

    def reset(self):
        self.records.clear()
        self.next_source_step = 0

    def append(self, state, executed_action, *, source_step: int, pose=None, images=None, visual_tokens=None):
        self.spec.indices(source_step)
        if source_step != self.next_source_step:
            raise ValueError(f"Execution feedback must cover every step in order; expected {self.next_source_step}")
        record = {
            "state": finite_array(state, (14,), "raw state"),
            "executed_actions": finite_array(executed_action, (14,), "executed action"),
            "source_step": int(source_step),
        }
        if visual_tokens is None:
            visual_tokens = (np.stack([image_to_memory_tokens(image) for image in images])
                             if images is not None else np.zeros(MEMORY_VISUAL_SHAPE, np.uint8))
        visual_tokens = np.asarray(visual_tokens)
        if visual_tokens.shape != MEMORY_VISUAL_SHAPE or visual_tokens.dtype != np.uint8:
            raise ValueError(f"visual_tokens must be uint8 {MEMORY_VISUAL_SHAPE}, got {visual_tokens.shape}/{visual_tokens.dtype}")
        record["visual_tokens"] = visual_tokens.copy()
        if self.spec.use_pose:
            record["eef_pose"] = finite_array(pose, (2, 4, 4), "dual-arm FK pose")
        elif pose is not None:
            raise ValueError("Pose feedback is disabled by the memory contract")
        self.records.append(record)
        self.next_source_step += 1

    def build(self, current_step: int) -> dict:
        if current_step != self.next_source_step:
            raise ValueError("History must contain all and only feedback strictly before current step")
        result = self.spec.empty()
        by_step = {r["source_step"]: r for r in self.records}
        for slot, step in enumerate(self.spec.indices(current_step)):
            if step < 0:
                continue
            record = by_step[int(step)]
            for key in ("state", "executed_actions", "visual_tokens", "eef_pose"):
                if key in result:
                    result[key][slot] = record[key]
            result["valid_mask"][slot] = True
            result["source_frame_index"][slot] = step
            result["elapsed_env_steps"][slot] = current_step - step
        return result

    def build_burnin(self, current_step: int) -> dict:
        """Return committed stride records before the recent memory window."""
        current_step = int(current_step)
        cutoff = max(0, current_step - self.spec.decision_stride * self.spec.recent_records)
        records = [r for r in self.records if r["source_step"] < cutoff]
        template = self.spec.empty()
        result = {}
        for key, value in template.items():
            shape = (len(records), *value.shape[1:])
            if records:
                result[key] = np.stack([r[key] if key in r else value[0] for r in records]).astype(value.dtype, copy=False)
            else:
                result[key] = np.zeros(shape, dtype=value.dtype)
        for index, record in enumerate(records):
            result["valid_mask"][index] = True
            result["source_frame_index"][index] = record["source_step"]
            result["elapsed_env_steps"][index] = self.spec.decision_stride
        return result

"""Shared causal LIBERO memory record helpers for training and evaluation."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image


MEMORY_GRID_SIZE = 8
MEMORY_PATCH_ROWS = 2
MEMORY_PATCH_COLS = 4
MEMORY_VISUAL_TOKENS = MEMORY_PATCH_ROWS * MEMORY_PATCH_COLS
MEMORY_PATCH_SIZE = 8
MEMORY_VISUAL_FEATURE_DIM = 3
MEMORY_VISUAL_VIEWS = 2
MEMORY_RECENT_RECORDS = 32
MEMORY_DECISION_STRIDE = 8


def image_to_memory_tokens(image: Any) -> np.ndarray:
    """Encode an RGB frame as eight spatially ordered 8x8 RGB patches.

    This compact, deterministic representation intentionally uses no external
    vision model. The row/column ordering preserves coarse image layout.
    """
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"Expected an RGB image [H,W,3], got {value.shape}")
        if value.dtype != np.uint8:
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError("RGB image must contain finite numeric values")
            value = np.clip(value, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(value, mode="RGB")
    width, height = rgb.size
    patch_width = width // MEMORY_PATCH_COLS
    patch_height = height // MEMORY_PATCH_ROWS
    tokens = np.empty((MEMORY_VISUAL_TOKENS, MEMORY_PATCH_SIZE, MEMORY_PATCH_SIZE, 3), dtype=np.uint8)
    token_index = 0
    for row in range(MEMORY_PATCH_ROWS):
        for col in range(MEMORY_PATCH_COLS):
            crop = rgb.crop(
                (
                    col * patch_width,
                    row * patch_height,
                    (col + 1) * patch_width if col + 1 < MEMORY_PATCH_COLS else width,
                    (row + 1) * patch_height if row + 1 < MEMORY_PATCH_ROWS else height,
                )
            )
            tokens[token_index] = np.asarray(
                crop.resize((MEMORY_PATCH_SIZE, MEMORY_PATCH_SIZE), Image.Resampling.BOX), dtype=np.uint8
            )
            token_index += 1
    return tokens


class LiberoMemoryBuffer:
    """Right-aligned, strictly-past memory window used by LIBERO evaluators."""

    def __init__(
        self,
        recent_records: int = MEMORY_RECENT_RECORDS,
        decision_stride: int = MEMORY_DECISION_STRIDE,
    ) -> None:
        self.recent_records = int(recent_records)
        self.decision_stride = int(decision_stride)
        if self.recent_records < 1 or self.decision_stride < 1:
            raise ValueError("recent_records and decision_stride must be positive")
        self.records: deque[dict[str, Any]] = deque(maxlen=self.recent_records)

    def reset(self) -> None:
        self.records.clear()

    def build(self, current_step: int) -> dict[str, np.ndarray]:
        current_step = int(current_step)
        visual = np.zeros(
            (
                self.recent_records,
                MEMORY_VISUAL_VIEWS,
                MEMORY_VISUAL_TOKENS,
                MEMORY_PATCH_SIZE,
                MEMORY_PATCH_SIZE,
                MEMORY_VISUAL_FEATURE_DIM,
            ),
            dtype=np.uint8,
        )
        state = np.zeros((self.recent_records, 8), dtype=np.float32)
        actions = np.zeros((self.recent_records, 7), dtype=np.float32)
        poses = np.broadcast_to(np.eye(4, dtype=np.float32), (self.recent_records, 4, 4)).copy()
        elapsed = np.zeros((self.recent_records,), dtype=np.float32)
        valid = np.zeros((self.recent_records,), dtype=np.bool_)
        source_indices = np.full((self.recent_records,), -1, dtype=np.int64)

        first_slot = self.recent_records - len(self.records)
        for slot, record in enumerate(self.records, start=first_slot):
            source_step = int(record["source_step"])
            if source_step >= current_step:
                raise ValueError(
                    f"Memory is not strictly past: record step {source_step} >= current step {current_step}"
                )
            visual[slot] = record["visual_tokens"]
            state[slot] = record["state"]
            actions[slot] = record["executed_actions"]
            poses[slot] = record["eef_pose"]
            elapsed[slot] = float(current_step - source_step)
            valid[slot] = True
            source_indices[slot] = source_step

        return {
            "visual_tokens": visual,
            "state": state,
            "executed_actions": actions,
            "eef_pose": poses,
            "elapsed_env_steps": elapsed,
            "valid_mask": valid,
            "source_frame_index": source_indices,
        }

    def append(self, example: dict, executed_action: np.ndarray, source_step: int) -> None:
        source_step = int(source_step)
        if self.records:
            previous_step = int(self.records[-1]["source_step"])
            if source_step - previous_step != self.decision_stride:
                raise ValueError(
                    f"Memory records must be spaced by {self.decision_stride} environment steps; "
                    f"got {previous_step} then {source_step}"
                )
        images = example.get("image")
        if not isinstance(images, (list, tuple)) or len(images) != MEMORY_VISUAL_VIEWS:
            raise ValueError(f"LIBERO memory requires {MEMORY_VISUAL_VIEWS} camera images")
        memory_state = np.asarray(example.get("memory_state"), dtype=np.float32).reshape(-1)
        action = np.asarray(executed_action, dtype=np.float32).reshape(-1)
        geometry = example.get("geometry_current")
        if not isinstance(geometry, dict) or "eef_pose" not in geometry:
            raise ValueError("LIBERO memory requires geometry_current.eef_pose")
        pose = np.asarray(geometry["eef_pose"], dtype=np.float32)
        if memory_state.shape != (8,) or action.shape != (7,) or pose.shape != (4, 4):
            raise ValueError(
                f"Invalid memory source shapes: state={memory_state.shape}, action={action.shape}, pose={pose.shape}"
            )
        if not np.isfinite(memory_state).all() or not np.isfinite(action).all() or not np.isfinite(pose).all():
            raise ValueError("Memory source values must be finite")
        self.records.append(
            {
                "visual_tokens": np.stack([image_to_memory_tokens(image) for image in images]),
                "state": memory_state,
                "executed_actions": action,
                "eef_pose": pose,
                "source_step": source_step,
            }
        )

    def override_latest_action(self, executed_action: np.ndarray, source_step: int) -> None:
        """Replace the predicted action with the action actually sent to the environment."""
        if not self.records or int(self.records[-1]["source_step"]) != int(source_step):
            raise ValueError(f"No memory record exists for source step {source_step}")
        action = np.asarray(executed_action, dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Executed LIBERO action must be finite [7], got {action.shape}")
        self.records[-1]["executed_actions"] = action


__all__ = [
    "LiberoMemoryBuffer",
    "MEMORY_DECISION_STRIDE",
    "MEMORY_GRID_SIZE",
    "MEMORY_PATCH_SIZE",
    "MEMORY_PATCH_COLS",
    "MEMORY_PATCH_ROWS",
    "MEMORY_RECENT_RECORDS",
    "MEMORY_VISUAL_FEATURE_DIM",
    "MEMORY_VISUAL_TOKENS",
    "MEMORY_VISUAL_VIEWS",
    "image_to_memory_tokens",
]

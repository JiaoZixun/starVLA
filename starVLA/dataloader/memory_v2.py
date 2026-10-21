"""Shared stride-8 execution-event lifecycle used by LIBERO train and eval."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image

MEMORY_CAPACITY_EVENTS = 128
MEMORY_STRIDE = 8
ACTION_HORIZON = 8
VISUAL_VIEWS = 2
VISUAL_POOL_GRID = (4, 4)
VISUAL_TOKENS_PER_VIEW = VISUAL_POOL_GRID[0] * VISUAL_POOL_GRID[1]
VISUAL_FEATURE_DIM = 3
VISUAL_IMAGE_SIZE = (224, 224)


def image_to_memory_visual_tokens(image: Any) -> np.ndarray:
    """Convert a training/evaluation RGB view to the shared 4x4 event feature."""
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
    if rgb.size != VISUAL_IMAGE_SIZE:
        rgb = rgb.resize(VISUAL_IMAGE_SIZE, Image.Resampling.BICUBIC)
    pooled = rgb.resize(VISUAL_POOL_GRID, Image.Resampling.BOX)
    return np.asarray(pooled, dtype=np.float32).reshape(VISUAL_TOKENS_PER_VIEW, VISUAL_FEATURE_DIM) / 255.0


def _array(value: Any, shape: tuple[int, ...], name: str, dtype=np.float32) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def pose_in_episode_frame(initial_pose: np.ndarray, pose: np.ndarray) -> np.ndarray:
    initial_pose = _array(initial_pose, (4, 4), "initial_pose")
    pose = _array(pose, (4, 4), "pose")
    return (np.linalg.inv(initial_pose) @ pose).astype(np.float32)


def build_execution_event(
    *,
    start_step: int,
    end_step: int,
    visual_start: Any,
    visual_end: Any,
    state_start: Any,
    state_end: Any,
    executed_actions: Any,
    pose_start_in_episode: Any,
    pose_end_in_episode: Any,
) -> dict[str, Any]:
    """Validate and pack an observed, completed interval; no success labels are inferred."""
    start_step, end_step = int(start_step), int(end_step)
    if start_step < 0 or end_step - start_step != MEMORY_STRIDE:
        raise ValueError(f"Execution event must span exactly {MEMORY_STRIDE} steps: {start_step}->{end_step}")
    visual_start = np.asarray(visual_start)
    visual_end = np.asarray(visual_end)
    expected_visual_shape = (VISUAL_VIEWS, VISUAL_TOKENS_PER_VIEW, VISUAL_FEATURE_DIM)
    if visual_start.shape != expected_visual_shape:
        raise ValueError(f"visual_start must have shape {expected_visual_shape}, got {visual_start.shape}")
    if visual_end.shape != visual_start.shape:
        raise ValueError(f"visual_end shape {visual_end.shape} != visual_start shape {visual_start.shape}")
    if not np.isfinite(visual_start).all() or not np.isfinite(visual_end).all():
        raise ValueError("Visual features must be finite")
    if visual_start.min(initial=0) < 0 or visual_start.max(initial=0) > 1 or visual_end.min(initial=0) < 0 or visual_end.max(initial=0) > 1:
        raise ValueError("Memory visual RGB features must be scaled to [0,1]")
    return {
        "start_step": start_step,
        "end_step": end_step,
        "visual_start": visual_start,
        "visual_end": visual_end,
        "state_start": _array(state_start, (8,), "state_start"),
        "state_end": _array(state_end, (8,), "state_end"),
        "executed_actions": _array(executed_actions, (ACTION_HORIZON, 7), "executed_actions"),
        "pose_start_in_episode": _array(pose_start_in_episode, (4, 4), "pose_start_in_episode"),
        "pose_end_in_episode": _array(pose_end_in_episode, (4, 4), "pose_end_in_episode"),
        "elapsed_steps": float(end_step - start_step),
    }


class EpisodeEventHistory:
    """Keep only the latest completed events and expose a strictly causal prefix."""

    def __init__(self, capacity: int = MEMORY_CAPACITY_EVENTS, stride: int = MEMORY_STRIDE):
        if capacity < 1 or stride != ACTION_HORIZON:
            raise ValueError("MEMORY-v2 requires positive capacity and an 8-step action/event interval")
        self.capacity = int(capacity)
        self.stride = int(stride)
        self._events: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    def reset(self) -> None:
        self._events.clear()

    def append(self, event: dict[str, Any]) -> None:
        start, end = int(event["start_step"]), int(event["end_step"])
        if end - start != self.stride:
            raise ValueError(f"Expected a complete {self.stride}-step event, got {start}->{end}")
        if self._events and start != int(self._events[-1]["end_step"]):
            raise ValueError("Events must be appended in order without gaps or episode mixing")
        self._events.append(event)

    def available(self, query_step: int) -> list[dict[str, Any]]:
        query_step = int(query_step)
        eligible = [
            event
            for event in self._events
            if int(event["start_step"]) < query_step and int(event["end_step"]) <= query_step
        ]
        return eligible[-self.capacity :]

    def __len__(self) -> int:
        return len(self._events)


class LiberoExecutionEventBuffer:
    """Build the same completed 8-step events online that training builds offline."""

    def __init__(self, capacity: int = MEMORY_CAPACITY_EVENTS):
        self.history = EpisodeEventHistory(capacity=capacity)
        self.pending: dict[str, Any] | None = None
        self.executed_actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.history.reset()
        self.pending = None
        self.executed_actions.clear()

    @staticmethod
    def _pose(example: dict[str, Any]) -> np.ndarray:
        geometry = example.get("geometry_current")
        if not isinstance(geometry, dict) or "eef_pose" not in geometry:
            raise ValueError("LIBERO execution memory requires geometry_current.eef_pose")
        return _array(geometry["eef_pose"], (4, 4), "geometry_current.eef_pose")

    @staticmethod
    def _state(example: dict[str, Any]) -> np.ndarray:
        # ``state`` would be appended to the Qwen prompt by QwenSpatialOFT; the
        # memory path reads its own key so the frozen Spatial prompt is unchanged.
        if "memory_state" not in example:
            raise ValueError("LIBERO execution memory requires memory_state")
        return _array(np.asarray(example["memory_state"]).reshape(-1), (8,), "memory_state")

    def begin_query(self, example: dict[str, Any], env_step: int) -> list[dict[str, Any]]:
        """Close the previous interval, then stage this observation for the next one."""
        env_step = int(env_step)
        if env_step < 0 or env_step % MEMORY_STRIDE:
            raise ValueError(f"LIBERO memory queries must align to the {MEMORY_STRIDE}-step stride: {env_step}")
        visual = np.stack([image_to_memory_visual_tokens(image) for image in example["image"]])
        state = self._state(example)
        pose = self._pose(example)
        if self.pending is not None:
            start = int(self.pending["step"])
            if env_step - start != MEMORY_STRIDE:
                raise ValueError(f"Execution event must span {MEMORY_STRIDE} steps: {start}->{env_step}")
            if len(self.executed_actions) != ACTION_HORIZON:
                raise ValueError(
                    f"Expected {ACTION_HORIZON} executed actions before query {env_step}, "
                    f"got {len(self.executed_actions)}"
                )
            event = build_execution_event(
                start_step=start,
                end_step=env_step,
                visual_start=self.pending["visual"],
                visual_end=visual,
                state_start=self.pending["state"],
                state_end=state,
                executed_actions=np.stack(self.executed_actions),
                pose_start_in_episode=self.pending["pose_in_episode"],
                pose_end_in_episode=pose_in_episode_frame(self.pending["initial_pose"], pose),
            )
            self.history.append(event)
            self.executed_actions.clear()
        elif env_step != 0:
            raise ValueError(f"The first LIBERO memory query must start at step 0, got {env_step}")

        initial_pose = self.pending["initial_pose"] if self.pending is not None else pose.copy()
        pose_in_episode = pose_in_episode_frame(initial_pose, pose)
        self.pending = {
            "step": env_step,
            "visual": visual,
            "state": state,
            "pose_in_episode": pose_in_episode,
            "initial_pose": initial_pose,
        }
        return self.history.available(env_step)

    def record_executed_action(self, action: Any, env_step: int) -> None:
        """Record exactly one finite 7-D command after LIBERO accepted the step."""
        if self.pending is None:
            raise RuntimeError("Cannot record an action before the first memory query")
        env_step = int(env_step)
        expected = int(self.pending["step"]) + len(self.executed_actions)
        if env_step != expected:
            raise ValueError(f"Executed action step out of order: expected {expected}, got {env_step}")
        value = _array(action, (7,), "executed_action")
        self.executed_actions.append(value.copy())


__all__ = [
    "ACTION_HORIZON",
    "EpisodeEventHistory",
    "LiberoExecutionEventBuffer",
    "MEMORY_CAPACITY_EVENTS",
    "MEMORY_STRIDE",
    "VISUAL_POOL_GRID",
    "VISUAL_FEATURE_DIM",
    "VISUAL_IMAGE_SIZE",
    "VISUAL_TOKENS_PER_VIEW",
    "VISUAL_VIEWS",
    "build_execution_event",
    "image_to_memory_visual_tokens",
    "pose_in_episode_frame",
]

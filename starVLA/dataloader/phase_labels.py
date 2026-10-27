"""Manipulation-phase labels shared by the robot (LIBERO) and human (DexYCB) domains.

Phases (index = class id, matching ``QwenSpatialOFT.PHASES``):
0 approach  - not holding, and not about to grasp or just released
1 grasp     - not holding, within ``window`` frames before a hold starts
2 transport - holding
3 release   - not holding, within ``window`` frames after a hold ends
"""

from __future__ import annotations

import numpy as np

APPROACH, GRASP, TRANSPORT, RELEASE = 0, 1, 2, 3


def phases_from_holding(holding: np.ndarray, window: int) -> np.ndarray:
    """Per-frame phase ids [T] from a per-frame holding mask [T]."""
    holding = np.asarray(holding, dtype=np.bool_)
    if holding.ndim != 1:
        raise ValueError(f"holding must be 1-D, got shape {holding.shape}")
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    count = len(holding)
    phase = np.full(count, APPROACH, dtype=np.int64)
    phase[holding] = TRANSPORT
    edges = np.diff(holding.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1  # first holding frame
    ends = np.flatnonzero(edges == -1) + 1  # first free frame after holding
    free = ~holding
    for end in ends:
        span = slice(end, min(count, end + window))
        phase[span] = np.where(free[span], RELEASE, phase[span])
    # Grasp wins over release when a regrasp follows within the window.
    for start in starts:
        span = slice(max(0, start - window), start)
        phase[span] = np.where(free[span], GRASP, phase[span])
    return phase


def robot_holding_from_gripper_action(gripper_action: np.ndarray) -> np.ndarray:
    """LIBERO LeRobot gripper command: 1 = open, 0 = close; closed command = holding."""
    gripper_action = np.asarray(gripper_action, dtype=np.float64)
    if not np.isin(gripper_action, (0.0, 1.0)).all():
        raise ValueError("Expected binary LIBERO gripper actions in {0, 1}")
    return gripper_action < 0.5


def slot_phases(phase: np.ndarray, future_indices) -> tuple[np.ndarray, np.ndarray]:
    """Phase at each future frame index; out-of-range slots are invalid (label 0)."""
    future = np.asarray(future_indices, dtype=np.int64)
    valid = (future >= 0) & (future < len(phase))
    labels = np.zeros(len(future), dtype=np.int64)
    labels[valid] = phase[future[valid]]
    return labels, valid

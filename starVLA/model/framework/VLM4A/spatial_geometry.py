"""Float32 SE(3) utilities shared by spatial OFT training and inference."""

from __future__ import annotations

import numpy as np
import torch


def axis_angle_to_matrix_numpy(axis_angle: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to SO(3) matrices using Rodrigues' formula."""
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) axis-angle vectors, got {value.shape}")
    theta = np.linalg.norm(value, axis=-1)
    x, y, z = np.moveaxis(value, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(*value.shape[:-1], 3, 3)
    theta2 = theta * theta
    a = np.divide(np.sin(theta), theta, out=np.ones_like(theta), where=theta > 1e-8)
    b = np.divide(1.0 - np.cos(theta), theta2, out=np.full_like(theta, 0.5), where=theta > 1e-8)
    eye = np.broadcast_to(np.eye(3), (*value.shape[:-1], 3, 3))
    return (eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)).astype(np.float32)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors to SO(3) matrices in float32."""
    value = axis_angle.float()
    if value.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) axis-angle vectors, got {tuple(value.shape)}")
    theta2 = (value * value).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    a = torch.where(theta2 > 1e-8, torch.sin(theta) / theta, 1.0 - theta2 / 6.0 + theta2.square() / 120.0)
    b = torch.where(
        theta2 > 1e-8,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-16),
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
    )
    x, y, z = value.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*value.shape[:-1], 3, 3)
    eye = torch.eye(3, device=value.device, dtype=torch.float32).expand(*value.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def pose_from_position_axis_angle(position: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    """Build homogeneous transforms with p_world = R p_local + t."""
    position = np.asarray(position, dtype=np.float32)
    if position.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) positions, got {position.shape}")
    rotation = axis_angle_to_matrix_numpy(axis_angle)
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (*position.shape[:-1], 4, 4)).copy()
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = position
    return pose


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """PyTorch3D-compatible 6D rotation encoding (first two matrix rows)."""
    return matrix.float()[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """PyTorch3D-compatible Gram-Schmidt 6D rotation decoding."""
    value = rotation_6d.float()
    if value.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6) 6D rotations, got {tuple(value.shape)}")
    a1, a2 = value[..., :3], value[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def relative_pose(current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Return inverse(current) @ future for column-vector SE(3) transforms."""
    current = current.float()
    future = future.float()
    current_rotation = current[..., :3, :3]
    future_rotation = future[..., :3, :3]
    current_position = current[..., :3, 3]
    future_position = future[..., :3, 3]
    relative_rotation = current_rotation.transpose(-1, -2) @ future_rotation
    relative_position = (current_rotation.transpose(-1, -2) @ (future_position - current_position).unsqueeze(-1)).squeeze(-1)
    result = torch.eye(4, device=current.device, dtype=torch.float32).expand(*relative_position.shape[:-1], 4, 4).clone()
    result[..., :3, :3] = relative_rotation
    result[..., :3, 3] = relative_position
    return result


def relative_pose_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Translation Huber and rotation-matrix losses with safe all-invalid masks."""
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    if not torch.any(valid):
        zero = prediction.float().sum() * 0.0
        return zero, zero
    pred = prediction.float()[valid]
    truth = target.float()[valid]
    translation_error = (pred[..., :3, 3] - truth[..., :3, 3]) / 0.1
    translation = torch.nn.functional.huber_loss(translation_error, torch.zeros_like(translation_error), delta=1.0)
    rotation_error = pred[..., :3, :3] - truth[..., :3, :3]
    rotation = rotation_error.square().sum(dim=(-2, -1)).mean() / 9.0
    return translation, rotation

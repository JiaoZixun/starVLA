"""Small, dependency-light SE(3) helpers used by FastVLA spatial training."""
from __future__ import annotations

import math
import torch


def inverse_se3(t: torch.Tensor) -> torch.Tensor:
    r, p = t[..., :3, :3], t[..., :3, 3]
    out = torch.zeros_like(t)
    out[..., :3, :3] = r.transpose(-1, -2)
    out[..., :3, 3] = -(r.transpose(-1, -2) @ p.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def relative_se3(current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    return inverse_se3(current) @ future


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(r: torch.Tensor) -> torch.Tensor:
    return r[..., :, :2].transpose(-1, -2).reshape(*r.shape[:-2], 6)


def so3_geodesic(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    rel = r1.transpose(-1, -2) @ r2
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6))


def pose_to_features(t: torch.Tensor) -> torch.Tensor:
    """Translation + first two rotation columns, preserving explicit geometry."""
    return torch.cat((t[..., :3, 3], matrix_to_rot6d(t[..., :3, :3])), dim=-1)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(value.dtype)
    denom = m.sum().clamp_min(1.0)
    return (value * m).sum() / denom

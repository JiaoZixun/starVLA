"""Action-query bridge over multi-layer Qwen features and measured proprioception."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenSpatialOFT import WaypointDecoderBlock
from starVLA.model.framework.VLM4A.spatial_geometry import matrix_to_rotation_6d

# Token types seen by the bridge: text, primary (agent-view) image, wrist image,
# and the proprioception token.
TOKEN_TEXT, TOKEN_PRIMARY, TOKEN_WRIST, TOKEN_PROPRIO = range(4)
PROPRIO_DIM = 11  # eef position (3) + eef rotation 6D (3x2) + gripper qpos (2)


@dataclass(frozen=True)
class BridgeConfig:
    hidden_layers: tuple[int, ...] = (6, 12, 18, 24)
    width: int = 256
    heads: int = 4
    ffn_width: int = 1024
    layers: int = 2
    proprio_mean: tuple[float, ...] = (0.0,) * PROPRIO_DIM
    proprio_std: tuple[float, ...] = (1.0,) * PROPRIO_DIM

    @classmethod
    def from_dict(cls, values: dict) -> "BridgeConfig":
        config = cls(
            hidden_layers=tuple(int(v) for v in values.get("hidden_layers", cls.hidden_layers)),
            width=int(values.get("width", cls.width)),
            heads=int(values.get("heads", cls.heads)),
            ffn_width=int(values.get("ffn_width", cls.ffn_width)),
            layers=int(values.get("layers", cls.layers)),
            proprio_mean=tuple(float(v) for v in values.get("proprio_mean", cls.proprio_mean)),
            proprio_std=tuple(float(v) for v in values.get("proprio_std", cls.proprio_std)),
        )
        if not config.hidden_layers or len(set(config.hidden_layers)) != len(config.hidden_layers):
            raise ValueError(f"bridge.hidden_layers must be unique and non-empty, got {config.hidden_layers}")
        if len(config.proprio_mean) != PROPRIO_DIM or len(config.proprio_std) != PROPRIO_DIM:
            raise ValueError(f"bridge.proprio_mean/std must have {PROPRIO_DIM} values")
        if min(config.proprio_std) <= 0:
            raise ValueError("bridge.proprio_std must be positive")
        return config


def proprio_features(eef_pose: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
    """Measured proprioception from the same inputs train and eval already send.

    Rotation uses the continuous 6D form of the measured TCP pose instead of the
    raw axis-angle, which flips sign near pi.
    """
    if eef_pose.shape[-2:] != (4, 4) or memory_state.shape[-1] != 8:
        raise ValueError(
            f"Expected eef_pose [B,4,4] and memory_state [B,8], got {tuple(eef_pose.shape)}/{tuple(memory_state.shape)}"
        )
    return torch.cat(
        (
            eef_pose[..., :3, 3].float(),
            matrix_to_rotation_6d(eef_pose[..., :3, :3].float()),
            memory_state[..., 6:8].float(),
        ),
        dim=-1,
    )


def bridge_token_types(input_ids: torch.Tensor, image_token_id: int, views: int = 2) -> torch.Tensor:
    """Label every sequence position as text, primary-view, or wrist-view image.

    Image tokens appear in camera order, and both views share one resolution, so
    each view owns an equal contiguous share of a sample's image tokens.
    """
    if views != 2:
        raise ValueError("The bridge expects exactly the primary and wrist views")
    is_image = input_ids == int(image_token_id)
    counts = is_image.sum(dim=1)
    if torch.any(counts % views) or torch.any(counts == 0):
        raise ValueError(f"Per-sample image token counts {counts.tolist()} do not split into {views} views")
    order = torch.cumsum(is_image.long(), dim=1) - 1
    wrist = is_image & (order >= (counts // views)[:, None])
    types = torch.full_like(input_ids, TOKEN_TEXT)
    types = torch.where(is_image, torch.full_like(types, TOKEN_PRIMARY), types)
    return torch.where(wrist, torch.full_like(types, TOKEN_WRIST), types)


class ProprioLayerBridge(nn.Module):
    """Cross-attend action queries to several Qwen layers plus a proprio token.

    The zero-initialized output keeps the reused Spatial checkpoint's action
    path unchanged at step 0.
    """

    def __init__(self, vlm_width: int, config: BridgeConfig):
        super().__init__()
        self.config = config
        width = config.width
        layer_count = len(config.hidden_layers)
        # Intermediate residual streams are not normalized; per-layer norms put
        # every layer on a common scale before the shared projection.
        self.layer_norms = nn.ModuleList(nn.LayerNorm(vlm_width) for _ in range(layer_count))
        self.feature_projection = nn.Linear(vlm_width, width)
        self.layer_embedding = nn.Embedding(layer_count, width)
        self.type_embedding = nn.Embedding(4, width)
        self.register_buffer("proprio_mean", torch.tensor(config.proprio_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("proprio_std", torch.tensor(config.proprio_std, dtype=torch.float32), persistent=False)
        self.proprio_projection = nn.Sequential(nn.Linear(PROPRIO_DIM, width), nn.GELU(), nn.Linear(width, width))
        self.query_projection = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.blocks = nn.ModuleList(
            WaypointDecoderBlock(width, config.heads, config.ffn_width) for _ in range(config.layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, vlm_width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action_queries: torch.Tensor,
        layer_features: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor | None,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, layer_count, tokens, _ = layer_features.shape
        if layer_count != len(self.layer_norms):
            raise ValueError(f"Expected {len(self.layer_norms)} bridge layers, got {layer_count}")
        if token_types.shape != (batch, tokens) or proprio.shape != (batch, PROPRIO_DIM):
            raise ValueError("Bridge token types/proprioception do not match the feature batch")
        type_embedding = self.type_embedding(token_types)
        keys = torch.cat(
            [
                self.feature_projection(norm(layer_features[:, index]))
                + self.layer_embedding.weight[index]
                + type_embedding
                for index, norm in enumerate(self.layer_norms)
            ],
            dim=1,
        )
        normalized = (proprio.float() - self.proprio_mean) / self.proprio_std
        proprio_token = self.proprio_projection(normalized.to(keys.dtype)) + self.type_embedding.weight[TOKEN_PROPRIO]
        keys = torch.cat((keys, proprio_token.unsqueeze(1)), dim=1)
        padding = None
        if attention_mask is not None:
            valid = attention_mask.to(torch.bool).repeat(1, layer_count)
            padding = ~torch.cat((valid, valid.new_ones((batch, 1))), dim=1)
        query = self.query_projection(action_queries)
        for block in self.blocks:
            query = block(query, keys, padding)
        residual = self.output_projection(self.output_norm(query))
        return action_queries + residual, residual


__all__ = [
    "BridgeConfig",
    "PROPRIO_DIM",
    "ProprioLayerBridge",
    "bridge_token_types",
    "proprio_features",
]

"""Train/eval shared causal execution-event encoder and memory reader."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.spatial_geometry import matrix_to_rotation_6d


@dataclass(frozen=True)
class MemoryConfig:
    width: int = 256
    heads: int = 4
    ffn_width: int = 1024
    event_encoder_layers: int = 1
    reader_layers: int = 2
    event_tokens: int = 8
    summary_tokens: int = 1
    capacity_events: int = 128
    visual_pool_grid: tuple[int, int] = (4, 4)
    action_horizon: int = 8
    dropout: float = 0.0
    # Training-only regularizers against copying the previous action chunk.
    history_dropout: float = 0.0
    event_action_dropout: float = 0.0

    @property
    def tokens_per_event(self) -> int:
        return self.event_tokens + self.summary_tokens


class EventEncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_width, width)
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        update, _ = self.cross_attention(
            self.query_norm(query), self.context_norm(context), self.context_norm(context), need_weights=False
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class ExecutionEventEncoder(nn.Module):
    """Encode one completed interval from shared Qwen visual tokens and observed actions."""

    def __init__(self, visual_dim: int, config: MemoryConfig, visual_input_norm: bool = False):
        super().__init__()
        self.config = config
        width = config.width
        visual_tokens = 2 * config.visual_pool_grid[0] * config.visual_pool_grid[1]
        # Unit-scale RGB summaries need no normalization; high-dimensional Qwen
        # features are normalized per frame before the start/end/delta concat.
        self.visual_input_norm = nn.LayerNorm(visual_dim) if visual_input_norm else None
        self.visual_projection = nn.Linear(3 * visual_dim, width)
        self.visual_position = nn.Parameter(torch.zeros(1, visual_tokens, width))
        self.view_embedding = nn.Embedding(2, width)
        self.action_projection = nn.Linear(7, width)
        self.action_position = nn.Embedding(config.action_horizon, width)
        # Replaces the executed-action context of events whose actions were
        # dropped during training; never used at inference.
        self.masked_action_token = nn.Parameter(torch.zeros(1, 1, width))
        self.state_projection = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.pose_projection = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
        self.delta_pose_projection = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.elapsed_projection = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.event_queries = nn.Parameter(torch.randn(1, config.tokens_per_event, width) * 0.02)
        self.encoder = nn.ModuleList(
            EventEncoderBlock(width, config.heads, config.ffn_width, config.dropout)
            for _ in range(config.event_encoder_layers)
        )

    def forward(
        self,
        visual_start: torch.Tensor,
        visual_end: torch.Tensor,
        states_start: torch.Tensor,
        states_end: torch.Tensor,
        actions: torch.Tensor,
        pose_start_in_episode: torch.Tensor,
        pose_end_in_episode: torch.Tensor,
        elapsed_steps: torch.Tensor,
        action_dropped: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, views, patches, dim = visual_start.shape
        expected_patches = self.config.visual_pool_grid[0] * self.config.visual_pool_grid[1]
        if views != 2 or patches != expected_patches or visual_end.shape != visual_start.shape:
            raise ValueError(
                f"Expected paired visual tokens [B,2,{expected_patches},D], got "
                f"{tuple(visual_start.shape)} and {tuple(visual_end.shape)}"
            )
        if actions.shape != (batch, self.config.action_horizon, 7):
            raise ValueError(
                f"Completed event actions must be [B,{self.config.action_horizon},7], got {tuple(actions.shape)}"
            )
        if self.visual_input_norm is not None:
            visual_start = self.visual_input_norm(visual_start.float())
            visual_end = self.visual_input_norm(visual_end.float())
        visual_delta = visual_end - visual_start
        visual = torch.cat((visual_start, visual_end, visual_delta), dim=-1).reshape(batch, 2 * patches, 3 * dim)
        view_ids = torch.arange(views, device=visual.device).repeat_interleave(patches)
        visual_context = self.visual_projection(visual) + self.visual_position + self.view_embedding(view_ids)[None]

        action_ids = torch.arange(self.config.action_horizon, device=actions.device)
        action_context = self.action_projection(actions.float()) + self.action_position(action_ids)[None]
        if action_dropped is not None:
            masked = self.masked_action_token + self.action_position(action_ids)[None]
            action_context = torch.where(
                action_dropped.view(batch, 1, 1), masked.to(action_context.dtype), action_context
            )

        relative = torch.linalg.inv(pose_start_in_episode.float()) @ pose_end_in_episode.float()
        pose_features = torch.cat(
            (
                pose_start_in_episode[..., :3, 3],
                matrix_to_rotation_6d(pose_start_in_episode[..., :3, :3]),
                pose_end_in_episode[..., :3, 3],
                matrix_to_rotation_6d(pose_end_in_episode[..., :3, :3]),
            ),
            dim=-1,
        )
        delta_features = torch.cat(
            (relative[..., :3, 3], matrix_to_rotation_6d(relative[..., :3, :3])), dim=-1
        )
        state_context = torch.stack(
            (self.state_projection(states_start.float()), self.state_projection(states_end.float())), dim=1
        )
        pose_context = self.pose_projection(pose_features).unsqueeze(1)
        delta_context = self.delta_pose_projection(delta_features).unsqueeze(1)
        elapsed_context = self.elapsed_projection(elapsed_steps.float().reshape(batch, 1, 1))
        context = torch.cat(
            (visual_context, action_context, state_context, pose_context, delta_context, elapsed_context), dim=1
        )
        query = self.event_queries.expand(batch, -1, -1)
        for block in self.encoder:
            query = block(query, context)
        return query


class MemoryReaderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_width, width)
        )

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, disallow_mask: torch.Tensor
    ) -> torch.Tensor:
        # MultiheadAttention uses True for disallowed positions.  Null memory is
        # always allowed, so no query ever has an all-masked softmax row.
        batch, queries, _ = query.shape
        heads = self.cross_attention.num_heads
        mask = disallow_mask[:, None].expand(batch, heads, queries, -1).reshape(batch * heads, queries, -1)
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross_attention(
            self.query_norm(query), normalized_memory, normalized_memory,
            attn_mask=mask, need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class CausalMemoryReader(nn.Module):
    """Read per-query histories using event-level causal masks and a null token."""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.token_type_embedding = nn.Embedding(config.tokens_per_event, config.width)
        self.null_memory = nn.Parameter(torch.zeros(1, 1, config.width))
        self.layers = nn.ModuleList(
            MemoryReaderBlock(config.width, config.heads, config.ffn_width, config.dropout)
            for _ in range(config.reader_layers)
        )

    def forward(
        self,
        queries: torch.Tensor,
        event_tokens: torch.Tensor,
        event_valid: torch.Tensor,
        query_steps: torch.Tensor,
        event_start_steps: torch.Tensor,
        event_end_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = queries.shape
        events = event_tokens.shape[1]
        if event_tokens.shape[2] != self.config.tokens_per_event:
            raise ValueError("Event token count differs from the memory configuration")
        allowed = (
            event_valid[:, None, :]
            & (event_start_steps[:, None, :] < query_steps[:, :, None])
            & (event_end_steps[:, None, :] <= query_steps[:, :, None])
        )
        # Keep the most recent capacity independently for every query.
        if events > self.config.capacity_events:
            ranks = torch.arange(events, device=queries.device)[None, None, :].expand(batch, query_count, -1)
            eligible_rank = torch.where(allowed, ranks, torch.full_like(ranks, -1))
            last_rank = eligible_rank.max(dim=-1, keepdim=True).values
            allowed = allowed & (ranks > last_rank - self.config.capacity_events)

        if events:
            type_ids = torch.arange(self.config.tokens_per_event, device=queries.device)
            memory = event_tokens + self.token_type_embedding(type_ids)[None, None]
            memory = memory.reshape(batch, events * self.config.tokens_per_event, -1)
            disallowed_events = ~allowed.repeat_interleave(self.config.tokens_per_event, dim=-1)
        else:
            memory = queries.new_zeros((batch, 0, self.config.width))
            disallowed_events = torch.ones((batch, query_count, 0), device=queries.device, dtype=torch.bool)
        null_memory = self.null_memory.expand(batch, -1, -1).to(dtype=queries.dtype)
        memory = torch.cat((memory.to(dtype=queries.dtype), null_memory), dim=1)
        disallow_mask = torch.cat(
            (disallowed_events, torch.zeros((batch, query_count, 1), device=queries.device, dtype=torch.bool)), dim=-1
        )
        context = queries
        for layer in self.layers:
            context = layer(context, memory, disallow_mask)
        has_history = allowed.any(dim=-1, keepdim=True)
        return context * has_history.to(context.dtype), allowed


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Stable SE(3) exponential for twist vectors ordered as translation, rotation."""
    twist = twist.float()
    if twist.shape[-1] != 6:
        raise ValueError(f"Expected (...,6) twists, got {tuple(twist.shape)}")
    v, omega = twist[..., :3], twist[..., 3:]
    theta2 = omega.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    a = torch.where(theta2 > 1e-8, torch.sin(theta) / theta, 1.0 - theta2 / 6.0 + theta2.square() / 120.0)
    b = torch.where(
        theta2 > 1e-8,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-16),
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
    )
    c = torch.where(
        theta2 > 1e-8,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(1e-16),
        1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
    )
    x, y, z = omega.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*twist.shape[:-1], 3, 3)
    skew2 = skew @ skew
    eye3 = torch.eye(3, dtype=torch.float32, device=twist.device).expand(*twist.shape[:-1], 3, 3)
    rotation = eye3 + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * skew2
    v_matrix = eye3 + b.unsqueeze(-1) * skew + c.unsqueeze(-1) * skew2
    translation = (v_matrix @ v.unsqueeze(-1)).squeeze(-1)
    result = torch.eye(4, dtype=torch.float32, device=twist.device).expand(*twist.shape[:-1], 4, 4).clone()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    return result


__all__ = [
    "CausalMemoryReader",
    "ExecutionEventEncoder",
    "MemoryConfig",
    "se3_exp",
]

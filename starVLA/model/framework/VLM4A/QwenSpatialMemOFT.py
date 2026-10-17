"""Causal memory extension for the dual-arm RoboDojo spatial OFT model.

Memory inputs are a fixed window of source rows strictly older than the
current observation.  The model never receives future action/pose labels as
inputs; those remain available only to the auxiliary spatial losses.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from starVLA.model.framework.VLM4A.QwenSpatialOFT import QwenSpatialOFT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.utils.robodojo_memory import (
    MEMORY_VISUAL_SHAPE, RoboDojoMemorySpec,
)


class CausalExecutionMemory(nn.Module):
    """Encode a bounded causal history and retrieve it for action queries."""

    def __init__(
        self,
        hidden_size: int,
        state_dim: int = 14,
        action_dim: int = 14,
        pose_dim: int = 0,
        width: int = 256,
        heads: int = 4,
        layers: int = 2,
        slots: int = 8,
        detach_every: int = 8,
        record_dropout: float = 0.0,
    ):
        super().__init__()
        self.width = int(width)
        self.state_dim, self.action_dim, self.pose_dim = int(state_dim), int(action_dim), int(pose_dim)
        self.slots = int(slots)
        self.detach_every = max(1, int(detach_every))
        self.record_dropout = float(record_dropout)
        # Flatten all camera views and patches into one shared record feature.
        visual_dim = int(np.prod(MEMORY_VISUAL_SHAPE))
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, self.width), nn.GELU(), nn.Linear(self.width, self.width)
        )
        self.record_projection = nn.Sequential(
            nn.LayerNorm(self.state_dim + self.action_dim + self.pose_dim + 1),
            nn.Linear(self.state_dim + self.action_dim + self.pose_dim + 1, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.record_norm = nn.LayerNorm(self.width)
        self.slot_seed = nn.Parameter(torch.empty(self.slots, self.width))
        nn.init.normal_(self.slot_seed, mean=0.0, std=0.02)
        self.slot_update = nn.GRUCell(self.width, self.width)
        self.retrieval_query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.width))
        self.retrieval_layers = nn.ModuleList(
            [nn.MultiheadAttention(self.width, heads, batch_first=True) for _ in range(int(layers))]
        )
        self.retrieval_norms = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(int(layers))])
        self.output_projection = nn.Linear(self.width, hidden_size)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size + self.width + 1),
            nn.Linear(hidden_size + self.width + 1, self.width),
            nn.Sigmoid(),
        )
        # Preserve the spatial checkpoint's action path at initialization.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _record_features(self, records: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        visual = torch.as_tensor(records["visual_tokens"], device=device)
        if visual.ndim != 7 or tuple(visual.shape[-5:]) != MEMORY_VISUAL_SHAPE:
            raise ValueError(f"Memory visual tokens must be [B,R,{MEMORY_VISUAL_SHAPE}], got {tuple(visual.shape)}")
        visual = self.visual_projection(visual.float().div(255.0).flatten(start_dim=2))
        state = torch.as_tensor(records["state"], device=device, dtype=torch.float32)
        action = torch.as_tensor(records["executed_actions"], device=device, dtype=torch.float32)
        elapsed = torch.as_tensor(records["elapsed_env_steps"], device=device, dtype=torch.float32).unsqueeze(-1)
        values = [state, action, elapsed / 128.0]
        if self.pose_dim:
            pose = torch.as_tensor(records["eef_pose"], device=device, dtype=torch.float32)
            if pose.shape != (*state.shape[:2], 2, 4, 4):
                raise ValueError("Pose memory must have shape [B,R,2,4,4]")
            values.insert(2, pose.flatten(start_dim=2))
        if state.ndim != 3 or state.shape[-1] != self.state_dim or action.shape != (*state.shape[:2], self.action_dim):
            raise ValueError("Memory requires batched dual-arm state/action [B,R,14]")
        # Serving can cast the entire framework to BF16, including LayerNorm
        # parameters. Match that dtype before normalization; training keeps
        # FP32 master parameters and still uses autocast for the linear/GRU path.
        values = torch.cat(values, dim=-1).to(self.record_projection[0].weight.dtype)
        record = self.record_norm(self.record_projection(values) + visual)
        valid = torch.as_tensor(records["valid_mask"], device=device, dtype=torch.bool)
        return record, valid

    def _update_slots(self, slots: torch.Tensor, records: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch = records.shape[0]
        for index in range(records.shape[1]):
            active = valid[:, index].view(batch, 1, 1)
            inputs = records[:, index].unsqueeze(1).expand(-1, slots.shape[1], -1)
            updated = self.slot_update(inputs.reshape(-1, self.width), slots.reshape(-1, self.width)).view_as(slots)
            slots = torch.where(active, updated, slots)
        return slots

    def forward(
        self,
        action_queries: torch.Tensor,
        records: dict[str, Any] | None,
        burnin_records: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = action_queries.shape[0]
        if records is None:
            return action_queries, action_queries.new_zeros((batch, self.width)), action_queries.new_zeros(())
        record, valid = self._record_features(records, action_queries.device)
        if record.shape[0] != batch:
            raise ValueError(f"Memory batch mismatch: records={record.shape[0]} queries={batch}")
        slots = self.slot_seed.unsqueeze(0).expand(batch, -1, -1)
        if self.training and self.record_dropout:
            valid = valid & (torch.rand_like(valid.float()) >= self.record_dropout)
        if burnin_records is not None:
            with torch.no_grad():
                burn, burn_valid = self._record_features(burnin_records, action_queries.device)
                slots = self._update_slots(slots, burn, burn_valid)
            slots = slots.detach()
        for start in range(0, record.shape[1], self.detach_every):
            slots = self._update_slots(slots, record[:, start:start + self.detach_every], valid[:, start:start + self.detach_every])
            if start + self.detach_every < record.shape[1]:
                slots = slots.detach()
        retrieval = torch.cat((record, slots), dim=1)
        retrieval_valid = torch.cat(
            (valid, torch.ones((batch, slots.shape[1]), device=valid.device, dtype=torch.bool)), dim=1
        )
        query = self.retrieval_query(action_queries)
        for attention, norm in zip(self.retrieval_layers, self.retrieval_norms):
            update, _ = attention(query, retrieval, retrieval, key_padding_mask=~retrieval_valid, need_weights=False)
            query = norm(query + update)
        summary = query.mean(dim=1)
        # One gate per query controls the residual. No labels enter this path.
        elapsed_summary = torch.as_tensor(records["elapsed_env_steps"], device=action_queries.device, dtype=torch.float32)
        elapsed_summary = (elapsed_summary * valid.float()).sum(1, keepdim=True) / valid.float().sum(1, keepdim=True).clamp_min(1)
        gate = self.gate(torch.cat((action_queries, query, elapsed_summary[:, None].expand(-1, query.shape[1], -1) / 128.0), dim=-1))
        conditioned = action_queries + self.output_projection(gate * query)
        return conditioned, summary, gate.detach().float().mean()


@FRAMEWORK_REGISTRY.register("QwenSpatialMemOFT")
class QwenSpatialMemOFT(QwenSpatialOFT):
    """QwenSpatialOFT with a causal dual-arm execution-memory adapter."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        spatial = self.spatial_config
        memory = self.config.framework.get("memory", {})
        self.memory_recent_records = int(memory.get("recent_records", spatial.get("memory_recent_records", 16)))
        self.memory_decision_stride = int(memory.get("decision_stride_env_steps", spatial.get("memory_decision_stride", 8)))
        self.memory_required = bool(memory.get("required", spatial.get("memory_required", True)))
        if not self.memory_required:
            raise ValueError("QwenSpatialMemOFT requires memory during both training and inference")
        self.memory_state_dim = int(memory.get("state_dim", 14))
        self.memory_action_dim = int(memory.get("action_dim", 14))
        self.memory_use_pose = bool(memory.get("use_pose", False))
        self.memory_bptt_records = int(memory.get("bptt_records", memory.get("detach_every", 8)))
        self.memory_dropout = float(memory.get("dropout", memory.get("record_dropout", 0.0)))
        self.memory_spec = RoboDojoMemorySpec(self.memory_recent_records, self.memory_decision_stride, self.memory_use_pose)
        if self.memory_state_dim != 14 or self.memory_action_dim != 14:
            raise ValueError("RoboDojo memory requires state_dim=action_dim=14")
        data = self.config.datasets.vla_data
        if (int(data.get("memory_recent_records", 16)), int(data.get("memory_decision_stride", 8)), bool(data.get("memory_use_pose", False))) != (self.memory_recent_records, self.memory_decision_stride, self.memory_use_pose):
            raise ValueError("Model and training dataset memory configurations differ")
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.execution_memory = CausalExecutionMemory(
            hidden_size,
            state_dim=self.memory_state_dim,
            action_dim=self.memory_action_dim,
            width=int(memory.get("width", spatial.get("memory_width", 256))),
            heads=int(memory.get("heads", spatial.get("memory_heads", 4))),
            layers=int(memory.get("layers", spatial.get("memory_layers", 2))),
            pose_dim=32 if self.memory_use_pose else 0,
            slots=int(memory.get("slots", 8)),
            detach_every=self.memory_bptt_records,
            record_dropout=self.memory_dropout,
        )
        self.memory_waypoint_projection = nn.Linear(self.execution_memory.width, hidden_size)
        nn.init.zeros_(self.memory_waypoint_projection.weight)
        nn.init.zeros_(self.memory_waypoint_projection.bias)
        # A spatial checkpoint is a valid initialization for memory training.
        # The loader accepts these prefixes as newly initialized parameters.
        self.allow_missing_checkpoint_prefixes = tuple(self.allow_missing_checkpoint_prefixes) + (
            "execution_memory.", "memory_waypoint_projection."
        )

    def _batch_memory_records(self, examples: list[dict]) -> tuple[dict, None]:
        if not all(isinstance(e.get("memory_records"), dict) for e in examples):
            raise ValueError("QwenSpatialMemOFT requires causal memory_records for every example")
        records = [e["memory_records"] for e in examples]
        for example in examples:
            contract = example.get("memory_contract")
            if not isinstance(contract, dict):
                raise ValueError("Every memory sample requires memory_contract")
            self.memory_spec.validate(example["memory_records"], contract)
        keys = tuple(self.memory_spec.empty())
        recent = {key: np.stack([r[key] for r in records]) for key in keys}
        burnins = [e.get("memory_burnin_records") for e in examples]
        if all(b is None for b in burnins):
            return recent, None
        if any(not isinstance(b, dict) for b in burnins):
            raise ValueError("Every memory sample must provide memory_burnin_records")
        max_len = max(len(b["valid_mask"]) for b in burnins)
        burn = {}
        for key in keys:
            value = np.asarray(burnins[0][key])
            arr = np.zeros((len(examples), max_len, *value.shape[1:]), dtype=value.dtype)
            for index, b in enumerate(burnins):
                if np.asarray(b[key]).shape[1:] != value.shape[1:]:
                    raise ValueError(f"Inconsistent burn-in field {key}")
                arr[index, :len(b["valid_mask"])] = b[key]
            burn[key] = arr
        return recent, burn

    def _spatial_forward(self, hidden, action_hidden, examples):
        records, burnin = self._batch_memory_records(examples)
        if records is None:
            return super()._spatial_forward(hidden, action_hidden, examples)
        conditioned, summary, _ = self.execution_memory(action_hidden, records, burnin)
        waypoint_hidden = hidden + self.memory_waypoint_projection(summary).unsqueeze(1)
        adapted, wp_loss, rel_loss, geom_loss = super()._spatial_forward(waypoint_hidden, conditioned, examples)
        return adapted, wp_loss, rel_loss, geom_loss


__all__ = ["CausalExecutionMemory", "QwenSpatialMemOFT"]

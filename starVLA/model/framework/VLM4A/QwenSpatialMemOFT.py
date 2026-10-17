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
from starVLA.utils.robodojo_memory import RoboDojoMemorySpec


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
    ):
        super().__init__()
        self.width = int(width)
        self.state_dim, self.action_dim, self.pose_dim = int(state_dim), int(action_dim), int(pose_dim)
        self.record_projection = nn.Sequential(
            nn.LayerNorm(self.state_dim + self.action_dim + self.pose_dim + 1),
            nn.Linear(self.state_dim + self.action_dim + self.pose_dim + 1, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.slot_seed = nn.Parameter(torch.empty(8, self.width))
        nn.init.normal_(self.slot_seed, mean=0.0, std=0.02)
        self.slot_update = nn.GRU(self.width, self.width, batch_first=True)
        self.retrieval_query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.width))
        self.retrieval_layers = nn.ModuleList(
            [nn.MultiheadAttention(self.width, heads, batch_first=True) for _ in range(int(layers))]
        )
        self.retrieval_norms = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(int(layers))])
        self.output_projection = nn.Linear(self.width, hidden_size)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size + self.width),
            nn.Linear(hidden_size + self.width, self.width),
            nn.Sigmoid(),
        )
        # Preserve the spatial checkpoint's action path at initialization.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _record_features(self, records: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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
        record = self.record_projection(values)
        valid = torch.as_tensor(records["valid_mask"], device=device, dtype=torch.bool)
        return record, valid

    def _update_slots(self, slots: torch.Tensor, records: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Right-aligned padding -> left-aligned sequences. Run all eight slots
        # and all history steps in one cuDNN GRU, then gather each last valid
        # output. This is equivalent to skipping padded GRUCell updates, with
        # no Python loop over history or GPU->CPU length synchronization.
        batch, length, width = records.shape
        count = valid.sum(1)
        indices = (torch.arange(length, device=records.device)[None] + length - count[:, None]) % length
        compact = records.gather(1, indices[..., None].expand(-1, -1, width))
        nslots = slots.shape[1]
        inputs = compact[:, None].expand(-1, nslots, -1, -1).reshape(batch * nslots, length, width).contiguous()
        initial = slots.reshape(1, batch * nslots, width).contiguous()
        outputs, _ = self.slot_update(inputs, initial)
        outputs = outputs.view(batch, nslots, length, width)
        last = (count - 1).clamp_min(0)[:, None, None, None].expand(-1, nslots, 1, width)
        updated = outputs.gather(2, last).squeeze(2)
        return torch.where((count > 0)[:, None, None], updated, slots)

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
        if burnin_records is not None:
            raise ValueError("RoboDojo uses a bounded window; external burn-in is forbidden")
        slots = self._update_slots(slots, record, valid)
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
        gate = self.gate(torch.cat((action_queries, query), dim=-1))
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
            if example.get("memory_burnin_records") is not None:
                raise ValueError("External burn-in is forbidden for bounded RoboDojo memory")
            contract = example.get("memory_contract")
            if not isinstance(contract, dict):
                raise ValueError("Every memory sample requires memory_contract")
            self.memory_spec.validate(example["memory_records"], contract)
        keys = tuple(self.memory_spec.empty())
        return {key: np.stack([r[key] for r in records]) for key in keys}, None

    def _spatial_forward(self, hidden, action_hidden, examples):
        records, burnin = self._batch_memory_records(examples)
        if records is None:
            return super()._spatial_forward(hidden, action_hidden, examples)
        conditioned, summary, _ = self.execution_memory(action_hidden, records, burnin)
        waypoint_hidden = hidden + self.memory_waypoint_projection(summary).unsqueeze(1)
        adapted, wp_loss, rel_loss, geom_loss = super()._spatial_forward(waypoint_hidden, conditioned, examples)
        return adapted, wp_loss, rel_loss, geom_loss


__all__ = ["CausalExecutionMemory", "QwenSpatialMemOFT"]

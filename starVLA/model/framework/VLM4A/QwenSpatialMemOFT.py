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
        slots: int = 8,
        detach_every: int = 8,
    ):
        super().__init__()
        self.width = int(width)
        self.slots = int(slots)
        self.detach_every = max(1, int(detach_every))
        self.state_dim, self.action_dim, self.pose_dim = int(state_dim), int(action_dim), int(pose_dim)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(3 * 8 * 8 * 8 * 3),
            nn.Linear(3 * 8 * 8 * 8 * 3, self.width), nn.GELU(), nn.Linear(self.width, self.width),
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
        patches = torch.as_tensor(records["visual_tokens"], device=device)
        if patches.ndim != 7 or tuple(patches.shape[-5:]) != (3, 8, 8, 8, 3):
            raise ValueError(f"visual_tokens must be [B,R,3,8,8,8,3], got {tuple(patches.shape)}")
        visual = self.visual_projection(patches.float().div(255.0).flatten(start_dim=2))
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
            active = valid[:, index, None, None]
            value = records[:, index, None, :].expand(-1, self.slots, -1)
            updated = self.slot_update(value.reshape(-1, self.width), slots.reshape(-1, self.width))
            slots = torch.where(active, updated.view(batch, self.slots, self.width), slots)
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
        if burnin_records is not None and burnin_records["valid_mask"].shape[1]:
            with torch.no_grad():
                burnin, burnin_valid = self._record_features(burnin_records, action_queries.device)
                slots = self._update_slots(slots, burnin, burnin_valid)
            slots = slots.detach()
        for start in range(0, record.shape[1], self.detach_every):
            stop = min(start + self.detach_every, record.shape[1])
            slots = self._update_slots(slots, record[:, start:stop], valid[:, start:stop])
            if stop % self.detach_every == 0:
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
        elapsed = torch.as_tensor(records["elapsed_env_steps"], device=action_queries.device, dtype=torch.float32)
        elapsed = (elapsed * valid.float()).sum(dim=1, keepdim=True) / valid.float().sum(dim=1, keepdim=True).clamp_min(1)
        gate = self.gate(torch.cat((action_queries, query, elapsed[:, None, :].expand(-1, query.shape[1], -1)), dim=-1))
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

    def _batch_memory_records(self, examples: list[dict]) -> tuple[dict, dict | None]:
        if not all(isinstance(e.get("memory_records"), dict) for e in examples):
            raise ValueError("QwenSpatialMemOFT requires causal memory_records for every example")
        records = [e["memory_records"] for e in examples]
        for example in examples:
            contract = example.get("memory_contract")
            if not isinstance(contract, dict):
                raise ValueError("Every memory sample requires memory_contract")
            self.memory_spec.validate(example["memory_records"], contract)
        keys = tuple(self.memory_spec.empty())
        batched_records = {key: np.stack([r[key] for r in records]) for key in keys}

        burnin_records = [example.get("memory_burnin_records") for example in examples]
        if all(record is None for record in burnin_records):
            return batched_records, None
        if not all(isinstance(record, dict) for record in burnin_records):
            raise ValueError("A batch cannot mix examples with and without memory burn-in records")

        burnin_keys = tuple(self.memory_spec.empty_burnin(0))
        expected_indices = [self.memory_spec.burnin_indices(example["memory_contract"]["source_frame_index"])
                            for example in examples]
        max_length = max((len(indices) for indices in expected_indices), default=0)
        batched_burnin = {}
        for key in burnin_keys:
            template = self.memory_spec.empty_burnin(0)[key]
            fill_value = -1 if key == "source_frame_index" else False if key == "valid_mask" else 0
            batched_burnin[key] = np.full(
                (len(examples), max_length) + template.shape[1:], fill_value, dtype=template.dtype
            )

        for batch_index, (record, indices, example) in enumerate(zip(burnin_records, expected_indices, examples)):
            if set(record) != set(burnin_keys):
                raise ValueError(f"Burn-in fields must be {burnin_keys}")
            count = len(indices)
            if np.asarray(record["valid_mask"]).shape != (count,) or not np.asarray(record["valid_mask"]).dtype == np.bool_:
                raise ValueError("Burn-in valid_mask must be boolean and match the expected record count")
            if not np.asarray(record["valid_mask"]).all():
                raise ValueError("Unbatched burn-in records may not contain padded slots")
            source_indices = np.asarray(record["source_frame_index"])
            if not np.issubdtype(source_indices.dtype, np.integer) or not np.array_equal(source_indices, indices):
                raise ValueError("Burn-in must cover the exact strictly-past stride grid")
            current = int(example["memory_contract"]["source_frame_index"])
            elapsed = np.asarray(record["elapsed_env_steps"])
            if not np.array_equal(elapsed, current - indices):
                raise ValueError("Burn-in elapsed steps disagree with source indices")
            for key in burnin_keys:
                value = np.asarray(record[key])
                template = self.memory_spec.empty_burnin(count)[key]
                if value.shape != template.shape:
                    raise ValueError(f"Burn-in {key} must have shape {template.shape}, got {value.shape}")
                if value.dtype != template.dtype:
                    raise ValueError(f"Burn-in {key} must have dtype {template.dtype}, got {value.dtype}")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError(f"Burn-in {key} must contain finite values")
                if count:
                    batched_burnin[key][batch_index, :count] = value

        return batched_records, batched_burnin

    def _spatial_forward(self, hidden, action_hidden, examples):
        records, burnin = self._batch_memory_records(examples)
        if records is None:
            return super()._spatial_forward(hidden, action_hidden, examples)
        conditioned, summary, _ = self.execution_memory(action_hidden, records, burnin)
        waypoint_hidden = hidden + self.memory_waypoint_projection(summary).unsqueeze(1)
        adapted, wp_loss, rel_loss, geom_loss = super()._spatial_forward(waypoint_hidden, conditioned, examples)
        return adapted, wp_loss, rel_loss, geom_loss


__all__ = ["CausalExecutionMemory", "QwenSpatialMemOFT"]

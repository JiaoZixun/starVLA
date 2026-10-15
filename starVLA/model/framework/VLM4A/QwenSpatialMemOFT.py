"""Compatibility name for the causal memory-enabled Spatial OFT model.

The implementation lives in :mod:`QwenSpatialOFT` so there is one model
definition and one state-dict contract.  Older launch files use the
``QwenSpatialMemOFT`` name and the ``framework.memory`` section; those values
are translated into the shared spatial configuration here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenSpatialOFT import (
    CausalExecutionMemory,
    QwenSpatialOFT,
    SpatialAdapter,
    WaypointModel,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenSpatialMemOFT")
class QwenSpatialMemOFT(QwenSpatialOFT):
    """Backward-compatible alias with an explicit memory contract."""

    def __init__(self, config=None, **kwargs):
        if config is not None and hasattr(config, "framework") and hasattr(config.framework, "memory"):
            memory = config.framework.memory
            if not hasattr(config.framework, "spatial"):
                config.framework.spatial = {}
            spatial = config.framework.spatial
            mapping = {
                "width": "memory_width",
                "heads": "memory_heads",
                "layers": "memory_layers",
                "recent_records": "memory_recent_records",
                "bptt_records": "memory_bptt_records",
                "dropout": "memory_dropout",
            }
            for source, target in mapping.items():
                if source in memory and target not in spatial:
                    spatial[target] = memory[source]
            if "decision_stride_env_steps" in memory:
                config.datasets.vla_data.memory_decision_stride = memory.decision_stride_env_steps
            if "visual_tokens" in memory:
                config.datasets.vla_data.memory_visual_tokens = memory.visual_tokens
        super().__init__(config=config, **kwargs)
        memory_cfg = config.framework.memory if config is not None and hasattr(config.framework, "memory") else {}
        spatial_cfg = self.config.framework.get("spatial", {})
        self.memory_width = int(memory_cfg.get("width", spatial_cfg.get("memory_width", 256)))
        self.memory_recent_records = int(memory_cfg.get("recent_records", spatial_cfg.get("memory_recent_records", 32)))
        self.memory_decision_stride = int(memory_cfg.get("decision_stride_env_steps", 8))
        data_cfg = getattr(getattr(config, "datasets", None), "vla_data", None)
        if data_cfg is not None:
            self.memory_recent_records = int(getattr(data_cfg, "memory_recent_records", self.memory_recent_records))
            self.memory_decision_stride = int(getattr(data_cfg, "memory_decision_stride_env_steps", self.memory_decision_stride))

        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.execution_memory = CausalExecutionMemory(
            hidden_size,
            width=self.memory_width,
            heads=int(memory_cfg.get("heads", spatial_cfg.get("memory_heads", 4))),
            slots=int(spatial_cfg.get("memory_slots", 8)),
            layers=int(memory_cfg.get("layers", spatial_cfg.get("memory_layers", 2))),
            detach_every=int(memory_cfg.get("bptt_records", spatial_cfg.get("memory_bptt_records", 8))),
            record_dropout=float(spatial_cfg.get("memory_dropout", 0.0)),
        )
        self.memory_waypoint_projection = nn.Linear(self.memory_width, hidden_size)
        nn.init.zeros_(self.memory_waypoint_projection.weight)
        nn.init.zeros_(self.memory_waypoint_projection.bias)

        visual_tokens = int(memory_cfg.get("visual_tokens", 8))
        visual_views = int(memory_cfg.get("visual_views", 2))
        visual_dim = int(memory_cfg.get("visual_feature_dim", 3))
        if (visual_tokens, visual_views, visual_dim) != (8, 2, 3):
            raise ValueError("LIBERO spatial memory requires visual_views=2, visual_tokens=8, visual_feature_dim=3")

    @property
    def memory_encoder(self):
        """Compatibility alias without registering the same module twice."""
        return self.execution_memory

    def _batch_memory_records(self, examples: list[dict]) -> dict[str, np.ndarray]:
        records = []
        for sample_index, example in enumerate(examples):
            record = example.get("memory_records")
            contract = example.get("memory_contract")
            if not isinstance(record, dict) or not isinstance(contract, dict):
                raise ValueError("QwenSpatialMemOFT requires memory_records and memory_contract")
            if contract.get("history_is_strictly_past") is not True:
                raise ValueError(f"Sample {sample_index} memory is not declared strictly past")
            current = int(contract.get("source_frame_index", -1))
            valid = np.asarray(record.get("valid_mask"), dtype=np.bool_)
            source = np.asarray(record.get("source_frame_index"), dtype=np.int64)
            if valid.shape != (self.memory_recent_records,) or source.shape != valid.shape:
                raise ValueError("Memory valid/source indices do not match recent_records")
            if np.any(source[valid] < 0) or np.any(source[valid] >= current):
                raise ValueError(f"Sample {sample_index} contains a current/future memory record")
            if int(contract.get("decision_stride_env_steps", self.memory_decision_stride)) != self.memory_decision_stride:
                raise ValueError("Memory decision stride differs between sample and model")
            records.append(record)
        keys = ("visual_tokens", "state", "executed_actions", "eef_pose", "elapsed_env_steps", "valid_mask")
        return {key: np.stack([np.asarray(record[key]) for record in records]) for key in keys}

    def _condition_action_queries(
        self,
        examples: list[dict],
        action_queries: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
        gripper: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        conditioned, spatial_residual, spatial_gate = self.spatial_adapter(
            action_queries, translation, rotation_6d, gripper, tau
        )
        return conditioned, spatial_residual, {"spatial_gate_mean": spatial_gate}

    def _condition_memory(
        self,
        examples: list[dict],
        last_hidden: torch.Tensor,
        action_queries: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        memory_records = self._batch_memory_records(examples)
        burnin_records = []
        for example in examples:
            burnin = example.get("memory_burnin_records")
            if not isinstance(burnin, dict):
                raise ValueError("QwenSpatialMemOFT requires memory_burnin_records for every example")
            burnin_records.append(burnin)
        burnin_keys = tuple(burnin_records[0].keys())
        if any(tuple(record.keys()) != burnin_keys for record in burnin_records):
            raise ValueError("Memory burn-in records must have the same fields in every batch example")
        burnin_lengths = [len(record["valid_mask"]) for record in burnin_records]
        max_burnin = max(burnin_lengths, default=0)
        burnin_batch = {}
        for key in burnin_keys:
            sample_value = np.asarray(burnin_records[0][key])
            batched_value = np.zeros(
                (len(examples), max_burnin, *sample_value.shape[1:]), dtype=sample_value.dtype
            )
            for index, (record, length) in enumerate(zip(burnin_records, burnin_lengths)):
                value = np.asarray(record[key])
                if value.shape[0] != length or value.shape[1:] != sample_value.shape[1:]:
                    raise ValueError(f"Inconsistent memory burn-in field {key!r}: {value.shape}")
                batched_value[index, :length] = value
            burnin_batch[key] = batched_value
        # _condition_memory is called before the autocast block in the base
        # class (_predict_from_features:382), so open autocast here to match
        # the bf16 dtype of execution_memory weights on CUDA.
        device = action_queries.device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            memory_conditioned, _, _, memory_summary, memory_channel_gate, memory_gate = self.execution_memory(
                action_queries, memory_records, burnin_batch
            )
            # The same retrieved history conditions waypoint decoding and the
            # action path.  The projection is zero-initialized so the original
            # spatial checkpoint remains the optimization starting point.
            waypoint_features = last_hidden + self.memory_waypoint_projection(memory_summary).unsqueeze(1)
        return waypoint_features, memory_conditioned, attention_mask, {
            "memory_gate_mean": memory_gate.detach().float().mean(),
            "memory_gate_channel_mean": memory_channel_gate.detach().float().mean(),
        }


__all__ = [
    "CausalExecutionMemory",
    "QwenSpatialMemOFT",
    "SpatialAdapter",
    "WaypointModel",
]

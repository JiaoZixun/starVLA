"""Spatially conditioned QwenOFT with a zero-initialized residual adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT, masked_l1_loss
from starVLA.model.framework.VLM4A.spatial_geometry import (
    matrix_to_rotation_6d,
    relative_pose,
    relative_pose_loss,
    rotation_6d_to_matrix,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class WaypointDecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, memory_padding_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.query_norm(queries)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + attended
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.cross_query_norm(queries), normalized_memory, normalized_memory,
            key_padding_mask=memory_padding_mask, need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class CausalExecutionMemory(nn.Module):
    """Encode strictly-past records and retrieve them for action queries.

    The visual input is a compact uint8 patch sidecar rather than a second
    Qwen call.  Eight trainable recurrent slots summarize older records; the
    recent records remain individually addressable.  ``detach_every`` bounds
    the history graph and does not change the chronological order.
    """

    def __init__(self, vlm_width: int, width: int = 256, heads: int = 4, slots: int = 8, layers: int = 2, detach_every: int = 8, record_dropout: float = 0.0):
        super().__init__()
        self.width = int(width)
        self.slots = int(slots)
        self.detach_every = max(1, int(detach_every))
        self.record_dropout = float(record_dropout)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(2 * 8 * 8 * 8 * 3),
            nn.Linear(2 * 8 * 8 * 8 * 3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.record_projection = nn.Sequential(
            nn.Linear(8 + 7 + 16 + 1, width), nn.GELU(), nn.Linear(width, width)
        )
        self.record_norm = nn.LayerNorm(width)
        # Slot-specific seeds prevent the long-term slots from collapsing to
        # identical states when every slot receives the same record input.
        self.slot_seed = nn.Parameter(torch.empty(slots, width))
        nn.init.normal_(self.slot_seed, mean=0.0, std=0.02)
        self.slot_update = nn.GRUCell(width, width)
        self.retrieval_query = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.retrieval_layers = nn.ModuleList(
            [nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers)]
        )
        self.retrieval_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.output_projection = nn.Linear(width, vlm_width)
        self.gate = nn.Sequential(
            nn.LayerNorm(vlm_width + width + 1),
            nn.Linear(vlm_width + width + 1, width),
            nn.Sigmoid(),
        )
        # Keep the original OFT action path intact at the insertion point.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _record_features(self, records: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        patches = torch.as_tensor(records["visual_tokens"], device=device)
        if patches.ndim != 7:
            raise ValueError(f"memory_records.visual_tokens must be [B,R,V,T,8,8,3], got {tuple(patches.shape)}")
        # One ordered vector per record preserves both camera and 2x4 spatial layout.
        expected_tail = (2, 8, 8, 8, 3)
        if tuple(patches.shape[-5:]) != expected_tail:
            raise ValueError(f"Expected memory visual tail {expected_tail}, got {tuple(patches.shape[-5:])}")
        visual = self.visual_projection(patches.float().div(255.0).flatten(start_dim=2))
        state = torch.as_tensor(records["state"], device=device, dtype=torch.float32)
        action = torch.as_tensor(records["executed_actions"], device=device, dtype=torch.float32)
        pose = torch.as_tensor(records["eef_pose"], device=device, dtype=torch.float32).flatten(-2)
        elapsed = torch.as_tensor(records["elapsed_env_steps"], device=device, dtype=torch.float32).unsqueeze(-1)
        record_values = torch.cat((state, action, pose, elapsed), dim=-1)
        record = self.record_projection(record_values)
        record = self.record_norm(record + visual)
        valid = torch.as_tensor(records["valid_mask"], device=device, dtype=torch.bool)
        return record, valid

    def _update_slots(
        self, slots: torch.Tensor, record: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch, _, _ = record.shape
        for index in range(record.shape[1]):
            active = valid[:, index].view(batch, 1, 1)
            slot_input = record[:, index].unsqueeze(1).expand(-1, self.slots, -1)
            updated = self.slot_update(
                slot_input.reshape(-1, self.width), slots.reshape(-1, self.width)
            ).view(batch, self.slots, self.width)
            slots = torch.where(active, updated, slots)
        return slots

    def forward(
        self,
        action_queries: torch.Tensor,
        records: dict | None,
        burnin_records: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, vlm_width = action_queries.shape
        if records is None:
            memory_tokens = action_queries.new_zeros((batch, 1, self.width))
            memory_valid = torch.zeros((batch, 1), device=action_queries.device, dtype=torch.bool)
            return (
                action_queries,
                memory_tokens,
                memory_valid,
                action_queries.new_zeros((batch, vlm_width)),
                action_queries.new_zeros((batch, self.width)),
                action_queries.new_zeros((batch, action_queries.shape[1], self.width)),
            )
        record, valid = self._record_features(records, action_queries.device)
        if record.shape[0] != batch:
            raise ValueError(f"Memory batch mismatch: records={record.shape[0]} queries={batch}")
        if self.training and self.record_dropout > 0:
            keep = torch.rand(valid.shape, device=valid.device) >= self.record_dropout
            valid = valid & keep
        # Burn-in reconstructs long-term state from the episode prefix without
        # retaining its graph. Only the recent window contributes BPTT.
        slots = self.slot_seed.unsqueeze(0).expand(batch, -1, -1)
        if burnin_records is not None and burnin_records["valid_mask"].shape[1] > 0:
            with torch.no_grad():
                burnin, burnin_valid = self._record_features(burnin_records, action_queries.device)
                if self.training and self.record_dropout > 0:
                    keep = torch.rand(burnin_valid.shape, device=burnin_valid.device) >= self.record_dropout
                    burnin_valid = burnin_valid & keep
                slots = self._update_slots(slots, burnin, burnin_valid)
            slots = slots.detach()

        # Recent records remain differentiable, with periodic graph truncation.
        for start in range(0, record.shape[1], self.detach_every):
            stop = min(start + self.detach_every, record.shape[1])
            slots = self._update_slots(slots, record[:, start:stop], valid[:, start:stop])
            if stop % self.detach_every == 0:
                slots = slots.detach()
        # Retrieval keys are recent records followed by long-term slots.
        retrieval = torch.cat((record, slots), dim=1)
        retrieval_valid = torch.cat(
            (valid, torch.ones((batch, self.slots), device=valid.device, dtype=torch.bool)), dim=1
        )
        query = self.retrieval_query(action_queries)
        padding = ~retrieval_valid
        for attention, norm in zip(self.retrieval_layers, self.retrieval_norms):
            update, _ = attention(query, retrieval, retrieval, key_padding_mask=padding, need_weights=False)
            query = norm(query + update)
        # The same per-query memory gate controls the summary consumed by the
        # waypoint path and the residual consumed by the action path.
        retrieval_summary = query.mean(dim=1)
        elapsed_summary = torch.as_tensor(records["elapsed_env_steps"], device=action_queries.device, dtype=torch.float32)
        elapsed_summary = (elapsed_summary * valid.float()).sum(dim=1, keepdim=True) / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        gate_input = torch.cat((action_queries, retrieval_summary.unsqueeze(1).expand(-1, action_queries.shape[1], -1), elapsed_summary.unsqueeze(1).expand(-1, action_queries.shape[1], -1)), dim=-1)
        gates = self.gate(gate_input)
        summary = (query * gates).mean(dim=1)
        conditioned = action_queries + self.output_projection(gates * query)
        return conditioned, retrieval, retrieval_valid, summary, gates.mean(dim=1), gates.detach()


class WaypointModel(nn.Module):
    """Predict relative SE(3) waypoints from task-conditioned action-token features."""

    def __init__(self, vlm_width: int, width: int = 256, heads: int = 4, ffn_width: int = 1024, layers: int = 2, num_waypoints: int = 4, predict_gripper: bool = True):
        super().__init__()
        self.width = width
        self.predict_gripper = bool(predict_gripper)
        self.memory_projection = nn.Linear(vlm_width, width)
        self.anchor_projection = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.arm_embedding = nn.Embedding(2, width)
        self.time_projection = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.decoder = nn.ModuleList([WaypointDecoderBlock(width, heads, ffn_width) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.translation_head = nn.Linear(width, 3)
        self.rotation_head = nn.Linear(width, 6)
        self.num_waypoints = num_waypoints
        nn.init.zeros_(self.translation_head.weight)
        nn.init.zeros_(self.translation_head.bias)
        nn.init.zeros_(self.rotation_head.weight)
        if self.predict_gripper:
            self.gripper_head = nn.Linear(width, 1)
            nn.init.zeros_(self.gripper_head.weight)
            nn.init.zeros_(self.gripper_head.bias)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(
        self,
        vlm_features: torch.Tensor,
        anchor_pose: torch.Tensor,
        tau: torch.Tensor,
        memory_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = vlm_features.shape[0]
        rotation_6d = matrix_to_rotation_6d(anchor_pose[..., :3, :3])
        anchor_features = torch.cat((anchor_pose[..., :3, 3].float(), rotation_6d), dim=-1)
        times = tau.float().view(1, -1, 1).expand(batch_size, -1, -1)
        query = (
            self.anchor_projection(anchor_features).unsqueeze(1)
            + self.arm_embedding(torch.zeros((batch_size, 1), device=vlm_features.device, dtype=torch.long))
            + self.time_projection(times)
        )
        memory = self.memory_projection(vlm_features)
        padding_mask = None if memory_attention_mask is None else ~memory_attention_mask.to(torch.bool)
        for layer in self.decoder:
            query = layer(query, memory, padding_mask)
        features = self.output_norm(query)
        translation = self.translation_head(features)
        rotation = self.rotation_head(features)
        if self.predict_gripper:
            gripper = self.gripper_head(features).squeeze(-1)
        else:
            gripper = translation.new_zeros(translation.shape[:-1])
        return translation, rotation, gripper


class SpatialAdapter(nn.Module):
    """Cross-attend action queries to explicit metric waypoint tokens."""

    def __init__(self, vlm_width: int, width: int = 256, heads: int = 4, num_waypoints: int = 4, gated: bool = True):
        super().__init__()
        # v1 (ungated) has no gripper token channel and no spatial gate.
        self.gated = bool(gated)
        self.action_projection = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.spatial_projection = nn.Sequential(
            nn.Linear(12 if self.gated else 11, width), nn.GELU(), nn.Linear(width, width)
        )
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        if self.gated:
            self.spatial_gate = nn.Sequential(
                nn.LayerNorm(vlm_width + width),
                nn.Linear(vlm_width + width, width),
                nn.Sigmoid(),
            )
        self.output_projection = nn.Linear(width, vlm_width)
        self.num_waypoints = num_waypoints
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action_features: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
        gripper: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = action_features.shape[0]
        normalized_time = tau.float().view(1, -1, 1).expand(batch_size, -1, -1)
        arm_id = torch.zeros_like(normalized_time)
        # The token includes predicted metric coordinates. It does not use future labels or decoder-only latents.
        values = [translation.float() / 0.1, rotation_6d.float()]
        if self.gated:
            values.append(gripper.float().unsqueeze(-1))
        token_values = torch.cat((*values, normalized_time, arm_id), dim=-1)
        queries = self.action_projection(action_features)
        keys = self.spatial_projection(token_values)
        update, _ = self.cross_attention(queries, keys, keys, need_weights=False)
        if self.gated:
            gates = self.spatial_gate(torch.cat((action_features, update), dim=-1))
        else:
            gates = torch.ones_like(update)
        residual = self.output_projection(gates * update)
        return action_features + residual, residual, gates.detach()


@dataclass
class QwenSpatialOFTDefaultConfig:
    name: str = "QwenSpatialOFT"
    qwenvl: dict = field(default_factory=dict)
    action_model: dict = field(default_factory=dict)
    spatial: dict = field(
        default_factory=lambda: {
            "width": 256,
            "heads": 4,
            "ffn_width": 1024,
            "layers": 2,
            "tau": [1, 4, 8, 16],
            "waypoint_weight": 1.0,
            "waypoint_rotation_weight": 0.25,
            "gripper_weight": 0.10,
            "action_geometry_weight": 0.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenSpatialOFT")
class QwenSpatialOFT(Qwenvl_OFT):
    """QwenOFT preserving the original VLM and action-head state-dict names."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        spatial = self.config.framework.get("spatial", {})
        self.waypoint_tau = tuple(int(value) for value in spatial.get("tau", [1, 4, 8, 16]))
        if not self.waypoint_tau or tuple(sorted(set(self.waypoint_tau))) != self.waypoint_tau:
            raise ValueError(f"spatial.tau must be a non-empty, strictly increasing list; got {self.waypoint_tau}")
        if self.waypoint_tau[0] < 1:
            raise ValueError(f"spatial.tau={self.waypoint_tau} must be positive")
        self.spatial_width = int(spatial.get("width", 256))
        heads = int(spatial.get("heads", 4))
        ffn_width = int(spatial.get("ffn_width", 1024))
        layers = int(spatial.get("layers", 2))
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        # "v1" reproduces the adapter of checkpoints trained before the gripper
        # head / spatial gate were added (e.g. qwen35_0p8b_spatial_oft_libero_50k);
        # "v2" is the current gated adapter.
        self.adapter_version = str(spatial.get("adapter_version", "v2"))
        if self.adapter_version not in ("v1", "v2"):
            raise ValueError(f"spatial.adapter_version must be v1 or v2, got {self.adapter_version}")
        gated = self.adapter_version == "v2"
        self.waypoint_model = WaypointModel(
            hidden_size, self.spatial_width, heads, ffn_width, layers, len(self.waypoint_tau), predict_gripper=gated
        )
        self.spatial_adapter = SpatialAdapter(hidden_size, self.spatial_width, heads, len(self.waypoint_tau), gated=gated)
        self.waypoint_weight = float(spatial.get("waypoint_weight", 1.0))
        self.waypoint_rotation_weight = float(spatial.get("waypoint_rotation_weight", 0.25))
        self.gripper_weight = float(spatial.get("gripper_weight", 0.10 if gated else 0.0))
        if not gated and self.gripper_weight != 0.0:
            raise ValueError("spatial.adapter_version=v1 has no gripper head; gripper_weight must be 0")
        self.effect_weight = float(spatial.get("effect_weight", 0.0))
        self.action_geometry_weight = float(spatial.get("action_geometry_weight", 0.0))
        if self.effect_weight != 0.0:
            raise NotImplementedError(
                "Effect supervision is disabled until the LIBERO action/observation phase contract is verified; "
                "keep spatial.effect_weight=0.0"
            )
        if self.action_geometry_weight != 0.0:
            raise NotImplementedError(
                "LIBERO action-geometry supervision is not implemented because the action-row/pre-action "
                "pose phase contract has not been verified; keep spatial.action_geometry_weight=0.0"
            )

    def _build_inputs(self, examples: list[dict[str, Any]]) -> tuple[dict, torch.Tensor]:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        if state is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, state)
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return qwen_inputs, outputs.hidden_states[-1]

    def _batch_geometry_current(self, examples: list[dict], device: torch.device) -> torch.Tensor:
        values = []
        for example in examples:
            current = example.get("geometry_current")
            if not isinstance(current, dict) or "eef_pose" not in current:
                raise ValueError("QwenSpatialOFT requires geometry_current.eef_pose for every example")
            pose = np.asarray(current["eef_pose"], dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"geometry_current.eef_pose must be finite [4,4], got {pose.shape}")
            values.append(pose)
        return torch.as_tensor(np.stack(values), device=device, dtype=torch.float32)

    def _condition_action_queries(
        self,
        examples: list[dict],
        action_queries: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
        gripper: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        conditioned, residual, spatial_gate = self.spatial_adapter(
            action_queries, translation, rotation_6d, gripper, tau
        )
        spatial_gate = spatial_gate.float()
        return conditioned, residual, {
            "spatial_gate_mean": spatial_gate.mean(),
            "spatial_gate_std": spatial_gate.std(unbiased=False),
            "spatial_gate_min": spatial_gate.min(),
            "spatial_gate_max": spatial_gate.max(),
        }

    def _condition_memory(
        self,
        examples: list[dict],
        last_hidden: torch.Tensor,
        action_queries: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Hook for memory-enabled variants; base Spatial OFT stays unchanged."""
        return last_hidden, action_queries, attention_mask, {}

    def _predict_from_features(self, examples: list[dict], qwen_inputs: dict, last_hidden: torch.Tensor, *, train: bool):
        device = last_hidden.device
        anchor_pose = self._batch_geometry_current(examples, device)
        tau = torch.tensor(self.waypoint_tau, device=device, dtype=torch.float32) / float(max(self.waypoint_tau))
        # Keep the OFT action queries for the original action head and adapter, while the
        # waypoint decoder attends to the full task-conditioned VLM sequence as specified
        # by the spatial architecture. This reuses the existing VLM forward without a
        # second vision-language pass.
        action_queries = self._gather_action_token_embeddings(
            last_hidden,
            qwen_inputs.get("input_ids"),
            action_token_id=self.action_token_id,
        )
        memory_attention_mask = qwen_inputs.get("attention_mask")
        if memory_attention_mask is not None and memory_attention_mask.shape != last_hidden.shape[:2]:
            raise ValueError(
                "Qwen attention mask must align with the full VLM hidden sequence: "
                f"mask={tuple(memory_attention_mask.shape)}, hidden={tuple(last_hidden.shape[:2])}"
            )
        waypoint_features, action_queries, memory_attention_mask, memory_metrics = self._condition_memory(
            examples, last_hidden, action_queries, memory_attention_mask
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            translation, rotation_6d, waypoint_gripper = self.waypoint_model(
                waypoint_features,
                anchor_pose,
                tau,
                memory_attention_mask,
            )
            conditioned_queries, spatial_residual, extra_metrics = self._condition_action_queries(
                examples,
                action_queries,
                translation,
                rotation_6d,
                waypoint_gripper,
                tau,
            )
            pred_actions = self.action_model.predict_action(conditioned_queries)
        waypoint_prediction = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(len(examples), len(self.waypoint_tau), 1, 1)
        waypoint_prediction[..., :3, :3] = rotation_6d_to_matrix(rotation_6d)
        waypoint_prediction[..., :3, 3] = translation.float()
        if not train:
            return pred_actions, waypoint_prediction, spatial_residual, {**memory_metrics, **extra_metrics}

        actions = torch.as_tensor(np.stack([example["action"] for example in examples]), device=device, dtype=pred_actions.dtype)
        actions_target = actions[:, -self.action_horizon :, :]
        action_masks = [example.get("action_mask") for example in examples]
        if any(mask is not None for mask in action_masks) and not all(mask is not None for mask in action_masks):
            raise ValueError("action_mask must be present for every example in a batch or for none")
        action_mask = None
        if action_masks[0] is not None:
            action_mask = torch.as_tensor(np.asarray(action_masks), device=device, dtype=torch.bool)[:, -self.action_horizon :, :]
        bc_loss = masked_l1_loss(pred_actions, actions_target, action_mask)

        labels = [example.get("geometry_labels") for example in examples]
        if any(label is None for label in labels):
            raise ValueError("Training QwenSpatialOFT requires geometry_labels for every example")
        target_pose = torch.as_tensor(
            np.stack([label["relative_waypoints"] for label in labels]), device=device, dtype=torch.float32
        )
        future_valid = torch.as_tensor(
            np.stack([label["future_valid"] for label in labels]), device=device, dtype=torch.bool
        )
        if target_pose.shape != waypoint_prediction.shape or future_valid.shape != waypoint_prediction.shape[:2]:
            raise ValueError(
                f"Invalid waypoint label shapes: poses={tuple(target_pose.shape)}, masks={tuple(future_valid.shape)}, "
                f"expected={tuple(waypoint_prediction.shape)}"
            )
        translation_loss, rotation_loss = relative_pose_loss(waypoint_prediction, target_pose, future_valid)
        waypoint_loss = translation_loss + self.waypoint_rotation_weight * rotation_loss
        gripper_target = torch.as_tensor(
            np.stack([label["future_gripper"] for label in labels]), device=device, dtype=waypoint_gripper.dtype
        )
        gripper_valid = torch.as_tensor(
            np.stack([label["future_gripper_valid"] for label in labels]), device=device, dtype=torch.bool
        )
        if gripper_target.shape != waypoint_gripper.shape or gripper_valid.shape != waypoint_gripper.shape:
            raise ValueError(
                f"Invalid future gripper labels: target={tuple(gripper_target.shape)}, "
                f"mask={tuple(gripper_valid.shape)}, prediction={tuple(waypoint_gripper.shape)}"
            )
        if torch.any(gripper_valid):
            gripper_loss = torch.abs(waypoint_gripper.float() - gripper_target.float())[gripper_valid].mean()
        else:
            gripper_loss = bc_loss.new_zeros(())
        relation_loss = bc_loss.new_zeros(())
        action_geometry_loss = bc_loss.new_zeros(())
        auxiliary_loss = (
            self.waypoint_weight * waypoint_loss
            + self.gripper_weight * gripper_loss
            + self.action_geometry_weight * action_geometry_loss
        )
        total_loss = bc_loss + auxiliary_loss
        residual_ratio = spatial_residual.detach().float().square().mean().sqrt() / (
            action_queries.detach().float().square().mean().sqrt() + 1e-8
        )
        return {
            "total_loss": total_loss,
            "action_loss": bc_loss,
            "bc_loss": bc_loss,
            "waypoint_loss": waypoint_loss,
            "waypoint_translation_loss": translation_loss,
            "waypoint_rotation_loss": rotation_loss,
            "gripper_loss": gripper_loss,
            "auxiliary_loss": auxiliary_loss,
            "relation_loss": relation_loss,
            "action_geometry_loss": action_geometry_loss,
            "spatial_residual_ratio": residual_ratio,
            **memory_metrics,
            **extra_metrics,
        }

    def forward(self, examples: list[dict] = None, **kwargs):
        qwen_inputs, last_hidden = self._build_inputs(examples)
        return self._predict_from_features(examples, qwen_inputs, last_hidden, train=True)

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] = None, **kwargs):
        if type(examples) is not list:
            examples = [examples]
        qwen_inputs, last_hidden = self._build_inputs(examples)
        pred_actions, _, _, _ = self._predict_from_features(examples, qwen_inputs, last_hidden, train=False)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}


__all__ = ["QwenSpatialOFT", "WaypointModel", "SpatialAdapter"]

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


class WaypointModel(nn.Module):
    """Predict relative SE(3) waypoints from task-conditioned action-token features."""

    def __init__(self, vlm_width: int, width: int = 256, heads: int = 4, ffn_width: int = 1024, layers: int = 2, num_waypoints: int = 4):
        super().__init__()
        self.width = width
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
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(
        self,
        vlm_features: torch.Tensor,
        anchor_pose: torch.Tensor,
        tau: torch.Tensor,
        memory_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return translation, rotation


class SpatialAdapter(nn.Module):
    """Cross-attend action queries to explicit metric waypoint tokens."""

    def __init__(self, vlm_width: int, width: int = 256, heads: int = 4, num_waypoints: int = 4):
        super().__init__()
        self.action_projection = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.spatial_projection = nn.Sequential(nn.Linear(11, width), nn.GELU(), nn.Linear(width, width))
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output_projection = nn.Linear(width, vlm_width)
        self.num_waypoints = num_waypoints
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        action_features: torch.Tensor,
        translation: torch.Tensor,
        rotation_6d: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = action_features.shape[0]
        normalized_time = tau.float().view(1, -1, 1).expand(batch_size, -1, -1)
        arm_id = torch.zeros_like(normalized_time)
        # The token includes predicted metric coordinates. It does not use future labels or decoder-only latents.
        token_values = torch.cat((translation.float() / 0.1, rotation_6d.float(), normalized_time, arm_id), dim=-1)
        queries = self.action_projection(action_features)
        keys = self.spatial_projection(token_values)
        update, _ = self.cross_attention(queries, keys, keys, need_weights=False)
        residual = self.output_projection(update)
        return action_features + residual, residual


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
            "tau": [1, 2, 4, 8],
            "waypoint_weight": 1.0,
            "waypoint_rotation_weight": 0.25,
            "action_geometry_weight": 0.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenSpatialOFT")
class QwenSpatialOFT(Qwenvl_OFT):
    """QwenOFT preserving the original VLM and action-head state-dict names."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        spatial = self.config.framework.get("spatial", {})
        self.waypoint_tau = tuple(int(value) for value in spatial.get("tau", [1, 2, 4, 8]))
        if not self.waypoint_tau or tuple(sorted(set(self.waypoint_tau))) != self.waypoint_tau:
            raise ValueError(f"spatial.tau must be a non-empty, strictly increasing list; got {self.waypoint_tau}")
        if self.waypoint_tau[0] < 1 or self.waypoint_tau[-1] > self.action_horizon:
            raise ValueError(f"spatial.tau={self.waypoint_tau} must be in [1, action_horizon={self.action_horizon}]")
        self.spatial_width = int(spatial.get("width", 256))
        heads = int(spatial.get("heads", 4))
        ffn_width = int(spatial.get("ffn_width", 1024))
        layers = int(spatial.get("layers", 2))
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.waypoint_model = WaypointModel(hidden_size, self.spatial_width, heads, ffn_width, layers, len(self.waypoint_tau))
        self.spatial_adapter = SpatialAdapter(hidden_size, self.spatial_width, heads, len(self.waypoint_tau))
        self.waypoint_weight = float(spatial.get("waypoint_weight", 1.0))
        self.waypoint_rotation_weight = float(spatial.get("waypoint_rotation_weight", 0.25))
        self.action_geometry_weight = float(spatial.get("action_geometry_weight", 0.0))
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

    def _predict_from_features(self, examples: list[dict], qwen_inputs: dict, last_hidden: torch.Tensor, *, train: bool):
        device = last_hidden.device
        anchor_pose = self._batch_geometry_current(examples, device)
        tau = torch.tensor(self.waypoint_tau, device=device, dtype=torch.float32) / float(self.action_horizon)
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
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            translation, rotation_6d = self.waypoint_model(
                last_hidden,
                anchor_pose,
                tau,
                memory_attention_mask,
            )
            conditioned_queries, spatial_residual = self.spatial_adapter(
                action_queries,
                translation,
                rotation_6d,
                tau,
            )
            pred_actions = self.action_model.predict_action(conditioned_queries)
        waypoint_prediction = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(len(examples), len(self.waypoint_tau), 1, 1)
        waypoint_prediction[..., :3, :3] = rotation_6d_to_matrix(rotation_6d)
        waypoint_prediction[..., :3, 3] = translation.float()
        if not train:
            return pred_actions, waypoint_prediction, spatial_residual

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
        relation_loss = bc_loss.new_zeros(())
        action_geometry_loss = bc_loss.new_zeros(())
        total_loss = bc_loss + self.waypoint_weight * waypoint_loss + self.action_geometry_weight * action_geometry_loss
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
            "relation_loss": relation_loss,
            "action_geometry_loss": action_geometry_loss,
            "spatial_residual_ratio": residual_ratio,
        }

    def forward(self, examples: list[dict] = None, **kwargs):
        qwen_inputs, last_hidden = self._build_inputs(examples)
        return self._predict_from_features(examples, qwen_inputs, last_hidden, train=True)

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] = None, **kwargs):
        if type(examples) is not list:
            examples = [examples]
        qwen_inputs, last_hidden = self._build_inputs(examples)
        pred_actions, _, _ = self._predict_from_features(examples, qwen_inputs, last_hidden, train=False)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}


__all__ = ["QwenSpatialOFT", "WaypointModel", "SpatialAdapter"]

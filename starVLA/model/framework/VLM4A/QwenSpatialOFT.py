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


class BridgeAttentionBlock(nn.Module):
    """VLA-Adapter Bridge Attention over one Qwen layer.

    Queries cross-attend (a) to that layer's action-query latents plus extra
    condition tokens at full weight and (b) to all of that layer's raw tokens
    through a tanh(g) ratio initialised to zero, so raw features are blended in
    only as training finds them useful.
    """

    def __init__(self, vlm_width: int, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.aq_projection = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.raw_projection = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.aq_query_norm = nn.LayerNorm(width)
        self.aq_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.raw_query_norm = nn.LayerNorm(width)
        self.raw_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.raw_gate = nn.Parameter(torch.zeros(()))
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))

    def forward(
        self,
        queries: torch.Tensor,
        aq_features: torch.Tensor,
        raw_features: torch.Tensor,
        raw_padding_mask: torch.Tensor | None,
        condition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.query_norm(queries)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + attended
        aq_keys = self.aq_projection(aq_features)
        if condition_tokens is not None:
            aq_keys = torch.cat((aq_keys, condition_tokens.to(aq_keys.dtype)), dim=1)
        attended, _ = self.aq_attention(self.aq_query_norm(queries), aq_keys, aq_keys, need_weights=False)
        raw_keys = self.raw_projection(raw_features)
        raw_attended, _ = self.raw_attention(
            self.raw_query_norm(queries), raw_keys, raw_keys,
            key_padding_mask=raw_padding_mask, need_weights=False,
        )
        queries = queries + attended + torch.tanh(self.raw_gate).to(raw_attended.dtype) * raw_attended
        return queries + self.ffn(self.ffn_norm(queries))


ARM_ID = {"robot": 0, "human": 1}
ANCHOR_MODES = ("absolute", "gravity")
PHASES = ("approach", "grasp", "transport", "release")


def anchor_features(anchor_pose: torch.Tensor, mode: str = "absolute") -> torch.Tensor:
    """9-d anchor input [B,9] of the current end-effector pose.

    ``absolute``: world position + rot6d (frame dependent).
    ``gravity``: world -z expressed in the end-effector frame, R^T [0,0,-1], zero padded
    to 9 so the projection keeps its shape. It encodes only the tilt, which is defined the
    same way in any z-up world, and drops position and heading.
    """
    if mode == "absolute":
        rotation_6d = matrix_to_rotation_6d(anchor_pose[..., :3, :3])
        return torch.cat((anchor_pose[..., :3, 3].float(), rotation_6d), dim=-1)
    if mode == "gravity":
        # R^T @ [0,0,-1] is minus the third row of R.
        gravity_local = -anchor_pose[..., 2, :3].float()
        return torch.cat((gravity_local, gravity_local.new_zeros((*gravity_local.shape[:-1], 6))), dim=-1)
    raise ValueError(f"anchor_mode must be one of {ANCHOR_MODES}, got {mode!r}")


class WaypointModel(nn.Module):
    """Predict relative SE(3) waypoints from task-conditioned action-token features."""

    def __init__(
        self, vlm_width: int, width: int = 256, heads: int = 4, ffn_width: int = 1024, layers: int = 2, num_waypoints: int = 4,
        predict_gripper: bool = True, bridge_layers: int = 0, anchor_mode: str = "absolute", num_phases: int = 0,
        predict_object: bool = False,
    ):
        super().__init__()
        self.width = width
        self.predict_gripper = bool(predict_gripper)
        self.bridge = int(bridge_layers) > 0
        if anchor_mode not in ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {ANCHOR_MODES}, got {anchor_mode!r}")
        self.anchor_mode = anchor_mode
        self.anchor_projection = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.arm_embedding = nn.Embedding(2, width)
        self.time_projection = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        if self.bridge:
            # One Bridge Attention block per selected Qwen layer (shallow -> deep).
            self.decoder = nn.ModuleList(
                [BridgeAttentionBlock(vlm_width, width, heads, ffn_width) for _ in range(int(bridge_layers))]
            )
        else:
            self.memory_projection = nn.Linear(vlm_width, width)
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
        # Optional auxiliary heads on the per-slot features; absent unless enabled so
        # checkpoints trained without them still load strictly.
        self.num_phases = int(num_phases)
        if self.num_phases:
            self.phase_head = nn.Linear(width, self.num_phases)
        if predict_object:
            # Grasped-object centre in the current end-effector frame at each future slot, in 0.1 m units.
            self.object_head = nn.Linear(width, 3)

    def auxiliary_outputs(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        if hasattr(self, "phase_head"):
            outputs["phase_logits"] = self.phase_head(features)
        if hasattr(self, "object_head"):
            outputs["object_position"] = self.object_head(features) * 0.1
        return outputs

    def forward(
        self,
        vlm_features: torch.Tensor,
        anchor_pose: torch.Tensor,
        tau: torch.Tensor,
        memory_attention_mask: torch.Tensor | None = None,
        bridge_aq: list[torch.Tensor] | None = None,
        bridge_raw: list[torch.Tensor] | None = None,
        arm_id: int = ARM_ID["robot"],
        return_features: bool = False,
    ):
        """(translation, rotation_6d, gripper), plus the per-slot features [B,K,width] if requested."""
        batch_size = anchor_pose.shape[0]
        times = tau.float().view(1, -1, 1).expand(batch_size, -1, -1)
        anchor_token = self.anchor_projection(anchor_features(anchor_pose, self.anchor_mode)).unsqueeze(1)
        arm = torch.full((batch_size, 1), int(arm_id), device=anchor_pose.device, dtype=torch.long)
        query = anchor_token + self.arm_embedding(arm) + self.time_projection(times)
        padding_mask = None if memory_attention_mask is None else ~memory_attention_mask.to(torch.bool)
        if self.bridge:
            if bridge_aq is None or bridge_raw is None or len(bridge_aq) != len(self.decoder) or len(bridge_raw) != len(self.decoder):
                raise ValueError(f"Bridge WaypointModel needs {len(self.decoder)} per-layer action-query and raw features")
            # The current pose (proprio) joins the action-query keys, as in VLA-Adapter.
            for layer, aq, raw in zip(self.decoder, bridge_aq, bridge_raw):
                query = layer(query, aq, raw, padding_mask, condition_tokens=anchor_token)
        else:
            memory = self.memory_projection(vlm_features)
            for layer in self.decoder:
                query = layer(query, memory, padding_mask)
        features = self.output_norm(query)
        translation = self.translation_head(features)
        rotation = self.rotation_head(features)
        if self.predict_gripper:
            gripper = self.gripper_head(features).squeeze(-1)
        else:
            gripper = translation.new_zeros(translation.shape[:-1])
        if return_features:
            return translation, rotation, gripper, features
        return translation, rotation, gripper


class SpatialAdapter(nn.Module):
    """Cross-attend action queries to explicit metric waypoint tokens."""

    def __init__(self, vlm_width: int, width: int = 256, heads: int = 4, num_waypoints: int = 4, gated: bool = True, bridge_layers: int = 0, ffn_width: int = 1024):
        super().__init__()
        # v1 (ungated) has no gripper token channel and no spatial gate.
        self.gated = bool(gated)
        self.bridge = int(bridge_layers) > 0
        if self.bridge and not self.gated:
            raise ValueError("Bridge SpatialAdapter requires the gated (v2) adapter")
        self.action_projection = nn.Sequential(nn.LayerNorm(vlm_width), nn.Linear(vlm_width, width))
        self.spatial_projection = nn.Sequential(
            nn.Linear(12 if self.gated else 11, width), nn.GELU(), nn.Linear(width, width)
        )
        if self.bridge:
            # Predicted metric waypoint tokens are the condition tokens of every Bridge block.
            self.blocks = nn.ModuleList(
                [BridgeAttentionBlock(vlm_width, width, heads, ffn_width) for _ in range(int(bridge_layers))]
            )
            self.output_norm = nn.LayerNorm(width)
        else:
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
        bridge_aq: list[torch.Tensor] | None = None,
        bridge_raw: list[torch.Tensor] | None = None,
        raw_padding_mask: torch.Tensor | None = None,
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
        if self.bridge:
            if bridge_aq is None or bridge_raw is None or len(bridge_aq) != len(self.blocks) or len(bridge_raw) != len(self.blocks):
                raise ValueError(f"Bridge SpatialAdapter needs {len(self.blocks)} per-layer action-query and raw features")
            update = queries
            for block, aq, raw in zip(self.blocks, bridge_aq, bridge_raw):
                update = block(update, aq, raw, raw_padding_mask, condition_tokens=keys)
            update = self.output_norm(update)
        else:
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
            "tau": [1, 2, 4, 8],
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
        # Bridge Attention (VLA-Adapter): hidden_states indices of the Qwen layers
        # read by both the waypoint decoder and the spatial action adapter.
        self.bridge_layers = tuple(int(v) for v in (spatial.get("bridge_layers") or ()))
        num_vlm_layers = int(self.qwen_vl_interface.model.config.get_text_config().num_hidden_layers)
        if self.bridge_layers:
            if tuple(sorted(set(self.bridge_layers))) != self.bridge_layers or not (
                1 <= self.bridge_layers[0] and self.bridge_layers[-1] <= num_vlm_layers
            ):
                raise ValueError(
                    f"spatial.bridge_layers must be increasing indices in [1, {num_vlm_layers}], got {self.bridge_layers}"
                )
        self.anchor_mode = str(spatial.get("anchor_mode", "absolute"))
        # Auxiliary supervision on the waypoint slots; heads exist only when weighted.
        self.phase_weight = float(spatial.get("phase_weight", 0.0))
        self.object_weight = float(spatial.get("object_weight", 0.0))
        if self.phase_weight < 0 or self.object_weight < 0:
            raise ValueError("spatial.phase_weight and spatial.object_weight must be non-negative")
        if (self.phase_weight or self.object_weight) and not self.bridge_layers:
            raise ValueError("Phase/object supervision is defined for the Bridge architecture only")
        self.waypoint_model = WaypointModel(
            hidden_size, self.spatial_width, heads, ffn_width, layers, len(self.waypoint_tau), predict_gripper=gated,
            bridge_layers=len(self.bridge_layers), anchor_mode=self.anchor_mode,
            num_phases=len(PHASES) if self.phase_weight > 0 else 0, predict_object=self.object_weight > 0,
        )
        self.spatial_adapter = SpatialAdapter(
            hidden_size, self.spatial_width, heads, len(self.waypoint_tau), gated=gated,
            bridge_layers=len(self.bridge_layers), ffn_width=ffn_width,
        )
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
        qwen_inputs, hidden_states = self._build_inputs_all_layers(examples)
        return qwen_inputs, hidden_states[-1]

    def _build_inputs_all_layers(self, examples: list[dict[str, Any]]) -> tuple[dict, tuple[torch.Tensor, ...]]:
        """Shared by the human and robot domains: same prompt, same action-query tokens."""
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
        return qwen_inputs, tuple(outputs.hidden_states)

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
        bridge_aq: list[torch.Tensor] | None = None,
        bridge_raw: list[torch.Tensor] | None = None,
        raw_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        conditioned, residual, spatial_gate = self.spatial_adapter(
            action_queries, translation, rotation_6d, gripper, tau,
            bridge_aq=bridge_aq, bridge_raw=bridge_raw, raw_padding_mask=raw_padding_mask,
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

    @staticmethod
    def _batch_domain(examples: list[dict]) -> str:
        domains = {str(example.get("domain", "robot")) for example in examples}
        if len(domains) != 1 or not domains <= {"robot", "human"}:
            raise ValueError(f"A batch must contain exactly one domain (robot or human), got {sorted(domains)}")
        return domains.pop()

    def _bridge_features(self, qwen_inputs: dict, hidden_states: tuple[torch.Tensor, ...]):
        """Per selected layer: action-query latents [B,chunk,H] and all raw tokens [B,L,H]."""
        if len(hidden_states) <= self.bridge_layers[-1]:
            raise ValueError(f"Qwen returned {len(hidden_states)} hidden states; bridge needs index {self.bridge_layers[-1]}")
        raw = [hidden_states[index] for index in self.bridge_layers]
        aq = [
            self._gather_action_token_embeddings(layer, qwen_inputs.get("input_ids"), action_token_id=self.action_token_id)
            for layer in raw
        ]
        return aq, raw

    def _predict_from_features(
        self,
        examples: list[dict],
        qwen_inputs: dict,
        last_hidden: torch.Tensor,
        *,
        train: bool,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
    ):
        device = last_hidden.device
        domain = self._batch_domain(examples)
        if domain == "human" and self.bridge_layers == ():
            raise ValueError("Human geometry pretraining is defined for the Bridge architecture only")
        anchor_pose = self._batch_geometry_current(examples, device)
        tau = torch.tensor(self.waypoint_tau, device=device, dtype=torch.float32) / float(max(self.waypoint_tau))
        bridge_aq = bridge_raw = None
        if self.bridge_layers:
            if hidden_states is None:
                raise ValueError("Bridge QwenSpatialOFT needs all Qwen hidden states")
            bridge_aq, bridge_raw = self._bridge_features(qwen_inputs, hidden_states)
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
            # The same WaypointModel.forward serves both domains; arm_id tells them apart.
            translation, rotation_6d, waypoint_gripper, slot_features = self.waypoint_model(
                waypoint_features,
                anchor_pose,
                tau,
                memory_attention_mask,
                bridge_aq=bridge_aq,
                bridge_raw=bridge_raw,
                arm_id=ARM_ID[domain],
                return_features=True,
            )
            auxiliary_outputs = self.waypoint_model.auxiliary_outputs(slot_features) if train else {}
            if domain == "robot":
                conditioned_queries, spatial_residual, extra_metrics = self._condition_action_queries(
                    examples,
                    action_queries,
                    translation,
                    rotation_6d,
                    waypoint_gripper,
                    tau,
                    bridge_aq=bridge_aq,
                    bridge_raw=bridge_raw,
                    raw_padding_mask=None if memory_attention_mask is None else ~memory_attention_mask.to(torch.bool),
                )
                pred_actions = self.action_model.predict_action(conditioned_queries)
        waypoint_prediction = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(len(examples), len(self.waypoint_tau), 1, 1)
        waypoint_prediction[..., :3, :3] = rotation_6d_to_matrix(rotation_6d)
        waypoint_prediction[..., :3, 3] = translation.float()
        if domain == "human":
            if not train:
                return None, waypoint_prediction, None, {}
            return self._human_geometry_loss(examples, waypoint_prediction, device, auxiliary_outputs)
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
        slot_losses = self._slot_supervision_losses(labels, auxiliary_outputs, device, domain)
        auxiliary_loss = (
            self.waypoint_weight * waypoint_loss
            + self.gripper_weight * gripper_loss
            + self.action_geometry_weight * action_geometry_loss
            + slot_losses.pop("slot_loss")
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
            **slot_losses,
            **memory_metrics,
            **extra_metrics,
        }

    def _bridge_metrics(self) -> dict[str, torch.Tensor]:
        if not self.bridge_layers:
            return {}
        ratios = [torch.tanh(block.raw_gate).detach() for block in self.waypoint_model.decoder]
        ratios += [torch.tanh(block.raw_gate).detach() for block in self.spatial_adapter.blocks]
        return {"bridge_raw_ratio_mean": torch.stack(ratios).abs().mean()}

    @staticmethod
    def _stack_slot_labels(labels: list[dict], key: str, valid_key: str, shape: tuple[int, ...], device, dtype):
        """Stack an optional per-slot label; the whole batch must provide it or none does."""
        present = [key in label for label in labels]
        if not any(present):
            return None, None
        if not all(present):
            raise ValueError(f"geometry_labels.{key} must be present for every example in a batch or for none")
        values = torch.as_tensor(np.stack([label[key] for label in labels]), device=device, dtype=dtype)
        valid = torch.as_tensor(np.stack([label[valid_key] for label in labels]), device=device, dtype=torch.bool)
        if tuple(values.shape) != (len(labels), *shape) or tuple(valid.shape) != (len(labels), shape[0]):
            raise ValueError(
                f"Invalid {key} labels: values={tuple(values.shape)}, valid={tuple(valid.shape)}, expected={(len(labels), *shape)}"
            )
        return values, valid

    def _slot_supervision_losses(self, labels: list[dict], outputs: dict, device, domain: str) -> dict:
        """Weighted phase (both domains) and object (where labelled) losses on the waypoint slots."""
        slots = len(self.waypoint_tau)
        zero = torch.zeros((), device=device, dtype=torch.float32)
        losses = {"slot_loss": zero}
        if self.phase_weight > 0:
            phase, valid = self._stack_slot_labels(labels, "future_phase", "future_phase_valid", (slots,), device, torch.long)
            if phase is None:
                raise ValueError(f"spatial.phase_weight > 0 needs geometry_labels.future_phase for the {domain} domain")
            logits = outputs["phase_logits"].float()
            if torch.any(valid):
                phase_loss = nn.functional.cross_entropy(logits[valid], phase[valid])
                phase_accuracy = (logits[valid].argmax(-1) == phase[valid]).float().mean()
            else:
                phase_loss, phase_accuracy = logits.sum() * 0.0, zero
            losses.update(phase_loss=phase_loss, phase_accuracy=phase_accuracy.detach())
            losses["slot_loss"] = losses["slot_loss"] + self.phase_weight * phase_loss
        if self.object_weight > 0:
            # LIBERO has no object pose labels; robot batches simply carry no object term.
            target, valid = self._stack_slot_labels(labels, "future_object", "future_object_valid", (slots, 3), device, torch.float32)
            if target is not None:
                prediction = outputs["object_position"].float()
                if torch.any(valid):
                    error = (prediction[valid] - target[valid]) / 0.1
                    object_loss = nn.functional.huber_loss(error, torch.zeros_like(error), delta=1.0)
                else:
                    object_loss = prediction.sum() * 0.0
                losses["object_loss"] = object_loss
                losses["slot_loss"] = losses["slot_loss"] + self.object_weight * object_loss
        return losses

    def _human_geometry_loss(
        self, examples: list[dict], waypoint_prediction: torch.Tensor, device: torch.device, auxiliary_outputs: dict | None = None
    ) -> dict:
        """Human batches: geometry (+ optional phase/object) supervision; no robot actions, no gripper labels."""
        labels = [example.get("geometry_labels") for example in examples]
        if any(label is None for label in labels):
            raise ValueError("Human geometry pretraining requires geometry_labels for every example")
        target_pose = torch.as_tensor(np.stack([label["relative_waypoints"] for label in labels]), device=device, dtype=torch.float32)
        future_valid = torch.as_tensor(np.stack([label["future_valid"] for label in labels]), device=device, dtype=torch.bool)
        if target_pose.shape != waypoint_prediction.shape or future_valid.shape != waypoint_prediction.shape[:2]:
            raise ValueError(
                f"Invalid human waypoint labels: poses={tuple(target_pose.shape)}, masks={tuple(future_valid.shape)}, "
                f"expected={tuple(waypoint_prediction.shape)}"
            )
        translation_loss, rotation_loss = relative_pose_loss(waypoint_prediction, target_pose, future_valid)
        waypoint_loss = translation_loss + self.waypoint_rotation_weight * rotation_loss
        slot_losses = self._slot_supervision_losses(labels, auxiliary_outputs or {}, device, "human")
        total_loss = self.waypoint_weight * waypoint_loss + slot_losses.pop("slot_loss")
        return {
            "total_loss": total_loss,
            "has_robot_action": False,
            "waypoint_loss": waypoint_loss,
            "waypoint_translation_loss": translation_loss,
            "waypoint_rotation_loss": rotation_loss,
            **slot_losses,
            **self._bridge_metrics(),
        }

    def forward(self, examples: list[dict] = None, **kwargs):
        qwen_inputs, hidden_states = self._build_inputs_all_layers(examples)
        outputs = self._predict_from_features(
            examples, qwen_inputs, hidden_states[-1], train=True, hidden_states=hidden_states
        )
        if outputs.get("has_robot_action", True):
            outputs.update(self._bridge_metrics())
        return outputs

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] = None, **kwargs):
        if type(examples) is not list:
            examples = [examples]
        qwen_inputs, hidden_states = self._build_inputs_all_layers(examples)
        pred_actions, _, _, _ = self._predict_from_features(
            examples, qwen_inputs, hidden_states[-1], train=False, hidden_states=hidden_states
        )
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}

    @torch.inference_mode()
    def predict_waypoints(self, examples: list[dict] = None, **kwargs) -> torch.Tensor:
        """Relative SE(3) waypoints [B,K,4,4] for either domain; used for geometry validation."""
        if type(examples) is not list:
            examples = [examples]
        qwen_inputs, hidden_states = self._build_inputs_all_layers(examples)
        _, waypoints, _, _ = self._predict_from_features(
            examples, qwen_inputs, hidden_states[-1], train=False, hidden_states=hidden_states
        )
        return waypoints


__all__ = ["QwenSpatialOFT", "WaypointModel", "SpatialAdapter"]

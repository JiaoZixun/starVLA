"""FastVLA spatial extension for the RoboDojo Qwen3.5-0.8B baseline.

The legacy QwenOFT path is kept intact.  The new branch adds a small waypoint
decoder and a zero-initialized residual adapter, so a checkpoint from QwenOFT
has exactly the same action output before spatial training starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT, masked_l1_loss
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY, get_training_dtype
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from .spatial_geometry import inverse_se3, relative_se3, rot6d_to_matrix, pose_to_features, masked_mean


@dataclass
class QwenSpatialDefaultConfig:
    name: str = "QwenSpatialOFT"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "Qwen/Qwen3.5-0.8B", "attn_implementation": "sdpa"})
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP", "action_dim": 14, "action_horizon": 50,
        "hidden_size": 1024, "add_pos_embed": True, "max_seq_len": 1024})
    spatial: dict = field(default_factory=lambda: {
        "enabled": True, "width": 256, "layers": 2, "heads": 4,
        "ffn_width": 1024, "tau": [1, 8, 16, 32, 50],
        "lambda_wp": 1.0, "lambda_rel": 0.25, "lambda_geom": 0.1})


class WaypointDecoder(nn.Module):
    def __init__(self, hidden_size: int, width: int, layers: int, heads: int, ffn_width: int, tau: list[int]):
        super().__init__()
        self.width, self.tau = width, tuple(int(x) for x in tau)
        self.memory = nn.Linear(hidden_size, width)
        layer = nn.TransformerDecoderLayer(width, heads, dim_feedforward=ffn_width,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.arm_embed = nn.Embedding(2, width)
        self.time_embed = nn.Embedding(max(self.tau) + 1, width)
        self.out = nn.Linear(width, 9)
        self.token_proj = nn.Linear(9, width)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[3] = 1.0
            self.out.bias[7] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b = features.shape[0]
        mem = self.memory(features)
        arms = torch.arange(2, device=features.device).repeat_interleave(len(self.tau))
        times = torch.tensor(self.tau, device=features.device).repeat(2)
        q = self.arm_embed(arms) + self.time_embed(times)
        q = q.unsqueeze(0).expand(b, -1, -1)
        return self.out(self.decoder(q, mem)).view(b, 2, len(self.tau), 9)


class SpatialAdapter(nn.Module):
    def __init__(self, hidden_size: int, width: int = 256, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, width)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.up = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, action_hidden: torch.Tensor, spatial_tokens: torch.Tensor) -> torch.Tensor:
        q = self.down(self.norm(action_hidden))
        u, _ = self.cross_attn(q, spatial_tokens, spatial_tokens, need_weights=False)
        return action_hidden + self.up(u)


@FRAMEWORK_REGISTRY.register("QwenSpatialOFT")
class QwenSpatialOFT(Qwenvl_OFT):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        # Build the legacy modules with the exact QwenOFT constructor/config.
        super().__init__(config)
        cfg = merge_framework_config(QwenSpatialDefaultConfig, config)
        self.spatial_config = cfg.framework.spatial
        self.spatial_enabled = bool(self.spatial_config.get("enabled", True))
        d = int(self.qwen_vl_interface.model.config.hidden_size)
        tau = list(self.spatial_config.get("tau", [1, 8, 16, 32, 50]))
        horizon = int(self.action_horizon)
        tau = sorted(set(max(1, min(horizon, int(t))) for t in tau))
        self.spatial_config.tau = tau
        self.waypoint_model = WaypointDecoder(d, int(self.spatial_config.get("width", 256)),
                                              int(self.spatial_config.get("layers", 2)),
                                              int(self.spatial_config.get("heads", 4)),
                                              int(self.spatial_config.get("ffn_width", 1024)), tau)
        self.spatial_adapter = SpatialAdapter(d, int(self.spatial_config.get("width", 256)),
                                              int(self.spatial_config.get("heads", 4)))
        self.register_buffer("_unit_pose", torch.eye(4), persistent=False)
        self.allow_missing_checkpoint_prefixes = ("waypoint_model.", "spatial_adapter.")

    def _inputs_and_hidden(self, examples, inference=False):
        images = [to_pil_preserve(e["image"]) if inference else e["image"] for e in examples]
        if inference:
            size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            if size: images = resize_images(images, target_size=size)
        instructions = [e["lang"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None
        if state is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, state)
        suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.action_token * self.chunk_len}<action>."
        qin = self.qwen_vl_interface.build_qwenvl_inputs(images=images,
                                                          instructions=[x + suffix for x in instructions])
        with torch.autocast("cuda", dtype=get_training_dtype()):
            # The spatial branch only consumes the final representation.  Asking
            # Transformers for all layer outputs retains one activation tensor per
            # layer until the forward returns and needlessly raises the peak.
            out = self.qwen_vl_interface(
                **qin,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.last_hidden_state
        ids = qin.get("input_ids")
        ah = self._gather_action_token_embeddings(hidden, ids, self.action_token_id)
        return qin, hidden, ah

    def _spatial_forward(self, hidden, action_hidden, examples):
        if not self.spatial_enabled or not all("geometry" in e and "current_pose" in e["geometry"] for e in examples):
            return action_hidden, hidden.new_zeros(()), hidden.new_zeros(()), hidden.new_zeros(())
        wp_raw = self.waypoint_model(hidden)
        # 9 = translation + rotation 6D; keep explicit numeric values in the token path.
        pred_pose_feat = wp_raw
        z = pred_pose_feat.reshape(pred_pose_feat.shape[0], -1, 9)
        z = torch.cat((z[..., :3] / 0.1, z[..., 3:]), dim=-1)
        # RoboDojo relation token is derived from both predicted arms and is differentiable.
        rel_token = (pred_pose_feat[:, 1] - pred_pose_feat[:, 0]).mean(1, keepdim=True)
        z = self.waypoint_model.token_proj(torch.cat((z, rel_token), dim=1))
        adapted = self.spatial_adapter(action_hidden, z)
        tau = self.waypoint_model.tau
        current = torch.as_tensor(np.stack([e["geometry"]["current_pose"] for e in examples]), device=hidden.device, dtype=torch.float32)
        if not all("future_pose" in e["geometry"] and "future_valid" in e["geometry"] for e in examples):
            return adapted, hidden.new_zeros(()), hidden.new_zeros(()), hidden.new_zeros(())
        target_list, valid_list = [], []
        for e in examples:
            g = e["geometry"]
            target_list.append(torch.as_tensor(g["future_pose"], device=hidden.device, dtype=torch.float32))
            valid_list.append(torch.as_tensor(g["future_valid"], device=hidden.device, dtype=torch.bool))
        future = torch.stack(target_list)
        valid = torch.stack(valid_list)
        target_rel = relative_se3(current[:, :, None], future)
        target_feat = pose_to_features(target_rel)
        pred_feat = wp_raw.float()
        trans = torch.nn.functional.smooth_l1_loss(pred_feat[..., :3] / 0.1, target_feat[..., :3] / 0.1, reduction="none").mean(-1)
        rot_pred = rot6d_to_matrix(pred_feat[..., 3:])
        rot_tgt = target_rel[..., :3, :3]
        rot = torch.nn.functional.smooth_l1_loss(rot_pred, rot_tgt, reduction="none").mean((-1, -2))
        wp_loss = masked_mean(trans + 0.25 * rot, valid)
        both = valid[:, 0] & valid[:, 1]
        # Build homogeneous predicted poses for the dual-arm relation target.
        ph = torch.eye(4, device=hidden.device, dtype=pred_feat.dtype).expand(*pred_feat.shape[:3], 4, 4).clone()
        ph[..., :3, :3], ph[..., :3, 3] = rot_pred, pred_feat[..., :3]
        pred_abs = current[:, :, None] @ ph
        rel_pred = inverse_se3(pred_abs[:, 0]) @ pred_abs[:, 1]
        rel_tgt = inverse_se3(future[:, 0]) @ future[:, 1]
        rel_loss = masked_mean(torch.nn.functional.smooth_l1_loss(rel_pred[..., :3, 3], rel_tgt[..., :3, 3], reduction="none").mean(-1), both)
        return adapted, wp_loss, rel_loss, hidden.new_zeros(())

    def forward(self, examples: List[dict] = None, **kwargs):
        _, hidden, ah = self._inputs_and_hidden(examples, inference=False)
        with torch.autocast("cuda", dtype=get_training_dtype()):
            adapted, wp_loss, rel_loss, geom_loss = self._spatial_forward(hidden, ah, examples)
            pred = self.action_model.predict_action(adapted)
            targets = torch.as_tensor(np.asarray([e["action"] for e in examples]), device=pred.device, dtype=torch.float32)[:, -self.action_horizon:]
            pred = pred.float()
            masks = [e.get("action_mask") for e in examples]
            mask = None if all(x is None for x in masks) else torch.as_tensor(np.asarray(masks), device=pred.device, dtype=torch.bool)[:, -self.action_horizon:]
            bc_loss = masked_l1_loss(pred, targets, mask)
        lam = self.spatial_config
        total = bc_loss + float(lam.get("lambda_wp", 1.0)) * wp_loss + float(lam.get("lambda_rel", .25)) * rel_loss + float(lam.get("lambda_geom", .1)) * geom_loss
        return {"action_loss": bc_loss, "bc_loss": bc_loss, "waypoint_loss": wp_loss,
                "relation_loss": rel_loss, "action_geometry_loss": geom_loss, "total_loss": total}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        if not isinstance(examples, list): examples = [examples]
        _, hidden, ah = self._inputs_and_hidden(examples, inference=True)
        with torch.autocast("cuda", dtype=get_training_dtype()):
            adapted, _, _, _ = self._spatial_forward(hidden, ah, examples)
            pred = self.action_model.predict_action(adapted)
        return {"normalized_actions": pred.detach().float().cpu().numpy()}

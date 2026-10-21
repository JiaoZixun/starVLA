"""Spatial OFT plus a shared-vision causal execution-event memory."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.memory_v2 import (
    ACTION_HORIZON,
    MEMORY_CAPACITY_EVENTS,
    MEMORY_STRIDE,
    VISUAL_FEATURE_DIM,
    VISUAL_TOKENS_PER_VIEW,
    VISUAL_VIEWS,
    image_to_memory_visual_tokens,
)
from starVLA.model.framework.VLM4A.QwenOFT import masked_l1_loss
from starVLA.model.framework.VLM4A.QwenSpatialOFT import QwenSpatialOFT
from starVLA.model.framework.VLM4A.execution_event_memory import (
    CausalMemoryReader,
    ExecutionEventEncoder,
    MemoryConfig,
    se3_exp,
)
from starVLA.model.framework.VLM4A.spatial_geometry import (
    matrix_to_rotation_6d,
    relative_pose_loss,
    rotation_6d_to_matrix,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenSpatialMemOFTDefaultConfig:
    name: str = "QwenSpatialMemOFT"
    qwenvl: dict = field(default_factory=dict)
    action_model: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)
    memory: dict = field(
        default_factory=lambda: {
            "width": 256,
            "heads": 4,
            "ffn_width": 1024,
            "event_encoder_layers": 1,
            "reader_layers": 2,
            "event_tokens": 8,
            "summary_tokens": 1,
            "capacity_events": 128,
            "visual_pool_grid": [4, 4],
            "dropout": 0.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenSpatialMemOFT")
class QwenSpatialMemOFT(QwenSpatialOFT):
    """Reuse original Spatial heads and add zero-initialized memory residuals."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        memory = self.config.framework.get("memory", {})
        grid = tuple(int(v) for v in memory.get("visual_pool_grid", [4, 4]))
        if grid != (4, 4):
            raise ValueError("Spatial-Mem v2 contract fixes post-merger visual pooling at 4x4 per view")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError("Spatial-Mem v2 requires the original 8-action chunk")
        self.memory_config = MemoryConfig(
            width=int(memory.get("width", 256)),
            heads=int(memory.get("heads", 4)),
            ffn_width=int(memory.get("ffn_width", 1024)),
            event_encoder_layers=int(memory.get("event_encoder_layers", 1)),
            reader_layers=int(memory.get("reader_layers", 2)),
            event_tokens=int(memory.get("event_tokens", 8)),
            summary_tokens=int(memory.get("summary_tokens", 1)),
            capacity_events=int(memory.get("capacity_events", MEMORY_CAPACITY_EVENTS)),
            visual_pool_grid=grid,
            action_horizon=self.action_horizon,
            dropout=float(memory.get("dropout", 0.0)),
        )
        width = self.memory_config.width
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.event_encoder = ExecutionEventEncoder(VISUAL_FEATURE_DIM, self.memory_config)
        self.memory_reader = CausalMemoryReader(self.memory_config)
        self.action_memory_projection = nn.Linear(hidden_size, width)
        self.action_memory_gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
        self.action_memory_output = nn.Linear(width, hidden_size)
        self.waypoint_memory_projection = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.Linear(width, width))
        self.proprio_memory_projection = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.task_memory_projection = nn.Linear(hidden_size, width)
        self.waypoint_twist_head = nn.Linear(width, 6)
        nn.init.zeros_(self.action_memory_output.weight)
        nn.init.zeros_(self.action_memory_output.bias)
        nn.init.zeros_(self.waypoint_twist_head.weight)
        nn.init.zeros_(self.waypoint_twist_head.bias)
        # Runtime-only encoded-event cache. Event IDs are episode-local, and
        # cached tokens never enter state_dict/checkpoints.
        self._eval_event_cache: OrderedDict[str, dict[tuple[int, int], np.ndarray]] = OrderedDict()
        self._max_cached_episodes = 64

    def _build_inputs_with_visual(self, examples: list[dict[str, Any]]) -> tuple[dict, torch.Tensor, torch.Tensor]:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if image_size:
            batch_images = resize_images(batch_images, target_size=image_size)
        instructions = [example["lang"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None
        if states is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, states)
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=inputs["input_ids"].device.type == "cuda"):
            hidden, _ = self.qwen_vl_interface.forward_features(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                pool_grid=self.memory_config.visual_pool_grid,
            )
        visual_rows = []
        for example in examples:
            visual = example.get("memory_visual_tokens")
            if visual is None:
                visual = np.stack([image_to_memory_visual_tokens(image) for image in example["image"]])
            visual = np.asarray(visual, dtype=np.float32)
            if visual.shape != (VISUAL_VIEWS, VISUAL_TOKENS_PER_VIEW, VISUAL_FEATURE_DIM):
                raise ValueError(f"memory_visual_tokens must be [2,16,3], got {visual.shape}")
            if not np.isfinite(visual).all() or visual.min(initial=0) < 0 or visual.max(initial=0) > 1:
                raise ValueError("memory_visual_tokens must be finite RGB values in [0,1]")
            visual_rows.append(visual)
        visual = torch.as_tensor(np.stack(visual_rows), device=hidden.device, dtype=torch.float32)
        return inputs, hidden, visual

    @staticmethod
    def _tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _encode_event_lists(
        self,
        event_lists: list[list[dict[str, Any]]],
        device: torch.device,
        *,
        train: bool,
        episode_ids: list[str | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(event_lists)
        max_events = max((len(events) for events in event_lists), default=0)
        token_count = self.memory_config.tokens_per_event
        if max_events == 0:
            return (
                torch.zeros((batch, 0, token_count, self.memory_config.width), device=device),
                torch.zeros((batch, 0), dtype=torch.bool, device=device),
                torch.zeros((batch, 0), dtype=torch.long, device=device),
                torch.zeros((batch, 0), dtype=torch.long, device=device),
            )

        valid = torch.zeros((batch, max_events), dtype=torch.bool, device=device)
        starts = torch.zeros((batch, max_events), dtype=torch.long, device=device)
        ends = torch.zeros((batch, max_events), dtype=torch.long, device=device)
        missing_events: list[dict[str, Any]] = []
        cached_tokens: dict[tuple[int, int], np.ndarray] = {}
        for b, events in enumerate(event_lists):
            previous_end = -1
            for index, event in enumerate(events):
                start, end = int(event["start_step"]), int(event["end_step"])
                if start < 0 or end - start != ACTION_HORIZON or (previous_end >= 0 and start != previous_end):
                    raise ValueError("Memory events must be ordered, contiguous 8-step intervals")
                previous_end = end
                starts[b, index], ends[b, index] = start, end
                valid[b, index] = True
                cache_key = (start, end)
                episode_cache = None
                if not train and episode_ids[b] is not None:
                    episode_cache = self._eval_event_cache.setdefault(str(episode_ids[b]), {})
                    if cache_key in episode_cache:
                        cached_tokens[(b, index)] = episode_cache[cache_key]
                        continue
                missing_events.append(event)

        encoded = torch.zeros(
            (batch, max_events, token_count, self.memory_config.width), device=device, dtype=torch.float32
        )
        if missing_events:
            encoded_missing = self.event_encoder(
                self._tensor(np.stack([event["visual_start"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["visual_end"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["state_start"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["state_end"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["executed_actions"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["pose_start_in_episode"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["pose_end_in_episode"] for event in missing_events]), device, torch.float32),
                self._tensor([event["elapsed_steps"] for event in missing_events], device, torch.float32),
            )
            missing_index = 0
            for b, events in enumerate(event_lists):
                for index, event in enumerate(events):
                    cache_key = (int(event["start_step"]), int(event["end_step"]))
                    if (b, index) in cached_tokens:
                        encoded[b, index] = self._tensor(cached_tokens[(b, index)], device, torch.float32)
                        continue
                    tokens = encoded_missing[missing_index]
                    encoded[b, index] = tokens
                    if not train and episode_ids[b] is not None:
                        episode_cache = self._eval_event_cache[str(episode_ids[b])]
                        episode_cache[cache_key] = tokens.detach().float().cpu().numpy().astype(np.float16)
                    missing_index += 1

        if not train:
            for episode_id in episode_ids:
                if episode_id is not None:
                    self._eval_event_cache.setdefault(str(episode_id), {})
                    self._eval_event_cache.move_to_end(str(episode_id))
            while len(self._eval_event_cache) > self._max_cached_episodes:
                self._eval_event_cache.popitem(last=False)
        return encoded, valid, starts, ends

    @property
    def memory_encoder(self):
        """Compatibility name used by the policy server's memory contract."""
        return self.event_encoder

    def _batch_anchor_poses(self, examples: list[dict], device: torch.device) -> torch.Tensor:
        poses = []
        for example in examples:
            current = example.get("geometry_current")
            if not isinstance(current, dict) or "eef_pose" not in current:
                raise ValueError("QwenSpatialMemOFT requires measured geometry_current.eef_pose")
            pose = np.asarray(current["eef_pose"], dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"geometry_current.eef_pose must be finite [4,4], got {pose.shape}")
            poses.append(pose)
        return torch.as_tensor(np.stack(poses), device=device, dtype=torch.float32)

    def _batch_states(self, examples: list[dict], device: torch.device) -> torch.Tensor:
        states = []
        for example in examples:
            if "state" not in example:
                raise ValueError("QwenSpatialMemOFT requires the same current 8D state in train and eval")
            state = np.asarray(example["state"], dtype=np.float32).reshape(-1)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError(f"Current state must be finite [8], got {state.shape}")
            states.append(state)
        return torch.as_tensor(np.stack(states), device=device, dtype=torch.float32)

    def _predict_from_features(
        self,
        examples: list[dict],
        qwen_inputs: dict,
        last_hidden: torch.Tensor,
        visual: torch.Tensor,
        *,
        train: bool,
        event_lists: list[list[dict[str, Any]]] | None = None,
    ):
        device = last_hidden.device
        batch = len(examples)
        anchor_pose = self._batch_anchor_poses(examples, device)
        states = self._batch_states(examples, device)
        tau = torch.tensor(self.waypoint_tau, device=device, dtype=torch.float32) / float(max(self.waypoint_tau))
        action_queries = self._gather_action_token_embeddings(
            last_hidden, qwen_inputs["input_ids"], action_token_id=self.action_token_id
        )
        attention_mask = qwen_inputs.get("attention_mask")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            base_translation, base_rotation, waypoint_gripper = self.waypoint_model(
                last_hidden, anchor_pose, tau, attention_mask
            )
            base_waypoints = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(
                batch, len(self.waypoint_tau), 1, 1
            )
            base_waypoints[..., :3, :3] = rotation_6d_to_matrix(base_rotation)
            base_waypoints[..., :3, 3] = base_translation.float()

            if event_lists is None:
                event_lists = [list(example.get("memory_events", [])) for example in examples]
            episode_ids = [example.get("memory_episode_id") for example in examples]
            event_tokens, event_valid, event_starts, event_ends = self._encode_event_lists(
                event_lists, device, train=train, episode_ids=episode_ids
            )
            query_steps = []
            for index, (example, events) in enumerate(zip(examples, event_lists)):
                contract = example.get("memory_contract")
                if not isinstance(contract, dict) or contract.get("history_is_strictly_past") is not True:
                    raise ValueError("QwenSpatialMemOFT requires a strictly-past memory_contract")
                if int(contract.get("decision_stride_env_steps", -1)) != MEMORY_STRIDE:
                    raise ValueError("Memory event stride differs from the model's 8-step action horizon")
                query_step = int(example.get("env_step", contract.get("source_frame_index", -1)))
                if query_step < 0 or query_step % MEMORY_STRIDE:
                    raise ValueError(f"Memory query step must be nonnegative and stride-aligned, got {query_step}")
                if any(int(event["start_step"]) >= query_step or int(event["end_step"]) > query_step for event in events):
                    raise ValueError(f"Sample {index} includes a current/future execution event")
                query_steps.append(query_step)
            query_steps = torch.as_tensor(query_steps, device=device, dtype=torch.long).view(batch, 1).expand(
                -1, len(self.waypoint_tau) + self.action_horizon
            )

            action_memory_queries = self.action_memory_projection(action_queries)
            base_pose_features = torch.cat(
                (
                    base_waypoints[..., :3, 3].float(),
                    matrix_to_rotation_6d(base_waypoints[..., :3, :3]),
                    tau.view(1, -1, 1).expand(batch, -1, -1),
                ),
                dim=-1,
            )
            task_features = self.task_memory_projection(action_queries.mean(dim=1)).unsqueeze(1)
            waypoint_memory_queries = (
                self.waypoint_memory_projection(base_pose_features)
                + self.proprio_memory_projection(states).unsqueeze(1)
                + task_features
            )
            memory_queries = torch.cat((action_memory_queries, waypoint_memory_queries), dim=1)
            memory_context, allowed = self.memory_reader(
                memory_queries,
                event_tokens.to(memory_queries.dtype),
                event_valid,
                query_steps,
                event_starts,
                event_ends,
            )
            action_context = memory_context[:, : self.action_horizon]
            waypoint_context = memory_context[:, self.action_horizon :]
            action_has_history = allowed[:, : self.action_horizon].any(dim=-1, keepdim=True)
            waypoint_has_history = allowed[:, self.action_horizon :].any(dim=-1, keepdim=True)
            action_gate = self.action_memory_gate(torch.cat((action_memory_queries, action_context), dim=-1))
            action_residual = self.action_memory_output(action_gate * action_context)
            action_residual = action_residual * action_has_history.to(action_residual.dtype)
            conditioned_queries = action_queries + action_residual
            twist = self.waypoint_twist_head(waypoint_context).float()
            with torch.autocast(device_type=device.type, enabled=False):
                corrected_with_history = base_waypoints.float() @ se3_exp(twist.float())
            corrected_waypoints = torch.where(
                waypoint_has_history.squeeze(-1)[..., None, None],
                corrected_with_history,
                base_waypoints.float(),
            )
            corrected_translation = corrected_waypoints[..., :3, 3]
            corrected_rotation = matrix_to_rotation_6d(corrected_waypoints[..., :3, :3])
            spatial_translation = torch.where(
                waypoint_has_history,
                corrected_translation,
                base_translation.float(),
            )
            spatial_rotation = torch.where(
                waypoint_has_history,
                corrected_rotation,
                base_rotation.float(),
            )
            conditioned_queries, spatial_residual, spatial_gates = self.spatial_adapter(
                conditioned_queries,
                spatial_translation,
                spatial_rotation,
                waypoint_gripper,
                tau,
            )
            pred_actions = self.action_model.predict_action(conditioned_queries)

        if not train:
            return pred_actions, corrected_waypoints, spatial_residual, allowed

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
            raise ValueError("Training QwenSpatialMemOFT requires geometry_labels")
        target_pose = torch.as_tensor(
            np.stack([label["relative_waypoints"] for label in labels]), device=device, dtype=torch.float32
        )
        future_valid = torch.as_tensor(
            np.stack([label["future_valid"] for label in labels]), device=device, dtype=torch.bool
        )
        translation_loss, rotation_loss = relative_pose_loss(corrected_waypoints, target_pose, future_valid)
        waypoint_loss = translation_loss + self.waypoint_rotation_weight * rotation_loss
        gripper_target = torch.as_tensor(
            np.stack([label["future_gripper"] for label in labels]), device=device, dtype=waypoint_gripper.dtype
        )
        gripper_valid = torch.as_tensor(
            np.stack([label["future_gripper_valid"] for label in labels]), device=device, dtype=torch.bool
        )
        gripper_loss = (
            torch.abs(waypoint_gripper.float() - gripper_target.float())[gripper_valid].mean()
            if torch.any(gripper_valid)
            else bc_loss.new_zeros(())
        )
        total_loss = bc_loss + self.waypoint_weight * waypoint_loss + self.gripper_weight * gripper_loss
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
            "spatial_residual_ratio": residual_ratio,
            "spatial_gate_mean": spatial_gates.float().mean(),
            "spatial_gate_std": spatial_gates.float().std(unbiased=False),
            "memory_events_visible": allowed.sum(dim=-1).float().mean(),
        }

    def forward(self, examples: list[dict] = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, hidden, visual = self._build_inputs_with_visual(examples)
        return self._predict_from_features(examples, qwen_inputs, hidden, visual, train=True)

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, hidden, visual = self._build_inputs_with_visual(examples)
        actions, _, _, _ = self._predict_from_features(examples, qwen_inputs, hidden, visual, train=False)
        return {"normalized_actions": actions.detach().float().cpu().numpy()}


__all__ = ["QwenSpatialMemOFT"]

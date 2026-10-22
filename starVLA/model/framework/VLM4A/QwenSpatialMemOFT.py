"""Spatial OFT plus a shared-vision causal execution-event memory."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import hashlib

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.memory_v2 import (
    ACTION_HORIZON,
    MEMORY_CAPACITY_EVENTS,
    MEMORY_STRIDE,
    QWEN_VISUAL_FEATURE_DIM,
    VISUAL_FEATURE_DIM,
    VISUAL_TOKENS_PER_VIEW,
    VISUAL_VIEWS,
    image_to_memory_visual_tokens,
)
from starVLA.model.framework.VLM4A.proprio_layer_bridge import (
    BridgeConfig,
    ProprioLayerBridge,
    bridge_token_types,
    proprio_features,
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


def vision_tower_fingerprint(qwen_vl_interface) -> str:
    """sha256 over the Qwen vision tower's bf16 weights, in name order.

    Shared by the Qwen visual sidecar writer and the model so the sidecar,
    the training weights, and the eval weights can be proven identical.
    """
    digest = hashlib.sha256()
    visual = qwen_vl_interface.model.model.visual
    for name, tensor in sorted(visual.state_dict().items()):
        value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.int16).numpy().tobytes())
    return digest.hexdigest()


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
            "history_dropout": 0.0,
            "event_action_dropout": 0.0,
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
            history_dropout=float(memory.get("history_dropout", 0.0)),
            event_action_dropout=float(memory.get("event_action_dropout", 0.0)),
        )
        for name in ("history_dropout", "event_action_dropout"):
            if not 0.0 <= getattr(self.memory_config, name) < 1.0:
                raise ValueError(f"memory.{name} must be in [0, 1)")
        width = self.memory_config.width
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        # "rgb": 4x4 RGB means (memv2 checkpoints).  "qwen": the frozen Qwen
        # vision tower's pooled post-merger tokens, from an offline sidecar in
        # training and from this model's own forward in eval.
        self.memory_visual_source = str(memory.get("visual_source", "rgb"))
        if self.memory_visual_source not in ("rgb", "qwen"):
            raise ValueError(f"memory.visual_source must be rgb or qwen, got {self.memory_visual_source!r}")
        qwen_visual = self.memory_visual_source == "qwen"
        if qwen_visual and hidden_size != QWEN_VISUAL_FEATURE_DIM:
            raise ValueError(f"Qwen visual memory expects {QWEN_VISUAL_FEATURE_DIM}-D features, got {hidden_size}")
        self.memory_visual_fingerprint = memory.get("visual_fingerprint")
        if qwen_visual and not self.memory_visual_fingerprint:
            raise ValueError("memory.visual_source=qwen requires memory.visual_fingerprint from the sidecar manifest")
        self._vision_fingerprint_verified = False
        self.event_encoder = ExecutionEventEncoder(
            QWEN_VISUAL_FEATURE_DIM if qwen_visual else VISUAL_FEATURE_DIM,
            self.memory_config,
            visual_input_norm=qwen_visual,
        )
        bridge = self.config.framework.get("bridge")
        self.bridge_config = BridgeConfig.from_dict(dict(bridge)) if bridge else None
        if self.bridge_config is not None:
            layer_count = int(self.qwen_vl_interface.model.config.text_config.num_hidden_layers)
            if max(self.bridge_config.hidden_layers) > layer_count:
                raise ValueError(f"bridge.hidden_layers exceed the {layer_count} Qwen decoder layers")
            self.proprio_layer_bridge = ProprioLayerBridge(hidden_size, self.bridge_config)
            self.image_token_id = int(self.qwen_vl_interface.model.config.image_token_id)
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
        # Qwen-visual eval cache: episode -> {env_step: fp16 [2,16,D]} for the
        # latest query, which is the start frame of the next event.
        self._eval_visual_cache: OrderedDict[str, dict[int, np.ndarray]] = OrderedDict()
        self._max_cached_episodes = 64

    def vision_tower_fingerprint(self) -> str:
        return vision_tower_fingerprint(self.qwen_vl_interface)

    def _verify_qwen_visual_contract(self, examples: list[dict[str, Any]], *, train: bool) -> None:
        """Training sidecar, training weights, and eval weights must share one vision tower."""
        if self.memory_visual_source != "qwen":
            return
        if train:
            trainable = [n for n, p in self.qwen_vl_interface.model.model.visual.named_parameters() if p.requires_grad]
            if trainable:
                raise ValueError(
                    "memory.visual_source=qwen requires a frozen vision tower "
                    "(trainer.freeze_modules: qwen_vl_interface.model.model.visual); "
                    f"trainable: {trainable[:3]}"
                )
            for example in examples:
                found = example.get("memory_contract", {}).get("visual_fingerprint")
                if found != self.memory_visual_fingerprint:
                    raise ValueError(f"Sample visual sidecar fingerprint {found} != {self.memory_visual_fingerprint}")
        if not self._vision_fingerprint_verified:
            actual = self.vision_tower_fingerprint()
            if actual != self.memory_visual_fingerprint:
                raise ValueError(
                    f"Vision tower fingerprint {actual} != memory.visual_fingerprint {self.memory_visual_fingerprint}; "
                    "the Qwen visual sidecar was computed from different vision weights"
                )
            self._vision_fingerprint_verified = True

    def _build_inputs_with_visual(
        self, examples: list[dict[str, Any]]
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if image_size:
            batch_images = resize_images(batch_images, target_size=image_size)
        instructions = [example["lang"] for example in examples]
        # The reused Spatial OFT checkpoint was trained without a state prompt.
        # Memory proprioception arrives as ``memory_state``; a stray ``state``
        # would silently change the frozen Qwen input, so reject it.
        if any("state" in example for example in examples):
            raise ValueError(
                "QwenSpatialMemOFT does not use a state prompt; pass proprioception as memory_state"
            )
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        hidden_layers = self.bridge_config.hidden_layers if self.bridge_config is not None else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=inputs["input_ids"].device.type == "cuda"):
            outputs = self.qwen_vl_interface.forward_features(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                pool_grid=self.memory_config.visual_pool_grid,
                hidden_layers=hidden_layers,
            )
        hidden, pooled = outputs[0], outputs[1]
        layer_features = outputs[2] if hidden_layers is not None else None
        if self.memory_visual_source == "qwen":
            # Same rounding as the fp16 sidecar used in training.
            visual = pooled.detach().to(torch.float16).float()
            return inputs, hidden, visual, layer_features
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
        return inputs, hidden, visual, layer_features

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
        visual_lookup=None,
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
        missing_keys: list[tuple[int, int, int]] = []  # (batch row, start, end)
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
                missing_keys.append((b, start, end))

        encoded = torch.zeros(
            (batch, max_events, token_count, self.memory_config.width), device=device, dtype=torch.float32
        )
        action_dropped = None
        if missing_events and train and self.memory_config.event_action_dropout > 0:
            action_dropped = (
                torch.rand(len(missing_events), device=device) < self.memory_config.event_action_dropout
            )
        if missing_events:
            if visual_lookup is not None:
                visual_start, visual_end = visual_lookup(missing_keys)
            else:
                visual_start = self._tensor(np.stack([event["visual_start"] for event in missing_events]), device, torch.float32)
                visual_end = self._tensor(np.stack([event["visual_end"] for event in missing_events]), device, torch.float32)
            encoded_missing = self.event_encoder(
                visual_start,
                visual_end,
                self._tensor(np.stack([event["state_start"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["state_end"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["executed_actions"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["pose_start_in_episode"] for event in missing_events]), device, torch.float32),
                self._tensor(np.stack([event["pose_end_in_episode"] for event in missing_events]), device, torch.float32),
                self._tensor([event["elapsed_steps"] for event in missing_events], device, torch.float32),
                action_dropped=action_dropped,
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
                        # Keep fp32 so cached and freshly encoded events are identical.
                        episode_cache[cache_key] = tokens.detach().float().cpu().numpy()
                    missing_index += 1

        if not train:
            for episode_id in episode_ids:
                if episode_id is not None:
                    self._eval_event_cache.setdefault(str(episode_id), {})
                    self._eval_event_cache.move_to_end(str(episode_id))
            while len(self._eval_event_cache) > self._max_cached_episodes:
                self._eval_event_cache.popitem(last=False)
        return encoded, valid, starts, ends

    def _qwen_visual_lookup(
        self,
        examples: list[dict[str, Any]],
        event_lists: list[list[dict[str, Any]]],
        episode_ids: list[str | None],
        current_visual: torch.Tensor,
        query_steps: list[int],
        *,
        train: bool,
    ):
        """Return a ``[(row, start, end)] -> (visual_start, visual_end)`` resolver.

        Rows with ``memory_visual_bank`` (training samples) read the fp16 Qwen
        sidecar; eval rows read the features this model produced at earlier
        queries of the same episode.  Both hold the same frozen-tower values.
        """
        device = current_visual.device
        bank_rows: dict[tuple[int, int], int] = {}
        bank_parts = []
        offset = 0
        for b, example in enumerate(examples):
            steps = example.get("memory_visual_bank_steps")
            if steps is None:
                if train and event_lists[b]:
                    raise ValueError("Training QwenSpatialMemOFT with qwen visuals requires memory_visual_bank")
                continue
            values = np.asarray(example["memory_visual_bank"])
            steps = np.asarray(steps).reshape(-1)
            if values.dtype != np.float16 or values.shape != (len(steps), VISUAL_VIEWS, VISUAL_TOKENS_PER_VIEW, QWEN_VISUAL_FEATURE_DIM):
                raise ValueError(f"memory_visual_bank must be fp16 [N,2,16,{QWEN_VISUAL_FEATURE_DIM}], got {values.shape}/{values.dtype}")
            bank_parts.append(torch.from_numpy(values))
            bank_rows.update({(b, int(step)): offset + index for index, step in enumerate(steps)})
            offset += len(steps)
        bank = torch.cat(bank_parts).to(device, non_blocking=True) if bank_parts else None

        for b, example in enumerate(examples):
            if example.get("memory_visual_bank_steps") is not None:
                continue
            if episode_ids[b] is None:
                # A sample without history (e.g. the first 8 steps of a
                # training episode) needs no visuals; eval clients always send
                # an episode id so later queries can find this frame.
                if event_lists[b]:
                    raise ValueError("Eval with qwen visual memory requires memory_episode_id")
                continue
            cache = self._eval_visual_cache.setdefault(str(episode_ids[b]), {})
            cache[int(query_steps[b])] = current_visual[b].detach().to(torch.float16).cpu().numpy()

        def lookup(keys: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
            def frame(b: int, step: int) -> torch.Tensor:
                if (b, step) in bank_rows:
                    return bank[bank_rows[(b, step)]]
                cache = self._eval_visual_cache.get(str(episode_ids[b]), {})
                if step not in cache:
                    raise KeyError(f"No Qwen visual feature for episode {episode_ids[b]} step {step}")
                return torch.from_numpy(cache[step]).to(device)

            if bank is not None and all((b, s) in bank_rows and (b, e) in bank_rows for b, s, e in keys):
                # Training: one gather on the GPU, no per-event host copies.
                starts = torch.as_tensor([bank_rows[(b, s)] for b, s, _ in keys], device=device)
                ends = torch.as_tensor([bank_rows[(b, e)] for b, _, e in keys], device=device)
                return bank[starts].float(), bank[ends].float()
            visual_start = torch.stack([frame(b, s) for b, s, _ in keys]).float()
            visual_end = torch.stack([frame(b, e) for b, _, e in keys]).float()
            return visual_start, visual_end

        return lookup

    def _prune_eval_visual_cache(self, episode_ids: list[str | None], query_steps: list[int]) -> None:
        # The next event starts at this query; older frames are never needed.
        for episode_id, step in zip(episode_ids, query_steps):
            if episode_id is None or str(episode_id) not in self._eval_visual_cache:
                continue
            cache = self._eval_visual_cache[str(episode_id)]
            for key in [key for key in cache if key != int(step)]:
                del cache[key]
            self._eval_visual_cache.move_to_end(str(episode_id))
        while len(self._eval_visual_cache) > self._max_cached_episodes:
            self._eval_visual_cache.popitem(last=False)

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
            if "memory_state" not in example:
                raise ValueError("QwenSpatialMemOFT requires the same current 8D memory_state in train and eval")
            state = np.asarray(example["memory_state"], dtype=np.float32).reshape(-1)
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
        layer_features: torch.Tensor | None = None,
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
        self._verify_qwen_visual_contract(examples, train=train)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            bridge_residual = None
            if self.bridge_config is not None:
                if layer_features is None:
                    raise ValueError("bridge is enabled but no Qwen layer features were computed")
                token_types = bridge_token_types(qwen_inputs["input_ids"], self.image_token_id)
                # Measured TCP pose + gripper qpos, the same values eval sends.
                action_queries, bridge_residual = self.proprio_layer_bridge(
                    action_queries,
                    layer_features,
                    token_types,
                    attention_mask,
                    proprio_features(anchor_pose, states),
                )
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
            if train and self.memory_config.history_dropout > 0:
                # Whole-history dropout falls back to the exact Spatial path
                # (no visible events), which is also what eval sees at step 0.
                keep = (torch.rand(batch) >= self.memory_config.history_dropout).tolist()
                event_lists = [events if kept else [] for events, kept in zip(event_lists, keep)]
            episode_ids = [example.get("memory_episode_id") for example in examples]
            query_steps = []
            for index, (example, events) in enumerate(zip(examples, event_lists)):
                contract = example.get("memory_contract")
                if not isinstance(contract, dict) or contract.get("history_is_strictly_past") is not True:
                    raise ValueError("QwenSpatialMemOFT requires a strictly-past memory_contract")
                if int(contract.get("decision_stride_env_steps", -1)) != MEMORY_STRIDE:
                    raise ValueError("Memory event stride differs from the model's 8-step action horizon")
                query_step = int(example.get("env_step", contract.get("source_frame_index", -1)))
                # Evaluators never send event_phase (phase 0; their buffer
                # enforces stride-aligned queries).  Training samples, including
                # the trainer's in-loop action eval, may use phase-shifted grids
                # that still end exactly at the query step.
                phase = int(contract.get("event_phase", 0))
                if query_step < 0 or not 0 <= phase < MEMORY_STRIDE or query_step % MEMORY_STRIDE != phase:
                    raise ValueError(f"Memory query step {query_step} does not match event phase {phase}")
                if any(int(event["start_step"]) >= query_step or int(event["end_step"]) > query_step for event in events):
                    raise ValueError(f"Sample {index} includes a current/future execution event")
                query_steps.append(query_step)
            visual_lookup = None
            if self.memory_visual_source == "qwen":
                visual_lookup = self._qwen_visual_lookup(
                    examples, event_lists, episode_ids, visual, query_steps, train=train
                )
            event_tokens, event_valid, event_starts, event_ends = self._encode_event_lists(
                event_lists, device, train=train, episode_ids=episode_ids, visual_lookup=visual_lookup
            )
            if visual_lookup is not None and not train:
                self._prune_eval_visual_cache(episode_ids, query_steps)
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
            if self.gripper_weight != 0.0 and torch.any(gripper_valid)
            else bc_loss.new_zeros(())
        )
        total_loss = bc_loss + self.waypoint_weight * waypoint_loss + self.gripper_weight * gripper_loss
        query_scale = action_queries.detach().float().square().mean().sqrt() + 1e-8
        residual_ratio = spatial_residual.detach().float().square().mean().sqrt() / query_scale
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
            # How much memory and the bridge actually move the action queries.
            "memory_residual_ratio": action_residual.detach().float().square().mean().sqrt() / query_scale,
            "waypoint_twist_norm": twist.detach().float().norm(dim=-1).mean(),
            **(
                {"bridge_residual_ratio": bridge_residual.detach().float().square().mean().sqrt() / query_scale}
                if bridge_residual is not None
                else {}
            ),
        }

    def forward(self, examples: list[dict] = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, hidden, visual, layer_features = self._build_inputs_with_visual(examples)
        return self._predict_from_features(
            examples, qwen_inputs, hidden, visual, train=True, layer_features=layer_features
        )

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, hidden, visual, layer_features = self._build_inputs_with_visual(examples)
        actions, _, _, _ = self._predict_from_features(
            examples, qwen_inputs, hidden, visual, train=False, layer_features=layer_features
        )
        return {"normalized_actions": actions.detach().float().cpu().numpy()}


__all__ = ["QwenSpatialMemOFT"]

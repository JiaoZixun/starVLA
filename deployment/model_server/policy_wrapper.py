# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper using the training-time modality transforms."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework, merge_config_overrides
from starVLA.model.framework.share_tools import read_mode_config
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.robodojo_memory_adapter import RoboDojoMemorySessions
from starVLA.utils.robodojo_memory import RoboDojoMemorySpec


def _training_obs_image_size(model_cfg: Dict[str, Any]) -> Optional[List[int]]:
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if size is None and "default_image_resolution" in vla_data_cfg:
        resolution = vla_data_cfg["default_image_resolution"]
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            size = resolution[-2:]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    return [int(size[0]), int(size[1])]


class PolicyServerWrapper:
    """Wrap a trained framework for websocket inference."""

    def __init__(self, ckpt_path: str, device: str = "cuda", use_bf16: bool = False,
                 unnorm_key: Optional[str] = None, config_overrides: Sequence[str] | None = None) -> None:
        self._ckpt_path = str(ckpt_path)
        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path, config_overrides=config_overrides)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        self._framework = framework.to(device).eval()

        model_cfg, norm_stats = read_mode_config(self._ckpt_path)
        model_cfg = merge_config_overrides(model_cfg, config_overrides)
        self._model_cfg = model_cfg
        memory = model_cfg.get("framework", {}).get("memory", {})
        self._memory_sessions = None
        if model_cfg.get("framework", {}).get("name") == "QwenSpatialMemOFT":
            self._memory_sessions = RoboDojoMemorySessions(RoboDojoMemorySpec(
                int(memory.get("recent_records", 16)),
                int(memory.get("decision_stride_env_steps", 8)),
                bool(memory.get("use_pose", False)),
            ))

        action_cfg = model_cfg["framework"]["action_model"]
        if "action_horizon" in action_cfg:
            self._action_chunk_size = int(action_cfg["action_horizon"])
        elif "future_action_window_size" in action_cfg:
            self._action_chunk_size = int(action_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(f"No action horizon found in model config for {self._ckpt_path}")

        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}
        self._available_unnorm_keys = list(norm_stats.keys())
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            processor = self._get_processor(unnorm_key)
            self._default_unnorm_key = processor.unnorm_key
            logging.info("PolicyServerWrapper ready: action_chunk_size=%d, unnorm_key=%s",
                         self._action_chunk_size, processor.unnorm_key)

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        key = unnorm_key if unnorm_key is not None else "__default__"
        if key not in self._norm_processors:
            self._norm_processors[key] = PolicyNormProcessor(self._ckpt_path, unnorm_key=unnorm_key)
        return self._norm_processors[key]

    def get_norm_processor(self, unnorm_key: Optional[str] = None) -> PolicyNormProcessor:
        return self._get_processor(unnorm_key if unnorm_key is not None else self._default_unnorm_key)

    @property
    def metadata(self) -> Dict[str, Any]:
        result = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "training_data_mix": self._model_cfg.get("datasets", {}).get("vla_data", {}).get("data_mix"),
            "training_obs_image_size": _training_obs_image_size(self._model_cfg),
            "eval_image_contract": (
                "Eval clients must explicitly provide the training camera count and order; "
                "the server does not infer or reorder camera views."
            ),
        }
        if self._default_unnorm_key is not None:
            processor = self._get_processor(self._default_unnorm_key)
            result["action_keys"] = processor.action_keys
            result["state_keys"] = processor.state_keys
        if self._memory_sessions is not None:
            spec = self._memory_sessions.spec
            result["execution_memory"] = {
                "recent_records": spec.recent_records,
                "stride": spec.decision_stride,
                "use_pose": spec.use_pose,
                "required_fields": ["env_id", "episode_id", "env_step", "raw_state", "memory_feedback"],
                "feedback": "Every completed environment step: source_step, raw state[14], actual executed_action[14]",
                "representation": "raw_dual_arm_abs_qpos_v1",
            }
        return result

    def predict_action(self, examples: List[dict], unnorm_key: Optional[str] = None, **kwargs) -> Dict[str, np.ndarray]:
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) != 1:
                raise ValueError(f"Specify unnorm_key from {self._available_unnorm_keys}")
            effective_key = self._available_unnorm_keys[0]
        processor = self._get_processor(effective_key)

        updates = None
        if self._memory_sessions is not None:
            expected_size = _training_obs_image_size(self._model_cfg)
            configured_size = kwargs.pop("eval_image_size", None)
            if configured_size is not None and expected_size is not None and list(configured_size) != expected_size:
                raise ValueError(f"Eval image size {configured_size} differs from training size {expected_size}")
            examples, updates = self._memory_sessions.prepare(examples, processor.apply_state, namespace=effective_key)
        output = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(output["normalized_actions"])
        if normalized.ndim != 3 or normalized.shape[0] != len(examples):
            raise ValueError(f"Expected normalized actions [B,T,D], got {normalized.shape}")
        actions = np.stack([processor.unapply_actions(normalized[b]) for b in range(normalized.shape[0])], axis=0)
        if updates is not None:
            self._memory_sessions.commit(updates)
        return {"actions": actions}

    def reset_memory(self) -> None:
        if self._memory_sessions is not None:
            self._memory_sessions.reset()

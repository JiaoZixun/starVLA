# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

import os
import time
from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


def resize_image_like_training(image: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    """Resize a LIBERO frame with the same PIL filter used by the train loader."""
    arr = np.asarray(image)
    if arr.shape[:2] == tuple(target_hw):
        return arr
    return np.asarray(
        Image.fromarray(arr).resize(
            (int(target_hw[1]), int(target_hw[0])), Image.Resampling.BICUBIC
        )
    )


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Sequence[int] = (224, 224),
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self._server_metadata = meta

        # This adapter implements the 7D LIBERO action contract and the
        # 8-step QwenOFT chunk protocol. Fail before running hundreds of
        # episodes if the server loaded a different checkpoint/config.
        if self.action_chunk_size != 8:
            raise ValueError(
                f"LIBERO QwenOFT expects action_chunk_size=8, got "
                f"{self.action_chunk_size}; server metadata={meta}"
            )
        expected_action_keys = [
            "action.x",
            "action.y",
            "action.z",
            "action.roll",
            "action.pitch",
            "action.yaw",
            "action.gripper",
        ]
        server_action_keys = meta.get("action_keys")
        if server_action_keys is not None and list(server_action_keys) != expected_action_keys:
            raise ValueError(
                f"LIBERO expects action keys {expected_action_keys}, got "
                f"{server_action_keys}; server metadata={meta}"
            )

        available_unnorm_keys = meta.get("available_unnorm_keys", [])
        resolved_unnorm_key = unnorm_key or meta.get("default_unnorm_key")
        if resolved_unnorm_key is None and len(available_unnorm_keys) == 1:
            resolved_unnorm_key = available_unnorm_keys[0]
        if resolved_unnorm_key is None:
            raise ValueError(
                "LIBERO requires an explicit action unnormalization key; "
                f"available keys={available_unnorm_keys}, server metadata={meta}"
            )
        if available_unnorm_keys and resolved_unnorm_key not in available_unnorm_keys:
            raise ValueError(
                f"unnorm_key={resolved_unnorm_key!r} is not available from the server: "
                f"{available_unnorm_keys}"
            )

        self.image_size: tuple = tuple(image_size)
        self.policy_setup = policy_setup
        self.unnorm_key = resolved_unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None
        self.record_inference_timing = os.getenv("LIBERO_RECORD_INFERENCE_TIMING", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self.inference_timing_by_task: dict[int, list[float]] = {}
        self.inference_timing_task_descriptions: dict[int, str] = {}
        self._timing_task_id: Optional[int] = None

    def set_timing_task_context(self, task_id: int, task_description: str) -> None:
        """Set the task label used by optional per-request inference timing."""
        self._timing_task_id = int(task_id)
        self.inference_timing_task_descriptions[self._timing_task_id] = str(task_description)
        self.inference_timing_by_task.setdefault(self._timing_task_id, [])

    def inference_timing_summary(self, episodes_per_task: int) -> dict:
        """Summarize client-observed action-chunk request latency in milliseconds."""
        def summarize(values: list[float]) -> dict:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return {"requests": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
            return {
                "requests": int(arr.size),
                "mean_ms": float(arr.mean()),
                "median_ms": float(np.median(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
            }

        all_values = [latency for values in self.inference_timing_by_task.values() for latency in values]
        return {
            "measurement": (
                "Wall-clock around WebsocketClientPolicy.predict_action, from client call to decoded response; "
                "includes client validation/serialization, websocket round trip, server batching wait and policy "
                "inference, and excludes simulator stepping and image/state preparation."
            ),
            "unit": "ms",
            "action_chunk_size": self.action_chunk_size,
            "episodes_per_task": int(episodes_per_task),
            "overall": summarize(all_values),
            "tasks": [
                {
                    "task_id": task_id,
                    "task_description": self.inference_timing_task_descriptions.get(task_id, ""),
                    **summarize(self.inference_timing_by_task[task_id]),
                }
                for task_id in sorted(self.inference_timing_by_task)
            ],
        }

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Match LeRobotDataset._pack_sample exactly: training resizes RGB frames
        # with PIL's default bicubic filter. Using bilinear here was a silent
        # train/eval visual-domain shift that affected both the released 4B
        # checkpoint and the currently trained 0.8B checkpoint.
        if self.image_size and example.get("image"):
            target_hw = self.image_size  # (H, W)
            resized = [resize_image_like_training(img, target_hw) for img in example["image"]]
            example = {**example, "image": resized}

        # Refresh chunk if needed.
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            # Embodied policies degrade SILENTLY (no error) when the eval-time observation
            # differs from what the model saw during TRAINING. Verify these match the
            # training config used for this checkpoint:
            #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
            #   - image size  : resize / crop resolution (e.g. 224x224)
            #   - image count : how many camera views are fed
            #   - image order : the ordering of those camera views
            #   - action normalization: unnorm_key must match the training dataset stats
            # ==============================================================================
            if self.record_inference_timing:
                inference_start = time.perf_counter()
                response = self.client.predict_action(vla_input)
                latency_ms = (time.perf_counter() - inference_start) * 1000.0
                if self._timing_task_id is not None:
                    self.inference_timing_by_task[self._timing_task_id].append(latency_ms)
            else:
                response = self.client.predict_action(vla_input)
            try:
                actions_batch = np.asarray(
                    response["data"]["actions"], dtype=np.float32
                )  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            expected_shape = (1, self.action_chunk_size, 7)
            if actions_batch.shape != expected_shape:
                raise ValueError(
                    f"Expected one unnormalized LIBERO action chunk with shape "
                    f"{expected_shape}, got {actions_batch.shape}; "
                    f"server metadata={self._server_metadata}"
                )
            if not np.isfinite(actions_batch).all():
                raise ValueError("Policy server returned non-finite actions")
            self.raw_actions = actions_batch[0]  # (T, D)

        chunk_index = step % self.action_chunk_size
        raw_actions = self.raw_actions[chunk_index][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {"raw_action": raw_action}

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

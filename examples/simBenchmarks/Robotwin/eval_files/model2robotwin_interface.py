from collections import deque
import uuid
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.dataloader.memory_visual import image_to_memory_tokens

try:
    from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    _STATE_TO_CONTRACT = np.asarray(
        [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.int64
    )
    _ACTION_TO_ENV = np.asarray(
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13], dtype=np.int64
    )

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None
        self.env_id = "robotwin"
        self.episode_id = None
        self.pending_memory_feedback = []
        self.last_observation_state = None
        self.last_observation_visual_tokens = None
        self.last_executed_action = None

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None
        self.episode_id = uuid.uuid4().hex
        self.pending_memory_feedback.clear()
        self.last_observation_state = None
        self.last_observation_visual_tokens = None
        self.last_executed_action = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        if not isinstance(example, dict):
            raise TypeError(f"RobotWin example must be a dict, got {type(example).__name__}")
        state = example.get("state")
        if state is None:
            raise ValueError("RobotWin memory evaluation requires a 14D state")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"RobotWin state must be finite shape (14,), got {state.shape}")
        contract_state = state[self._STATE_TO_CONTRACT]
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example.get("image")
        if not isinstance(images, (list, tuple)) or len(images) != 3:
            raise ValueError("RobotWin memory evaluation requires exactly 3 images: head, left wrist, right wrist")
        for index, image in enumerate(images):
            value = np.asarray(image)
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"RobotWin image[{index}] must be HWC RGB, got {value.shape}")
            if value.dtype != np.uint8:
                raise ValueError(f"RobotWin image[{index}] must be uint8 RGB, got {value.dtype}")

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        # Build memory patches from the original RGB observations using the
        # same shared transform as offline sidecar preprocessing. The policy's
        # ordinary vision input remains at the configured 224x224 resolution.
        visual_tokens = np.stack([
            image_to_memory_tokens(np.asarray(image), resize_size=256)
            for image in images
        ])
        images = [self._resize_image(image) for image in images]
        example["image"] = images
        if self.episode_id is None:
            self.reset(task_description)
        if step > 0 and self.last_observation_state is not None and self.last_executed_action is not None:
            if self.last_observation_visual_tokens is None:
                raise RuntimeError("Missing visual patches for the previous observation memory record")
            self.pending_memory_feedback.append({
                "source_step": int(step - 1),
                "state": self.last_observation_state.tolist(),
                "executed_action": self.last_executed_action.tolist(),
                "visual_tokens": self.last_observation_visual_tokens.copy(),
            })
        example_copy = {"image": example["image"], "lang": task_description,
                        "env_id": self.env_id, "episode_id": self.episode_id,
                        "env_step": int(step), "raw_state": contract_state.tolist(),
                        "memory_feedback": list(self.pending_memory_feedback)}
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
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
            response = self.client.predict_action(vla_input)
            # server already un-normalized via training-time transform
            raw_actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            if raw_actions.ndim != 3 or raw_actions.shape[0] != 1 or raw_actions.shape[2] != 14:
                raise ValueError(
                    "RobotWin policy must return actions with shape (1, horizon, 14), "
                    f"got {raw_actions.shape}"
                )
            raw_actions = raw_actions[0]
            if raw_actions.shape[0] != action_chunk_size:
                raise ValueError(
                    f"Policy returned horizon {raw_actions.shape[0]}, expected server action_chunk_size "
                    f"{action_chunk_size}"
                )
            self.pending_memory_feedback.clear()

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        # Dataset/model order [left arm, left gripper, right arm, right gripper]
        # is converted back to RoboTwin's environment order [left arm, right arm, grippers].
        if current_action.shape != (14,) or not np.isfinite(current_action).all():
            raise ValueError(f"Policy action must be finite shape (14,), got {current_action.shape}")
        env_action = current_action[self._ACTION_TO_ENV]
        self.last_observation_state = contract_state.copy()
        self.last_observation_visual_tokens = visual_tokens.copy()
        self.last_executed_action = current_action.astype(np.float32, copy=True)
        return env_action

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    # RoboTwin observations use [left arm, right arm, grippers].
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)

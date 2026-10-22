# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import os
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


def _resolve_local_qwen35_checkpoint(
    model_id: str,
    revision: Optional[str],
    *,
    cache_dir: Optional[str] = None,
) -> str:
    """Resolve a Qwen3.5 model id to a local snapshot without network access.

    Transformers 5.x may call ``huggingface_hub.model_info`` while constructing
    a tokenizer when it receives a Hub repo id.  That happens even when all
    model files are already cached.  Passing the concrete snapshot directory
    avoids that remote metadata lookup entirely.
    """

    candidate = Path(str(model_id)).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())

    try:
        snapshot = snapshot_download(
            repo_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception as exc:
        cache_hint = f" in cache `{cache_dir}`" if cache_dir else " in the local Hugging Face cache"
        raise FileNotFoundError(
            f"Qwen3.5 checkpoint `{model_id}`{cache_hint} is unavailable. "
            "Training is configured for offline loading and will not contact Hugging Face; "
            "set framework.qwenvl.local_model_path to a complete local snapshot."
        ) from exc

    resolved = Path(snapshot).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Resolved Qwen3.5 snapshot is not a directory: {resolved}")
    return str(resolved)


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        # Keep the Hub id for framework dispatch, but allow launchers to inject
        # an exact local snapshot path.  This is useful for offline training and
        # prevents Transformers from performing a remote model-info request.
        local_model_path = qwenvl_config.get("local_model_path")
        if local_model_path:
            model_id = str(local_model_path)
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        revision = qwenvl_config.get("revision")
        local_files_only = qwenvl_config.get("local_files_only", True)
        if isinstance(local_files_only, str):
            local_files_only = local_files_only.lower() in {"1", "true", "yes", "on"}
        # The offline environment variables are an additional safety net for
        # launch scripts.  They must never be able to turn an offline run back
        # into a network request.
        if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
            local_files_only = True

        if local_files_only:
            model_id = _resolve_local_qwen35_checkpoint(
                model_id,
                revision,
                cache_dir=qwenvl_config.get("cache_dir"),
            )
            hub_kwargs = {}
            logger.info(f"Using local Qwen3.5 checkpoint: {model_id}")
        else:
            hub_kwargs = {"revision": revision} if revision else {}

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
            **hub_kwargs,
        )
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only, **hub_kwargs)
        processor.tokenizer.padding_side = "left"

        if qwenvl_config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def _per_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        image_outputs = self.model.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        per_image_features = image_outputs.pooler_output
        if not isinstance(per_image_features, (tuple, list)):
            raise TypeError("Qwen3.5 image pooler_output must retain one tensor per image")
        return list(per_image_features)

    def _pool_image_features(
        self,
        per_image_features: list[torch.Tensor],
        image_grid_thw: torch.Tensor,
        batch_size: int,
        pool_grid: tuple[int, int],
    ) -> torch.Tensor:
        """Average post-merger image tokens into a fixed grid: [B, views, gh*gw, D].

        Shared by the training/eval forward and the offline memory sidecar so
        both produce the same features from the same vision tower.
        """
        image_count = len(per_image_features)
        if image_count % batch_size:
            raise ValueError(f"Image count {image_count} is not divisible by batch size {batch_size}")
        views = image_count // batch_size
        pooled = []
        merge = int(self.model.model.visual.spatial_merge_size)
        for features, grid in zip(per_image_features, image_grid_thw):
            grid_height, grid_width = int(grid[1]), int(grid[2])
            if grid_height % merge or grid_width % merge:
                raise ValueError(f"Image grid {grid.tolist()} is not divisible by merge size {merge}")
            height, width = grid_height // merge, grid_width // merge
            if features.shape[0] != height * width:
                raise ValueError(f"Image feature count disagrees with grid {grid.tolist()}")
            feature_map = features.transpose(0, 1).reshape(1, features.shape[-1], height, width).float()
            bins = torch.nn.functional.adaptive_avg_pool2d(feature_map, pool_grid)
            pooled.append(bins.flatten(2).transpose(1, 2).squeeze(0).to(features.dtype))
        return torch.stack(pooled, dim=0).reshape(batch_size, views, pool_grid[0] * pool_grid[1], -1)

    @torch.no_grad()
    def encode_pooled_images(self, images: list[list], pool_grid: tuple[int, int] = (4, 4)) -> torch.Tensor:
        """Vision-tower-only pooled features for ``images`` ([B][views] PIL images).

        Uses the same processor preprocessing and pooling as ``forward_features``;
        the vision tower does not see the text, so no prompt is required.
        """
        if not images or any(len(views) != len(images[0]) for views in images):
            raise ValueError("encode_pooled_images expects a non-empty batch with a fixed view count")
        flat = [image for views in images for image in views]
        processed = self.processor.image_processor(images=flat, return_tensors="pt")
        device = next(self.model.model.visual.parameters()).device
        pixel_values = processed["pixel_values"].to(device)
        image_grid_thw = processed["image_grid_thw"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            per_image = self._per_image_features(pixel_values, image_grid_thw)
            return self._pool_image_features(per_image, image_grid_thw, len(images), pool_grid)

    def forward_features(
        self,
        *,
        pool_grid: tuple[int, int] = (4, 4),
        hidden_layers: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """Return final text features without materializing all hidden states or logits.

        Spatial OFT regresses actions from action-token features. Calling the
        multimodal base model directly avoids the language-model head and, by
        default, the tuple of every layer's hidden states, which otherwise costs
        memory and time for a large batch. The vision tower runs once and
        supplies both the language sequence and compact pooled image features.

        ``hidden_layers`` selects entries of the language model's hidden-state
        tuple (0 = embeddings, i = output of decoder layer i; the last entry is
        post-norm) and returns them as a third value ``[B, L, T, D]``.
        """
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        input_ids = kwargs.pop("input_ids", None)
        if pixel_values is None or image_grid_thw is None or input_ids is None:
            raise ValueError("Qwen3.5 forward_features requires pixel_values, image_grid_thw and input_ids")

        base_model = self.model.model
        per_image_features = self._per_image_features(pixel_values, image_grid_thw)

        inputs_embeds = base_model.get_input_embeddings()(input_ids)
        image_embeds = torch.cat(per_image_features, dim=0).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        image_mask, _ = base_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        cache_position = kwargs.pop("cache_position", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("output_attentions", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("pixel_values_videos", None)
        kwargs.pop("video_grid_thw", None)
        if position_ids is None:
            position_ids = base_model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=mm_token_type_ids,
            )
        language_outputs = base_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=False,
            output_hidden_states=hidden_layers is not None,
            output_attentions=False,
            return_dict=True,
            **kwargs,
        )

        visual = self._pool_image_features(
            per_image_features, image_grid_thw, int(input_ids.shape[0]), pool_grid
        )
        if hidden_layers is None:
            return language_outputs.last_hidden_state, visual
        all_hidden = language_outputs.hidden_states
        if any(not 0 <= int(index) < len(all_hidden) for index in hidden_layers):
            raise ValueError(f"hidden_layers={hidden_layers} outside [0, {len(all_hidden)})")
        selected = torch.stack([all_hidden[int(index)] for index in hidden_layers], dim=1)
        return language_outputs.last_hidden_state, visual, selected

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass

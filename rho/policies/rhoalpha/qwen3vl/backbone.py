"""
Qwen3-VL backbone adapter for RhoAlphaModel.

Handles Qwen3-VL specific initialization, processing, and hidden state
extraction. Supports multiple model sizes (2B, 4B, 8B).

Key difference from Qwen2.5-VL: hidden_size lives under
``config.text_config.hidden_size`` rather than ``config.hidden_size``.
"""

import logging

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from rho.policies.rhoalpha.backbone import BackboneAdapter

logger = logging.getLogger(__name__)

QWEN3VL_DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

_VALID_QWEN3VL_MODELS = {
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
}


class Qwen3VLBackbone(BackboneAdapter):
    """
    Backbone adapter for Qwen3-VL.

    Differences from Qwen2.5-VL:
    - Supports configurable model sizes (2B/4B/8B) via vlm_model_name
    - Hidden size accessed via config.text_config.hidden_size (composite config)
    - Uses Qwen3VLForConditionalGeneration instead of Qwen2_5_VL
    """

    def get_model_id(self) -> str:
        return self.config.vlm_model_name or QWEN3VL_DEFAULT_MODEL_ID

    def get_hidden_size(self, backbone: nn.Module) -> int:
        return backbone.config.text_config.hidden_size

    def load_backbone(self) -> nn.Module:
        model_id = self.get_model_id()

        if model_id not in _VALID_QWEN3VL_MODELS:
            raise ValueError(
                f"Invalid vlm_model_name: '{model_id}'. Must be one of: {sorted(_VALID_QWEN3VL_MODELS)}"
            )

        logger.info(f"Loading Qwen3-VL backbone: {model_id}")
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )

        if self.config.enable_gradient_checkpointing:
            if hasattr(backbone, "gradient_checkpointing_enable"):
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                raise ValueError("Model does not support gradient checkpointing")

        return backbone

    def create_processor(self):
        return AutoProcessor.from_pretrained(self.get_model_id())

    def prepare_prompt(self, batch) -> list[str]:
        from rho.common.constants import OBSERVATION_LANG as OBS_TASK

        if OBS_TASK not in batch:
            logger.warning(f"Key {OBS_TASK} not found in batch. Returning empty prompt list.")
            return []

        num_images = len(self.config.image_features) * self.config.n_obs_steps
        image_token = "".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in range(num_images)])

        prefix = (
            "<|im_start|>system\nYou are a helpful assistant. "
            "You will be provided with images and a task description, "
            "output the next action to send to the robot.<|im_end|>\n"
        )
        user_prompt = "<|im_start|>user\n"
        assistant_prompt = "<|im_start|>assistant"
        prompt_suffix = "<|im_end|>"

        formatted = []
        for prompt in batch[OBS_TASK]:
            if not prompt.startswith(user_prompt):
                prompt = prefix + user_prompt + image_token + prompt + "\n"
            if not prompt.endswith(f"{prompt_suffix}{assistant_prompt}"):
                prompt = prompt + prompt_suffix + assistant_prompt
            formatted.append(prompt)
        return formatted

    def remove_audio_components(self, backbone: nn.Module):
        """No-op: Qwen3-VL has no audio components."""
        pass

    def get_language_model(self, backbone: nn.Module) -> nn.Module:
        return backbone.model.language_model

    def set_vision_requires_grad(self, backbone: nn.Module):
        if hasattr(backbone, "visual"):
            for param in backbone.visual.parameters():
                param.requires_grad = False

    def get_freezing_components(self, backbone: nn.Module) -> dict:
        base_model = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
        return {
            "VLM Backbone": backbone,
            "  Language Model": base_model.model if hasattr(base_model, "model") else None,
            "  Vision Encoder": base_model.visual if hasattr(base_model, "visual") else None,
        }

    def process_batch(
        self,
        processor,
        images,
        texts,
        image_mask,
        pad_sequence_fn=None,
        cat_with_pad_fn=None,
        max_length=8192,
    ) -> dict:
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False,
            padding_side="left",
            max_length=max_length,
        )
        return inputs

    def get_image_text_hidden_state(
        self,
        backbone: nn.Module,
        processor,
        vlm_projector: nn.Module,
        hidden_state_idx: int,
        image,
        text,
        image_mask,
        convert_image_fn,
        pad_sequence_fn,
        cat_with_pad_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.process_batch(processor, image, text, image_mask)
        inputs = {k: v.to(backbone.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        attention_masks = inputs["attention_mask"].to(backbone.device)

        outputs = backbone(**inputs, output_hidden_states=True)
        selected_hidden_state = outputs.hidden_states[hidden_state_idx]
        selected_hidden_state = vlm_projector(selected_hidden_state)

        return selected_hidden_state, attention_masks

    def get_all_hidden_states(
        self,
        backbone: nn.Module,
        processor,
        image,
        text,
        image_mask,
        convert_image_fn,
        pad_sequence_fn,
        cat_with_pad_fn,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        inputs = self.process_batch(processor, image, text, image_mask)
        inputs = {k: v.to(backbone.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        attention_masks = inputs["attention_mask"].to(backbone.device)
        outputs = backbone(**inputs, output_hidden_states=True)
        return list(outputs.hidden_states), attention_masks

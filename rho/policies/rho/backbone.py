"""
Backbone adapter interface for the Rho policy.

Internal policies can inject a different factory into the shared model
implementation.
"""

import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

logger = logging.getLogger(__name__)


def create_backbone_adapter(config) -> "BackboneAdapter":
    """Create the configured Rho backbone adapter."""
    if config.vlm_backend != "phi5":
        raise ValueError(f"Rho only supports the Phi5 backend, got {config.vlm_backend!r}")
    from rho.policies.rho.phi5.backbone import Phi5Backbone

    return Phi5Backbone(config)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class BackboneAdapter(ABC):
    """Abstract adapter interface for different VLM backbones."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

    @abstractmethod
    def load_backbone(self) -> nn.Module:
        """Load and configure the VLM backbone model."""

    @abstractmethod
    def create_processor(self):
        """Create and return the VLM processor for image/text tokenization."""

    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier used for loading generation config, etc."""

    @abstractmethod
    def prepare_prompt(self, batch, num_images_override: int | None = None) -> list[str]:
        """Format text prompts with image tokens for the specific VLM backend."""

    @abstractmethod
    def remove_audio_components(self, backbone: nn.Module):
        """Remove unused audio components from the backbone to save memory."""

    @abstractmethod
    def set_vision_requires_grad(self, backbone: nn.Module):
        """Apply vision-specific parameter freezing based on config."""

    @abstractmethod
    def get_hidden_size(self, backbone: nn.Module) -> int:
        """Return the hidden size of the VLM backbone's text model."""

    def supports_builtin_lora(self) -> bool:
        """Whether this backend has built-in LoRA adapters."""
        return False

    def get_language_model(self, backbone: nn.Module) -> nn.Module:
        """Return the language model submodule containing embed_tokens, layers, norm."""
        return backbone.model

    @abstractmethod
    def get_freezing_components(self, backbone: nn.Module) -> dict:
        """Return an ordered dict of component names -> modules for freezing status."""

    @abstractmethod
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
        """Process a batch of images and texts into a backbone-specific batch dict."""

    def compute_lm_loss(
        self,
        backbone: nn.Module,
        processor,
        images,
        prompts: list[str],
        target_texts: list[str],
        image_mask=None,
        pad_sequence_fn=None,
        cat_with_pad_fn=None,
    ) -> torch.Tensor:
        """Autoregressive cross-entropy loss on ``target_texts``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement compute_lm_loss.")

    def forward_lm_with_hidden_state(
        self,
        backbone: nn.Module,
        processor,
        images,
        prompts: list[str],
        target_texts: list[str],
        image_mask=None,
        pad_sequence_fn=None,
        cat_with_pad_fn=None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Single VLM forward returning all hidden states + prefix mask + LM CE loss."""
        raise NotImplementedError(f"{type(self).__name__} does not implement forward_lm_with_hidden_state.")

    @abstractmethod
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
        """Extract image and text hidden states from the backbone."""

    def get_image_text_embed(
        self,
        backbone: nn.Module,
        processor,
        hidden_state_idx: int,
        image,
        text,
        image_mask,
        convert_image_fn,
        pad_sequence_fn,
        cat_with_pad_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract raw (unprojected) image and text hidden states."""
        identity = nn.Identity()
        return self.get_image_text_hidden_state(
            backbone=backbone,
            processor=processor,
            vlm_projector=identity,
            hidden_state_idx=hidden_state_idx,
            image=image,
            text=text,
            image_mask=image_mask,
            convert_image_fn=convert_image_fn,
            pad_sequence_fn=pad_sequence_fn,
            cat_with_pad_fn=cat_with_pad_fn,
        )

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
        """Extract ALL layer hidden states from the backbone (unprojected)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement get_all_hidden_states.")

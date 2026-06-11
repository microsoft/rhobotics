"""
Abstract backbone adapter for VLM backends used by RhoAlphaModel.

Each VLM backend (Phi4MM, Phi5, etc.) implements this interface to handle
backbone-specific initialization, processing, and hidden state extraction.
"""

import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BackboneAdapter(ABC):
    """
    Abstract adapter interface for different VLM backbones.

    The RhoAlphaModel delegates backbone-specific operations to an adapter,
    allowing the same model architecture to work with different VLM backends.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

    @abstractmethod
    def load_backbone(self) -> nn.Module:
        """
        Load and configure the VLM backbone model.

        Returns:
            The loaded VLM backbone model, moved to the correct device/dtype.
        """

    @abstractmethod
    def create_processor(self):
        """
        Create and return the VLM processor for image/text tokenization.

        Returns:
            The VLM processor instance.
        """

    @abstractmethod
    def get_model_id(self) -> str:
        """
        Return the model identifier used for loading generation config, etc.

        Returns:
            Model ID string (HuggingFace model ID or local path).
        """

    @abstractmethod
    def prepare_prompt(self, batch) -> list[str]:
        """
        Format text prompts with image tokens for the specific VLM backend.

        Each backend uses different chat templates and image token formats.
        E.g., Phi4MM uses ``<|user|><|image_1|>...<|end|><|assistant|>``
        while Phi5 uses ``<|im_start|>user<|im_sep|><image>...<|im_end|>``.

        Args:
            batch: Dict containing at least OBS_TASK (list of prompt strings).

        Returns:
            List of formatted prompt strings ready for tokenization.
        """

    @abstractmethod
    def remove_audio_components(self, backbone: nn.Module):
        """
        Remove unused audio components from the backbone to save memory.

        Args:
            backbone: The VLM backbone model.
        """

    @abstractmethod
    def set_vision_requires_grad(self, backbone: nn.Module):
        """
        Apply vision-specific parameter freezing based on config.

        Handles backbone-specific vision architecture paths
        (e.g., embed_tokens_extend.image_embed vs vision_tower).

        Args:
            backbone: The VLM backbone model.
        """

    @abstractmethod
    def get_hidden_size(self, backbone: nn.Module) -> int:
        """
        Return the hidden size of the VLM backbone's text model.

        Different backends store this in different config locations
        (e.g., ``config.hidden_size`` vs ``config.text_config.hidden_size``).

        Args:
            backbone: The VLM backbone model.

        Returns:
            Hidden dimension size (int).
        """

    def supports_builtin_lora(self) -> bool:
        """Whether this backend has built-in LoRA adapters (e.g. Phi4MM vision LoRA).

        Backends that return False will skip LoRA configuration in
        ``RhoAlphaModel._configure_builtin_loras()``.
        Defaults to False; Phi4MM overrides to True.
        """
        return False

    def get_language_model(self, backbone: nn.Module) -> nn.Module:
        """Return the language model submodule containing embed_tokens, layers, norm.

        Phi4MM/Phi5: ``backbone.model`` (direct)
        Qwen: ``backbone.model.language_model`` (nested)

        Override in subclasses if the backbone nests the language model differently.
        """
        return backbone.model

    @abstractmethod
    def get_freezing_components(self, backbone: nn.Module) -> dict:
        """
        Return an ordered dict of component names -> modules for freezing status reporting.

        Each backbone has different vision component paths. This method
        provides the correct mapping for print_freezing_status().

        Args:
            backbone: The VLM backbone model.

        Returns:
            dict mapping display names to modules.
        """

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
        """
        Process a batch of images and texts into a backbone-specific batch dict.

        Different backends use different processor output field names
        (e.g., input_image_embeds vs pixel_values).

        Args:
            processor: The VLM processor.
            images: List[List[PIL.Image.Image]] - batch_size x num_images.
            texts: List[str] - batch_size text prompts.
            image_mask: Tensor mask for image selection.
            pad_sequence_fn: Utility function for padding sequences (optional,
                not needed by all backends).
            cat_with_pad_fn: Utility function for concatenating with padding
                (optional, not needed by all backends).
            max_length: Maximum sequence length.

        Returns:
            dict of tensors ready for backbone forward pass.
        """

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
        """
        Extract image and text hidden states from the backbone.

        Different backends have different hidden state extraction strategies
        (e.g., Phi4MM supports HD/non-HD branching, Phi5 uses a direct path).

        Args:
            backbone: The VLM backbone model.
            processor: The VLM processor.
            vlm_projector: Linear projection from backbone hidden size to embed_dim.
            hidden_state_idx: Which layer's hidden state to extract.
            image: Image input (List[List[PIL.Image]] or batch dict).
            text: List[str] text prompts.
            image_mask: Optional image masking tensor.
            convert_image_fn: Function to convert images to batch dict format.
            pad_sequence_fn: Utility function for padding sequences.
            cat_with_pad_fn: Utility function for concatenating with padding.

        Returns:
            Tuple of (hidden_states, attention_masks):
            - hidden_states: (batch_size, num_tokens, embed_dim)
            - attention_masks: (batch_size, num_tokens)
        """

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
        """
        Extract raw (unprojected) image and text hidden states from the backbone.

        Same as get_image_text_hidden_state but WITHOUT applying vlm_projector,
        returning features at the backbone's native hidden dimension. Useful for
        cross-attention where the action expert handles the dimension mismatch.

        Default implementation delegates to get_image_text_hidden_state with an
        identity projector. Override in subclasses if more efficient paths exist.

        Returns:
            Tuple of (hidden_states, attention_masks):
            - hidden_states: (batch_size, num_tokens, backbone_hidden_dim)
            - attention_masks: (batch_size, num_tokens)
        """
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
        """
        Extract ALL layer hidden states from the backbone (unprojected).

        Used by LayerwiseCrossAttentionExpert to cross-attend to different
        VLM layers at each action expert block.

        Default implementation runs the backbone once with
        output_hidden_states=True and collects all layers. Backends
        that need special handling (e.g. Phi4MM HD) should override.

        Returns:
            Tuple of (all_hidden_states, attention_masks):
            - all_hidden_states: list of (B, S, vlm_hidden_dim) tensors
            - attention_masks: (B, S)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_all_hidden_states. Override in subclass."
        )


def create_backbone_adapter(config) -> BackboneAdapter:
    """
    Factory function to create the appropriate backbone adapter.

    Args:
        config: RhoAlphaConfig with vlm_backend field.

    Returns:
        BackboneAdapter instance for the configured backend.
    """
    backend = getattr(config, "vlm_backend", "phi4mm")

    if backend == "phi4mm":
        from rho.policies.rhoalpha.phi4mm.backbone import Phi4MMBackbone

        return Phi4MMBackbone(config)
    elif backend == "phi5":
        from rho.policies.rhoalpha.phi5.backbone import Phi5Backbone

        return Phi5Backbone(config)
    elif backend == "qwen25vl":
        from rho.policies.rhoalpha.qwen25vl.backbone import Qwen25VLBackbone

        return Qwen25VLBackbone(config)
    elif backend == "qwen3vl":
        from rho.policies.rhoalpha.qwen3vl.backbone import Qwen3VLBackbone

        return Qwen3VLBackbone(config)
    else:
        raise ValueError(
            f"Unknown vlm_backend: '{backend}'. Supported: 'phi4mm', 'phi5', 'qwen25vl', 'qwen3vl'"
        )

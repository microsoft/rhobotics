"""
Phi5 backbone adapter for RhoAlphaModel.

Handles Phi-5 (Phi-4-vision-5B) specific initialization, processing, and
hidden state extraction, including:
- Configurable model path via vlm_backbone_folder
- Float16 loading with device_map
- Phi5-specific processor output field names (pixel_values, etc.)
- vision_tower-based architecture paths
"""

import logging
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from rho.policies.rhoalpha.backbone import BackboneAdapter
from rho.policies.rhoalpha.phi5.processing_phi5 import Phi5ImageProcessor

logger = logging.getLogger(__name__)

# Sentinel value used by BunnyPhi4ForCausalLM to locate image positions
# in the token sequence. prepare_inputs_labels_for_multimodal() searches
# for this value and splices in vision embeddings at those positions.
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"  # nosec B105


def _tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    """Tokenize prompt, replacing each <image> with IMAGE_TOKEN_INDEX.

    Port of tokenizer_image_token() from processing_bunny_phi4.py so we
    don't need to import the model's remote-code module at runtime.
    """
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def _interleave(xs, sep):
        out = []
        for a, b in zip(xs, [sep] * len(xs), strict=False):
            out.append(a)
            out.append(b)
        return out[:-1]  # drop trailing separator

    ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        ids.append(chunks[0][0])

    sep = [IMAGE_TOKEN_INDEX] * (offset + 1)
    for x in _interleave(chunks, sep):
        ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(ids, dtype=torch.long)
    return ids


class Phi5Backbone(BackboneAdapter):
    """
    Backbone adapter for Phi-5 (Phi-4-vision-5B).

    Handles Phi5's unique requirements:
    - Configurable model path (vlm_backbone_folder)
    - Float16 dtype, device_map loading
    - No compatibility patches needed
    - vision_tower-based architecture (vs embed_tokens_extend.image_embed)
    - Phi5-specific processor output field names (pixel_values, pixel_attention_mask, etc.)
    """

    def get_hidden_size(self, backbone: nn.Module) -> int:
        return backbone.config.hidden_size

    def get_model_id(self) -> str:
        return self.config.vlm_backbone_folder

    def prepare_prompt(self, batch) -> list[str]:
        from rho.common.constants import OBSERVATION_LANG as OBS_TASK

        if OBS_TASK not in batch:
            logger.warning(f"Key {OBS_TASK} not found in batch. Returning empty prompt list.")
            return []

        num_images = len(self.config.image_features) * self.config.n_obs_steps
        image_tokens = "<image>" * num_images

        formatted = []
        for prompt in batch[OBS_TASK]:
            text = (
                f"<|im_start|>user<|im_sep|>{image_tokens}{prompt}<|im_end|><|im_start|>assistant<|im_sep|>"
            )
            formatted.append(text)
        return formatted

    def _has_weight_files(self) -> bool:
        """Check whether the vlm_backbone_folder contains safetensors or bin model weights."""
        folder = Path(self.config.vlm_backbone_folder)
        if not folder.is_dir():
            return False
        weight_patterns = ("*.safetensors", "*.bin")
        for pattern in weight_patterns:
            for f in folder.glob(pattern):
                # Ignore non-model bins like training_args.bin
                if pattern == "*.bin" and "training_args" in f.name:
                    continue
                return True
        return False

    def load_backbone(self) -> nn.Module:
        logger.info("Loading VLM backbone from '%s'...", self.config.vlm_backbone_folder)
        if self._has_weight_files():
            logger.info("Found model weight files — loading pretrained backbone...")
            backbone = AutoModelForCausalLM.from_pretrained(
                self.config.vlm_backbone_folder,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            logger.warning(
                "No safetensors/model weight files found in '%s'. "
                "Creating backbone with uninitialized (random) weights. "
                "This is suitable for architecture validation but NOT for inference. "
                "To load real weights, set vlm_backbone_folder to a directory "
                "containing .safetensors files or set the VLM_BACKBONE_FOLDER env var.",
                self.config.vlm_backbone_folder,
            )
            logger.info("Loading model config from '%s'...", self.config.vlm_backbone_folder)
            config = AutoConfig.from_pretrained(
                self.config.vlm_backbone_folder,
                trust_remote_code=True,
            )
            logger.info("Creating backbone model from config (this may take a moment)...")
            backbone = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            # from_config() with delay_load=True creates the vision tower wrapper
            # but not the inner SiglipModel. Initialize it without pretrained weights
            # so the architecture matches what the pretrained checkpoint expects.
            vision_tower = backbone.get_vision_tower()
            if vision_tower is not None and not vision_tower.is_loaded:
                logger.info("Initializing vision tower architecture (without pretrained weights)...")
                vision_tower.load_model(skip_weights=True)

            self._backbone_has_uninitialized_weights = True

        if self.config is not None and self.config.enable_gradient_checkpointing:
            backbone.gradient_checkpointing_enable()

        logger.info("Moving backbone to device=%s, dtype=%s...", self.device, self.dtype)
        backbone.to(device=self.device, dtype=self.dtype)
        logger.info("Backbone loaded successfully.")
        return backbone

    def create_processor(self):
        original_processor = AutoProcessor.from_pretrained(
            self.config.vlm_backbone_folder, trust_remote_code=True
        )

        # Replace with GPU-native image processor to avoid PIL/CPU round-trips
        gpu_image_processor = Phi5ImageProcessor.from_siglip2_processor(
            original_processor.image_processor, device=self.device
        )
        original_processor.image_processor = gpu_image_processor

        return original_processor

    def remove_audio_components(self, backbone: nn.Module):
        """Remove speech LoRA from backbone (Phi5 has no audio_embed module)."""
        logger.info("Removing audio components...")

        base_model = backbone.model if hasattr(backbone, "model") else backbone

        # Remove audio/speech LoRA from all layers
        removed_count = 0
        for layer in base_model.layers:
            for proj_name in ["down_proj", "gate_up_proj"]:
                proj = getattr(layer.mlp, proj_name, None)
                if proj and hasattr(proj, "lora_A") and hasattr(proj.lora_A, "speech"):
                    del proj.lora_A.speech
                    del proj.lora_B.speech
                    removed_count += 1
            for proj_name in ["o_proj", "qkv_proj"]:
                proj = getattr(layer.self_attn, proj_name, None)
                if proj and hasattr(proj, "lora_A") and hasattr(proj.lora_A, "speech"):
                    del proj.lora_A.speech
                    del proj.lora_B.speech
                    removed_count += 1

        logger.info(f"  Removed audio LoRA ({removed_count} modules)")

    def set_vision_requires_grad(self, backbone: nn.Module):
        if self.config.freeze_vision_encoder:
            for params in backbone.model.vision_tower.parameters():
                params.requires_grad = False

    def get_freezing_components(self, backbone: nn.Module) -> dict:
        base_model = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
        return {
            "  Vision Encoder": base_model.model.vision_tower,
        }

    # Token ID used by the BunnyPhi4 model for image placeholders in the
    # tokenized sequence. Discovered empirically from the tokenizer vocab.
    _IMAGE_PLACEHOLDER_TOKEN_ID = 200010

    def process_batch(
        self,
        processor,
        images,
        texts,
        image_mask,
        pad_sequence_fn,
        cat_with_pad_fn,
        max_length=8192,
    ) -> dict:
        """
        Process a batch of images + texts into model inputs.

        Uses tokenizer_image_token() so that <image> placeholders become
        IMAGE_TOKEN_INDEX (-200) in input_ids — required by
        BunnyPhi4ForCausalLM.prepare_inputs_labels_for_multimodal()
        to splice vision embeddings into the sequence.

        Args:
            processor: BunnyPhi4Processor with Phi5ImageProcessor.
            images: List[List[Tensor]] — batch × cameras, each (C,H,W).
            texts: List[str] — batch prompts (with <image> tokens).
            image_mask: (batch, cameras) or None.
            pad_sequence_fn: padding utility.
            cat_with_pad_fn: cat-with-pad utility.
        """
        batch_size = len(images)
        n_imgs = len(images[0])
        image_processor = processor.image_processor

        if image_mask is None:
            image_mask = torch.ones(
                batch_size,
                n_imgs,
                dtype=torch.long,
                device=self.device,
            )

        # ==========================================================
        # 1. Batch image processing (NaFlex patch extraction on GPU)
        # ==========================================================
        flat_images = torch.stack([img for sample in images for img in sample])
        vision_out = image_processor.process_batched(flat_images)
        all_pixel_values = vision_out["pixel_values"]
        all_pixel_attention_mask = vision_out["pixel_attention_mask"]
        all_spatial_shapes = vision_out["spatial_shapes"]

        # ==========================================================
        # 2. Tokenize with IMAGE_TOKEN_INDEX (-200) sentinels
        # ==========================================================
        input_ids_list = []
        for text in texts:
            ids = _tokenizer_image_token(
                text,
                processor.tokenizer,
                return_tensors="pt",
            )
            input_ids_list.append(ids)

        # Pad to uniform length
        input_ids = pad_sequence_fn(
            input_ids_list,
            padding_side="right",
            padding_value=0,
        )
        # Attention mask: 1 for real tokens (including -200 sentinels)
        attention_mask = (input_ids != 0).long().to(self.device)

        return {
            "input_ids": input_ids.to(self.device),
            "labels": None,
            "attention_mask": attention_mask,
            "pixel_values": all_pixel_values,
            "pixel_attention_mask": all_pixel_attention_mask,
            "spatial_shapes": all_spatial_shapes,
        }

    def get_image_text_hidden_state(
        self,
        backbone,
        processor,
        vlm_projector,
        hidden_state_idx,
        image,
        text,
        image_mask,
        convert_image_fn,
        pad_sequence_fn,
        cat_with_pad_fn,
    ):
        assert 0 <= hidden_state_idx <= 33, (
            f"Layer {hidden_state_idx} is out of range. Must be between 0 and 33."
        )

        # printing shapes for debugging
        batch = self.process_batch(
            processor,
            image,
            text,
            image_mask=image_mask,
            pad_sequence_fn=pad_sequence_fn,
            cat_with_pad_fn=cat_with_pad_fn,
        )
        batch = {k: v.to(backbone.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Call prepare_inputs_labels_for_multimodal ourselves so we
        # can capture the *new* attention_mask (sequence length changes
        # when image embeddings are spliced in for -200 sentinel tokens).
        images = {
            "pixel_values": batch["pixel_values"],
            "pixel_attention_mask": batch["pixel_attention_mask"],
            "spatial_shapes": batch["spatial_shapes"],
        }
        (
            _,  # input_ids (None after splice)
            position_ids,
            attention_mask,
            _,  # past_key_values
            inputs_embeds,
            _,  # labels
        ) = backbone.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            None,  # position_ids
            batch["attention_mask"],
            None,  # past_key_values
            None,  # labels
            images,
        )

        # Forward with inputs_embeds so the model skips
        # prepare_inputs_labels_for_multimodal (already done above).
        outputs = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[hidden_state_idx]
        hidden = vlm_projector(hidden)

        return hidden, attention_mask.to(self.device)

    def get_all_hidden_states(
        self,
        backbone,
        processor,
        image,
        text,
        image_mask,
        convert_image_fn,
        pad_sequence_fn,
        cat_with_pad_fn,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch = self.process_batch(
            processor,
            image,
            text,
            image_mask=image_mask,
            pad_sequence_fn=pad_sequence_fn,
            cat_with_pad_fn=cat_with_pad_fn,
        )
        batch = {k: v.to(backbone.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        images = {
            "pixel_values": batch["pixel_values"],
            "pixel_attention_mask": batch["pixel_attention_mask"],
            "spatial_shapes": batch["spatial_shapes"],
        }
        (
            _,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
        ) = backbone.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            None,
            batch["attention_mask"],
            None,
            None,
            images,
        )
        outputs = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return list(outputs.hidden_states), attention_mask.to(self.device)

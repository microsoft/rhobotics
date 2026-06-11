"""
Phi4MM backbone adapter for RhoAlphaModel.

Handles all Phi-4 Multimodal specific initialization, processing, and
hidden state extraction, including:
- Compatibility patches for transformers API changes
- SiglipEncoder gradient checkpointing fix
- Custom Phi4MMImageProcessor with HD/non-HD support
- Manual image embedding for non-HD mode
- Phi4MM-specific processor output field names
"""

import logging

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor

from rho.common.transforms import ResizeWithPadding
from rho.policies.rhoalpha.backbone import BackboneAdapter
from rho.policies.rhoalpha.configuration_rhoalpha import PHI4MM_IMAGE_SIZE
from rho.policies.rhoalpha.phi4mm.processing_phi4mm import Phi4MMImageProcessor

logger = logging.getLogger(__name__)

# Resize transform for non-HD image processing (448x448 is Phi4MM's native resolution)
_phi4mm_resize_transform = ResizeWithPadding(height=PHI4MM_IMAGE_SIZE, width=PHI4MM_IMAGE_SIZE)

PHI4MM_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"


# ============================================================================
# Phi4MM Compatibility Patches
# ============================================================================
def _patch_transformers_compatibility():
    """
    Patch transformers for compatibility with Phi4MM remote code.

    The Phi4MM model from HuggingFace uses deprecated APIs that were removed
    in newer versions of transformers. This function patches them back.
    """
    from transformers import DynamicCache

    # In transformers >= 4.45, get_usable_length was renamed to get_seq_length
    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            """Compatibility shim: get_usable_length -> get_seq_length"""
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length
        logger.debug("Patched DynamicCache.get_usable_length for transformers compatibility.")


# Apply compatibility patches at module import time
_patch_transformers_compatibility()


def _patch_phi4mm_prepare_inputs():
    """
    Patch Phi4MM model classes to add prepare_inputs_for_generation if missing.

    This must be called BEFORE AutoModelForCausalLM.from_pretrained() because
    the remote code may call this method during model initialization (e.g. when
    PEFT wraps the inner model via get_peft_model).
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        classes_to_patch = [
            ("modeling_phi4mm.Phi4MMModel", "Phi4MMModel"),
            ("modeling_phi4mm.Phi4MMForCausalLM", "Phi4MMForCausalLM"),
        ]

        for class_path, class_name in classes_to_patch:
            try:
                model_class = get_class_from_dynamic_module(
                    class_path,
                    PHI4MM_MODEL_ID,
                    trust_remote_code=True,
                )

                if not hasattr(model_class, "prepare_inputs_for_generation"):

                    def _prepare_inputs_for_generation(self, *args, **kwargs):
                        """Passthrough for prepare_inputs_for_generation."""
                        return kwargs

                    model_class.prepare_inputs_for_generation = _prepare_inputs_for_generation
                    logger.debug(f"Patched prepare_inputs_for_generation on {class_name}.")
            except Exception as e:
                logger.warning(f"Could not patch {class_name}: {e}")
    except Exception as e:
        logger.warning(f"Could not pre-patch Phi4MM models: {e}")


class Phi4MMBackbone(BackboneAdapter):
    """
    Backbone adapter for Phi-4 Multimodal (Phi4MM).

    Handles Phi4MM's unique requirements:
    - Compatibility patches for transformers API
    - SiglipEncoder gradient checkpointing fix
    - Custom image processor with HD/non-HD support
    - Manual image embedding for non-HD mode
    - Phi4MM-specific processor output field names (input_image_embeds, etc.)
    """

    def get_hidden_size(self, backbone: nn.Module) -> int:
        return backbone.config.hidden_size

    def supports_builtin_lora(self) -> bool:
        return True

    def get_model_id(self) -> str:
        return self.config.vlm_backbone_folder or PHI4MM_MODEL_ID

    def prepare_prompt(self, batch) -> list[str]:
        from rho.common.constants import OBSERVATION_LANG as OBS_TASK

        if OBS_TASK not in batch:
            logger.warning(f"Key {OBS_TASK} not found in batch. Returning empty prompt list.")
            return []

        num_images = len(self.config.image_features) * self.config.n_obs_steps
        image_token = "".join([f"<|image_{i + 1}|>" for i in range(num_images)])

        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"

        formatted = []
        for prompt in batch[OBS_TASK]:
            if not prompt.startswith(user_prompt):
                prompt = user_prompt + image_token + prompt
            if not prompt.endswith(f"{prompt_suffix}{assistant_prompt}"):
                prompt = prompt + prompt_suffix + assistant_prompt
            formatted.append(prompt)
        return formatted

    def load_backbone(self) -> nn.Module:
        _patch_phi4mm_prepare_inputs()

        attn_impl = getattr(self.config, "attention_implementation", "flash_attention_2")
        model_id = self.get_model_id()

        backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation=attn_impl,
        ).to(self.device)

        if self.config is not None and self.config.enable_gradient_checkpointing:
            backbone.gradient_checkpointing_enable()

        # Fix for AttributeError: 'SiglipEncoder' object has no attribute '_gradient_checkpointing_func'
        siglip_encoder = backbone.model.embed_tokens_extend.image_embed.img_processor.encoder
        siglip_encoder.gradient_checkpointing = True
        if not hasattr(siglip_encoder, "_gradient_checkpointing_func"):
            siglip_encoder._gradient_checkpointing_func = checkpoint.checkpoint

        backbone.to(device=self.device, dtype=self.dtype)
        return backbone

    def create_processor(self):
        model_id = self.get_model_id()
        original_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Create GPU-optimized image processor with config-based HD settings
        # When use_hd_transform=False, use dynamic_hd=1 (single global crop only)
        # When use_hd_transform=True, use configured dynamic_hd value (default 36)
        if hasattr(self.config, "use_hd_transform") and not self.config.use_hd_transform:
            dynamic_hd = 1  # Single global crop only - much faster and less memory
        else:
            dynamic_hd = getattr(self.config, "dynamic_hd", 36)

        gpu_image_processor = Phi4MMImageProcessor(dynamic_hd=dynamic_hd, device=self.device)

        # Replace the image processor
        original_processor.image_processor = gpu_image_processor

        return original_processor

    def remove_audio_components(self, backbone: nn.Module):
        logger.info("Removing audio components...")

        base_model = backbone.model if hasattr(backbone, "model") else backbone

        # Remove audio encoder
        if hasattr(base_model, "embed_tokens_extend") and hasattr(
            base_model.embed_tokens_extend, "audio_embed"
        ):
            del base_model.embed_tokens_extend.audio_embed
            logger.info("  Removed audio encoder")

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
            for params in backbone.model.embed_tokens_extend.image_embed.parameters():
                params.requires_grad = False

        if self.config.freeze_vision_transformer:  # freeze only the SiglipVisionTransformer
            for params in backbone.model.embed_tokens_extend.image_embed.img_processor.parameters():
                params.requires_grad = False
        else:
            for params in backbone.model.embed_tokens_extend.image_embed.img_processor.parameters():
                backbone.model.embed_tokens_extend.image_embed.img_processor.train()
                params.requires_grad = True

        if self.config.freeze_vision_projector:  # freeze only the image projector
            for params in backbone.model.embed_tokens_extend.image_embed.img_projection.parameters():
                params.requires_grad = False
        else:
            for params in backbone.model.embed_tokens_extend.image_embed.img_projection.parameters():
                backbone.model.embed_tokens_extend.image_embed.img_projection.train()
                params.requires_grad = True

    def get_freezing_components(self, backbone: nn.Module) -> dict:
        base_model = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
        return {
            "  Vision Encoder": base_model.model.embed_tokens_extend.image_embed,
            "    Vision Transformer": base_model.model.embed_tokens_extend.image_embed.img_processor,
            "    Vision Projector": base_model.model.embed_tokens_extend.image_embed.img_projection,
        }

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
        batch_input_ids = []
        batch_input_image_embeds = []
        batch_image_attention_mask = []
        batch_image_sizes = []
        batch_img_token_mask = []

        if image_mask is None:
            image_mask = torch.ones(len(images), len(images[0]), dtype=torch.long, device=self.device)

        for img_list, text, mask_img in zip(images, texts, image_mask, strict=False):
            prompt = text
            inputs = processor(prompt, images=img_list, return_tensors="pt")
            input_ids = inputs.input_ids
            batch_input_ids.append(input_ids[0])
            n_imgs = image_mask[0].shape[0]
            n_tokens_per_img = torch.sum(inputs.input_ids == 200010) / n_imgs
            img_token_mask = mask_img.expand(int(n_tokens_per_img), -1).T.flatten().unsqueeze(0)
            batch_input_image_embeds.append(inputs.input_image_embeds)
            # The following has the correct shape but for some reason throws
            # an error in vlm_backbone forward
            # image_attention_mask = inputs.image_attention_mask * mask_img[:, None, None, None]
            # batch_image_attention_mask.append(image_attention_mask)
            batch_image_attention_mask.append(inputs.image_attention_mask)
            batch_image_sizes.append(inputs.image_sizes)
            batch_img_token_mask.append(img_token_mask)

        input_ids = pad_sequence_fn(batch_input_ids, padding_side="right", padding_value=0)
        # Here we assume that images come first and text comes second with no audio
        input_image_token_mask = cat_with_pad_fn(batch_img_token_mask, dim=0)
        n_img_tokens = input_image_token_mask.shape[1]
        n_tokens = input_ids.shape[1]
        text_attention_mask = (input_ids[..., n_img_tokens - n_tokens :] != 0).long().to(self.device)
        attention_mask = torch.cat((input_image_token_mask, text_attention_mask), dim=1)
        input_image_embeds = cat_with_pad_fn(batch_input_image_embeds, dim=0)
        image_attention_mask = cat_with_pad_fn(batch_image_attention_mask, dim=0)
        image_sizes = torch.cat(batch_image_sizes)

        batch_dict = {
            "input_ids": input_ids,
            "labels": None,  # labels are not used in inference
            "attention_mask": attention_mask,
            "input_image_embeds": input_image_embeds,
            "image_attention_mask": image_attention_mask,
            "image_sizes": image_sizes,
            "input_mode": 1,  # vision mode
        }
        return batch_dict

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

        # Two paths: HD (processor-based) vs non-HD (manual embedding)
        use_hd = getattr(self.config, "use_hd_transform", True)

        if not use_hd:
            return self._get_image_text_hidden_state_manual(
                backbone,
                processor,
                vlm_projector,
                hidden_state_idx,
                image,
                text,
                image_mask,
                convert_image_fn,
            )
        else:
            return self._get_image_text_hidden_state_hd(
                backbone,
                processor,
                vlm_projector,
                hidden_state_idx,
                image,
                text,
                image_mask,
                pad_sequence_fn,
                cat_with_pad_fn,
            )

    # ========================================================================
    # Phi4MM-specific: Non-HD manual image embedding
    # ========================================================================

    def prepare_images_manual(self, batch):
        """
        Prepare images from batch without using processor (for non-HD mode).
        Based on phi4mm_fast's prepare_images method.
        """
        from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES

        if OBS_IMAGES not in batch:
            return []

        consolidated_imgs = batch[OBS_IMAGES]
        if consolidated_imgs.shape[1] == 0:
            return []

        images_list = []
        num_cameras = consolidated_imgs.shape[1]

        for cam_idx in range(num_cameras):
            img = consolidated_imgs[:, cam_idx, :, :, :]
            img = _phi4mm_resize_transform(img)
            images_list.append(img)

        return images_list

    def embed_image_manual(self, backbone, images_list: list):
        """
        Embed images directly using Phi4MM's vision encoder (for non-HD mode).
        Based on phi4mm_fast's embed_image method.

        get_img_features() internally applies avg_pool (256 patches at 448x448).
        Optionally adds separator tokens (sub_GN, glb_GN) to match HD path's structure.
        """
        if not images_list:
            return None

        image_embed = backbone.model.embed_tokens_extend.image_embed
        vision_dtype = image_embed.img_processor.embeddings.patch_embedding.weight.dtype

        add_separators = getattr(self.config, "add_separator_tokens", True)
        sub_gn = image_embed.sub_GN if add_separators and hasattr(image_embed, "sub_GN") else None
        glb_gn = image_embed.glb_GN if add_separators and hasattr(image_embed, "glb_GN") else None

        all_image_embeds = []

        for img in images_list:
            batch_size = img.shape[0]

            # Normalize for SigLIP: [0,1] -> [-1,1]
            img_normalized = img * 2.0 - 1.0
            img_normalized = img_normalized.to(dtype=vision_dtype)

            img_embeds = image_embed.get_img_features(img_normalized)

            _, num_patches, hidden_dim = img_embeds.shape
            h = w = int(num_patches**0.5)

            img_embeds = img_embeds.view(batch_size, h, w, hidden_dim)

            if sub_gn is not None:
                row_seps = sub_gn.squeeze(0).expand(batch_size, h, 1, hidden_dim)
                img_embeds = torch.cat([img_embeds, row_seps], dim=2)

            img_embeds = img_embeds.reshape(batch_size, -1, hidden_dim)

            if glb_gn is not None:
                global_sep = glb_gn.expand(batch_size, 1, hidden_dim)
                img_embeds = torch.cat([global_sep, img_embeds], dim=1)

            img_embeds = image_embed.img_projection(img_embeds)

            all_image_embeds.append(img_embeds)

        image_embeds = torch.cat(all_image_embeds, dim=1) if all_image_embeds else None

        if image_embeds is not None and not hasattr(self, "_logged_nohd_tokens"):
            num_cameras = len(images_list)
            tokens_per_camera = image_embeds.shape[1] // num_cameras
            logger.debug(
                f"[nohd] Image tokens: {image_embeds.shape[1]} total, "
                f"{tokens_per_camera} per camera, {num_cameras} cameras"
            )
            self._logged_nohd_tokens = True

        return image_embeds

    def _get_image_text_hidden_state_manual(
        self,
        backbone,
        processor,
        vlm_projector,
        hidden_state_idx,
        image,
        text,
        image_mask,
        convert_image_fn,
    ):
        """Non-HD path: Manually embed images and process text separately."""
        batch_dict = convert_image_fn(image)
        images_list = self.prepare_images_manual(batch_dict)
        image_embeds = self.embed_image_manual(backbone, images_list)

        text_tokens = processor.tokenizer(
            text, add_special_tokens=True, return_tensors="pt", padding="longest", truncation=True
        )
        text_input_ids = text_tokens["input_ids"].to(self.device)
        text_attention_mask = text_tokens["attention_mask"].to(self.device)

        text_embeds = backbone.model.embed_tokens(text_input_ids)

        if image_embeds is not None:
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            img_mask = torch.ones(
                image_embeds.shape[0],
                image_embeds.shape[1],
                dtype=text_attention_mask.dtype,
                device=self.device,
            )
            combined_mask = torch.cat([img_mask, text_attention_mask], dim=1)
        else:
            combined_embeds = text_embeds
            combined_mask = text_attention_mask

        outputs = backbone(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            input_mode=1,
            output_hidden_states=True,
        )
        selected_hidden_state = outputs.hidden_states[hidden_state_idx]
        selected_hidden_state = vlm_projector(selected_hidden_state)

        return selected_hidden_state, combined_mask

    def _get_image_text_hidden_state_hd(
        self,
        backbone,
        processor,
        vlm_projector,
        hidden_state_idx,
        image,
        text,
        image_mask,
        pad_sequence_fn,
        cat_with_pad_fn,
    ):
        """HD path: Full processor + VLM forward (original behavior)."""
        batch = self.process_batch(
            processor,
            image,
            text,
            image_mask=image_mask,
            pad_sequence_fn=pad_sequence_fn,
            cat_with_pad_fn=cat_with_pad_fn,
        )
        batch = {k: v.to(backbone.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = backbone(**batch, output_hidden_states=True)
        selected_hidden_state = outputs.hidden_states[hidden_state_idx]
        selected_hidden_state = vlm_projector(selected_hidden_state)

        attention_masks = batch["attention_mask"].to(self.device)

        return selected_hidden_state, attention_masks

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
        """Return all VLM hidden states. Uses HD path."""
        batch = self.process_batch(
            processor,
            image,
            text,
            image_mask=image_mask,
            pad_sequence_fn=pad_sequence_fn,
            cat_with_pad_fn=cat_with_pad_fn,
        )
        batch = {k: v.to(backbone.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = backbone(**batch, output_hidden_states=True)
        attention_masks = batch["attention_mask"].to(self.device)
        return list(outputs.hidden_states), attention_masks

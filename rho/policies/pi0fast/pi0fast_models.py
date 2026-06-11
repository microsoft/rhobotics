#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PI0FAST Model - Core neural network implementation
"""

import logging
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import idct
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
from transformers.cache_utils import HybridCache, StaticCache
from transformers.models.auto import CONFIG_MAPPING

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_STATE as OBS_STATE
from rho.common.transforms import ResizeWithPadding

logger = logging.getLogger(__name__)

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def make_attn_mask(input_mask: torch.Tensor, ar_mask: torch.Tensor) -> torch.Tensor:
    """
    Create prefix-LM attention mask following JAX implementation.

    Tokens can attend to valid input tokens which have a cumulative ar_mask
    smaller or equal to theirs. This creates:
    - Bidirectional attention for AR=0 tokens (images, prefix)
    - Causal attention for AR=1 tokens (action suffix)

    Args:
        input_mask: bool[B, N] - true if part of input, false if padding
        ar_mask: bool[B, N] - true for causal tokens, false for bidirectional

    Returns:
        Attention mask [B, N, N] where True means can attend
    """
    cumsum = torch.cumsum(ar_mask, dim=1)  # [B, N]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # [B, N, N]
    return attn_mask & valid_mask.bool()


def block_causal_update_causal_mask(
    attention_mask,
    token_type_ids=None,
    past_key_values=None,
    cache_position=None,
    input_tensor=None,
    attn_implementation: str = "eager",
    dtype: torch.dtype = "float32",
):
    """
    Update the causal mask during training and generation. It can be customized to different attention masks.
    """
    if attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    using_static_cache = isinstance(past_key_values, StaticCache)
    min_dtype = torch.finfo(dtype).min

    if input_tensor is None:
        input_tensor = attention_mask

    inputs_lead_dim, sequence_length = input_tensor.shape[:2]

    if using_static_cache or isinstance(past_key_values, HybridCache):
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else cache_position[0] + sequence_length + 1
        )

    # Handle precomputed attention masks
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    # Causal mask initialization
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
    )

    # Standard causal masking (triu ensures tokens can only attend to past)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Apply block causal mask
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(causal_mask.device).bool()
            cumsum = torch.cumsum(token_type_ids, dim=1)
            block_causal_mask = cumsum[:, None, :] <= cumsum[:, :, None]

            # Combine causal_mask with block-wise attention mask
            causal_mask = torch.where(block_causal_mask, 0.0, causal_mask)
            causal_mask = causal_mask[:, None, :, :]
        else:
            # Apply past cache position constraint
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
    else:
        # Apply past cache position constraint
        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
            -1, 1
        )
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # Copy to contiguous memory for in-place edits
        mask_length = attention_mask.shape[-1]

        # Apply padding mask
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
            causal_mask.device
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    return causal_mask


def prepare_inputs_for_generation(
    input_ids,
    past_key_values=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    pixel_values=None,
    attention_mask=None,
    token_type_ids=None,
    use_cache=True,
    num_logits_to_keep=None,
    labels=None,
    self=None,
    **kwargs,
):
    # create block causal attention
    if cache_position[0] > 0 and input_ids.shape[1] > 0:
        input_tensor = input_ids[:, -1:]
        new_positions = (
            torch.ones(
                (position_ids.shape[0], input_ids.shape[1]),
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).cumsum(-1)
            + position_ids[:, -1:]
        )
        position_ids = torch.cat([position_ids, new_positions], dim=-1)
    else:
        input_tensor = inputs_embeds
    attention_mask = block_causal_update_causal_mask(
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        input_tensor=input_tensor,
        token_type_ids=token_type_ids,
        dtype=self.dtype,
        attn_implementation=self.config.text_config._attn_implementation,
    )
    # Overwritten -- custom `position_ids` and `pixel_values` handling
    model_inputs = self.language_model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=use_cache,
        num_logits_to_keep=num_logits_to_keep,
        token_type_ids=token_type_ids,
        **kwargs,
    )

    # Position_ids in Paligemma are 1-indexed
    if model_inputs.get("position_ids") is not None:
        model_inputs["position_ids"] += 1
    # If we're in cached decoding stage, pixel values should be None because input ids do not
    # contain special image token anymore. Otherwise we need pixel values to be passed to model.
    # NOTE: use_cache=False needs pixel_values always
    if cache_position[0] == 0:
        model_inputs["pixel_values"] = pixel_values
    is_training = token_type_ids is not None and labels is not None
    if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
        input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
        )
        model_inputs["attention_mask"] = causal_mask

    return model_inputs


class PI0FAST(nn.Module):
    """PI0FAST model - Core neural network for action prediction using PaliGemma backbone."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Tokenizer paths
        fast_tokenizer_path = "physical-intelligence/fast"
        pi0_paligemma_path = "google/paligemma-3b-pt-224"

        # Initialize tokenizers
        self.paligemma_tokenizer = AutoTokenizer.from_pretrained(pi0_paligemma_path)
        self.processor = AutoProcessor.from_pretrained(pi0_paligemma_path)
        self.fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

        # Model configuration
        self.fast_skip_tokens = self.config.fast_skip_tokens
        self.max_input_seq_len = self.config.max_input_seq_len
        self.action_horizon = self.config.chunk_size
        self.action_dim = (
            self.config.action_feature.shape[0]
            if hasattr(self.config, "action_feature")
            else self.config.max_action_dim
        )

        # Precision settings
        precision = config.precision
        torch_precision = PRECISION.get(precision, torch.float32)

        # Token IDs
        self.pad_token_id = (
            self.paligemma_tokenizer.pad_token_id
            if hasattr(self.paligemma_tokenizer, "pad_token_id")
            else self.paligemma_tokenizer.eos_token_id
        )

        # Create PaliGemma configuration
        paligemma_config = CONFIG_MAPPING["paligemma"](
            transformers_version="4.48.1",
            _vocab_size=257152,
            bos_token_id=2,
            eos_token_id=1,
            hidden_size=2048,
            image_token_index=257152,
            model_type="paligemma",
            pad_token_id=0,
            projection_dim=2048,
            text_config={
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2048,
                "intermediate_size": 16384,
                "model_type": "gemma",
                "num_attention_heads": 8,
                "num_hidden_layers": 18,
                "num_image_tokens": 256,
                "num_key_value_heads": 1,
                "torch_dtype": precision,
                "vocab_size": 257152,
                "_attn_implementation": self.config.attention_implementation,
            },
            vision_config={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "model_type": "siglip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "num_image_tokens": 256,
                "patch_size": 14,
                "projection_dim": 2048,
                "projector_hidden_act": "gelu_pytorch_tanh",
                "torch_dtype": precision,
                "vision_use_head": False,
            },
        )

        # Initialize PaliGemma model
        self.pi0_paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config)
        self.pi0_paligemma.prepare_inputs_for_generation = partial(
            prepare_inputs_for_generation, self=self.pi0_paligemma
        )

        # Convert important components to specified precision
        params_to_change_dtype = [
            "language_model",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.pi0_paligemma.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch_precision)

        self.set_requires_grad()
        self.image_keys = self.config.image_features.keys() if hasattr(self.config, "image_features") else []

        # TODO: Remove this once we bump transformers to >4.52.0 because the attribute will be removed
        # AttributeError: 'PaliGemmaConfig' object has no attribute 'ignore_index'
        self.ignore_index = self.pi0_paligemma.config.ignore_index
        self.padding_side = self.config.padding_side

        # Initialize resize transform if configured
        if self.config.resize_imgs_with_padding is not None:
            h, w = self.config.resize_imgs_with_padding
            self.resize_transform = ResizeWithPadding(height=h, width=w)
        else:
            self.resize_transform = None

    def set_requires_grad(self):
        """Set gradient requirements based on configuration."""
        if self.config.freeze_vision_encoder:
            self.pi0_paligemma.vision_tower.eval()
            for params in self.pi0_paligemma.vision_tower.parameters():
                params.requires_grad = False
        # To avoid unused params issue with distributed training
        if self.config.freeze_lm_head:
            for name, params in self.pi0_paligemma.named_parameters():
                if "embed_tokens" in name:  # lm heads and embedding layer are tied
                    params.requires_grad = False

    def embed_tokens(self, tokens: torch.Tensor):
        """Embed tokens using the language model."""
        return self.pi0_paligemma.language_model.model.embed_tokens(tokens)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation."""
        return self.pi0_paligemma.prepare_inputs_for_generation(*args, **kwargs)

    def prepare_images(self, batch):
        """Preprocess batch images into Pi0 inputs."""
        from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES

        # Handle case where images are already consolidated into OBS_IMAGES key
        # (e.g., when coming from prepare_batch_for_model in the policy)
        if OBS_IMAGES in batch:
            # Images are already consolidated and in format (batch, num_cameras, channels, height, width)
            consolidated_imgs = batch[OBS_IMAGES]

            # Check if we have any images
            if consolidated_imgs.shape[1] == 0:
                # No images - return empty lists
                return [], []

            images = []
            img_masks = []
            bsize = consolidated_imgs.shape[0]
            num_cameras = consolidated_imgs.shape[1]
            device = consolidated_imgs.device

            for cam_idx in range(num_cameras):
                img = consolidated_imgs[:, cam_idx, :, :, :]  # Extract each camera view

                if self.resize_transform is not None:
                    img = self.resize_transform(img)

                # Normalize from range [0,1] to [-1,1] as expected by siglip
                img = img * 2.0 - 1.0

                mask = torch.ones(bsize, dtype=torch.bool, device=device)
                images.append(img)
                img_masks.append(mask)

            return images, img_masks

        # Original logic: handle individual image feature keys
        images = []
        img_masks = []
        present_img_keys = [key for key in self.image_keys if key in batch]

        if len(present_img_keys) == 0 and len(self.image_keys) > 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        num_empty_cameras = 0
        for key in self.image_keys:
            if key in present_img_keys:
                img = batch[key]

                if self.resize_transform is not None:
                    img = self.resize_transform(img)

                # Normalize from range [0,1] to [-1,1] as expected by siglip
                img = img * 2.0 - 1.0

                bsize = img.shape[0]
                device = img.device
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            else:
                if num_empty_cameras >= self.config.empty_cameras:
                    continue
                img = torch.ones_like(img) * -1
                bsize = img.shape[0]
                device = img.device
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
                num_empty_cameras += 1

            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert action tokens to PaliGemma token space."""
        out = self.paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        return out

    def fast_tokenizer_wrapper(self, actions_norm):
        """
        A wrapper for self.fast_tokenizer that ensures batch processing,
        conversion to PyTorch tensors, and returns a dictionary without padding.
        """
        batch_tokens = self.fast_tokenizer(actions_norm)
        fast_out = self.processor.tokenizer.pad({"input_ids": batch_tokens}, return_tensors="pt")
        return fast_out

    def create_token_type_ids(self, padded_mask: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """Create token type IDs for block causal attention."""
        token_type_ids = torch.zeros_like(padded_mask, dtype=torch.bool)
        # Compute cumulative sum mask
        cumsum_mask = (padded_mask != 0).cumsum(dim=1)
        # Suffix block (everything after prefix_len)
        suffix_mask = cumsum_mask > prefix_len
        token_type_ids = suffix_mask
        return token_type_ids

    def create_input_tokens(self, state, lang_text, actions=None):
        """Create input tokens from state, language text, and optionally actions."""
        bsize = state.shape[0]
        device = state.device
        bins = torch.linspace(-1, 1, 256 + 1, device=device)[:-1]
        discretized = torch.bucketize(state, bins) - 1
        discretized = discretized[:, :32]

        prefix_texts = []
        state_text = []
        for txt, disc in zip(lang_text, discretized, strict=False):
            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(str(val.item()) for val in disc)
            prefix_texts.append(f"Task: {cleaned}, State: {state_str};\n")
            state_text.append(f"State: {state_str};\n")

        prefix_out = self.paligemma_tokenizer(
            prefix_texts, add_special_tokens=True, return_tensors="pt", padding="longest", truncation=False
        )
        prefix_ids = prefix_out["input_ids"].to(device)
        prefix_mask = prefix_out["attention_mask"].to(device)
        prefix_lens = prefix_mask.sum(dim=1)[:, None].cpu()

        if actions is not None:
            # Actions are already normalized by the framework using quantile normalization
            # No need to normalize again - just pad to max_action_dim
            actions_pad = F.pad(actions, (0, max(0, self.config.max_action_dim - actions.shape[2])), value=0)[
                :, :, : self.config.max_action_dim
            ]
            fast_out = self.fast_tokenizer_wrapper(
                actions_pad.cpu(),
            )
            act_ids = fast_out["input_ids"]
            act_mask = fast_out["attention_mask"].to(device)

            act_ids = self._act_tokens_to_paligemma_tokens(act_ids).to(device)
            # Replace action with 0 to pad tokens
            act_ids = torch.where(
                act_ids == self.paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens,
                self.pad_token_id,
                act_ids,
            )

            eos_token = torch.tensor(
                [self.paligemma_tokenizer.eos_token_id], dtype=torch.long, device=device
            ).expand(bsize, -1)
            eos_mask = torch.tensor([1], dtype=torch.long, device=device).expand(bsize, -1)
            bos = self.paligemma_tokenizer("Action: ", add_special_tokens=False, return_tensors="pt")
            bos_token = bos["input_ids"].expand(act_ids.shape[0], -1).to(device)
            bos_mask = bos["attention_mask"].expand(act_ids.shape[0], -1).to(device)

            # Track "Action: " token count for proper loss mask
            action_prompt_len = bos_token.shape[1]

            act_ids = torch.cat([bos_token, act_ids, eos_token], dim=1)
            act_mask = torch.cat([bos_mask, act_mask, eos_mask], dim=1)
            act_mask = act_mask.to(device)
        else:
            action_prompt_len = 0
            act_ids = torch.empty(bsize, 0, dtype=torch.long, device=device)
            act_mask = torch.empty(bsize, 0, dtype=torch.long, device=device)

        final_ids = torch.cat([prefix_ids, act_ids], dim=1)
        final_mask = torch.cat([prefix_mask, act_mask], dim=1)
        batch_inputs = {"input_ids": final_ids.tolist(), "attention_mask": final_mask.tolist()}

        # Use tokenizer pad function
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs, padding="longest", max_length=180, return_tensors="pt"
        )
        padded_mask = padded_output["attention_mask"]

        # AR mask: 1 for all suffix tokens (including "Action: ")
        att_mask = (padded_mask != 0).cumsum(dim=1) > prefix_lens

        token_type_ids = self.create_token_type_ids(padded_mask=padded_mask, prefix_len=prefix_lens)

        # Loss mask: compute loss on action tokens only, excluding "Action: " prefix and EOS
        loss_mask = torch.zeros_like(padded_mask, dtype=torch.bool)
        if actions is not None:
            for b in range(bsize):
                prefix_len_b = int(prefix_lens[b].item())
                # Start after prefix + "Action: "
                loss_start = prefix_len_b + action_prompt_len
                # End before EOS (which is the last non-pad token)
                valid_len = int(padded_mask[b].sum().item())
                loss_end = valid_len - 1  # Exclude EOS
                if loss_start < loss_end:
                    loss_mask[b, loss_start:loss_end] = True

        padded_output["padded_mask"] = padded_output.pop("attention_mask")
        padded_output["attention_mask"] = att_mask
        padded_output["loss_mask"] = loss_mask
        padded_output["token_type_ids"] = token_type_ids
        return padded_output

    def shift_padding_side(
        self,
        tokens: torch.Tensor,
        ar_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        targets: torch.Tensor,
        token_type_ids: torch.Tensor,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor]:
        """Shift padding to the specified side."""
        if padding_side not in ["right", "left"]:
            return tokens, ar_mask, padding_mask, loss_mask, targets, token_type_ids

        new_tokens = torch.empty_like(tokens)
        new_ar_masks = torch.empty_like(ar_mask)
        new_padding_mask = torch.empty_like(padding_mask)
        new_loss_mask = torch.empty_like(loss_mask)
        new_targets = torch.empty_like(targets)
        new_token_type_ids = torch.empty_like(token_type_ids)
        batch_size = tokens.shape[0]

        for i in range(batch_size):
            padding_indices = torch.where(padding_mask[i] == 0)[0]
            non_padding_indices = torch.where(padding_mask[i] == 1)[0]
            if padding_side == "left":
                new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
            else:
                new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
            new_tokens[i] = tokens[i].index_select(0, new_indices)
            new_ar_masks[i] = ar_mask[i].index_select(0, new_indices)
            new_padding_mask[i] = padding_mask[i].index_select(0, new_indices)
            new_loss_mask[i] = loss_mask[i].index_select(0, new_indices)
            new_targets[i] = targets[i].index_select(0, new_indices)
            new_token_type_ids[i] = token_type_ids[i].index_select(0, new_indices)

        return new_tokens, new_ar_masks, new_padding_mask, new_loss_mask, new_targets, new_token_type_ids

    def forward(self, batch: dict[str, Tensor]):
        """Forward pass for training."""
        images, img_masks = self.prepare_images(batch)

        padded_outs = self.create_input_tokens(
            state=batch[OBS_STATE],
            lang_text=batch["task"],
            actions=batch[ACTION],
        )

        # Store language token count before embedding
        num_lang_tokens = padded_outs["input_ids"].shape[1]

        embs, input_mask, ar_mask, targets, loss_mask, token_type_ids = self.embed_inputs(
            images,
            img_masks,
            padded_outs["input_ids"],
            padded_outs["padded_mask"],
            padded_outs["attention_mask"],
            padded_outs["loss_mask"],
            padded_outs["token_type_ids"],
            padding_side=self.padding_side,
        )

        # Create prefix-LM attention mask (following JAX implementation)
        # Remove last token for next-token prediction
        input_embs = embs[:, :-1]  # [B, seq_len-1, D]
        input_mask_trimmed = input_mask[:, :-1]  # [B, seq_len-1]
        ar_mask_trimmed = ar_mask[:, :-1]  # [B, seq_len-1]

        # Create 3D attention mask with prefix-LM pattern
        attn_mask_3d = make_attn_mask(input_mask_trimmed, ar_mask_trimmed)  # [B, seq_len-1, seq_len-1]

        # Convert to 4D format expected by HuggingFace
        # True -> 0.0 (can attend), False -> large negative (cannot attend)
        attn_mask_4d = attn_mask_3d.unsqueeze(1)  # [B, 1, seq_len-1, seq_len-1]
        min_dtype = torch.finfo(self.pi0_paligemma.dtype).min
        attn_mask_4d = torch.where(attn_mask_4d, 0.0, min_dtype).to(dtype=self.pi0_paligemma.dtype)

        # Compute position IDs
        position_ids = torch.cumsum(input_mask_trimmed, dim=-1) - 1

        # Forward through PaliGemma model to get hidden states
        outputs = self.pi0_paligemma.language_model.model(
            inputs_embeds=input_embs,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # [B, seq_len-1, hidden_dim]

        # CRITICAL: Only decode language token positions to vocabulary, not image positions
        # We removed last token, so extract last (num_lang_tokens - 1) positions
        num_lang_tokens_trimmed = num_lang_tokens - 1
        hidden_states_lang = hidden_states[:, -num_lang_tokens_trimmed:]  # [B, num_lang_tokens-1, hidden_dim]

        # Decode to vocabulary logits
        logits = self.pi0_paligemma.language_model.lm_head(
            hidden_states_lang
        )  # [B, num_lang_tokens-1, vocab_size]

        # Targets: language tokens shifted (predict next token)
        # Original language tokens: [tok0, tok1, ..., tokN]
        # We predict: [tok1, tok2, ..., tokN] from inputs [tok0, tok1, ..., tokN-1]
        targets_lang = targets[:, -num_lang_tokens:][:, 1:]  # [B, num_lang_tokens-1]
        loss_mask_lang = loss_mask[:, -num_lang_tokens:][:, 1:]  # [B, num_lang_tokens-1]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets_lang.reshape(-1))

        # Apply loss mask
        token_loss = token_loss * loss_mask_lang.reshape(-1).float()

        # Compute final loss
        loss = token_loss.sum() / torch.clamp(loss_mask_lang.sum(), min=1)

        # Return loss dictionary
        loss_dict = {"ce_loss": loss.item(), "loss": loss}
        return loss_dict

    def decode_actions_with_fast(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        relaxed_decoding: bool = True,
    ) -> np.array:
        """
        Adapt original decoding in FAST to always return actions instead of zeros.
        """
        self.time_horizon = (
            time_horizon or self.fast_tokenizer.time_horizon or self.fast_tokenizer.called_time_horizon
        )
        self.action_dim = (
            action_dim or self.fast_tokenizer.action_dim or self.fast_tokenizer.called_action_dim
        )

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert self.time_horizon is not None and self.action_dim is not None, (
            "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        )

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                if relaxed_decoding:
                    # Expected sequence length
                    expected_seq_len = self.time_horizon * self.action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]
                    # Apply truncation if too long
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]  # Truncate on the right
                    # Apply padding if too short
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert decoded_dct_coeff.shape == (
                    self.time_horizon,
                    self.action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, "
                    f"expected ({self.time_horizon}, {self.action_dim})"
                )
            except Exception as e:
                logger.error(f"Error decoding tokens: {e}")
                logger.error(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)

    def extract_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        """
        Extracts actions from predicted output tokens using the FAST model.

        Args:
            tokens (torch.Tensor): The input tensor of tokenized outputs.
            action_horizon (int): The number of timesteps for actions.
            action_dim (int): The dimensionality of each action.

        Returns:
            torch.Tensor: The extracted actions as a tensor of shape (action_horizon, action_dim).
        """
        # Decode predicted output tokens
        decoded_tokens = self.paligemma_tokenizer.batch_decode(tokens, skip_special_tokens=True)
        cleaned_tokens = [
            tokens_sequence.replace("Action:", "").replace(":", "").strip().split("|")[0].strip()
            for tokens_sequence in decoded_tokens
        ]
        raw_action_tokens = [
            self.processor.tokenizer.encode(sample_tokens, return_tensors="pt", padding=False)
            for sample_tokens in cleaned_tokens
        ]
        action_tokens = [
            self._act_tokens_to_paligemma_tokens(raw_action_token) for raw_action_token in raw_action_tokens
        ]
        # returns the tensor of decoded actions per sample in a list
        decoded_actions = [
            torch.tensor(
                self.decode_actions_with_fast(
                    tok.tolist(),
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                    relaxed_decoding=self.config.relaxed_action_decoding,
                ),
                device=tokens.device,
            ).squeeze(0)
            for tok in action_tokens
        ]

        return torch.stack(decoded_actions, dim=0)

    def generate_actions(self, batch: dict[str, Tensor]):
        """Generate actions for inference."""
        images, img_masks = self.prepare_images(batch)

        padded_outs = self.create_input_tokens(state=batch[OBS_STATE], lang_text=batch["task"], actions=None)
        embs, input_mask, ar_mask, targets, loss_mask, token_type_ids = self.embed_inputs(
            images,
            img_masks,
            padded_outs["input_ids"],
            padded_outs["padded_mask"],
            padded_outs["attention_mask"],
            padded_outs["loss_mask"],
            padded_outs["token_type_ids"],
            padding_side="left",
        )

        # Compute position IDs
        prefix_position_ids = torch.cumsum(input_mask, dim=-1) - 1

        # For generation, use 2D padding mask (HuggingFace .generate() expects this)
        # The model will use block_causal_update_causal_mask with token_type_ids for prefix-LM
        output_tokens = self.pi0_paligemma.generate(
            input_ids=None,
            attention_mask=input_mask,  # 2D padding mask
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=self.config.use_cache,
            max_new_tokens=self.config.max_decoding_steps,
            do_sample=False,
            num_beams=1,
            token_type_ids=token_type_ids.to(dtype=torch.int64),
        )
        actions = self.extract_actions(output_tokens, self.action_horizon, self.action_dim)
        return actions

    def embed_image(self, image: torch.Tensor):
        """Embed images using the vision encoder."""
        # Handle different transformers versions
        if hasattr(self.pi0_paligemma, "get_image_features"):
            return self.pi0_paligemma.get_image_features(image)
        else:
            return self.pi0_paligemma.model.get_image_features(image)

    def embed_inputs(
        self,
        images,
        img_masks,
        tokens,
        pad_mask,
        ar_mask,
        loss_mask,
        token_type_ids,
        padding_side: str = "right",
    ):
        """Embed all inputs (images and tokens) for the model."""
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        # images are a list of same size
        # vectorizing everything!
        device = images[0].device if images else tokens.device

        if images:
            image_embedding_dim = images[0].shape[-1]
            all_images = torch.stack(images, dim=1).to(device)
            b, n, c, h, w = all_images.shape
            all_images = all_images.view(b * n, c, h, w)
            embedded = self.embed_image(all_images).to(device)
            b_n, p, image_embedding_dim = embedded.shape  # Extract current dimensions
            m = b_n // b  # Compute the number of images per sample dynamically

            # Reshape dynamically
            embedded = embedded.view(b, m, p, image_embedding_dim)

            img_masks = torch.stack(img_masks, dim=1).unsqueeze(-1).to(device)
            num_img_emb = embedded.shape[2]
            img_pad_masks = img_masks.repeat(1, 1, num_img_emb).view(b, -1)

            img_ar_masks = torch.zeros((b, n, num_img_emb), dtype=torch.long, device=device).reshape(b, -1)

            image_target_tokens = (
                torch.ones((b, n, num_img_emb), dtype=torch.long, device=device) * self.pad_token_id
            ).reshape(b, -1)
            image_loss_mask = torch.zeros((b, n, num_img_emb), dtype=torch.long, device=device).reshape(b, -1)

            embedded = embedded.reshape(b, n * num_img_emb, image_embedding_dim)  # Shape: (B, N*P, D)
        else:
            # No images case
            b = tokens.shape[0]
            embedded = torch.empty(b, 0, self.pi0_paligemma.config.hidden_size, device=device)
            img_pad_masks = torch.empty(b, 0, dtype=torch.long, device=device)
            img_ar_masks = torch.empty(b, 0, dtype=torch.long, device=device)
            image_loss_mask = torch.empty(b, 0, dtype=torch.long, device=device)
            image_target_tokens = torch.empty(b, 0, dtype=torch.long, device=device)

        tokens_embs = self.embed_tokens(tokens.to(device))

        embs = torch.cat([embedded, tokens_embs], dim=1).to(device)
        pad_masks = torch.cat([img_pad_masks, pad_mask.to(device)], dim=1)
        ar_masks = torch.cat([img_ar_masks, ar_mask.to(device)], dim=1)
        loss_masks = torch.cat([image_loss_mask, loss_mask.to(device)], dim=1)
        targets = torch.cat([image_target_tokens, tokens.to(device)], dim=1)

        # Recreate full token_type_ids by concatenating image and language portions
        img_token_type_ids = torch.zeros(
            (b, n, num_img_emb) if images else (b, 0), dtype=torch.long, device=device
        ).reshape(b, -1)
        full_token_type_ids = torch.cat([img_token_type_ids, token_type_ids.to(device)], dim=1)

        # Shift pad tokens to the left (.generate()) or right (.train())
        embs, ar_masks, pad_masks, loss_masks, targets, full_token_type_ids = self.shift_padding_side(
            embs, ar_masks, pad_masks, loss_masks, targets, full_token_type_ids, padding_side=padding_side
        )

        targets = torch.where(targets == self.pad_token_id, self.ignore_index, targets)
        return embs, pad_masks, ar_masks, targets, loss_masks, full_token_type_ids

"""
Phi4MM FAST (Autoregressive) Model.

Implements discrete action prediction using FAST tokenizer with autoregressive training.
Based on Pi0FAST approach with Phi4MM backbone.

ARCHITECTURE:
- Embed images directly using vision encoder
- Create full token sequence: [image_tokens, task, state, "Action:", action_tokens]
- Forward pass through language model
- Compute loss on action tokens only
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import idct
from torch import Tensor
from transformers import AutoProcessor

from rho.common.transforms import ResizeWithPadding
from rho.policies.rhoalpha.configuration_rhoalpha import PHI4MM_IMAGE_SIZE, RhoAlphaConfig
from rho.policies.rhoalpha.phi4mm.processing_phi4mm import Phi4MMImageProcessor
from rho.policies.rhoalpha.rhoalpha_model import RhoAlphaModel

logger = logging.getLogger(__name__)
# Resize transform for non-HD image processing (448x448 is Phi4MM's native resolution)
_phi4mm_resize_transform = ResizeWithPadding(height=PHI4MM_IMAGE_SIZE, width=PHI4MM_IMAGE_SIZE)

# Phi4MM special token for images
_IMAGE_SPECIAL_TOKEN_ID = 200010


class RhoAlphaFASTModel(RhoAlphaModel):
    """
    Autoregressive model using FAST tokenizer for discrete action prediction with Phi4MM backbone.

    Attention Mask Options:
    - Prefix-LM (default): Bidirectional attention on images/text, causal on actions
      - Requires SDPA or eager attention (not flash_attention_2)
      - Set config.attention_implementation = "sdpa"
      - Better quality but higher memory usage

    - Full Causal (optional): Causal attention everywhere
      - Compatible with flash_attention_2 for memory efficiency
      - Set config.use_full_causal_mask = True
      - Set config.attention_implementation = "flash_attention_2"
      - May reduce quality but uses less memory
    """

    def __init__(self, config: RhoAlphaConfig, **kwargs):
        assert getattr(config, "vlm_backend", "phi4mm") == "phi4mm", (
            f"RhoAlphaFASTModel only supports the phi4mm backend, got vlm_backend='{config.vlm_backend}'"
        )
        super().__init__(config, **kwargs)

        # Initialize FAST processor
        self.fast_processor = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )

        # Skip tokens to avoid special token region
        self.fast_skip_tokens = 1100
        self.fast_vocab_size = self.fast_processor.bpe_tokenizer.vocab_size
        self.phi4_vocab_size = self.vlm_backbone.config.vocab_size

        # Token IDs
        self.pad_token_id = (
            self.vlm_backbone.config.pad_token_id if self.vlm_backbone.config.pad_token_id is not None else 0
        )
        self.eos_token_id = (
            self.vlm_backbone.config.eos_token_id if self.vlm_backbone.config.eos_token_id is not None else 2
        )
        self.bos_token_id = getattr(self.vlm_backbone.config, "bos_token_id", 1)

        # Ignore index for loss computation
        self.ignore_index = -100

        # Action configuration
        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.max_action_dim

        # Padding side for training (right) vs inference (left)
        self.padding_side = "right"

        # Fix Phi4MM generation_config max_length (defaults to 100, too short)
        self.vlm_backbone.generation_config.max_length = 4096

        # Initialize GPU-optimized image processor for HD path
        # This avoids PIL conversions and keeps everything on GPU
        if getattr(self.config, "use_hd_transform", True):
            dynamic_hd = getattr(self.config, "dynamic_hd", 36)
        else:
            dynamic_hd = 1  # Single global crop only
        self.gpu_image_processor = Phi4MMImageProcessor(dynamic_hd=dynamic_hd, device=self.device)

    def embed_tokens(self, tokens: torch.Tensor):
        """Embed tokens using the language model."""
        return self.vlm_backbone.model.embed_tokens(tokens)

    def embed_image(self, images_list: list):
        """
        Embed images using Phi4MM's vision encoder.

        Routes to HD or non-HD path based on config.use_hd_transform.

        Args:
            images_list: List of image tensors for each camera view (batch, C, H, W)

        Returns:
            Tuple:
            - image_embeds: (batch, num_image_tokens, hidden_size)
            - image_token_mask: (batch, num_image_tokens) where 1=valid, 0=padding
        """
        if not images_list:
            return None, None

        # Check if we should use HD transform
        use_hd = getattr(self.config, "use_hd_transform", True)

        if use_hd:
            return self._embed_image_hd(images_list)
        else:
            return self._embed_image_manual(images_list)

    def _embed_image_manual(self, images_list: list):
        """
        Non-HD path: Embed images directly using vision encoder.
        Fast and memory efficient - uses single-scale images.

        get_img_features() internally applies avg_pool (256 patches at 448x448).
        Optionally adds separator tokens (sub_GN, glb_GN) to match HD path's structure.
        This keeps token count low while maintaining closer alignment to pre-training distribution.
        """
        # Get the dtype that the vision encoder expects
        image_embed = self.vlm_backbone.model.embed_tokens_extend.image_embed
        vision_dtype = image_embed.img_processor.embeddings.patch_embedding.weight.dtype

        # Get separators if we're using them (matches HD path's structure)
        # sub_GN: marks end of each row in the patch grid
        # glb_GN: marks this as the "global" image (no sub-crops follow)
        add_separators = getattr(self.config, "add_separator_tokens", True)
        sub_gn = image_embed.sub_GN if add_separators and hasattr(image_embed, "sub_GN") else None
        glb_gn = image_embed.glb_GN if add_separators and hasattr(image_embed, "glb_GN") else None

        all_image_embeds = []

        for img in images_list:
            batch_size = img.shape[0]

            # Normalize for SigLIP: [0,1] -> [-1,1]
            img_normalized = img * 2.0 - 1.0
            img_normalized = img_normalized.to(dtype=vision_dtype)

            # Get image features from vision encoder
            # NOTE: get_img_features already applies avg_pool internally if configured
            # At 448x448 with avg_pool: returns (batch, 256, 1152) where 256 = 16x16 patches
            img_embeds = image_embed.get_img_features(img_normalized)

            _, num_patches, hidden_dim = img_embeds.shape
            h = w = int(num_patches**0.5)

            # Reshape to spatial: (batch, h, w, hidden_dim)
            img_embeds = img_embeds.view(batch_size, h, w, hidden_dim)

            # Add row separators (sub_GN) to match HD path structure
            # This adds one separator token at the end of each row: (batch, h, w) -> (batch, h, w+1)
            # Gives 256 + 16 = 272 tokens per image at 448x448
            if sub_gn is not None:
                # sub_gn is (1, 1, 1, hidden_dim), repeat to (batch, h, 1, hidden_dim)
                row_seps = sub_gn.squeeze(0).expand(batch_size, h, 1, hidden_dim)
                img_embeds = torch.cat([img_embeds, row_seps], dim=2)  # (batch, h, w+1, hidden_dim)

            # Flatten spatial dims
            img_embeds = img_embeds.reshape(batch_size, -1, hidden_dim)

            if glb_gn is not None:
                # Single image as "global": [glb_gn, patches+row_seps] = 273 tokens
                # In HD with sub_glb order: [sub, glb_gn, global] - so glb_gn comes before global
                global_sep = glb_gn.expand(batch_size, 1, hidden_dim)
                img_embeds = torch.cat([global_sep, img_embeds], dim=1)

            # Project to model hidden size
            img_embeds = image_embed.img_projection(img_embeds)

            all_image_embeds.append(img_embeds)

        # Concatenate all camera views
        image_embeds = torch.cat(all_image_embeds, dim=1) if all_image_embeds else None
        if image_embeds is None:
            return None, None

        # Log token count on first call for verification
        if not hasattr(self, "_logged_nohd_tokens"):
            num_cameras = len(images_list)
            tokens_per_camera = image_embeds.shape[1] // num_cameras
            logger.debug(
                f"[nohd] Image tokens: {image_embeds.shape[1]} total, "
                f"{tokens_per_camera} per camera, {num_cameras} cameras"
            )
            self._logged_nohd_tokens = True

        # Manual path: all tokens are valid
        image_token_mask = torch.ones(
            image_embeds.shape[0],
            image_embeds.shape[1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        return image_embeds, image_token_mask

    def _embed_image_hd(self, images_list: list):
        """
        HD path: Use GPU-optimized processor for multi-crop HD image processing.
        Higher quality but more memory intensive.

        Uses Phi4MMImageProcessor which keeps everything on GPU (no PIL conversion).
        Calls the full HD transform logic from image_embed.forward().
        """
        batch_size = images_list[0].shape[0]
        num_cameras = len(images_list)

        # Process images through GPU-optimized processor
        # The processor accepts tensors directly and keeps everything on GPU
        all_inputs = []
        for b in range(batch_size):
            # Collect all camera images for this batch element
            batch_camera_tensors = [images_list[cam_idx][b] for cam_idx in range(num_cameras)]
            # GPU processor handles tensor input directly
            inputs = self.gpu_image_processor.preprocess(images=batch_camera_tensors, return_tensors="pt")
            all_inputs.append(inputs)

        # Stack inputs across batch - GPU processor returns tensors already on GPU
        input_image_embeds = torch.cat([inp["input_image_embeds"] for inp in all_inputs], dim=0)
        image_sizes = torch.cat([inp["image_sizes"] for inp in all_inputs], dim=0)
        image_attention_mask = (
            torch.cat([inp["image_attention_mask"] for inp in all_inputs], dim=0)
            if "image_attention_mask" in all_inputs[0]
            else None
        )
        # Use processor-provided token counts (already accounts for any masked-out crops)
        # num_img_tokens is a list of ints, convert to tensor for easier manipulation
        tokens_per_image_tensor = torch.tensor(
            [tok for inp in all_inputs for tok in inp["num_img_tokens"]],
            device=self.device,
        )
        tokens_per_image = tokens_per_image_tensor.tolist()
        total_tokens = int(tokens_per_image_tensor.sum().item())

        # Log token count on first call for verification
        if not hasattr(self, "_logged_hd_tokens"):
            tokens_per_camera = total_tokens // (batch_size * num_cameras)
            logger.debug(
                f"[HD] Image tokens: {total_tokens} total, "
                f"{tokens_per_camera} per camera, {num_cameras} cameras"
            )
            self._logged_hd_tokens = True

        # Create input_ids with the EXACT number of image tokens needed
        dummy_input_ids = torch.full(
            (1, total_tokens), _IMAGE_SPECIAL_TOKEN_ID, dtype=torch.long, device=self.device
        )

        # Call image_embed.forward() which does the full HD transform
        # This returns hidden_states with image tokens replaced by HD features
        hidden_states = self.vlm_backbone.model.embed_tokens_extend.image_embed(
            input_ids=dummy_input_ids,
            input_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            wte=self.vlm_backbone.model.embed_tokens,
        )

        # hidden_states shape: (1, total_tokens, hidden_dim)
        # Remove batch dim: (total_tokens, hidden_dim)
        hidden_states = hidden_states.squeeze(0)

        # Split back into per-batch-sample chunks
        # Each batch sample has num_cameras images
        all_batch_embeds = []
        offset = 0
        for b in range(batch_size):
            batch_tokens = []
            for c in range(num_cameras):
                img_idx = b * num_cameras + c
                n_tokens = tokens_per_image[img_idx]
                batch_tokens.append(hidden_states[offset : offset + n_tokens])
                offset += n_tokens
            # Concatenate all cameras for this batch sample
            all_batch_embeds.append(torch.cat(batch_tokens, dim=0))

        # Pad to same length
        max_len = max(emb.shape[0] for emb in all_batch_embeds)
        padded_embeds = []
        padded_masks = []
        for emb in all_batch_embeds:
            current_len = emb.shape[0]
            if current_len < max_len:
                padding = torch.zeros(
                    (max_len - current_len, emb.shape[1]), dtype=emb.dtype, device=emb.device
                )
                emb = torch.cat([emb, padding], dim=0)
                mask = torch.cat(
                    [
                        torch.ones(current_len, dtype=torch.long, device=emb.device),
                        torch.zeros(max_len - current_len, dtype=torch.long, device=emb.device),
                    ],
                    dim=0,
                )
            else:
                mask = torch.ones(current_len, dtype=torch.long, device=emb.device)
            padded_embeds.append(emb)
            padded_masks.append(mask)

        # Stack into (batch, max_tokens, hidden_dim)
        img_embeds = torch.stack(padded_embeds, dim=0)
        img_mask = torch.stack(padded_masks, dim=0)

        return img_embeds, img_mask

    def _act_tokens_to_phi4_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert action tokens to Phi4 token space."""
        out = self.phi4_vocab_size - 1 - self.fast_skip_tokens - tokens
        return out

    def _phi4_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert Phi4 tokens back to action token space."""
        out = self.phi4_vocab_size - 1 - self.fast_skip_tokens - tokens
        return out

    def prepare_images(self, batch):
        """
        Prepare images from batch.

        Returns:
            images_list: List of image tensors (batch, C, H, W)
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

            # Resize to Phi4MM's expected size (448x448)
            img = _phi4mm_resize_transform(img)

            images_list.append(img)

        return images_list

    def create_input_tokens(
        self, state, lang_text, images_list, image_embeds=None, image_token_mask=None, actions=None
    ):
        """
        Create full input token sequence.

        Args:
            image_embeds: Actual image embeddings (batch, num_image_tokens, hidden_size).
                         If provided, we'll create the correct number of special tokens.

        Returns dict with:
            - input_ids: Token IDs
            - attention_mask: Padding mask
            - ar_mask: Autoregressive mask (0=bidirectional, 1=causal)
            - loss_mask: Which tokens to compute loss on
            - num_image_tokens: Actual number of image tokens (for tracking)
        """
        bsize = state.shape[0]
        device = state.device

        # Handle n_obs_steps dimension
        if state.ndim == 3:
            state = state[:, -1]

        # Discretize state
        bins = torch.linspace(-1, 1, 256 + 1, device=device)[:-1]
        discretized = torch.bucketize(state, bins) - 1
        discretized = discretized[:, :32]

        # Build prefix text with task and state
        prefix_texts = []
        for txt, disc in zip(lang_text, discretized, strict=False):
            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(str(val.item()) for val in disc)
            prefix_texts.append(f"Task: {cleaned}, State: {state_str};\n")

        # Tokenize prefix
        prefix_out = self.vlm_processor.tokenizer(
            prefix_texts, add_special_tokens=True, return_tensors="pt", padding="longest", truncation=False
        )
        prefix_ids = prefix_out["input_ids"].to(device)
        prefix_mask = prefix_out["attention_mask"].to(device)

        # Add image special tokens if we have images
        num_image_tokens = 0
        if image_embeds is not None:
            # Use ACTUAL number of image tokens from embeddings
            num_image_tokens = image_embeds.shape[1]

            # Use provided mask or default to all ones
            if image_token_mask is None:
                image_token_mask = torch.ones((bsize, num_image_tokens), dtype=torch.long, device=device)

            img_token_ids = torch.full(
                (bsize, num_image_tokens), _IMAGE_SPECIAL_TOKEN_ID, dtype=torch.long, device=device
            )

            # Prepend image tokens
            prefix_ids = torch.cat([img_token_ids, prefix_ids], dim=1)
            prefix_mask = torch.cat([image_token_mask, prefix_mask], dim=1)

        # Create action tokens if provided
        if actions is not None:
            # Pad actions to max_action_dim
            actions_pad = F.pad(actions, (0, max(0, self.config.max_action_dim - actions.shape[2])), value=0)[
                :, :, : self.config.max_action_dim
            ]

            # Tokenize with FAST
            batch_tokens = self.fast_processor(actions_pad.cpu())

            # Pad to same length
            max_len = max(len(seq) for seq in batch_tokens)
            padded_act_ids = []
            for seq in batch_tokens:
                padded_seq = seq + [0] * (max_len - len(seq))
                padded_act_ids.append(padded_seq)

            act_ids = torch.tensor(padded_act_ids, device=device)
            act_mask = (act_ids != 0).long()

            # Convert to Phi4 token space
            act_ids = self._act_tokens_to_phi4_tokens(act_ids)
            act_ids = torch.where(
                act_ids == self.phi4_vocab_size - 1 - self.fast_skip_tokens,
                self.pad_token_id,
                act_ids,
            )

            # Add "Action: " prefix and EOS
            action_prompt = self.vlm_processor.tokenizer(
                "Action: ", add_special_tokens=False, return_tensors="pt"
            )
            action_prompt_ids = action_prompt["input_ids"].to(device).expand(bsize, -1)
            action_prompt_mask = action_prompt["attention_mask"].to(device).expand(bsize, -1)

            eos_ids = torch.full((bsize, 1), self.eos_token_id, dtype=torch.long, device=device)
            eos_mask = torch.ones_like(eos_ids)

            # Concatenate: [prefix, "Action: ", action_tokens, EOS]
            final_ids = torch.cat([prefix_ids, action_prompt_ids, act_ids, eos_ids], dim=1)
            final_mask = torch.cat([prefix_mask, action_prompt_mask, act_mask, eos_mask], dim=1)

            # AR mask: 0 for prefix (bidirectional), 1 for actions (causal)
            prefix_ar = torch.zeros_like(prefix_mask)
            action_ar = torch.ones((bsize, action_prompt_ids.shape[1] + act_ids.shape[1] + 1), device=device)
            ar_mask = torch.cat([prefix_ar, action_ar], dim=1)

            # Loss mask: only on action tokens (not "Action: " prefix, not EOS)
            prefix_loss = torch.zeros_like(prefix_mask, dtype=torch.bool)
            action_prompt_loss = torch.zeros_like(action_prompt_mask, dtype=torch.bool)
            action_loss = act_mask.bool()  # Loss on actual action tokens
            eos_loss = torch.zeros_like(eos_mask, dtype=torch.bool)
            loss_mask = torch.cat([prefix_loss, action_prompt_loss, action_loss, eos_loss], dim=1)
        else:
            final_ids = prefix_ids
            final_mask = prefix_mask
            ar_mask = torch.zeros_like(prefix_mask)
            loss_mask = torch.zeros_like(prefix_mask, dtype=torch.bool)

        return {
            "input_ids": final_ids,
            "attention_mask": final_mask,
            "ar_mask": ar_mask,
            "loss_mask": loss_mask,
            "num_image_tokens": num_image_tokens,
        }

    def make_prefix_lm_mask(self, attention_mask: torch.Tensor, ar_mask: torch.Tensor) -> torch.Tensor:
        """
        Create prefix-LM attention mask.

        Args:
            attention_mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
            ar_mask: (batch, seq_len) - 0 for bidirectional, 1 for causal

        Returns:
            4D attention mask: (batch, 1, seq_len, seq_len)
        """
        # Cumulative sum of AR mask
        cumsum = torch.cumsum(ar_mask, dim=1)  # (batch, seq_len)

        # Token i can attend to token j if:
        # cumsum[j] <= cumsum[i] (respects causality for AR tokens)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # (batch, seq_len, seq_len)

        # Also respect padding
        valid_mask = attention_mask[:, None, :] * attention_mask[:, :, None]
        attn_mask = attn_mask & valid_mask.bool()

        # Convert to 4D format: True -> 0.0, False -> -inf
        attn_mask_4d = attn_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
        min_dtype = torch.finfo(self.dtype).min
        attn_mask_4d = torch.where(attn_mask_4d, 0.0, min_dtype).to(dtype=self.dtype)

        return attn_mask_4d

    def make_causal_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Create simple causal attention mask (compatible with flash attention).

        Args:
            attention_mask: (batch, seq_len) - 1 for valid tokens, 0 for padding

        Returns:
            4D attention mask: (batch, 1, seq_len, seq_len) or None for flash attention
        """
        # For flash attention, we can just return None and it will use causal mask
        if getattr(self.config, "attention_implementation", "sdpa") == "flash_attention_2":
            return None

        # For SDPA, create explicit causal mask
        batch_size, seq_len = attention_mask.shape

        # Create causal mask: lower triangular
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device, dtype=torch.bool))

        # Expand for batch
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply padding mask
        valid_mask = attention_mask[:, None, :] * attention_mask[:, :, None]
        causal_mask = causal_mask & valid_mask.bool()

        # Convert to 4D format: True -> 0.0, False -> -inf
        attn_mask_4d = causal_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
        min_dtype = torch.finfo(self.dtype).min
        attn_mask_4d = torch.where(attn_mask_4d, 0.0, min_dtype).to(dtype=self.dtype)

        return attn_mask_4d

    def get_attention_mask(self, attention_mask: torch.Tensor, ar_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get appropriate attention mask based on config.

        Args:
            attention_mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
            ar_mask: (batch, seq_len) - 0 for bidirectional, 1 for causal (only for prefix-LM)

        Returns:
            4D attention mask or None
        """
        use_full_causal = getattr(self.config, "use_full_causal_mask", False)

        if use_full_causal:
            return self.make_causal_mask(attention_mask)
        else:
            if ar_mask is None:
                raise ValueError("ar_mask required for prefix-LM attention")
            return self.make_prefix_lm_mask(attention_mask, ar_mask)

    def forward(
        self,
        image,
        prompt,
        state,
        actions,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=None,
    ):
        """
        Training forward pass with autoregressive next-token prediction.

        Single forward pass through Phi4MM (like pi0fast).
        """
        # Prepare images
        # batch_dict = self.convert_image_to_batch_dict(image)
        batch_dict = {"observation.images": image} if isinstance(image, torch.Tensor) else image
        images_list = self.prepare_images(batch_dict)

        # Get image embeddings directly (no forward pass)
        image_embeds, image_token_mask = self.embed_image(images_list) if images_list else (None, None)

        # Create input tokens with actual image embeddings
        token_dict = self.create_input_tokens(
            state,
            prompt,
            images_list,
            image_embeds=image_embeds,
            image_token_mask=image_token_mask,
            actions=actions,
        )
        input_ids = token_dict["input_ids"]
        attention_mask = token_dict["attention_mask"]
        ar_mask = token_dict["ar_mask"]
        loss_mask = token_dict["loss_mask"]

        # Warn if we're exceeding configured max sequence length
        max_seq = getattr(self.config, "max_seq_len", None)
        if max_seq is not None and input_ids.shape[1] > max_seq:
            logger.warning(
                f"Sequence length {input_ids.shape[1]} exceeds max_seq_len {max_seq}. "
                "Consider reducing dynamic_hd or prompt length."
            )

        # Embed tokens
        token_embeds = self.embed_tokens(input_ids)

        # Replace image special tokens with actual image embeddings
        if image_embeds is not None:
            # Clone to avoid in-place operation on leaf variable
            # token_embeds = token_embeds.clone()

            # Find positions of image tokens
            is_image_token = input_ids == _IMAGE_SPECIAL_TOKEN_ID

            # Replace in embeddings
            for b in range(input_ids.shape[0]):
                img_positions = torch.where(is_image_token[b])[0]
                if len(img_positions) > 0:
                    # Take first N positions where N = number of image embeddings
                    n_img_embeds = min(len(img_positions), image_embeds.shape[1])
                    token_embeds[b, img_positions[:n_img_embeds]] = image_embeds[b, :n_img_embeds]

        # Prepare for next-token prediction: remove last token
        input_embeds = token_embeds[:, :-1]
        input_mask = attention_mask[:, :-1]
        ar_mask_trimmed = ar_mask[:, :-1]

        # Create attention mask (prefix-LM or full causal based on config)
        attn_mask_4d = self.get_attention_mask(input_mask, ar_mask_trimmed)

        # Position IDs
        position_ids = torch.cumsum(input_mask, dim=-1) - 1

        # Forward through language model
        outputs = self.vlm_backbone.model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len-1, hidden_dim)

        # Decode to logits
        logits = self.vlm_backbone.lm_head(hidden_states)  # (batch, seq_len-1, vocab_size)

        # Targets: shifted tokens
        targets = input_ids[:, 1:]  # (batch, seq_len-1)
        loss_mask_shifted = loss_mask[:, 1:]  # (batch, seq_len-1)

        # Compute loss
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # Apply loss mask
        token_loss = token_loss * loss_mask_shifted.reshape(-1).float()

        # Compute mean loss
        num_loss_tokens = loss_mask_shifted.sum()
        loss = token_loss.sum() / num_loss_tokens if num_loss_tokens > 0 else token_loss.sum()

        # Return in expected format (batch, n_action_steps, 1)
        return loss.unsqueeze(0).unsqueeze(1).expand(actions.shape[0], actions.shape[1], 1)

    def decode_actions_with_fast(
        self,
        tokens: list[list[int]],
        time_horizon: int | None = None,
        action_dim: int | None = None,
        relaxed_decoding: bool = True,
    ) -> np.array:
        """Decode action tokens using FAST (following pi0fast implementation)."""
        time_horizon = time_horizon or self.action_horizon
        action_dim = action_dim or self.action_dim

        decoded_actions = []
        for token in tokens:
            try:
                # Handle empty token lists
                if not token:
                    raise ValueError("Empty token list")

                decoded_tokens = self.fast_processor.bpe_tokenizer.decode(token)
                # Clip ord() values to prevent extreme unicode characters from causing huge DCT coefficients
                # Valid DCT coefficients for FAST are typically in range corresponding to ord() < 1000
                ord_values = np.array([min(ord(c), 512) for c in decoded_tokens])
                decoded_dct_coeff = ord_values + self.fast_processor.min_token

                if relaxed_decoding:
                    expected_seq_len = time_horizon * action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
            except Exception:
                # Fallback to zeros for any decoding errors
                decoded_dct_coeff = np.zeros((time_horizon, action_dim))

            # Clip DCT coefficients to reasonable range to prevent extreme action values
            # FAST training data typically has DCT coefficients in range ~[-128, 128]
            # Clipping here prevents out-of-distribution FAST tokens from causing huge values
            if decoded_dct_coeff.max() > 1000 or decoded_dct_coeff.min() < -1000:
                min_val = decoded_dct_coeff.min()
                max_val = decoded_dct_coeff.max()
                logger.warning(f"Extreme DCT coefficients detected: [{min_val:.2f}, {max_val:.2f}]")
                logger.warning(f"Input FAST tokens (first 30): {token[:30]}")
                logger.warning("Clipping to [-512, 512] range")
                decoded_dct_coeff = np.clip(decoded_dct_coeff, -512, 512)

            action = idct(decoded_dct_coeff / self.fast_processor.scale, axis=0, norm="ortho")
            decoded_actions.append(action)

        return np.stack(decoded_actions)

    def extract_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        """
        Extract actions from generated tokens.

        NOTE: Unlike pi0fast, we CANNOT use decode-then-reencode because Phi4MM's
        action tokens are not mapped to decodable strings in the tokenizer.
        Instead, we directly extract and convert token IDs.
        """
        # Define action token range
        action_token_min = self.phi4_vocab_size - 1 - self.fast_skip_tokens - self.fast_vocab_size
        action_token_max = self.phi4_vocab_size - 1 - self.fast_skip_tokens

        # Step 1: Extract only tokens in the action range (skip decode entirely)
        action_tokens_filtered = []
        for sample_tokens in tokens:
            # Find tokens in the action range
            mask = (sample_tokens >= action_token_min) & (sample_tokens <= action_token_max)
            action_toks = sample_tokens[mask]

            # Convert to FAST space
            if len(action_toks) == 0:
                action_tokens_filtered.append([])
                continue

            fast_toks = self._phi4_tokens_to_act_tokens(action_toks)

            # Filter to valid FAST range [0, fast_vocab_size)
            valid_toks = [t.item() for t in fast_toks if 0 <= t < self.fast_vocab_size]
            action_tokens_filtered.append(valid_toks)

        # Step 2: Decode with FAST
        decoded_actions = [
            torch.tensor(
                self.decode_actions_with_fast(
                    [tok],  # wrap in list for batch
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                    relaxed_decoding=True,
                ),
                device=tokens.device,
            ).squeeze(0)
            for tok in action_tokens_filtered
        ]

        return torch.stack(decoded_actions, dim=0)

    def sample_actions(self, image, prompt, state, noise=None, image_mask=None) -> Tensor:
        """
        Inference: Generate actions autoregressively.
        """
        device = state.device
        bsize = state.shape[0] if state.ndim == 2 else state.shape[0]

        # Prepare images
        images_list = self.prepare_images(
            {"observation.images": image} if isinstance(image, torch.Tensor) else image
        )

        # Get image embeddings
        image_embeds, image_token_mask = self.embed_image(images_list) if images_list else (None, None)

        # Create input tokens with actual image embeddings (no actions, but we'll add "Action: " prompt)
        token_dict = self.create_input_tokens(
            state,
            prompt,
            images_list,
            image_embeds=image_embeds,
            image_token_mask=image_token_mask,
            actions=None,
        )
        input_ids = token_dict["input_ids"]
        attention_mask = token_dict["attention_mask"]

        # Add "Action: " prompt at the end to signal the model to generate actions
        action_prompt = self.vlm_processor.tokenizer(
            "Action: ", add_special_tokens=False, return_tensors="pt"
        )
        action_prompt_ids = action_prompt["input_ids"].to(device).expand(bsize, -1)
        action_prompt_mask = action_prompt["attention_mask"].to(device).expand(bsize, -1)

        # Append to input
        input_ids = torch.cat([input_ids, action_prompt_ids], dim=1)
        attention_mask = torch.cat([attention_mask, action_prompt_mask], dim=1)

        # Embed tokens
        token_embeds = self.embed_tokens(input_ids)

        # Replace image special tokens
        if image_embeds is not None:
            # Clone to avoid in-place operation on leaf variable
            token_embeds = token_embeds.clone()

            is_image_token = input_ids == _IMAGE_SPECIAL_TOKEN_ID
            for b in range(input_ids.shape[0]):
                img_positions = torch.where(is_image_token[b])[0]
                if len(img_positions) > 0:
                    n_img_embeds = min(len(img_positions), image_embeds.shape[1])
                    token_embeds[b, img_positions[:n_img_embeds]] = image_embeds[b, :n_img_embeds]

        # Manual autoregressive generation loop (Phi4MM's .generate() is broken with inputs_embeds)
        max_new = getattr(self.config, "max_decoding_steps", 100)

        # Track prefix length for prefix-LM attention
        # Everything before "Action: " is prefix (bidirectional)
        # Everything from "Action: " onwards is suffix (causal)
        prefix_len = input_ids.shape[1] - action_prompt_ids.shape[1]

        generated_ids = []
        current_embeds = token_embeds
        current_mask = attention_mask

        for _ in range(max_new):
            # Create ar_mask for prefix-LM attention
            # 0 = bidirectional (prefix), 1 = causal (suffix starting from "Action: ")
            ar_mask = torch.zeros_like(current_mask)
            ar_mask[:, prefix_len:] = 1  # Everything from "Action: " onwards is causal

            # Create attention mask for prefill (prefix-LM or full causal based on config)
            attn_mask_4d = self.get_attention_mask(current_mask, ar_mask)

            # Create position IDs
            position_ids = torch.cumsum(current_mask, dim=-1) - 1

            # Forward pass with prefix-LM attention
            outputs = self.vlm_backbone.model(
                inputs_embeds=current_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                return_dict=True,
            )

            # Get logits for last position
            logits = self.vlm_backbone.lm_head(outputs.last_hidden_state[:, -1:, :])  # (batch, 1, vocab)

            # Greedy sampling
            next_token_ids = logits.argmax(dim=-1)  # (batch, 1)
            generated_ids.append(next_token_ids)

            # Check if all sequences hit EOS
            if (next_token_ids == self.eos_token_id).all():
                break

            # Prepare for next iteration
            next_token_embeds = self.embed_tokens(next_token_ids)
            current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(bsize, 1, device=device)], dim=1)

        # Concatenate input_ids with generated tokens
        if generated_ids:
            generated_ids = torch.cat(generated_ids, dim=1)  # (batch, num_generated)
            output_tokens = torch.cat([input_ids, generated_ids], dim=1)
        else:
            output_tokens = input_ids

        # Use extract_actions method
        # Unlike pi0fast, we directly extract action tokens without decode-then-reencode
        decoded_actions = self.extract_actions(output_tokens, self.action_horizon, self.action_dim)

        return decoded_actions

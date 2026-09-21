"""
Phi5 backbone adapter for the Rho policy.

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

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from rho.policies.rho.backbone import BackboneAdapter
from rho.policies.rho.phi5.processing_phi5 import Phi5ImageProcessor

logger = logging.getLogger(__name__)

# Sentinel value used by BunnyPhi4ForCausalLM to locate image positions
# in the token sequence. prepare_inputs_labels_for_multimodal() searches
# for this value and splices in vision embeddings at those positions.
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"  # nosec B105


def _restore_pretrained_rotary_buffers(language_model: nn.Module, source_dtype: torch.dtype) -> None:
    """Recreate nonpersistent RoPE buffers through the pretrained load dtype."""
    rotary_emb = language_model.rotary_emb
    if rotary_emb.rope_type != "default":
        raise ValueError(f"Unsupported Phi5 RoPE type: {rotary_emb.rope_type!r}")

    inv_freq, _ = rotary_emb.compute_default_rope_parameters(rotary_emb.config, rotary_emb.inv_freq.device)
    inv_freq = inv_freq.to(source_dtype).to(rotary_emb.inv_freq.dtype)
    rotary_emb.inv_freq.copy_(inv_freq)
    rotary_emb.original_inv_freq.copy_(inv_freq)


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

    def prepare_prompt(self, batch, num_images_override: int | None = None) -> list[str]:
        from rho.common.constants import OBSERVATION_LANG as OBS_TASK

        if OBS_TASK not in batch:
            logger.warning(f"Key {OBS_TASK} not found in batch. Returning empty prompt list.")
            return []

        if num_images_override is not None:
            num_images = num_images_override
        else:
            num_images = len(self.config.image_features) * self.config.n_obs_steps
        image_tokens = "<image>" * num_images

        formatted = []
        for prompt in batch[OBS_TASK]:
            # Most robot/LLaVA prompts do not need explicit image placement, so
            # we prepend the required image sentinels. ERQA-style prompts carry
            # interleaved `<image>` placeholders; preserve those positions and
            # only repair the count if the manifest is malformed.
            prompt_image_count = prompt.count("<image>")
            if prompt_image_count:
                if prompt_image_count < num_images:
                    prompt = ("<image>" * (num_images - prompt_image_count)) + prompt
                elif prompt_image_count > num_images:
                    extra = prompt_image_count - num_images
                    while extra > 0:
                        prompt = prompt.replace("<image>", "", 1)
                        extra -= 1
                image_prefix = ""
                cleaned = prompt.lstrip("\n ")
            else:
                image_prefix = image_tokens
                cleaned = prompt.lstrip("\n ")
            text = (
                f"<|im_start|>user<|im_sep|>{image_prefix}{cleaned}<|im_end|><|im_start|>assistant<|im_sep|>"
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
        attn_impl = getattr(self.config, "attention_implementation", "flash_attention_2")
        if self._has_weight_files():
            logger.info("Found model weight files — loading pretrained backbone...")
            backbone = AutoModelForCausalLM.from_pretrained(
                self.config.vlm_backbone_folder,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
        else:
            logger.warning(
                "No safetensors/model weight files found in '%s'. "
                "Creating the backbone architecture without initializing weights. "
                "This is suitable for architecture validation but NOT for inference. "
                "To load real weights, set vlm_backbone_folder to a directory "
                "containing .safetensors files or set the VLM_BACKBONE_FOLDER env var.",
                self.config.vlm_backbone_folder,
            )
            logger.info("Loading model config from '%s'...", self.config.vlm_backbone_folder)
            config = AutoConfig.from_pretrained(
                self.config.vlm_backbone_folder,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            logger.info("Creating backbone model from config (this may take a moment)...")
            # Every backbone tensor is replaced by the policy checkpoint before
            # use, so initializing billions of temporary random values is wasted.
            with no_init_weights():
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

        # transformers 5.x SigLIP2 numeric-parity compat.  No-op on
        # transformers 4.x.  See scratch/REPORT_transformers_4_vs_5_parity.md
        # for the investigation and rationale.
        self._apply_transformers_5x_siglip2_compat(backbone)

        if self.config is not None and self.config.enable_gradient_checkpointing:
            # use_reentrant=False is required for DDP without
            # static_graph=True (cotraining alternates which head's params
            # get touched per iteration, so static_graph isn't usable; the
            # reentrant variant tries to mark the same param ready twice
            # under DDP without static_graph and crashes).
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # transformers 5.x unconditionally registers an embed_tokens
            # forward hook that calls `requires_grad_(True)` on its output
            # whenever gradient checkpointing is enabled and main_input_name
            # is "input_ids" (modeling_utils.py:3109-3117). For our frozen
            # LM + trainable action-expert setup this forces every Phi3
            # decoder layer to build a useless autograd graph, doubling
            # forward-activation memory and per-step time. We never train
            # input embeddings here, so undo the hook.
            #
            # Older transformers (<5.0) only register this hook under PEFT,
            # so disable_input_require_grads() will raise AttributeError on
            # the missing `_require_grads_hook(s)` attribute. Guard on the
            # hook(s) actually being present.
            has_grad_hook = hasattr(backbone, "_require_grads_hook") or getattr(
                backbone, "_require_grads_hooks", None
            )
            if has_grad_hook and hasattr(backbone, "disable_input_require_grads"):
                backbone.disable_input_require_grads()

        logger.info("Moving backbone to device=%s, dtype=%s...", self.device, self.dtype)
        backbone.to(device=self.device, dtype=self.dtype)
        if getattr(self, "_backbone_has_uninitialized_weights", False):
            # These buffers are absent from checkpoints. Match the normal
            # from_pretrained(float16) -> policy dtype conversion path.
            _restore_pretrained_rotary_buffers(self.get_language_model(backbone), source_dtype=torch.float16)
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

    def _apply_transformers_5x_siglip2_compat(self, backbone: nn.Module) -> None:
        """Compatibility shims for transformers >= 5.x when running with SigLIP2.

        No-op on transformers 4.x.  Two orthogonal regressions surfaced in
        transformers 5.5 relative to 4.57.6 that would silently change model
        outputs for Phi-4-vision-5B-* checkpoints trained under 4.57:

        1. ``Siglip2VisionTransformer.forward`` now routes the vision attention
           mask through ``create_bidirectional_mask``, which returns ``None``
           when the mask is a no-op.  With no mask, the SDPA dispatcher picks
           a different fp16 kernel than 4.57's explicit-mask path, and drift
           accumulates through 27 encoder layers.  Fix: pin the vision
           tower's SDPA backend to EFFICIENT_ATTENTION (with a MATH fallback)
           via ``torch.nn.attention.sdpa_kernel`` wrapped around each
           ``Siglip2EncoderLayer.forward``.  Wrapping at the encoder-layer
           granularity is required so the pin survives ``torch.utils.checkpoint``
           recompute on backward (the outer-tower wrap would be exited by the
           time gradient checkpointing recomputes the encoder layer, causing
           a Flash-vs-Efficient RNG-state metadata mismatch and
           ``CheckpointError``).  The scope is still just the vision tower —
           LM SDPA (when ``attention_implementation="sdpa"``) and
           action-expert ``nn.MultiheadAttention`` retain their native
           backend selection.

        2. The new ``_can_record_outputs`` capture mechanism doubles
           ``hidden_states`` (2N+1 entries instead of N+1, each layer output
           captured twice).  The Bunny remote code's ``feature_select`` uses
           ``select_layer=-2`` which now points to the *final* layer instead
           of the second-to-last.  Fix: remap ``select_layer`` so it lands
           on the same semantic tensor 4.57 grabbed (``2k+1`` for negative
           ``k``, ``2k-1`` for positive ``k``).

        Together these close the SDPA-selection gap between 4.57 and 5.x on
        the vision tower.  Verified end-to-end via ``scratch/dump_option_b.py``
        against a checkpoint trained under 4.57.
        """
        try:
            import transformers

            major = int(transformers.__version__.split(".")[0])
        except (ImportError, ValueError, IndexError):
            return
        if major < 5:
            return

        # Regression #1: pin vision-tower SDPA backend so 4.57's explicit-mask
        # path and 5.x's mask=None path land on the same kernel.  We wrap
        # each SigLIP2 encoder layer's forward (not the outer vision-tower
        # forward), because ``torch.utils.checkpoint(..., use_reentrant=False)``
        # only preserves RNG state across the forward-vs-recompute boundary
        # — not the ``sdpa_kernel`` context.  If the wrap sat at the outer
        # tower forward, gradient-checkpointed encoder layers would recompute
        # *outside* the context on backward, the SDPA dispatcher would fall
        # back to Flash (which saves a ``(seed, offset)`` uint64 CUDA tensor
        # for fused dropout) whereas forward saved efficient-attention's
        # scalar CPU state, and torch would raise ``CheckpointError`` on the
        # metadata mismatch.  Wrapping at the encoder-layer granularity means
        # the wrap *is* the checkpointed unit — recompute re-enters the same
        # context.  LM and action-expert attention paths remain untouched.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def _wrap_forward_with_pinned_sdpa(module: nn.Module) -> None:
            _orig_forward = module.forward

            def _forward(*args, **kwargs):
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    return _orig_forward(*args, **kwargs)

            module.forward = _forward

        vt_wrapper = backbone.model.vision_tower
        inner_vt = vt_wrapper.vision_tower  # Siglip2VisionModel
        vision_model = getattr(inner_vt, "vision_model", inner_vt)
        encoder_layers = vision_model.encoder.layers  # 27x Siglip2EncoderLayer
        for layer in encoder_layers:
            _wrap_forward_with_pinned_sdpa(layer)
        # Optional pooling head has its own SDPA call; disabled by default in
        # this checkpoint but wrap defensively so the fix stays correct if a
        # future config enables it.
        head = getattr(vision_model, "head", None)
        if head is not None:
            _wrap_forward_with_pinned_sdpa(head)
        logger.info(
            "transformers>=5 SigLIP2 compat: pinned SDPA backend to EFFICIENT_ATTENTION "
            "on %d Siglip2EncoderLayer%s%s (survives gradient-checkpointing recompute; "
            "LM and action-expert MHA unaffected).",
            len(encoder_layers),
            "" if len(encoder_layers) == 1 else "s",
            " and pooling head" if head is not None else "",
        )

        # Regression #2: hidden_states tuple doubled by _can_record_outputs.
        old_k = vt_wrapper.select_layer
        if old_k < 0:
            new_k = 2 * old_k + 1  # -2 -> -3, -1 -> -1
        elif old_k > 0:
            new_k = 2 * old_k - 1  # 1 -> 1, 2 -> 3
        else:
            new_k = 0  # embedding output, unchanged
        vt_wrapper.select_layer = new_k
        logger.info(
            "transformers>=5 SigLIP2 compat: remapped Siglip2VisionTower.select_layer %s → %s.",
            old_k,
            new_k,
        )

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

    def _process_images(self, processor, images, image_mask) -> dict:
        """Extract NaFlex patches and apply per-image validity masks."""
        if isinstance(images, torch.Tensor):
            if images.ndim != 5:
                raise ValueError(
                    "Phi5 images must have shape (batch, images, channels, height, width), "
                    f"got {tuple(images.shape)}."
                )
            batch_size, n_imgs = images.shape[:2]
            flat_images = images.flatten(0, 1)  # (B, N, C, H, W) -> (B*N, C, H, W)
        else:
            batch_size = len(images)
            n_imgs = len(images[0])
            flat_images = torch.stack(
                [img for sample in images for img in sample]
            )  # B * N * (C, H, W) -> (B*N, C, H, W)
        if image_mask is not None:
            if not isinstance(image_mask, torch.Tensor):
                image_mask = torch.stack(image_mask)
            if image_mask.shape != (batch_size, n_imgs):
                raise ValueError(
                    f"Phi5 image_mask must have shape {(batch_size, n_imgs)}, got {tuple(image_mask.shape)}."
                )

        vision_out = processor.image_processor.process_batched(flat_images)
        if image_mask is not None:
            patch_mask = vision_out["pixel_attention_mask"]
            valid_images = image_mask.reshape(-1, 1).to(device=patch_mask.device, dtype=torch.bool)
            # Keep slots aligned with sentinels; SigLIP removes all masked patches before LM splicing.
            vision_out["pixel_attention_mask"] = patch_mask * valid_images
        return vision_out

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
        IMAGE_TOKEN_INDEX (-200) in input_ids, allowing the backbone to splice
        valid vision embeddings into the sequence.

        Args:
            processor: BunnyPhi4Processor with Phi5ImageProcessor.
            images: Tensor shaped (batch, images, C, H, W), or the legacy
                nested-list representation.
            texts: List[str] of batch prompts with <image> tokens.
            image_mask: (batch, images) validity mask (1 = valid), or None.
            pad_sequence_fn: padding utility.
            cat_with_pad_fn: cat-with-pad utility.
        """
        vision_out = self._process_images(processor, images, image_mask)
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
            "pixel_values": vision_out["pixel_values"],
            "pixel_attention_mask": vision_out["pixel_attention_mask"],
            "spatial_shapes": vision_out["spatial_shapes"],
        }

    def _build_lm_inputs(
        self,
        processor,
        images,
        prompts: list[str],
        target_texts: list[str],
        image_mask,
        pad_sequence_fn,
    ) -> dict:
        """Tokenize prompt+target with -100 labels over the prompt span.

        Shared by ``compute_lm_loss`` and ``forward_lm_with_hidden_state``.
        """
        if len(prompts) != len(target_texts):
            raise ValueError(f"prompts/target_texts length mismatch: {len(prompts)} vs {len(target_texts)}")
        if pad_sequence_fn is None:
            raise ValueError("_build_lm_inputs requires pad_sequence_fn")

        tokenizer = processor.tokenizer
        eos_id = tokenizer.eos_token_id

        context_len = getattr(tokenizer, "model_max_length", 8192) or 8192
        if context_len > 8192 or context_len < 0:
            context_len = 8192
        configured_len = int(getattr(self.config, "vl_lm_max_length", context_len) or context_len)
        max_seq_len = max(1, min(configured_len, context_len))

        full_input_ids: list[torch.Tensor] = []
        full_labels: list[torch.Tensor] = []
        raw_lengths: list[int] = []
        target_lengths: list[int] = []
        truncated = 0
        for prompt, target in zip(prompts, target_texts, strict=True):
            prompt_ids = _tokenizer_image_token(prompt, tokenizer)
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids = list(target_ids) + [eos_id]

            seq = list(prompt_ids) + list(target_ids)
            label_seq = [-100] * len(prompt_ids) + list(target_ids)
            raw_lengths.append(len(seq))
            target_lengths.append(len(target_ids))

            if len(seq) > max_seq_len:
                truncated += 1
                logger.warning(
                    "_build_lm_inputs: truncating sample from %d -> %d tokens "
                    "(prompt=%d target=%d). prompt[:200]=%r target[:200]=%r",
                    len(seq),
                    max_seq_len,
                    len(prompt_ids),
                    len(target_ids),
                    prompt[:200],
                    target[:200] if isinstance(target, str) else str(target)[:200],
                )
                # Left-truncate to preserve the answer span on the right.
                seq = seq[-max_seq_len:]
                label_seq = label_seq[-max_seq_len:]

            full_input_ids.append(torch.tensor(seq, dtype=torch.long))
            full_labels.append(torch.tensor(label_seq, dtype=torch.long))

        input_ids = pad_sequence_fn(full_input_ids, padding_side="right", padding_value=0)
        labels = pad_sequence_fn(full_labels, padding_side="right", padding_value=-100)
        attention_mask = (input_ids != 0).long().to(self.device)
        self._last_lm_input_stats = {
            "vl_lm/raw_max_tokens": float(max(raw_lengths) if raw_lengths else 0),
            "vl_lm/raw_mean_tokens": float(sum(raw_lengths) / len(raw_lengths) if raw_lengths else 0),
            "vl_lm/target_max_tokens": float(max(target_lengths) if target_lengths else 0),
            "vl_lm/padded_tokens": float(input_ids.shape[1]),
            "vl_lm/max_length": float(max_seq_len),
            "vl_lm/truncated_samples": float(truncated),
        }

        vision_out = self._process_images(processor, images, image_mask)

        return {
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "attention_mask": attention_mask,
            "pixel_values": vision_out["pixel_values"],
            "pixel_attention_mask": vision_out["pixel_attention_mask"],
            "spatial_shapes": vision_out["spatial_shapes"],
        }

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
        """Phi5 (BunnyPhi4ForCausalLM) LM-loss path.

        Tokenizes ``prompt`` with ``<image>`` -> -200 sentinels (same as
        ``process_batch``), then appends ``target`` tokens (no image
        sentinels). Builds ``labels`` as -100 for prompt + sentinel tokens
        and the real target token IDs for the answer span. Calls
        ``backbone(..., labels=labels)`` and returns ``outputs.loss``.
        Image embeddings get spliced in by
        ``BunnyPhi4ForCausalLM.prepare_inputs_labels_for_multimodal()``,
        which the standard backbone forward calls internally.
        """
        batch = self._build_lm_inputs(processor, images, prompts, target_texts, image_mask, pad_sequence_fn)
        outputs = backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            pixel_attention_mask=batch["pixel_attention_mask"],
            spatial_shapes=batch["spatial_shapes"],
            labels=batch["labels"],
            return_dict=True,
            use_cache=False,
        )
        return outputs.loss

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
        """Single VLM forward returning all hidden states + prefix mask + LM CE loss.

        Splices image embeddings ourselves (via
        ``prepare_inputs_labels_for_multimodal``) so we capture the
        post-splice attention_mask and labels, then run the backbone
        forward with ``output_hidden_states=True`` and ``labels=...``.
        Prefix positions are everywhere the spliced labels are -100 AND
        the spliced attention_mask is 1 (i.e., not answer span and not
        padding). Returns the full per-layer hidden-state tuple raw
        (no projection); the caller picks which layer(s) it needs.
        """
        batch = self._build_lm_inputs(processor, images, prompts, target_texts, image_mask, pad_sequence_fn)

        image_inputs = {
            "pixel_values": batch["pixel_values"],
            "pixel_attention_mask": batch["pixel_attention_mask"],
            "spatial_shapes": batch["spatial_shapes"],
        }
        (
            _,
            position_ids,
            spliced_attention_mask,
            _,
            inputs_embeds,
            spliced_labels,
        ) = backbone.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            None,
            batch["attention_mask"],
            None,
            batch["labels"],
            image_inputs,
        )

        outputs = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=spliced_attention_mask,
            position_ids=position_ids,
            labels=spliced_labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # Prefix mask: 1 where this is a valid context position (image or
        # prompt token) AND not part of the answer span / padding.
        # Spliced labels are -100 over (prompt + image-spliced positions
        # + padding); the answer span carries real token IDs.
        attn_mask_long = spliced_attention_mask.to(self.device).long()
        is_prefix = (spliced_labels == -100).long()
        prefix_mask = is_prefix * attn_mask_long

        return list(outputs.hidden_states), prefix_mask, outputs.loss

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
        language_model = self.get_language_model(backbone)
        num_hidden_layers = min(language_model.config.num_hidden_layers, len(language_model.layers))
        assert 0 <= hidden_state_idx <= num_hidden_layers, (
            f"Layer {hidden_state_idx} is out of range. Must be between 0 and {num_hidden_layers}."
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

        original_num_hidden_layers = language_model.config.num_hidden_layers
        try:
            # Transformers replaces the final captured hidden state with the
            # post-norm output, so run one extra layer to preserve index parity.
            language_model.config.num_hidden_layers = min(hidden_state_idx + 1, original_num_hidden_layers)
            outputs = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
        finally:
            language_model.config.num_hidden_layers = original_num_hidden_layers
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
            use_cache=False,
        )
        return list(outputs.hidden_states), attention_mask.to(self.device)

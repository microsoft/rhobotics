import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .processing_pi0 import _get_safe_dtype, _preprocess_observation_pytorch

logger = logging.getLogger(__name__)


@dataclass
class _GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def _get_gemma_config(variant: str) -> _GemmaConfig:
    # Matches `openpi.models.gemma.get_config` numerics.
    if variant == "dummy":
        return _GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return _GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return _GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unknown gemma variant: {variant}")


class _PaliGemmaWithExpertModel(nn.Module):
    """Directly adapted from OpenPI's `PaliGemmaWithExpertModel` (PyTorch).

    Uses HF `PaliGemmaForConditionalGeneration` and `GemmaForCausalLM`.
    """

    def __init__(
        self,
        vlm_config: _GemmaConfig,
        action_expert_config: _GemmaConfig,
        use_adarms: list[bool] | None,
        precision: torch.dtype,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.gemma import modeling_gemma

        self._modeling_gemma = modeling_gemma

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self._to_precision(precision)

    def _to_precision(self, precision: torch.dtype):
        if precision == torch.bfloat16:
            self.to(dtype=torch.bfloat16)
        elif precision == torch.float32:
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Unsupported dtype for PI0: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            prefix_output = None
            prefix_past_key_values = None
            suffix_output = suffix_output.last_hidden_state
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = self._modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                att_output, _ = self._modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )

                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                num_heads = getattr(self.paligemma.language_model.layers[layer_idx].self_attn, "num_heads", 8)
                att_output = att_output.reshape(batch_size, -1, 1 * num_heads * head_dim)

                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    out_emb = self._modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])

                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    out_emb = self._modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    *,
    device: torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time is expected to be shape (batch_size,)")

    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


if TYPE_CHECKING:
    from .configuration_pi0 import PI0Config


class _PI0Pytorch(nn.Module):
    """OpenPI PI0 flow-matching model (PyTorch), adapted to avoid JAX deps."""

    IMAGE_KEYS = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    IMAGE_RESOLUTION = (224, 224)

    def __init__(self, config: "PI0Config"):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _get_gemma_config(config.paligemma_variant)
        action_expert_config = _get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = _PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        if config.compile_model:
            try:
                self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            except Exception as e:  # pragma: no cover
                logger.warning(f"torch.compile failed, continuing without compile: {e}")

        self.gradient_checkpointing_enabled = False

        if config.require_transformers_replace:
            msg = (
                "transformers_replace is required for PI0 in this setup. "
                "Install transformers==4.53.2 and patch transformers with OpenPI's transformers_replace."
            )
            try:
                from transformers.models.siglip import check

                if not check.check_whether_transformers_replace_is_installed_correctly():
                    raise ValueError(msg)
            except ImportError:
                raise ValueError(msg) from None

    @property
    def dtype(self) -> torch.dtype:
        """Module dtype. Matches RhoAlphaModel.dtype so rho.hil.noise_inverse_map
        can treat both flow models through the same interface."""
        return self.config.dtype

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    # ------------------------------------------------------------------
    # Shared flow-model interface for rho.hil.noise_inverse_map
    #
    # The inverter calls get_image_text_hidden_state -> velocity_eval (many
    # times, once per denoise step) -> sample_actions_from_precomputed (once,
    # for round-trip verification). RhoAlphaFlowMatchingModel exposes the same
    # three methods. ``precomputed_hidden_state`` is treated opaquely by the
    # inverter -- rhoalpha threads (image_text_embed, image_text_mask), pi0
    # threads (prefix_pad_masks, past_key_values). Both encode the same
    # prefix conditioning, just at different points in the transformer stack.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_image_text_hidden_state(self, image, prompt):
        """Run the VLM prefix forward once and cache its KV.

        Args:
            image: ``(images_list, img_masks_list)`` from ``PI0Policy.prepare_image``.
                ``images_list`` is a list of (B, 3, H, W) tensors, one per camera.
            prompt: ``(lang_tokens, lang_masks)`` from ``PI0Policy.prepare_prompt``.

        Returns:
            ``(prefix_pad_masks, past_key_values)`` -- pi0's flavor of the
            opaque "precomputed prefix conditioning" the inverter threads
            through velocity_eval / sample_actions_from_precomputed.
        """
        from .processing_pi0 import _resize_with_pad_torch

        images, img_masks = image

        # Resize each camera to self.IMAGE_RESOLUTION before siglip. pi0's
        # standard inference path does this inside _preprocess_observation;
        # we replicate it here so the flowdagger inversion entry point can't
        # bypass it. Training data may live at 448x448 (e.g. fr3
        # bimanual_plug_0522), which siglip would happily turn into 1024
        # tokens per image -- the downstream attention shapes assume 256.
        target_h, target_w = self.IMAGE_RESOLUTION
        resized = []
        for img in images:
            if img.shape[-2:] != (target_h, target_w):
                # _resize_with_pad_torch operates on BHWC.
                img = img.permute(0, 2, 3, 1)
                img = _resize_with_pad_torch(img, target_h, target_w)
                img = img.permute(0, 3, 1, 2)
            resized.append(img)
        images = resized

        lang_tokens, lang_masks = prompt

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return (prefix_pad_masks, past_key_values)

    @torch.no_grad()
    def velocity_eval(self, state, x_t, time_scalar, precomputed_hidden_state):
        """Evaluate v(x_t, t) for a single denoise step. Same role as
        RhoAlphaFlowMatchingModel.velocity_eval.

        Args:
            state: Robot state tensor (B, n_obs_steps, state_dim).
            x_t:   Current latent (B, chunk_size, max_action_dim).
            time_scalar: Float in [0, 1].
            precomputed_hidden_state: ``(prefix_pad_masks, past_key_values)``
                from ``get_image_text_hidden_state``.

        Returns:
            v_t: Velocity (B, chunk_size, max_action_dim) in x_t's dtype.
        """
        prefix_pad_masks, past_key_values = precomputed_hidden_state
        bsize = state.shape[0]
        device = state.device

        # pi0 is mixed-precision: _PaliGemmaWithExpertModel runs in bf16 (via
        # _to_precision), but the outer projection heads (action_in_proj,
        # time_mlp_in/out, etc.) stay in their checkpoint dtype, typically
        # fp32. The inverter casts x_t to flow_model.dtype (=config.dtype,
        # bf16); we recast to action_in_proj's actual weight dtype so the
        # mat1/mat2 dtypes line up. Pi0's own sample_actions never hits this
        # because it threads the fp32 noise tensor through unchanged.
        x_t_in = x_t.to(dtype=self.action_in_proj.weight.dtype)
        # Match pi0's sample_actions: timestep is fp32; embed_suffix casts
        # internally where needed.
        timestep = torch.full((bsize,), float(time_scalar), dtype=torch.float32, device=device)
        v_t = self._denoise_step(state, prefix_pad_masks, past_key_values, x_t_in, timestep)
        return v_t.to(x_t.dtype)

    @torch.no_grad()
    def sample_actions_from_precomputed(self, state, precomputed_hidden_state, noise, num_steps=None):
        """Full denoise loop given the precomputed prefix conditioning.

        Lifts the denoise loop from ``sample_actions`` (lines ~625-633) so the
        inverter can do its round-trip MSE check without re-running the VLM.

        Args:
            state: Robot state tensor.
            precomputed_hidden_state: ``(prefix_pad_masks, past_key_values)``.
            noise: Initial noise tensor (B, chunk_size, max_action_dim).
            num_steps: Override denoise steps. None = ``config.num_inference_steps``.

        Returns:
            Denoised actions tensor.
        """
        prefix_pad_masks, past_key_values = precomputed_hidden_state
        n_steps = num_steps if num_steps is not None else self.config.num_inference_steps
        bsize = state.shape[0]
        device = state.device

        # See velocity_eval: action_in_proj.weight is the dtype that matters
        # for x_t entering the suffix path; outer projection heads aren't
        # cast by _to_precision.
        target_dtype = self.action_in_proj.weight.dtype
        dt = torch.tensor(-1.0 / n_steps, dtype=torch.float32, device=device)
        x_t = noise.to(dtype=target_dtype)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train: bool):
        observation = _preprocess_observation_pytorch(
            observation,
            train=train,
            image_keys=self.IMAGE_KEYS,
            image_resolution=self.IMAGE_RESOLUTION,
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        time_beta = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed(img_in):
                return self.paligemma_with_expert.embed_image(img_in)

            img_emb = self._apply_checkpoint(image_embed, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed(tokens_in):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens_in)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj(state_in):
                return self.state_proj(state_in)

            state_emb = self._apply_checkpoint(state_proj, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            self.config.min_period,
            self.config.max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)

        def action_proj(x):
            return self.action_in_proj(x)

        action_emb = self._apply_checkpoint(action_proj, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp(x):
                x = self.action_time_mlp_in(x)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp, action_time_emb)
            adarms_cond = None
        else:

            def time_mlp(x):
                x = self.time_mlp_in(x)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions: torch.Tensor, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs_in, suffix_embs_in, att_mask_in, pos_ids_in, adarms_cond_in):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_mask_in,
                position_ids=pos_ids_in,
                past_key_values=None,
                inputs_embeds=[prefix_embs_in, suffix_embs_in],
                use_cache=False,
                adarms_cond=[None, adarms_cond_in],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)

        def out_proj(x):
            return self.action_out_proj(x)

        v_t = self._apply_checkpoint(out_proj, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device: torch.device, observation, noise=None, num_steps: int = 10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def _denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

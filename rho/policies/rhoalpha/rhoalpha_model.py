"""
Base RhoAlpha Model with pluggable VLM backbone support.

This module provides the foundation for all VLM-based robotics models, including:
- VLM backbone initialization via pluggable BackboneAdapter (Phi4MM, Phi5, etc.)
- Image/text processing utilities
- Shared helper modules (TransformerBlock, RMSNorm, etc.)
- State projection (common across all robot tasks)
- Abstract forward() and sample_actions() for specialized heads to implement

The VLM backend is selected via config.vlm_backend (\"phi4mm\" or \"phi5\").
Backend-specific logic lives in rho/policies/rhoalpha/phi4mm/ and phi5/.
"""

import logging
import math

import torch
import torch.utils.checkpoint as checkpoint
from peft import LoraConfig, TaskType
from torch import nn
from transformers import GenerationConfig

from rho.policies.rhoalpha.backbone import create_backbone_adapter
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig

logger = logging.getLogger(__name__)


def get_sinusoidal_encoding(seq_len, embed_dim, device="cpu"):
    """Build sinusoidal positional encoding in fp32 for precision."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(seq_len, embed_dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # keep as fp32; cast at use site


def weight_init(m):
    """Orthogonal weight initialization (BAKU-style)."""
    if isinstance(m, nn.Linear):
        # Cast to float32 for orthogonal init
        if m.weight.dtype != torch.float32:
            orig_dtype = m.weight.dtype
            m.weight.data = m.weight.data.float()
            nn.init.orthogonal_(m.weight.data)
            m.weight.data = m.weight.data.to(orig_dtype)
        else:
            nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data") and m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        if m.weight.dtype != torch.float32:
            orig_dtype = m.weight.dtype
            m.weight.data = m.weight.data.float()
            nn.init.orthogonal_(m.weight.data, gain)
            m.weight.data = m.weight.data.to(orig_dtype)
        else:
            nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data") and m.bias is not None:
            m.bias.data.fill_(0.0)


# ============================================================================
# Helper Modules
# ============================================================================


class GELUTanh(nn.Module):
    def forward(self, x):
        # return torch.tanh(torch.nn.functional.gelu(x))
        return torch.nn.functional.gelu(x, approximate="tanh")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute normalization in fp32 for stability, cast back
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = norm * self.weight.float()
        return out.to(dtype=x.dtype)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with DiT-style AdaLN-Zero.

    Outputs 3 modulation signals (scale, shift, gate).
    Uses zero initialization so the block starts as identity.

    Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    """

    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        # Single projection for all 3 signals: scale, shift, gate
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        # Zero-initialize so block starts as identity (AdaLN-Zero)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: Input tensor to normalize
            cond: Conditioning tensor (e.g., timestep embedding)

        Returns:
            Tuple of (normalized_modulated_x, gate)
            - normalized_modulated_x: LayerNorm(x) * (1 + scale) + shift
            - gate: Gate signal for gated residual connection
        """
        x_norm = self.layer_norm(x)
        # Get all 3 modulation signals
        modulation = self.modulation(cond)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        # Apply scale and shift to normalized input
        # Use unsqueeze(0) to broadcast over sequence dimension (seq, batch, embed)
        x_mod = x_norm * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x_mod, gate


class TransformerBlock(nn.Module):
    """
    Transformer block with DiT-style gated residuals when using adaptive norm.

    When norm="adaptive", uses AdaLN-Zero with gated residual connections:
        x = x + gate * sublayer(x)

    This allows the network to dynamically modulate update magnitude based on timestep,
    enabling straighter flow paths and fewer inference steps.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, norm, dropout_p):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.norm = norm
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=self.dropout_p, bias=False
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            GELUTanh(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(p=self.dropout_p),
        )
        if self.norm == "adaptive":
            self.pre_norm = AdaptiveLayerNorm(embed_dim)
            self.post_norm = AdaptiveLayerNorm(embed_dim)
        elif self.norm == "rms":
            self.pre_norm = GemmaRMSNorm(embed_dim)
            self.post_norm = GemmaRMSNorm(embed_dim)
        else:
            raise ValueError(f"Unknown norm type: {self.norm}")

    def forward(self, x, time_emb, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            time_emb: Timestep embedding for AdaLN conditioning
            key_padding_mask: Optional mask of shape (batch_size, seq_len)
                            where True indicates positions to IGNORE (padding tokens)
        """
        if self.norm == "adaptive":
            # AdaLN-Zero with gated residuals
            x_norm, gate_attn = self.pre_norm(x, time_emb)
            attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            # Gated residual: x = x + gate * attn_output
            x = x + gate_attn.unsqueeze(0) * self.dropout(attn_output)

            x_norm, gate_ff = self.post_norm(x, time_emb)
            ff_out = self.feedforward(x_norm)
            # Gated residual: x = x + gate * ff_out
            x = x + gate_ff.unsqueeze(0) * self.dropout(ff_out)
        else:
            # Standard residuals for non-adaptive norm
            x_norm = self.pre_norm(x)
            attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            x = x + self.dropout(attn_output)

            x_norm = self.post_norm(x)
            ff_out = self.feedforward(x_norm)
            x = x + self.dropout(ff_out)

        return x


class CrossAttentionTransformerBlock(nn.Module):
    """
    Cross-attention transformer block with DiT-style gated residuals:
    1. Query self-attention (with gated residual for adaptive norm)
    2. Cross-attention (Q=query, K=V=vlm_context) with gated residual
    3. Query feedforward with gated residual

    Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    """

    def __init__(self, embed_dim, num_heads, ff_dim, norm="rms", dropout_p=0.2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.norm = norm
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Self-attention for query tokens only
        self.query_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=self.dropout_p, bias=False
        )

        # Cross-attention for Q attending to KV context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=self.dropout_p, bias=False
        )

        # Feedforward for query tokens only
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            GELUTanh(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(p=self.dropout_p),
        )

        # Layer norms - simplified: only for query stream and KV input to cross-attn
        if norm == "adaptive":
            self.q_norm_1 = AdaptiveLayerNorm(embed_dim)  # for query self-attention
            self.q_norm_2 = AdaptiveLayerNorm(embed_dim)  # for query in cross-attention
            self.kv_norm = AdaptiveLayerNorm(embed_dim)  # for KV in cross-attention
            self.q_norm_3 = AdaptiveLayerNorm(embed_dim)  # for feedforward
        elif norm == "rms":
            self.q_norm_1 = GemmaRMSNorm(embed_dim)
            self.q_norm_2 = GemmaRMSNorm(embed_dim)
            self.kv_norm = GemmaRMSNorm(embed_dim)
            self.q_norm_3 = GemmaRMSNorm(embed_dim)
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

    def forward(self, query_tokens, key_value_tokens, time_emb, query_mask=None, kv_mask=None):
        """
        Args:
            query_tokens: (seq_len_q, batch_size, embed_dim) - tokens that will attend
            key_value_tokens: (seq_len_kv, batch_size, embed_dim) - context from VLM
            time_emb: Timestep embedding for AdaLN conditioning
            query_mask: mask for query tokens
            kv_mask: mask for key-value tokens
        """
        if self.norm == "adaptive":
            # Query self-attention with gated residual
            q_norm, gate_self = self.q_norm_1(query_tokens, time_emb)
            self_attn_output, _ = self.query_self_attention(
                q_norm, q_norm, q_norm, key_padding_mask=query_mask
            )
            query_tokens = query_tokens + gate_self.unsqueeze(0) * self.dropout(self_attn_output)

            # Cross-attention with gated residual
            q_norm, gate_cross = self.q_norm_2(query_tokens, time_emb)
            kv_norm, _ = self.kv_norm(key_value_tokens, time_emb)  # gate unused for KV
            cross_attn_output, _ = self.cross_attention(q_norm, kv_norm, kv_norm, key_padding_mask=kv_mask)
            query_tokens = query_tokens + gate_cross.unsqueeze(0) * self.dropout(cross_attn_output)

            # Feedforward with gated residual
            q_norm, gate_ff = self.q_norm_3(query_tokens, time_emb)
            ff_out = self.feedforward(q_norm)
            query_tokens = query_tokens + gate_ff.unsqueeze(0) * self.dropout(ff_out)
        else:
            # Standard residuals for non-adaptive norm
            q_norm = self.q_norm_1(query_tokens)
            self_attn_output, _ = self.query_self_attention(
                q_norm, q_norm, q_norm, key_padding_mask=query_mask
            )
            query_tokens = query_tokens + self.dropout(self_attn_output)

            q_norm = self.q_norm_2(query_tokens)
            kv_norm = self.kv_norm(key_value_tokens)
            cross_attn_output, _ = self.cross_attention(q_norm, kv_norm, kv_norm, key_padding_mask=kv_mask)
            query_tokens = query_tokens + self.dropout(cross_attn_output)

            q_norm = self.q_norm_3(query_tokens)
            ff_out = self.feedforward(q_norm)
            query_tokens = query_tokens + self.dropout(ff_out)

        return query_tokens


class SimpleActionExpert(nn.Module):
    """
    Transformer-based action expert that processes embeddings.

    Assumes embeddings have already been projected into a shared embed_dim space.
    Takes image/lang embeddings from phi4mm backbone and robot state,
    passes through transformer blocks, and provides output embeddings.
    """

    def __init__(self, config: RhoAlphaConfig):
        super().__init__()
        self.config = config

        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.ff_dim = self.config.ff_dim
        self.action_dim = self.config.max_action_dim
        self.num_action_steps = self.config.n_action_steps
        self.num_blocks = self.config.num_blocks
        self.max_seq_len = self.config.max_seq_len
        self.enable_gradient_checkpointing = self.config.enable_gradient_checkpointing
        self.norm = self.config.norm if self.config.norm is not None else "rms"
        self.dropout_p = self.config.dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_emb_method = self.config.pos_emb_method

        if self.pos_emb_method == "sinusoidal":
            pos_emb = get_sinusoidal_encoding(self.max_seq_len, self.embed_dim, device=self.config.device)
            self.register_buffer("positional_encoding", pos_emb)
        elif self.pos_emb_method == "learned":
            self.positional_encoding = nn.Parameter(torch.randn(self.max_seq_len, self.embed_dim) * 0.02)
        else:
            raise NotImplementedError(f"Unknown pos_emb_method: {self.pos_emb_method}")

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.norm, self.dropout_p)
                for _ in range(self.num_blocks)
            ]
        )

        if self.norm == "adaptive":
            self.final_norm = AdaptiveLayerNorm(self.embed_dim)
        elif self.norm == "rms":
            self.final_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown norm type: {self.norm}")

    def forward(self, image_text_embed, state_emb, time_emb, image_text_attn_mask=None):
        """
        image_text_embed: (batch_size, num_image_tokens, embed_dim)
        state_emb: (batch_size, num_state_tokens, embed_dim)
        time_emb: (batch_size, num_time_tokens, embed_dim)
        image_text_attn_mask: (batch_size, num_image_tokens) or None - mask for image/text tokens
                              where 1 = valid token, 0 = padding/masked token

        Note: time_emb is being passed in for AdaptiveLayerNorm only
        """
        # Stack along num_token dimension: (batch, total_tokens, embed_dim)
        x = torch.cat((image_text_embed, state_emb), dim=1)
        x = x.permute(1, 0, 2)  # (total_tokens, batch_size, embed_dim) for attention

        # Construct key_padding_mask if image_text_attn_mask is provided
        # key_padding_mask expects True for positions to IGNORE
        key_padding_mask = None
        if image_text_attn_mask is not None:
            batch_size = image_text_embed.size(0)
            num_state_tokens = state_emb.size(1)

            # Create full mask: image_text tokens use provided mask, state tokens are always unmasked
            state_mask = torch.ones(
                batch_size,
                num_state_tokens,
                dtype=image_text_attn_mask.dtype,
                device=image_text_attn_mask.device,
            )
            full_mask = torch.cat([image_text_attn_mask, state_mask], dim=1)  # (batch_size, total_tokens)

            # Convert to key_padding_mask: True where we should IGNORE (i.e., where mask is 0)
            key_padding_mask = full_mask == 0  # (batch_size, total_tokens)

        # Add positional encoding
        assert x.size(0) < self.max_seq_len, (
            f"Input sequence length ({x.size(0)}) exceeds max_seq_len ({self.max_seq_len})."
        )
        # Cast PE to match x.dtype at use-site (PE stored in fp32 for accuracy)
        pe = self.positional_encoding[: x.size(0), :].unsqueeze(1).to(dtype=x.dtype)
        x = self.dropout(x + pe)  # TODO optional dropout here

        for block in self.transformer_blocks:
            x = (
                checkpoint.checkpoint(block, x, time_emb, key_padding_mask)
                if self.enable_gradient_checkpointing
                else block(x, time_emb, key_padding_mask)
            )

        if self.norm == "adaptive":
            x, _ = self.final_norm(x, time_emb)  # Discard gate at final output
        else:
            x = self.final_norm(x)
        x = x.permute(1, 0, 2)  # (batch_size, num_tokens, embed_dim)
        return x


class CrossAttentionActionExpert(nn.Module):
    def __init__(self, config: RhoAlphaConfig):
        super().__init__()
        self.config = config

        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.ff_dim = self.config.ff_dim
        self.action_dim = self.config.max_action_dim
        self.num_action_steps = self.config.n_action_steps
        self.num_blocks = self.config.num_blocks
        self.max_seq_len = self.config.max_seq_len
        self.enable_gradient_checkpointing = self.config.enable_gradient_checkpointing
        self.norm = self.config.norm if self.config.norm is not None else "rms"
        self.dropout_p = self.config.dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.pos_emb_method = self.config.pos_emb_method

        if self.pos_emb_method == "sinusoidal":
            pos_emb = get_sinusoidal_encoding(self.max_seq_len, self.embed_dim, device=self.config.device)
            self.register_buffer("positional_encoding", pos_emb)
        elif self.pos_emb_method == "learned":
            self.positional_encoding = nn.Parameter(torch.randn(self.max_seq_len, self.embed_dim) * 0.02)
        else:
            raise NotImplementedError(f"Unknown pos_emb_method: {self.pos_emb_method}")

        self.transformer_blocks = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(
                    self.embed_dim, self.num_heads, self.ff_dim, norm=self.norm, dropout_p=self.dropout_p
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.norm == "adaptive":
            self.final_norm = AdaptiveLayerNorm(self.embed_dim)
        elif self.norm == "rms":
            self.final_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown norm type: {self.norm}")

    def forward(self, image_text_embed, state_emb, time_emb, image_text_attn_mask=None):
        """
        image_text_embed: (batch_size, num_image_tokens, embed_dim)
        state_emb: (batch_size, num_state_tokens, embed_dim)
        time_emb: (batch_size, 1, embed_dim)
        image_text_attn_mask: (batch_size, num_image_tokens) or None - mask for image/text tokens
                              where 1 = valid token, 0 = padding/masked token
        """
        # assuming image_emb, text_emb, state_emb are already in the shared embed_dim

        ### WORKING VERSION ###
        image_text_tokens = image_text_embed.permute(
            1, 0, 2
        )  # (total_tokens, batch_size, embed_dim) for attention
        state_tokens = state_emb.permute(1, 0, 2)

        image_text_pe = (
            self.positional_encoding[: image_text_tokens.size(0), :]
            .unsqueeze(1)
            .to(dtype=image_text_tokens.dtype)
        )
        state_pe = (
            self.positional_encoding[: state_tokens.size(0), :].unsqueeze(1).to(dtype=state_tokens.dtype)
        )

        image_text_tokens = self.dropout(image_text_tokens + image_text_pe)
        state_tokens = self.dropout(state_tokens + state_pe)

        img_text_key_mask = None
        if image_text_attn_mask is not None:
            img_text_key_mask = image_text_attn_mask == 0  # (batch_size, num_image_text_tokens)

        for block in self.transformer_blocks:  # (num_tokens, batch_size, embed_dim)
            state_tokens = (
                checkpoint.checkpoint(
                    block, state_tokens, image_text_tokens, time_emb, None, img_text_key_mask
                )
                if self.enable_gradient_checkpointing
                else block(state_tokens, image_text_tokens, time_emb, None, img_text_key_mask)
            )

        if self.norm == "adaptive":
            state_tokens, _ = self.final_norm(state_tokens, time_emb)  # Discard gate at final output
        else:
            state_tokens = self.final_norm(state_tokens)
        state_tokens = state_tokens.permute(1, 0, 2)  # (batch_size, num_tokens, embed_dim)

        return state_tokens


class LayerwiseCrossAttentionExpert(nn.Module):
    """
    Action expert that cross-attends to per-layer VLM hidden states.

    Each action expert block cross-attends to a different VLM layer's
    hidden states, allowing the action model to read from progressively
    more abstract VLM representations. The VLM is frozen and unaware of
    the action expert (unidirectional information flow).

    Layer mapping: if the VLM has V layers and the expert has E blocks,
    we select E layers from the VLM using either:
      - "first": layers [0, 1, ..., E-1]
      - "last":  layers [V-E, V-E+1, ..., V-1]
      - "stride": evenly spaced layers across the full depth
    """

    def __init__(self, config: RhoAlphaConfig):
        super().__init__()
        self.config = config

        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.ff_dim = self.config.ff_dim
        self.num_blocks = self.config.num_blocks
        self.max_seq_len = self.config.max_seq_len
        self.enable_gradient_checkpointing = self.config.enable_gradient_checkpointing
        self.norm = self.config.norm if self.config.norm is not None else "rms"
        self.dropout_p = self.config.dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.pos_emb_method = self.config.pos_emb_method
        self.cross_attention_dim = self.config.cross_attention_dim
        self.vlm_layer_select = getattr(self.config, "vlm_layer_select", "last")

        if self.pos_emb_method == "sinusoidal":
            pos_emb = get_sinusoidal_encoding(
                self.max_seq_len,
                self.embed_dim,
                device=self.config.device,
            )
            self.register_buffer("positional_encoding", pos_emb)
        elif self.pos_emb_method == "learned":
            self.positional_encoding = nn.Parameter(torch.randn(self.max_seq_len, self.embed_dim) * 0.02)
        else:
            raise NotImplementedError(f"Unknown pos_emb_method: {self.pos_emb_method}")

        self.transformer_blocks = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(
                    self.embed_dim,
                    self.num_heads,
                    self.ff_dim,
                    norm=self.norm,
                    dropout_p=self.dropout_p,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Projection(s) from VLM hidden dim to embed_dim
        self.shared_vlm_projection = getattr(self.config, "shared_vlm_projection", False)
        if self.shared_vlm_projection:
            self.vlm_proj_shared = nn.Linear(self.cross_attention_dim, self.embed_dim)
        else:
            self.vlm_projections = nn.ModuleList(
                [nn.Linear(self.cross_attention_dim, self.embed_dim) for _ in range(self.num_blocks)]
            )

        if self.norm == "adaptive":
            self.final_norm = AdaptiveLayerNorm(self.embed_dim)
        elif self.norm == "rms":
            self.final_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown norm type: {self.norm}")

    def _select_vlm_layers(self, all_hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        """Select which VLM layers to use based on config."""
        num_vlm_layers = len(all_hidden_states)
        num_expert_blocks = self.num_blocks

        if self.vlm_layer_select == "first":
            indices = list(range(num_expert_blocks))
        elif self.vlm_layer_select == "last":
            start = num_vlm_layers - num_expert_blocks
            indices = list(range(start, num_vlm_layers))
        elif self.vlm_layer_select == "stride":
            indices = [
                round(i * (num_vlm_layers - 1) / (num_expert_blocks - 1))
                if num_expert_blocks > 1
                else num_vlm_layers - 1
                for i in range(num_expert_blocks)
            ]
        else:
            raise ValueError(f"Unknown vlm_layer_select: {self.vlm_layer_select}")

        return [all_hidden_states[i] for i in indices]

    def forward(
        self,
        vlm_hidden_states: list[torch.Tensor],
        state_emb: torch.Tensor,
        time_emb: torch.Tensor,
        image_text_attn_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            vlm_hidden_states: list of (B, S, vlm_hidden_dim) tensors,
                one per VLM layer.
            state_emb: (B, num_state_tokens, embed_dim)
            time_emb: (B, embed_dim)
            image_text_attn_mask: (B, S) or None
                where 1 = valid, 0 = padding
        """
        selected = self._select_vlm_layers(vlm_hidden_states)

        state_tokens = state_emb.permute(1, 0, 2)

        state_pe = (
            self.positional_encoding[: state_tokens.size(0), :].unsqueeze(1).to(dtype=state_tokens.dtype)
        )
        state_tokens = self.dropout(state_tokens + state_pe)

        img_text_key_mask = None
        if image_text_attn_mask is not None:
            img_text_key_mask = image_text_attn_mask == 0

        projs = (
            [self.vlm_proj_shared] * self.num_blocks if self.shared_vlm_projection else self.vlm_projections
        )

        for block, vlm_proj, vlm_h in zip(
            self.transformer_blocks,
            projs,
            selected,
            strict=True,
        ):
            # Project this VLM layer to embed_dim
            kv_tokens = vlm_proj(vlm_h)
            kv_tokens = kv_tokens.permute(1, 0, 2)
            if getattr(self.config, "kv_pos_encoding", True):
                kv_pe = (
                    self.positional_encoding[: kv_tokens.size(0), :].unsqueeze(1).to(dtype=kv_tokens.dtype)
                )
                kv_tokens = kv_tokens + kv_pe

            if self.enable_gradient_checkpointing:
                state_tokens = checkpoint.checkpoint(
                    block,
                    state_tokens,
                    kv_tokens,
                    time_emb,
                    None,
                    img_text_key_mask,
                )
            else:
                state_tokens = block(
                    state_tokens,
                    kv_tokens,
                    time_emb,
                    None,
                    img_text_key_mask,
                )

        if self.norm == "adaptive":
            state_tokens, _ = self.final_norm(state_tokens, time_emb)
        else:
            state_tokens = self.final_norm(state_tokens)
        state_tokens = state_tokens.permute(1, 0, 2)

        return state_tokens


# ============================================================================
# Base RhoAlpha Model
# ============================================================================


class RhoAlphaModel(nn.Module):
    """
    Base VLM Model with pluggable backbone support.

    This class provides the foundation for all VLM-based robotics models:
    - Initializes and manages the VLM backbone via BackboneAdapter
    - Supports multiple backends (Phi4MM, Phi5) selected via config.vlm_backend
    - Provides image/text processing utilities
    - Projects robot state into embedding space
    - Defines abstract methods for specialized heads to implement

    Subclasses should implement:
    - forward(): Training forward pass with loss computation
    - sample_actions(): Inference forward pass
    """

    def __init__(
        self,
        config: RhoAlphaConfig,
        vlm_backbone=None,
        vlm_processor=None,
        generation_config=None,
        vlm_projector=None,
        state_projector=None,
    ):
        """
        Initialize RhoAlphaModel.

        Args:
            config: Model configuration
            vlm_backbone: Optional pre-initialized VLM backbone to share (saves ~7-14GB)
            vlm_processor: Optional pre-initialized VLM processor to share
            generation_config: Optional pre-initialized generation config to share
            vlm_projector: Optional pre-initialized VLM projector to share (saves ~5-10MB)
            state_projector: Optional pre-initialized state projector to share (saves ~28KB)
        """
        super().__init__()
        self.config = config

        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.ff_dim = self.config.ff_dim
        self.max_state_dim = self.config.max_state_dim
        self.max_action_dim = self.config.max_action_dim
        self.hidden_state_idx = self.config.hidden_state_idx
        self.num_action_steps = self.config.n_action_steps
        self.num_blocks = self.config.num_blocks
        self.max_seq_len = self.config.max_seq_len
        self.dtype = self.config.dtype
        self.device = self.config.device

        # Create backbone adapter for the configured VLM backend
        self._backend = create_backbone_adapter(config)

        # Use provided VLM components or initialize new ones
        if vlm_backbone is None:
            self.vlm_backbone = self._backend.load_backbone()
        else:
            self.vlm_backbone = vlm_backbone

        # Configure Phi4MM's built-in vision/speech LoRAs
        # These are created by Phi4MM's __init__ and loaded from checkpoint
        self._configure_builtin_loras()

        if vlm_processor is None:
            self.vlm_processor = self._backend.create_processor()
        else:
            self.vlm_processor = vlm_processor

        if generation_config is None:
            self.generation_config = GenerationConfig.from_pretrained(
                self._backend.get_model_id(), "generation_config.json"
            )
        else:
            self.generation_config = generation_config

        if vlm_projector is None:
            vlm_hidden_size = self._backend.get_hidden_size(self.vlm_backbone)
            self.vlm_projector = nn.Linear(vlm_hidden_size, self.embed_dim).to(
                device=self.device, dtype=self.dtype
            )
        else:
            self.vlm_projector = vlm_projector

        # Auto-populate cross_attention_dim from the backbone when needed
        if self.config.cross_attention_dim is None and self.config.attention_type == "layerwise_cross":
            self.config.cross_attention_dim = self._backend.get_hidden_size(self.vlm_backbone)

        if state_projector is None:
            # State projector (shared across all robot tasks)
            self.state_projector = nn.Linear(self.max_state_dim, self.embed_dim).to(
                device=self.device, dtype=self.dtype
            )
        else:
            self.state_projector = state_projector

        # LoRA if configured (must be done BEFORE _set_requires_grad because
        # add_adapter/set_adapter can reset requires_grad on base model params)
        # If action_lora_init_from_scratch=True, defer attachment until after checkpoint loading
        self._action_lora_deferred = False
        if hasattr(self.config, "use_action_lora") and self.config.use_action_lora:
            if getattr(self.config, "action_lora_init_from_scratch", True):
                # Will be attached after checkpoint loading via _post_checkpoint_load()
                self._action_lora_deferred = True
                logger.info("Action LoRA will be attached after checkpoint loading (init from scratch)")
            else:
                self._add_action_lora()

        self._set_requires_grad()

        # Freeze action LoRA if configured (must be done AFTER _set_requires_grad)
        # Skip if LoRA is deferred - will be handled in _post_checkpoint_load()
        if (
            hasattr(self.config, "use_action_lora")
            and self.config.use_action_lora
            and getattr(self.config, "freeze_action_lora", False)
            and not self._action_lora_deferred
        ):
            for name, param in self.vlm_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
            logger.info("Action LoRA frozen (contributes but not trainable)")

    def _configure_builtin_loras(self):
        """
        Configure built-in LoRAs for backends that support them (e.g. Phi4MM).

        For backends with built-in LoRA (Phi4MM):
        - Removes audio encoder and audio LoRA (saves ~600M params, not used by us)
        - Handles vision LoRA activation or merging

        For backends without built-in LoRA (Phi5, Qwen):
        - Calls remove_audio_components (no-op for backends without audio)
        - Skips LoRA configuration entirely
        """
        # Remove audio components if the backend has them (no-op for Qwen/Phi5)
        self._backend.remove_audio_components(self.vlm_backbone)

        # Only handle vision LoRA for backends that ship with built-in adapters
        if not self._backend.supports_builtin_lora():
            return

        # Handle vision LoRA
        merge_vision = getattr(self.config, "merge_vision_lora", False)
        if merge_vision:
            self._merge_vision_lora()
        else:
            # Activate vision LoRA (make sure it's the active adapter)
            if hasattr(self.vlm_backbone, "set_lora_adapter"):
                self.vlm_backbone.set_lora_adapter("vision")
            logger.info("Vision LoRA active (trainable)")

            # Warn if language model is unfrozen - LoRA is meant for parameter-efficient training
            if not getattr(self.config, "freeze_language_model", False):
                logger.warning(
                    "Using vision LoRA with unfrozen language model. "
                    "LoRA is typically used when the base model is frozen. "
                    "Consider setting freeze_language_model=True or merging vision LoRA."
                )

    def _merge_vision_lora(self):
        """Merge vision LoRA into base weights and remove adapter params."""
        logger.info("Merging vision LoRA into base weights...")
        from peft.tuners.lora.layer import LoraLayer

        merged_count = 0
        for _, module in self.vlm_backbone.named_modules():
            if isinstance(module, LoraLayer) and hasattr(module, "lora_A") and "vision" in module.lora_A:
                module.merge(adapter_names=["vision"])
                merged_count += 1

        logger.info(f"  Merged vision LoRA ({merged_count} layers)")

        # Manually delete vision LoRA params after merging to free memory
        # and avoid counting them in trainable params
        deleted_count = 0
        for _, module in self.vlm_backbone.named_modules():
            if isinstance(module, LoraLayer) and hasattr(module, "lora_A") and "vision" in module.lora_A:
                del module.lora_A["vision"]
                del module.lora_B["vision"]
                deleted_count += 1
        logger.info(f"  Deleted vision LoRA params ({deleted_count} layers)")

    def _add_action_lora(self):
        """
        Add action LoRA adapter to VLM backbone.
        """
        # Warn if vision LoRA not merged - it will be disabled
        merge_vision = getattr(self.config, "merge_vision_lora", False)
        if not merge_vision:
            logger.warning(
                "Adding action LoRA without merging vision LoRA. "
                "Vision LoRA will be disabled. Consider setting merge_vision_lora=True."
            )

        # Warn if language model is unfrozen - LoRA is meant for parameter-efficient training
        if not getattr(self.config, "freeze_language_model", False):
            logger.warning(
                "Using action LoRA with unfrozen language model. "
                "LoRA is typically used when the base model is frozen. "
                "Consider setting freeze_language_model=True or disabling action LoRA."
            )

        action_lora_config = LoraConfig(
            r=self.config.action_lora_r,
            lora_alpha=self.config.action_lora_alpha,
            target_modules=self.config.action_lora_modules,
            lora_dropout=self.config.action_lora_dropout,
            layers_to_transform=list(range(self.config.action_lora_start, self.config.action_lora_end + 1)),
            task_type=TaskType.CAUSAL_LM,
        )

        # Phi4MM always has peft_config (creates vision/speech LoRAs in __init__)
        # So we just add our action adapter to the existing PEFT model
        # Note: transformers integration uses (config, name) order
        self.vlm_backbone.add_adapter(action_lora_config, adapter_name="action")
        logger.info(
            f"Added action LoRA (r={self.config.action_lora_r}, "
            f"layers {self.config.action_lora_start}-{self.config.action_lora_end})"
        )

        # Set action as the active adapter
        self.vlm_backbone.set_adapter("action")
        logger.info("Set active adapter to 'action'")

    def _post_checkpoint_load(self):
        """
        Called after checkpoint loading to attach deferred components.

        If action_lora_init_from_scratch=True, LoRA attachment was deferred during __init__
        so that pretrained weights (without LoRA) can be loaded first. Now we attach fresh
        LoRA adapters on top of the loaded weights.
        """
        if self._action_lora_deferred:
            logger.info("Attaching action LoRA after checkpoint load (fresh initialization)")
            self._add_action_lora()
            self._action_lora_deferred = False

            # Re-run requires_grad setup since adding LoRA can change things
            self._set_requires_grad()

            # Handle freeze_action_lora if configured
            if getattr(self.config, "freeze_action_lora", False):
                for name, param in self.vlm_backbone.named_parameters():
                    if "lora_" in name and "action" in name:
                        param.requires_grad = False
                logger.info("Action LoRA frozen (contributes but not trainable)")

    def _set_requires_grad(self):
        """Set gradient requirements for model parameters based on config."""
        self.train()
        for params in self.parameters():
            params.requires_grad = True

        # Always freeze the token embeddings
        lang_model = self._backend.get_language_model(self.vlm_backbone)
        for params in lang_model.embed_tokens.parameters():
            params.requires_grad = False

        if self.config.freeze_vlm_backbone:  # freeze the entire VLM backbone
            self.vlm_backbone.eval()
            for param in self.vlm_backbone.parameters():
                param.requires_grad = False

        if self.config.freeze_language_model:
            for params in lang_model.layers.parameters():
                params.requires_grad = False
            for params in lang_model.norm.parameters():
                params.requires_grad = False
            for params in self.vlm_backbone.lm_head.parameters():
                params.requires_grad = False

            # If vision LoRA is not merged, unfreeze it so it can still be trained
            # (freeze_language_model freezes all layer params including LoRA)
            if not getattr(self.config, "merge_vision_lora", False):
                for name, param in self.vlm_backbone.named_parameters():
                    if "lora_" in name and "vision" in name:
                        param.requires_grad = True

            # If action LoRA is used and not frozen, unfreeze it
            if getattr(self.config, "use_action_lora", False) and not getattr(
                self.config, "freeze_action_lora", False
            ):
                for name, param in self.vlm_backbone.named_parameters():
                    if "lora_" in name and "action" in name:
                        param.requires_grad = True

        if self.config.freeze_vision_encoder:
            self._backend.set_vision_requires_grad(self.vlm_backbone)

        if self.config.train_expert_only:
            self.vlm_backbone.eval()
            for param in self.vlm_backbone.parameters():
                param.requires_grad = False

    def print_freezing_status(self):
        """Print which components are frozen vs trainable."""
        base_model = (
            self.vlm_backbone.get_base_model()
            if hasattr(self.vlm_backbone, "get_base_model")
            else self.vlm_backbone
        )
        lang_model = self._backend.get_language_model(base_model)

        # Start with common components
        components = {
            "VLM Backbone": self.vlm_backbone,
            "  Language Encoder": lang_model.embed_tokens,
            "  Transformer Layers": lang_model.layers,
        }

        # Add backend-specific vision components
        components.update(self._backend.get_freezing_components(self.vlm_backbone))

        # Add remaining common components
        components["State Projector"] = self.state_projector
        components["Model"] = self

        logger.info("\nFreezing Status:")
        logger.info("-" * 50)

        # Print component counts, excluding LoRA params to avoid double-counting
        for name, module in components.items():
            try:
                total = sum(p.numel() for n, p in module.named_parameters() if "lora_" not in n)
                trainable = sum(
                    p.numel() for n, p in module.named_parameters() if "lora_" not in n and p.requires_grad
                )
                if total == 0:
                    continue
                status = "🔒 FROZEN" if trainable == 0 else f"🔓 {trainable / total * 100:.0f}% trainable"
                logger.info(f"{name:20}\t: {status:15} ({trainable:>8,}/{total:>8,})")
            except Exception as e:
                logger.info(f"{name:20}: ❌ NOT FOUND. {e}")

        # Print LoRA parameters (after Model)
        logger.info("-" * 50)
        lora_params = sum(p.numel() for n, p in self.vlm_backbone.named_parameters() if "lora_" in n)
        lora_trainable = sum(
            p.numel() for n, p in self.vlm_backbone.named_parameters() if "lora_" in n and p.requires_grad
        )
        if lora_params > 0:
            # Get active adapter info from Phi4MM's internal PEFT model
            active_adapter = None
            if hasattr(self.vlm_backbone, "model") and hasattr(self.vlm_backbone.model, "active_adapter"):
                adapter_list = self.vlm_backbone.model.active_adapter
                if isinstance(adapter_list, list) and adapter_list:
                    active_adapter = ", ".join(adapter_list)
            # Fallback: get from first LoraLayer's active adapter
            if active_adapter is None:
                from peft.tuners.lora.layer import LoraLayer

                for module in self.vlm_backbone.modules():
                    if isinstance(module, LoraLayer) and hasattr(module, "active_adapter"):
                        active_adapter = module.active_adapter
                        break

            active_str = f" (active: {active_adapter})" if active_adapter else ""
            pct = lora_trainable / lora_params * 100 if lora_params > 0 else 0
            status = "🔒 FROZEN" if lora_trainable == 0 else f"🔓 {pct:.0f}% trainable"
            logger.info(
                f"{'LoRA Adapters':20}\t: {status:15} ({lora_trainable:>8,}/{lora_params:>8,}){active_str}"
            )

        # Print total
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = total_trainable / total_params * 100 if total_params > 0 else 0
        status = f"🔓 {pct:.0f}% trainable"
        logger.info("-" * 50)
        logger.info(f"{'Total':20}\t: {status:15} ({total_trainable:>8,}/{total_params:>8,})")

    def convert_image_to_batch_dict(self, image) -> dict:
        """
        Convert various image input formats to a standardized batch dict.

        Handles:
        - dict with observation.images key (passthrough)
        - List[List[torch.Tensor]] format
        - List[List[PIL.Image]] format
        - torch.Tensor in (batch, num_cameras, C, H, W) format

        Args:
            image: Image input in one of the supported formats

        Returns:
            dict with OBSERVATION_IMAGE key containing tensor of shape (batch, num_cameras, C, H, W)

        Raises:
            ValueError: If image format is not supported
        """
        from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES

        if isinstance(image, dict):
            return image
        elif isinstance(image, torch.Tensor):
            return {OBS_IMAGES: image}
        elif isinstance(image, list) and len(image) > 0:
            first_elem = image[0]

            # Handle List[List[...]] format
            if isinstance(first_elem, list) and len(first_elem) > 0:
                batch_size = len(image)
                num_cameras = len(first_elem)
                first_img = first_elem[0]

                if isinstance(first_img, torch.Tensor):
                    # List[List[torch.Tensor]] format
                    tensor_images = []
                    for batch_idx in range(batch_size):
                        batch_cameras = []
                        for cam_idx in range(num_cameras):
                            img_tensor = image[batch_idx][cam_idx].to(self.device)
                            batch_cameras.append(img_tensor)
                        tensor_images.append(torch.stack(batch_cameras))  # (num_cameras, C, H, W)
                    consolidated_imgs = torch.stack(tensor_images)  # (batch, num_cameras, C, H, W)
                    return {OBS_IMAGES: consolidated_imgs}

                elif hasattr(first_img, "convert"):
                    # List[List[PIL.Image]] format
                    import torchvision.transforms.functional as TF

                    tensor_images = []
                    for batch_idx in range(batch_size):
                        batch_cameras = []
                        for cam_idx in range(num_cameras):
                            pil_img = image[batch_idx][cam_idx]
                            img_tensor = TF.to_tensor(pil_img).to(self.device)
                            batch_cameras.append(img_tensor)
                        tensor_images.append(torch.stack(batch_cameras))
                    consolidated_imgs = torch.stack(tensor_images)
                    return {OBS_IMAGES: consolidated_imgs}
                else:
                    raise ValueError(f"Unsupported image element type: {type(first_img)}")
            else:
                raise ValueError(f"Expected List[List[...]] format, got List[{type(first_elem)}]")
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")

    def get_image_text_hidden_state(self, image, text, image_mask=None):
        """
        Extract image and text hidden states from VLM backbone.

        Delegates to the backend adapter which handles backbone-specific processing
        (e.g., Phi4MM HD/non-HD branching, Phi5 direct path).

        Args:
            image: List[List[PIL.Image.Image]] OR dict with observation.images - batch_size x num_images
            text: List[str] - batch_size text prompts
            image_mask: Optional tensor for image masking

        Returns:
            Tuple of (hidden_states, attention_masks)
            - hidden_states: (batch_size, num_tokens, embed_dim)
            - attention_masks: (batch_size, num_tokens)
        """
        return self._backend.get_image_text_hidden_state(
            backbone=self.vlm_backbone,
            processor=self.vlm_processor,
            vlm_projector=self.vlm_projector,
            hidden_state_idx=self.hidden_state_idx,
            image=image,
            text=text,
            image_mask=image_mask,
            convert_image_fn=self.convert_image_to_batch_dict,
            pad_sequence_fn=self.pad_sequence,
            cat_with_pad_fn=self.cat_with_pad,
        )

    def get_image_text_embed(self, image, text, image_mask=None):
        """
        Extract raw (unprojected) image and text hidden states from VLM backbone.

        Same as get_image_text_hidden_state but returns features at the backbone's
        native hidden dimension (without vlm_projector). Useful for cross-attention
        where the action expert handles the dimension mismatch via kdim/vdim.

        Args:
            image: List[List[PIL.Image.Image]] OR dict with observation.images
            text: List[str] - batch_size text prompts
            image_mask: Optional tensor for image masking

        Returns:
            Tuple of (hidden_states, attention_masks)
            - hidden_states: (batch_size, num_tokens, backbone_hidden_dim)
            - attention_masks: (batch_size, num_tokens)
        """
        return self._backend.get_image_text_embed(
            backbone=self.vlm_backbone,
            processor=self.vlm_processor,
            hidden_state_idx=self.hidden_state_idx,
            image=image,
            text=text,
            image_mask=image_mask,
            convert_image_fn=self.convert_image_to_batch_dict,
            pad_sequence_fn=self.pad_sequence,
            cat_with_pad_fn=self.cat_with_pad,
        )

    def get_all_hidden_states(self, image, text, image_mask=None):
        """
        Extract ALL layer hidden states from the VLM backbone (unprojected).

        Returns:
            Tuple of (all_hidden_states, attention_masks)
            - all_hidden_states: list of (B, S, vlm_hidden_dim) tensors
            - attention_masks: (B, S)
        """
        return self._backend.get_all_hidden_states(
            backbone=self.vlm_backbone,
            processor=self.vlm_processor,
            image=image,
            text=text,
            image_mask=image_mask,
            convert_image_fn=self.convert_image_to_batch_dict,
            pad_sequence_fn=self.pad_sequence,
            cat_with_pad_fn=self.cat_with_pad,
        )

    def process_batch(self, processor, images, texts, image_mask=None, max_length=8192):
        """
        Process a batch of images and texts into model inputs.

        Delegates to the backend adapter which handles backbone-specific
        processor output field names.

        Args:
            processor: The VLM processor.
            images: List[List[PIL.Image.Image]] - batch_size x num_images.
            texts: List[str] - batch_size text prompts.
            image_mask: Optional tensor for image masking.
            max_length: Maximum sequence length.

        Returns:
            dict of tensors ready for backbone forward pass.
        """
        return self._backend.process_batch(
            processor=processor,
            images=images,
            texts=texts,
            image_mask=image_mask,
            pad_sequence_fn=self.pad_sequence,
            cat_with_pad_fn=self.cat_with_pad,
            max_length=max_length,
        )

    def pad_sequence(self, sequences, padding_side="right", padding_value=0):
        """Pad sequences to the same length."""
        max_len = max(seq.size(0) for seq in sequences)
        batch_size = len(sequences)
        output = sequences[0].new_full((batch_size, max_len), padding_value)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            if padding_side == "right":
                output.data[i, :length] = seq
            else:
                output.data[i, -length:] = seq
        return output

    def cat_with_pad(self, tensors, dim, padding_value=0):
        """Concatenate tensors with padding."""
        ndim = tensors[0].dim()
        out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
        out_size[dim] = sum(t.shape[dim] for t in tensors)
        output = tensors[0].new_full(out_size, padding_value)
        index = 0
        for t in tensors:
            slices = [slice(0, t.shape[d]) for d in range(ndim)]
            slices[dim] = slice(index, index + t.shape[dim])
            output[tuple(slices)] = t
            index += t.shape[dim]
        return output

    def forward(self, *args, **kwargs):
        """Abstract method - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def sample_actions(self, *args, **kwargs):
        """Abstract method - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement sample_actions()")

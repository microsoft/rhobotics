"""
VLM model components for Rho flow-matching policies.

Contains the full shared implementation stack:
- Helper modules (RMSNorm, AdaptiveLayerNorm, TransformerBlock, etc.)
- Action experts (SimpleActionExpert, CrossAttentionActionExpert, LayerwiseCrossAttentionExpert)
- BaseVLMModel: pluggable-backbone base with state projection and image processing
- FlowMatchingModel: flow-matching training/sampling on top of BaseVLMModel
- RhoModel: thin checkpoint-namespace wrapper (model.flow_model.* keys)

The implementation retains compatibility with legacy Rho checkpoint
namespaces while exposing the public Rho model classes.
"""

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint as checkpoint
from lerobot.utils.device_utils import get_safe_dtype
from peft import LoraConfig, TaskType
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from transformers import GenerationConfig

from rho.policies.base import PolicyConfig
from rho.policies.rho.backbone import create_backbone_adapter
from rho.training.hidden_state_stats import hidden_state_stats

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

    Two modes:
    - shared=False: owns a Linear(dim, 3*dim) modulation map; forward cond is
      the timestep embedding.
    - shared=True (adaLN-single): adds either a learned constant offset or a
      low-rank, timestep-conditioned residual to a modulation map shared
      across blocks.

    Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    and adaLN-single from PixArt-alpha (Chen et al., 2023).
    """

    def __init__(self, dim, shared=False, lora_rank=None):
        super().__init__()
        self.shared = shared
        self.lora_rank = lora_rank
        self.layer_norm = nn.LayerNorm(dim)
        if shared:
            if lora_rank is None:
                self.offset = nn.Parameter(torch.zeros(3 * dim))
            else:
                self.lora = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim, lora_rank),
                    nn.Linear(lora_rank, 3 * dim),
                )
        else:
            # Single projection for all 3 signals: scale, shift, gate
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
            # Zero-initialize so block starts as identity (AdaLN-Zero)
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: Input tensor to normalize
            cond: Timestep embedding (shared=False), precomputed base
                modulation (shared offset mode), or ``(base, timestep_emb)``
                for shared low-rank adaptation.

        Returns:
            Tuple of (normalized_modulated_x, gate)
            - normalized_modulated_x: LayerNorm(x) * (1 + scale) + shift
            - gate: Gate signal for gated residual connection
        """
        x_norm = self.layer_norm(x)
        # Get all 3 modulation signals
        if self.shared and self.lora_rank is not None:
            base, raw_cond = cond
            modulation = base + self.lora(raw_cond)
        elif self.shared:
            modulation = cond + self.offset
        else:
            modulation = self.modulation(cond)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        # Apply scale and shift to normalized input
        # Use unsqueeze(0) to broadcast over sequence dimension (seq, batch, embed)
        x_mod = x_norm * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x_mod, gate


def build_shared_adaln_maps(embed_dim, roles):
    """One zero-init modulation map per AdaLN site role, shared across blocks (adaLN-single)."""
    mods = nn.ModuleDict(
        {role: nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 3 * embed_dim)) for role in roles}
    )
    for m in mods.values():
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return mods


def shared_adaln_cond(time_emb, role, mode):
    """Select a shared modulation, optionally paired with raw conditioning."""
    base = time_emb[role]
    return (base, time_emb["_raw"]) if mode == "shared_lora" else base


class GroupedQueryAttention(nn.Module):
    """Sequence-first grouped-query attention with MultiheadAttention-compatible output."""

    def __init__(self, embed_dim, num_heads, gqa_groups, dropout=0.0, bias=False):
        super().__init__()
        if gqa_groups < 1:
            raise ValueError(f"gqa_groups must be >= 1, got {gqa_groups}")
        if num_heads % gqa_groups != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by gqa_groups={gqa_groups}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gqa_groups = gqa_groups
        self.num_kv_heads = num_heads // gqa_groups
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _project_q(self, query):
        seq_len, batch_size, _ = query.shape
        return (
            self.q_proj(query)
            .permute(1, 0, 2)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _project_kv(self, tokens, proj):
        seq_len, batch_size, _ = tokens.shape
        return (
            proj(tokens)
            .permute(1, 0, 2)
            .reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, **_kwargs):
        q = self._project_q(query)
        k = self._project_kv(key, self.k_proj).repeat_interleave(self.gqa_groups, dim=1)
        v = self._project_kv(value, self.v_proj).repeat_interleave(self.gqa_groups, dim=1)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).view(key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1])

        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        batch_size, _, query_len, _ = attn.shape
        out = attn.transpose(1, 2).reshape(batch_size, query_len, self.embed_dim)
        return self.out_proj(out).permute(1, 0, 2), None


def build_action_attention(embed_dim, num_heads, dropout_p, gqa_groups=1):
    if gqa_groups == 1:
        return nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_p, bias=False)
    return GroupedQueryAttention(embed_dim, num_heads, gqa_groups, dropout=dropout_p, bias=False)


class TransformerBlock(nn.Module):
    """
    Transformer block with DiT-style gated residuals when using adaptive norm.

    When norm="adaptive", uses AdaLN-Zero with gated residual connections:
        x = x + gate * sublayer(x)

    This allows the network to dynamically modulate update magnitude based on timestep,
    enabling straighter flow paths and fewer inference steps.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        norm,
        dropout_p,
        adaln_mode="per_block",
        adaln_lora_rank=256,
        gqa_groups=1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.norm = norm
        self.adaln_mode = adaln_mode
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.self_attention = build_action_attention(embed_dim, num_heads, self.dropout_p, gqa_groups)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            GELUTanh(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(p=self.dropout_p),
        )
        if self.norm == "adaptive":
            shared = adaln_mode in ("shared", "shared_lora")
            lora_rank = adaln_lora_rank if adaln_mode == "shared_lora" else None
            self.pre_norm = AdaptiveLayerNorm(embed_dim, shared=shared, lora_rank=lora_rank)
            self.post_norm = AdaptiveLayerNorm(embed_dim, shared=shared, lora_rank=lora_rank)
        elif self.norm == "rms":
            self.pre_norm = GemmaRMSNorm(embed_dim)
            self.post_norm = GemmaRMSNorm(embed_dim)
        else:
            raise ValueError(f"Unknown norm type: {self.norm}")

    def forward(self, x, time_emb, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            time_emb: Timestep embedding for AdaLN conditioning; for
                adaln_mode="shared" a dict of precomputed base modulations
                keyed by site role ("pre", "post")
            key_padding_mask: Optional mask of shape (batch_size, seq_len)
                            where True indicates positions to IGNORE (padding tokens)
        """
        if self.norm == "adaptive":
            if self.adaln_mode in ("shared", "shared_lora"):
                cond_pre = shared_adaln_cond(time_emb, "pre", self.adaln_mode)
                cond_post = shared_adaln_cond(time_emb, "post", self.adaln_mode)
            else:
                cond_pre = cond_post = time_emb

            # AdaLN-Zero with gated residuals
            x_norm, gate_attn = self.pre_norm(x, cond_pre)
            attn_output, _ = self.self_attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            # Gated residual: x = x + gate * attn_output
            x = x + gate_attn.unsqueeze(0) * self.dropout(attn_output)

            x_norm, gate_ff = self.post_norm(x, cond_post)
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

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        norm="rms",
        dropout_p=0.2,
        adaln_mode="per_block",
        adaln_lora_rank=256,
        gqa_groups=1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.norm = norm
        self.adaln_mode = adaln_mode
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Self-attention for query tokens only
        self.query_self_attention = build_action_attention(embed_dim, num_heads, self.dropout_p, gqa_groups)

        # Cross-attention for Q attending to KV context
        self.cross_attention = build_action_attention(embed_dim, num_heads, self.dropout_p, gqa_groups)

        # Feedforward for query tokens only
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            GELUTanh(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(p=self.dropout_p),
        )

        # Layer norms - simplified: only for query stream and KV input to cross-attn
        if norm == "adaptive":
            shared = adaln_mode in ("shared", "shared_lora")
            lora_rank = adaln_lora_rank if adaln_mode == "shared_lora" else None
            self.q_norm_1 = AdaptiveLayerNorm(embed_dim, shared=shared, lora_rank=lora_rank)
            self.q_norm_2 = AdaptiveLayerNorm(embed_dim, shared=shared, lora_rank=lora_rank)
            self.kv_norm = AdaptiveLayerNorm(embed_dim, shared=shared, lora_rank=lora_rank)
            self.q_norm_3 = AdaptiveLayerNorm(embed_dim, shared=shared, lora_rank=lora_rank)
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
            time_emb: Timestep embedding for AdaLN conditioning; for
                adaln_mode="shared" a dict of precomputed base modulations
                keyed by site role ("q1", "q2", "kv", "q3")
            query_mask: mask for query tokens
            kv_mask: mask for key-value tokens
        """
        if self.norm == "adaptive":
            if self.adaln_mode in ("shared", "shared_lora"):
                cond_q1, cond_q2, cond_kv, cond_q3 = (
                    shared_adaln_cond(time_emb, r, self.adaln_mode) for r in ("q1", "q2", "kv", "q3")
                )
            else:
                cond_q1 = cond_q2 = cond_kv = cond_q3 = time_emb

            # Query self-attention with gated residual
            q_norm, gate_self = self.q_norm_1(query_tokens, cond_q1)
            self_attn_output, _ = self.query_self_attention(
                q_norm, q_norm, q_norm, key_padding_mask=query_mask
            )
            query_tokens = query_tokens + gate_self.unsqueeze(0) * self.dropout(self_attn_output)

            # Cross-attention with gated residual
            q_norm, gate_cross = self.q_norm_2(query_tokens, cond_q2)
            kv_norm, _ = self.kv_norm(key_value_tokens, cond_kv)  # gate unused for KV
            cross_attn_output, _ = self.cross_attention(q_norm, kv_norm, kv_norm, key_padding_mask=kv_mask)
            query_tokens = query_tokens + gate_cross.unsqueeze(0) * self.dropout(cross_attn_output)

            # Feedforward with gated residual
            q_norm, gate_ff = self.q_norm_3(query_tokens, cond_q3)
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

    def __init__(self, config: PolicyConfig):
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

        self.adaln_mode = getattr(self.config, "adaln_mode", "per_block")
        self.adaln_lora_rank = getattr(self.config, "adaln_lora_rank", 256)
        self.gqa_groups = getattr(self.config, "gqa_groups", 1)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.embed_dim,
                    self.num_heads,
                    self.ff_dim,
                    self.norm,
                    self.dropout_p,
                    adaln_mode=self.adaln_mode,
                    adaln_lora_rank=self.adaln_lora_rank,
                    gqa_groups=self.gqa_groups,
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.norm == "adaptive" and self.adaln_mode in ("shared", "shared_lora"):
            self.shared_mod = build_shared_adaln_maps(self.embed_dim, ("pre", "post"))

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
        x = self.dropout(x + pe)

        # For shared adaLN, compute the base modulations once, outside the
        # gradient-checkpointed region, and pass them to every block.
        block_cond = time_emb
        if self.norm == "adaptive" and self.adaln_mode in ("shared", "shared_lora"):
            block_cond = {role: mod(time_emb) for role, mod in self.shared_mod.items()}
            if self.adaln_mode == "shared_lora":
                block_cond["_raw"] = time_emb

        for block in self.transformer_blocks:
            x = (
                # use_reentrant=False required for DDP without static_graph;
                # the reentrant variant (default) marks the same params
                # ready in multiple backward passes per step which DDP only
                # tolerates under static_graph=True. Cotraining alternates
                # which head's params are touched per step, so static_graph
                # isn't usable; non-reentrant checkpointing is the path.
                checkpoint.checkpoint(block, x, block_cond, key_padding_mask, use_reentrant=False)
                if self.enable_gradient_checkpointing
                else block(x, block_cond, key_padding_mask)
            )

        if self.norm == "adaptive":
            x, _ = self.final_norm(x, time_emb)  # Discard gate at final output
        else:
            x = self.final_norm(x)
        x = x.permute(1, 0, 2)  # (batch_size, num_tokens, embed_dim)
        return x


class CrossAttentionActionExpert(nn.Module):
    def __init__(self, config: PolicyConfig):
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

        self.adaln_mode = getattr(self.config, "adaln_mode", "per_block")
        self.adaln_lora_rank = getattr(self.config, "adaln_lora_rank", 256)
        self.gqa_groups = getattr(self.config, "gqa_groups", 1)

        self.transformer_blocks = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(
                    self.embed_dim,
                    self.num_heads,
                    self.ff_dim,
                    norm=self.norm,
                    dropout_p=self.dropout_p,
                    adaln_mode=self.adaln_mode,
                    adaln_lora_rank=self.adaln_lora_rank,
                    gqa_groups=self.gqa_groups,
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.norm == "adaptive" and self.adaln_mode in ("shared", "shared_lora"):
            # adaLN-single: one modulation map per site role, shared across all
            # blocks. Blocks only own zero-init offsets (see AdaptiveLayerNorm).
            self.shared_mod = build_shared_adaln_maps(self.embed_dim, ("q1", "q2", "kv", "q3"))

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

        # For shared adaLN, compute the base modulations once, outside the
        # gradient-checkpointed region, and pass them to every block.
        block_cond = time_emb
        if self.norm == "adaptive" and self.adaln_mode in ("shared", "shared_lora"):
            block_cond = {role: mod(time_emb) for role, mod in self.shared_mod.items()}
            if self.adaln_mode == "shared_lora":
                block_cond["_raw"] = time_emb

        for block in self.transformer_blocks:  # (num_tokens, batch_size, embed_dim)
            state_tokens = (
                checkpoint.checkpoint(
                    block,
                    state_tokens,
                    image_text_tokens,
                    block_cond,
                    None,
                    img_text_key_mask,
                    use_reentrant=False,
                )
                if self.enable_gradient_checkpointing
                else block(state_tokens, image_text_tokens, block_cond, None, img_text_key_mask)
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

    def __init__(self, config: PolicyConfig):
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

        self.adaln_mode = getattr(self.config, "adaln_mode", "per_block")
        self.adaln_lora_rank = getattr(self.config, "adaln_lora_rank", 256)
        self.gqa_groups = getattr(self.config, "gqa_groups", 1)

        self.transformer_blocks = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(
                    self.embed_dim,
                    self.num_heads,
                    self.ff_dim,
                    norm=self.norm,
                    dropout_p=self.dropout_p,
                    adaln_mode=self.adaln_mode,
                    adaln_lora_rank=self.adaln_lora_rank,
                    gqa_groups=self.gqa_groups,
                )
                for _ in range(self.num_blocks)
            ]
        )

        if self.norm == "adaptive" and self.adaln_mode in ("shared", "shared_lora"):
            self.shared_mod = build_shared_adaln_maps(self.embed_dim, ("q1", "q2", "kv", "q3"))

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

        # For shared adaLN, compute the base modulations once, outside the
        # gradient-checkpointed region, and pass them to every block.
        block_cond = time_emb
        if self.norm == "adaptive" and self.adaln_mode in ("shared", "shared_lora"):
            block_cond = {role: mod(time_emb) for role, mod in self.shared_mod.items()}
            if self.adaln_mode == "shared_lora":
                block_cond["_raw"] = time_emb

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
                # use_reentrant=False so gradients reach tensors passed inside
                # the shared-adaLN modulation dict.
                state_tokens = checkpoint.checkpoint(
                    block,
                    state_tokens,
                    kv_tokens,
                    block_cond,
                    None,
                    img_text_key_mask,
                    use_reentrant=False,
                )
            else:
                state_tokens = block(
                    state_tokens,
                    kv_tokens,
                    block_cond,
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


class BaseVLMModel(nn.Module):
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
        config: PolicyConfig,
        vlm_backbone=None,
        vlm_processor=None,
        generation_config=None,
        vlm_projector=None,
        state_projector=None,
        backbone_factory=create_backbone_adapter,
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
        self._backend = backbone_factory(config)

        # Use provided VLM components or initialize new ones.
        owns_vlm_backbone = vlm_backbone is None
        if vlm_backbone is None:
            self.vlm_backbone = self._backend.load_backbone()
        else:
            self.vlm_backbone = vlm_backbone

        # Configure Phi4MM's built-in vision/speech LoRAs
        # These are created by Phi4MM's __init__ and loaded from checkpoint
        if owns_vlm_backbone:
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
        return torch_pad_sequence(
            sequences,
            batch_first=True,
            padding_value=padding_value,
            padding_side=padding_side,
        )

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


_rtc_execution_horizon_mismatch_warned = False


class FlowMatchingModel(BaseVLMModel):
    """
    Flow Matching model for continuous action prediction.

    Extends BaseVLMModel with:
    - Action projection and time embedding networks
    - Action expert transformer
    - Action prediction head
    - Flow matching training and sampling
    """

    def __init__(self, config: PolicyConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Flow matching specific components
        self.action_in_proj = nn.Linear(self.max_action_dim, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )
        self.action_head = nn.Linear(self.embed_dim, self.max_action_dim).to(
            device=self.device, dtype=self.dtype
        )

        self.action_time_mlp_in = nn.Linear(self.embed_dim * 2, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )
        self.action_time_mlp_out = nn.Linear(self.embed_dim, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )

        self.time_mlp_in = nn.Linear(self.embed_dim, self.embed_dim).to(device=self.device, dtype=self.dtype)
        self.time_mlp_out = nn.Linear(self.embed_dim, self.embed_dim).to(device=self.device, dtype=self.dtype)

        if config.attention_type == "self":
            self.action_expert = SimpleActionExpert(config).to(device=self.device, dtype=self.dtype)
        elif config.attention_type == "layerwise_cross":
            self.action_expert = LayerwiseCrossAttentionExpert(config).to(
                device=self.device, dtype=self.dtype
            )
        else:
            self.action_expert = CrossAttentionActionExpert(config).to(device=self.device, dtype=self.dtype)
        # Preserve positional encodings in fp32
        self.action_expert.positional_encoding = self.action_expert.positional_encoding.float()

        self._use_layerwise_cross = config.attention_type == "layerwise_cross"

        # Stash for hidden-state stats; consumed by the policy forward method
        # and merged into the wandb loss_dict. Reset per forward.
        self._last_hidden_state_stats: dict[str, float] = {}

    def _compute_hidden_state_stats(self, image_text_embed) -> dict[str, float]:
        """Compute scalar stats on the extracted VLM hidden state.

        For non-layerwise attention, ``image_text_embed`` is a single
        (B, S, embed_dim) tensor (post-projection). For layerwise attention it
        is a list of (B, S, vlm_hidden_dim) tensors (pre-projection); we stat
        the layer at ``hidden_state_idx`` to keep the signal directly
        comparable across attention types.
        """
        if isinstance(image_text_embed, (list, tuple)):
            idx = min(int(self.hidden_state_idx), len(image_text_embed) - 1)
            tensor = image_text_embed[idx]
        else:
            tensor = image_text_embed
        return hidden_state_stats(tensor, prefix="hidden_state")

    def sample_noise(self, shape, device):
        """Sample noise for flow matching."""
        return torch.randn(shape, dtype=torch.float32, device=device).to(self.dtype)

    def sample_beta(self, alpha, beta, bsize, device):
        """Sample from Beta distribution for time sampling."""
        g1 = torch.empty((bsize,), device=device, dtype=torch.float32).uniform_(0, 1).pow(1 / alpha)
        g2 = torch.empty((bsize,), device=device, dtype=torch.float32).uniform_(0, 1).pow(1 / beta)
        return g1 / (g1 + g2)

    def sample_time(self, bsize, device):
        """Sample time steps for flow matching."""
        strategy = self.config.time_sampling_strategy

        if strategy == "beta_reverse":
            t = self.sample_beta(1.0, 1.5, bsize, device)
        elif strategy == "uniform":
            t = torch.empty((bsize,), device=device, dtype=torch.float32).uniform_(0, 1)
        elif strategy == "logit_normal":
            t = torch.sigmoid(torch.randn((bsize,), device=device, dtype=torch.float32))
        else:  # default: "beta"
            t = self.sample_beta(1.5, 1.0, bsize, device)  # fp32
        t = t * 0.999 + 0.001

        return t.to(self.dtype)  # cast once at the boundary

    def create_sinusoidal_pos_embedding(
        self, time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
    ) -> Tensor:
        """Computes sine-cosine positional embedding vectors for scalar positions."""
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

        dtype = get_safe_dtype(torch.float64, device.type)
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction

        # Compute the outer product
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time.float()[:, None]
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        return pos_emb.to(dtype=self.dtype)

    def embed_state(self, state, noisy_actions, timestep):  # based on embed_suffix() in modeling_pi0.py
        """Embed state, noisy_actions, timestep to prepare for further processing."""
        embs = []

        # Store as bf16 for memory; linears will run under autocast bf16
        state = state.to(device=self.device, dtype=self.dtype)
        noisy_actions = noisy_actions.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device, dtype=self.dtype)

        # Embed state
        state_emb = self.state_projector(state)  # (batch_size, n_obs_steps, embed_dim)
        embs.append(state_emb)
        dtype = state_emb.dtype
        device = state_emb.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = self.create_sinusoidal_pos_embedding(
            timestep.to(self.device), self.embed_dim, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)  # (batch_size, chunk_size, embed_dim)

        embs = torch.cat(embs, dim=1)
        return embs

    def embed_time_for_cond(self, timestep):
        timestep = timestep.to(device=self.device, dtype=self.dtype)

        time_emb = self.create_sinusoidal_pos_embedding(
            timestep.to(self.device), self.embed_dim, min_period=4e-3, max_period=4.0, device=timestep.device
        )

        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = time_emb.type(dtype=self.dtype)
        return time_emb

    @staticmethod
    def _expand_batch_for_flow_samples(value, num_flow_samples):
        """Expand tensors from B to B*K, keeping each example's K rows adjacent."""
        if value is None or num_flow_samples == 1:
            return value
        if isinstance(value, torch.Tensor):
            expand_shape = (-1, num_flow_samples, *([-1] * (value.ndim - 1)))
            return (
                value.unsqueeze(1)
                .expand(*expand_shape)
                .reshape(value.shape[0] * num_flow_samples, *value.shape[1:])
            )
        if isinstance(value, tuple):
            return tuple(
                FlowMatchingModel._expand_batch_for_flow_samples(item, num_flow_samples) for item in value
            )
        if isinstance(value, list):
            return [
                FlowMatchingModel._expand_batch_for_flow_samples(item, num_flow_samples) for item in value
            ]
        raise TypeError(f"Unsupported flow-conditioning type: {type(value).__name__}")

    @staticmethod
    def _group_explicit_flow_samples(value, base_shape, num_flow_samples, name):
        """Normalize an explicitly supplied noise/time tensor to (B, K, ...)."""
        grouped_shape = (base_shape[0], num_flow_samples, *base_shape[1:])
        flat_shape = (base_shape[0] * num_flow_samples, *base_shape[1:])
        if tuple(value.shape) == grouped_shape:
            return value
        if tuple(value.shape) == flat_shape:
            return value.reshape(grouped_shape)
        if tuple(value.shape) == base_shape:
            expand_shape = (-1, num_flow_samples, *([-1] * (value.ndim - 1)))
            return value.unsqueeze(1).expand(*expand_shape)
        raise ValueError(
            f"{name} must have shape {base_shape}, {grouped_shape}, or {flat_shape} "
            f"when num_flow_samples={num_flow_samples}; got {tuple(value.shape)}"
        )

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
        precomputed_hidden_state=None,
    ):
        """
        Training forward pass with flow matching.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            actions: Ground truth actions
            noise: Optional noise (sampled if None)
            time: Optional time steps (sampled if None)
            image_mask: Optional image masking
            training_mode: Training mode (ignored, always uses flow matching)
            precomputed_hidden_state: Optional tuple of (hidden_state, mask) to avoid recomputing

        Returns:
            losses: Per-element flow matching losses
        """
        # Store inputs in bf16 for memory
        actions = actions.to(dtype=self.dtype)

        batch_size = actions.shape[0]
        num_flow_samples = int(self.config.num_flow_samples)
        if num_flow_samples == 1:
            # Keep the original K=1 sampling and arithmetic path unchanged.
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            else:
                noise = noise.to(dtype=self.dtype)

            time = self.sample_time(batch_size, actions.device) if time is None else time.to(dtype=self.dtype)

            # Upcast to fp32 for arithmetic
            a32 = actions.float()
            n32 = noise.float()
            t32 = time.float()

            time_expanded32 = t32[:, None, None]
            x_t32 = time_expanded32 * n32 + (1 - time_expanded32) * a32
            u_t32 = n32 - a32
        else:
            grouped_action_shape = (batch_size, num_flow_samples, *actions.shape[1:])
            actions_grouped = actions.unsqueeze(1).expand(grouped_action_shape)

            if noise is None:
                noise_grouped = self.sample_noise(grouped_action_shape, actions.device)
            else:
                noise_grouped = self._group_explicit_flow_samples(
                    noise.to(dtype=self.dtype), tuple(actions.shape), num_flow_samples, "noise"
                )

            if time is None:
                time_grouped = self.sample_time(batch_size * num_flow_samples, actions.device).reshape(
                    batch_size, num_flow_samples
                )
            else:
                time_grouped = self._group_explicit_flow_samples(
                    time.to(dtype=self.dtype), (batch_size,), num_flow_samples, "time"
                )

            # Use the existing reversed-time convention: x_t=t*noise+(1-t)*action,
            # with target noise-action. This is equivalent to the paper after t -> 1-t.
            a32 = actions_grouped.float()
            n32 = noise_grouped.float()
            t32 = time_grouped.float()
            time_expanded32 = t32[:, :, None, None]
            x_t32 = time_expanded32 * n32 + (1 - time_expanded32) * a32
            u_t32 = n32 - a32

            x_t32 = x_t32.reshape(batch_size * num_flow_samples, *actions.shape[1:])
            u_t32 = u_t32.reshape(batch_size * num_flow_samples, *actions.shape[1:])
            time = time_grouped.reshape(batch_size * num_flow_samples)

        # Cast x_t/time back for downstream bf16 modules
        x_t = x_t32.to(self.dtype)

        # Prepare all inputs - use precomputed if available
        if precomputed_hidden_state is not None:
            image_text_embed, image_text_mask = precomputed_hidden_state
        else:
            if self._use_layerwise_cross:
                image_text_embed, image_text_mask = self.get_all_hidden_states(
                    image, prompt, image_mask=image_mask
                )
            else:
                image_text_embed, image_text_mask = self.get_image_text_hidden_state(
                    image, prompt, image_mask=image_mask
                )

        if getattr(self.config, "log_hidden_state_stats", False) and self.training:
            self._last_hidden_state_stats = self._compute_hidden_state_stats(image_text_embed)

        if num_flow_samples > 1:
            # The VLM ran above exactly once. Only expand its cached context for
            # the vectorized B*K action-expert pass.
            image_text_embed = self._expand_batch_for_flow_samples(image_text_embed, num_flow_samples)
            image_text_mask = self._expand_batch_for_flow_samples(image_text_mask, num_flow_samples)
            state = self._expand_batch_for_flow_samples(state, num_flow_samples)

        state_embed = self.embed_state(state, x_t, time)

        time_embed = self.embed_time_for_cond(time)

        # Forward through action expert
        output_embed = self.action_expert.forward(
            image_text_embed, state_embed, time_embed, image_text_attn_mask=image_text_mask
        )

        action_token = output_embed[:, -self.config.chunk_size :]
        v_t = self.action_head(action_token)

        # Compute loss in fp32 for stability
        losses = F.mse_loss(u_t32.to(self.device), v_t.float(), reduction="none")

        if num_flow_samples > 1:
            losses = losses.reshape(batch_size, num_flow_samples, *losses.shape[1:]).mean(dim=1)

        return losses

    def velocity_eval(self, state, x_t, time_scalar, precomputed_hidden_state):
        """Evaluate the flow velocity v(x_t, t) at a single denoise step.

        Shared interface used by rho.hil.noise_inverse_map. Other flow models
        implement this with matching semantics; the inverter
        treats ``precomputed_hidden_state`` opaquely.

        Mirrors the inner body of ``sample_actions``'s denoise loop.

        Args:
            state: Robot state tensor (B, n_obs_steps, state_dim).
            x_t:   Current noise/action latent (B, chunk_size, max_action_dim).
            time_scalar: Float in [0, 1] -- the current diffusion time.
            precomputed_hidden_state: ``(image_text_embed, image_text_mask)`` from
                ``get_image_text_hidden_state``. Reused across steps without
                re-running the VLM.

        Returns:
            v_t: Velocity tensor (B, chunk_size, max_action_dim) in x_t's dtype.
        """
        image_text_embed, image_text_mask = precomputed_hidden_state
        bsize = state.shape[0]
        device = state.device
        time = (
            torch.tensor(float(time_scalar), dtype=torch.float32, device=device).expand(bsize).to(self.dtype)
        )
        state_embed = self.embed_state(state, x_t.to(self.dtype), time)
        time_emb = self.embed_time_for_cond(time)
        output_embed = self.action_expert.forward(
            image_text_embed, state_embed, time_emb, image_text_attn_mask=image_text_mask
        )
        action_token = output_embed[:, -self.config.chunk_size :]
        return self.action_head(action_token).to(x_t.dtype)

    def sample_actions_from_precomputed(self, state, precomputed_hidden_state, noise, num_steps=None):
        """Round-trip denoising entry point used by inversion verification.

        Thin wrapper over ``sample_actions`` that names the precomputed prefix
        conditioning explicitly and drops the image/prompt args (unused when
        ``precomputed_hidden_state`` is provided).
        """
        return self.sample_actions(
            image=None,
            prompt=None,
            state=state,
            noise=noise,
            precomputed_hidden_state=precomputed_hidden_state,
            num_steps=num_steps,
        )

    def sample_actions(
        self,
        image,
        prompt,
        state,
        noise=None,
        image_mask=None,
        precomputed_hidden_state=None,
        num_steps=None,
    ) -> Tensor:
        """
        Inference: denoise from t=1 to t=0 to generate actions.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            noise: Optional initial noise (sampled if None)
            image_mask: Optional image masking
            precomputed_hidden_state: Optional ``(image_text_embed, image_text_mask)``
                tuple. When provided, skips the VLM forward pass and reuses these
                features -- useful when iterating Euler steps many times with the
                same conditioning.
            num_steps: Override the denoising step count. None = use the
                configured ``self.config.num_steps``.

        Returns:
            Final denoised actions
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            x_t = self.sample_noise(actions_shape, device)
        else:
            x_t = noise.to(dtype=self.dtype)

        # Stash the initial noise so the policy wrapper can return it to the
        # DSRL trainer for transition recording.
        self._last_initial_noise = x_t.detach().float().cpu().numpy()

        n_steps = num_steps if num_steps is not None else self.config.num_steps
        dt32 = torch.tensor(-1.0 / n_steps, dtype=torch.float32, device=device)
        time32 = torch.tensor(1.0, dtype=torch.float32, device=device)
        if precomputed_hidden_state is not None:
            image_text_embed, image_text_mask = precomputed_hidden_state
        elif self._use_layerwise_cross:
            image_text_embed, image_text_mask = self.get_all_hidden_states(
                image, prompt, image_mask=image_mask
            )
        else:
            image_text_embed, image_text_mask = self.get_image_text_hidden_state(
                image, prompt, image_mask=image_mask
            )

        # Denoise from t=1 to t=0
        while time32 >= -dt32 / 2:
            expanded_time_bf16 = time32.expand(bsize).to(self.dtype)
            state_embed = self.embed_state(state, x_t.to(self.device), expanded_time_bf16)

            time_emb = self.embed_time_for_cond(expanded_time_bf16)

            output_embed = self.action_expert.forward(
                image_text_embed, state_embed, time_emb, image_text_attn_mask=image_text_mask
            )
            action_token = output_embed[:, -self.config.chunk_size :]
            v_t = self.action_head(action_token)

            # Euler step in fp32
            x_t32 = x_t.float() + dt32 * v_t.float()
            x_t = x_t32.to(self.dtype)
            time32 = time32 + dt32

        return x_t  # Final denoised actions (bf16)

    @staticmethod
    def _rtc_guidance_coeff(tau: float, beta: float, schedule: str = "paper") -> float:
        """Compute the guidance coefficient for one RTC denoising step.

        Implements Equations 1 and 4 from arXiv:2506.07339.

        Eq. 1 integration step:
            A_{τ+1/n} = A_τ + (1/n) v_π + coeff(τ) · g
            coeff(τ)  = min(β, (1−τ)/(τ · r²_τ))

        Eq. 4 continuity weight:
            r²_τ = (1−τ)² / (τ² + (1−τ)²)

        ``beta`` is an upper-bound / clip for numerical stability at small τ;
        it is NOT the coefficient itself.

        Args:
            tau: Flow time in the paper's convention (τ ∈ [0, 1]).
            beta: Clip / upper bound on the coefficient (Eq. 1).
            schedule: ``"paper"`` applies Eqs. 1 and 4; ``"constant"`` returns
                ``beta`` for every step (preserves the pre-fix behaviour for
                A/B comparison).

        Returns:
            Scalar guidance coefficient ≥ 0.

        Raises:
            ValueError: If ``schedule`` is not ``"paper"`` or ``"constant"``.
        """
        if schedule == "constant":
            return beta
        if schedule == "paper":
            if tau <= 0.0:
                return beta
            r2_tau = (1.0 - tau) ** 2 / (tau**2 + (1.0 - tau) ** 2)
            return min(beta, (1.0 - tau) / (tau * r2_tau))
        raise ValueError(f"Unknown guidance_schedule {schedule!r}. Valid choices are 'paper' and 'constant'.")

    @torch.no_grad
    def sample_actions_rtc(
        self,
        image,
        prompt,
        state,
        inference_delay,
        execution_horizon,
        prev_actions=None,
        noise=None,
        beta=40.0,
        guidance_schedule: str = "paper",
        image_mask=None,
    ) -> Tensor:
        """
        Guided inference algorithm for RTC as described in the paper (https://www.physicalintelligence.company/download/real_time_chunking.pdf).
        Original Algorithm:
        24: compute W using Eq. 5; right-pad prev_actions to length H; initialize A_0 ~ N(0,I)
        25: for τ=0 to 1 with stepsize 1/n do
        26:     f_A^1 = A' → A' + (1-τ)v_π(A',o,τ)  ⊳ Define denoising function (Eq. 3)
        27:     e = (prev_actions - f_A^1(A_τ))^T diag(W)  ⊳ Weighted error term from Eq. 2
        28:     g = e · ∂f_A^1/∂A'|_{A'=A_τ}  ⊳ Compute vector-Jacobian product from Eq. 2 via autodiff
        29:     A_{τ+1/n} = A_τ + 1/n v_π(A_τ,o,τ) + min(β, (1-τ)/τ · r^2/τ) g  ⊳ Integration step (Eq. 1)
        return A_1

        Args:
            obs: Dictionary containing observations
            prev_actions: Previous actions tensor (batch_size, chunk_size - s, action_dim)
            inference_delay: Inference delay ``d`` — rows 0..d-1 of W are set to 1.0.
            execution_horizon: Nominal number of steps executed since last inference.
                               The mask boundary is derived from ``prev_actions.shape[-2]``
                               per Algorithm 1 line 14 (``s_eff = H - len(A_prev)``); this
                               argument is retained for API compatibility and is used only as
                               a consistency check (a one-time warning fires on mismatch).
            beta: Maximum guidance strength
            r: Guidance scaling factor

        Returns:
            torch.Tensor: Final denoised actions
        """
        bsize = state.shape[0]
        device = state.device

        # Right-pad prev_actions to length H
        H = self.config.chunk_size  # noqa: N806
        action_dim = self.max_action_dim

        prev_actions_padded, w = self._prepare_rtc_mask(prev_actions, inference_delay, execution_horizon)

        # our denoising process goes from t = 1 to t = 0, as opposed to PI which goes from τ = 0 to τ = 1
        # Initialize A_1 ~ N(0, I)
        actions_shape = (bsize, H, action_dim)
        A_tau = (  # noqa: N806
            self.sample_noise(actions_shape, device).float()
            if noise is None
            else noise.to(device=device).float()
        )

        # Get image-text embeddings once (they don't change during denoising)
        if self._use_layerwise_cross:
            image_text_embed, image_text_mask = self.get_all_hidden_states(
                image, prompt, image_mask=image_mask
            )
        else:
            image_text_embed, image_text_mask = self.get_image_text_hidden_state(
                image, prompt, image_mask=image_mask
            )

        # Step 25: Denoising loop from t = 1 to t = 0
        n_steps = self.config.num_steps
        dt = -1.0 / n_steps

        for step in range(n_steps):
            tau = 1 + dt * step  # current time t
            tau_tensor = torch.tensor(tau, dtype=torch.float32, device=device).expand(bsize).to(self.dtype)

            # Step 26: Define denoising function f_A^0 - this is where we estimate
            # the final denoised version (i.e. A_0)
            def denoising_function(A_prime, tau_=tau, tau_tensor_=tau_tensor):  # noqa: N803, B023
                """f_A^1(A') = A' + (1-τ)v_π(A',o,τ)"""
                state_embed = self.embed_state(state, A_prime.to(self.dtype), tau_tensor_)
                time_embed = self.embed_time_for_cond(tau_tensor_)
                output_embed = self.action_expert.forward(
                    image_text_embed,
                    state_embed,
                    time_embed,
                    image_text_attn_mask=image_text_mask,
                )
                action_token = output_embed[:, -H:]
                v_pi = self.action_head(action_token).float()
                return A_prime - (tau_) * v_pi, v_pi

            A_0, vjp_fn, v_pi = torch.func.vjp(denoising_function, A_tau, has_aux=True)  # noqa: N806

            error = (prev_actions_padded - A_0) * w.unsqueeze(0)
            (vjp_result,) = vjp_fn(error)

            # flip tau to paper convention (τ = 0..1) to compute the guidance coefficient
            tau = 1 - tau
            guidance_coeff = self._rtc_guidance_coeff(tau, beta, guidance_schedule)

            # Step 29: Integration step
            # A_{τ+1/n} = A_τ + (1/n)(v_π + min(β, (1-τ)/(τ·r²_τ))g)
            # changed this to subtract vjp result since in our case dt is negative,
            # in PI's version dt is positive
            A_tau = A_tau + dt * (v_pi - (guidance_coeff * vjp_result))  # noqa: N806

        return A_tau

    def _prepare_rtc_mask(self, prev_actions, inference_delay, execution_horizon, action_dim=None):
        """
        Pad prev_actions to full chunk shape and compute the soft W mask per Algorithm 1.

        The mask's zero boundary is derived from the actual overlap length of prev_actions
        (Algorithm 1, line 14: ``s_eff = H - len(A_prev)``), not from the caller-supplied
        ``execution_horizon``.  This enforces the paper's invariant ``H - s == len(A_prev)``
        regardless of how re-inference is scheduled at runtime.

        The ``execution_horizon`` argument is retained for signature compatibility with all
        call sites; it is used only as a consistency check against the derived ``s_eff``.
        When the two disagree, a one-time warning is emitted to flag the misconfiguration.

        Args:
            prev_actions: (batch_size, overlap_len, feature_action_dim) tensor of previous actions.
            inference_delay: Inference delay parameter ``d`` for the W-matrix formula (Eq. 5).
            execution_horizon: Expected number of steps already executed; used only for a
                consistency check against the overlap derived from ``prev_actions``.
            action_dim: Width of the W matrix and padded output tensor.  Defaults to
                ``self.max_action_dim`` when not supplied, which preserves the behaviour of all
                existing callers.  Pass an explicit value when the effective action space is
                wider than ``max_action_dim`` (e.g. ``max_action_dim + max_tactile_dim`` for
                combined action-tactile heads).

        Returns:
            prev_actions_padded: (batch_size, H, action_dim) zero-padded tensor
            w: (H, action_dim) weight matrix with padded rows and columns zeroed
        """
        global _rtc_execution_horizon_mismatch_warned

        H = self.config.chunk_size  # noqa: N806
        if action_dim is None:
            action_dim = self.max_action_dim
        bsize = prev_actions.shape[0]

        overlap_len = prev_actions.shape[-2]

        # Clamp defensively so s_eff stays in [0, H].
        overlap_len_clamped = max(0, min(overlap_len, H))
        s_eff = H - overlap_len_clamped

        if execution_horizon != s_eff and not _rtc_execution_horizon_mismatch_warned:
            logger.warning(
                "RTC mask: caller-supplied execution_horizon=%d disagrees with the overlap "
                "derived from prev_actions (len=%d → s_eff=%d, H=%d).  The mask boundary will "
                "be set from the actual overlap per Algorithm 1 line 14.  To silence this "
                "warning, pass execution_horizon=%d.",
                execution_horizon,
                overlap_len,
                s_eff,
                H,
                s_eff,
            )
            _rtc_execution_horizon_mismatch_warned = True

        prev_actions_padded = torch.zeros(
            bsize, H, action_dim, dtype=prev_actions.dtype, device=prev_actions.device
        )
        prev_actions_padded[:, : prev_actions.shape[-2], : prev_actions.shape[-1]] = prev_actions

        w = self.compute_W_matrix_rtc(inference_delay, s_eff, action_dim)  # (H, action_dim)

        # Zero out padded timesteps (rows) and padded action dimensions (columns) separately.
        # A single 2-D slice `w[rows:, cols:]` only zeros the rectangle where BOTH are out of
        # range, leaving the padded-row and padded-column bands intact.
        w[prev_actions.shape[-2] :, :] = 0.0  # zero rows for padded timesteps
        w[:, prev_actions.shape[-1] :] = 0.0  # zero columns for padded action dims

        return prev_actions_padded, w

    def compute_W_matrix_rtc(self, inference_delay, execution_horizon, action_dim):  # noqa: N802
        """
        Compute weight matrix W for RTC (Real Time Chunking) according to (https://www.physicalintelligence.company/download/real_time_chunking.pdf).

        Wi = {
            1                     if i < d
            (ci * e^(ci) - 1) / (e - 1)        if d ≤ i < H-s
            0                     if i ≥ H-s
        }

        where ci = (H-s-i) / (H-s-d+1), i ∈ {0,...,H-1}
        H = self.config.chunk_size (horizon length)

        Args:
            d: Threshold parameter
            s: Step parameter

        Returns:
            torch.Tensor: Weight matrix of shape (chunk_size, max_action_dim)
        """
        horizon = self.config.chunk_size  # noqa: N806

        # Create weight vector for the temporal dimension
        weights = torch.zeros(horizon, dtype=self.dtype, device=self.device)

        for i in range(horizon):
            if i < inference_delay:
                weights[i] = 1.0
            elif inference_delay <= i < horizon - execution_horizon:
                # ci = (horizon-s-i) / (horizon-s-d+1)
                ci = (horizon - execution_horizon - i) / (horizon - execution_horizon - inference_delay + 1)
                weights[i] = (ci * (math.exp(ci) - 1)) / (math.exp(1) - 1)
            else:  # i >= horizon - execution_horizon
                weights[i] = 0.0

        # Repeat to matrix shape (chunk_size, max_action_dim)
        # Each action dimension gets the same temporal weighting.
        # Use repeat (not expand) to produce a contiguous tensor that supports in-place writes.
        W_matrix = weights.unsqueeze(1).repeat(1, action_dim)  # noqa: N806

        return W_matrix


class RhoModel(nn.Module):
    """Single-head Rho model with a stable RhoAlpha-compatible key layout."""

    def __init__(self, config):
        super().__init__()
        self.flow_model = FlowMatchingModel(config)

    def forward(self, image, prompt, state, actions, noise=None, time=None, image_mask=None):
        return self.flow_model(image, prompt, state, actions, noise=noise, time=time, image_mask=image_mask)

    def sample_actions(self, image, prompt, state, noise=None, image_mask=None) -> Tensor:
        return self.flow_model.sample_actions(image, prompt, state, noise, image_mask)

    def sample_actions_rtc(
        self,
        image,
        prompt,
        state,
        inference_delay,
        execution_horizon,
        prev_actions=None,
        noise=None,
        beta=40.0,
        guidance_schedule: str = "paper",
        image_mask=None,
    ) -> Tensor:
        return self.flow_model.sample_actions_rtc(
            image=image,
            prompt=prompt,
            state=state,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            prev_actions=prev_actions,
            noise=noise,
            beta=beta,
            guidance_schedule=guidance_schedule,
            image_mask=image_mask,
        )

    def print_freezing_status(self) -> None:
        self.flow_model.print_freezing_status()

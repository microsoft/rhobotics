"""Tests for shared AdaLN and grouped-query attention in action experts."""

import draccus
import pytest
import torch

from rho.policies.base import PolicyConfig
from rho.policies.rho.configuration_rho import RhoConfig
from rho.policies.rho.rho_model import (
    AdaptiveLayerNorm,
    CrossAttentionActionExpert,
    CrossAttentionTransformerBlock,
    GroupedQueryAttention,
    LayerwiseCrossAttentionExpert,
    SimpleActionExpert,
    TransformerBlock,
)

# attention_type -> (expert class, number of AdaLN sites per block)
EXPERTS = {
    "self": (SimpleActionExpert, 2),
    "cross": (CrossAttentionActionExpert, 4),
    "layerwise_cross": (LayerwiseCrossAttentionExpert, 4),
}

D = 64
HEADS = 4
BLOCKS = 3
FF = 128
VLM_DIM = 96  # cross_attention_dim for layerwise experts


def make_config(
    adaln_mode,
    attention_type="cross",
    embed_dim=D,
    num_heads=HEADS,
    num_blocks=BLOCKS,
    ff_dim=FF,
    device="cpu",
    adaln_lora_rank=8,
    gqa_groups=1,
):
    return RhoConfig(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ff_dim=ff_dim,
        norm="adaptive",
        attention_type=attention_type,
        adaln_mode=adaln_mode,
        adaln_lora_rank=adaln_lora_rank,
        gqa_groups=gqa_groups,
        cross_attention_dim=VLM_DIM,
        enable_gradient_checkpointing=False,
        dropout_p=0.0,
    )


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def modulation_savings(embed_dim, num_blocks, n_sites):
    """Exact param delta between per_block and shared adaLN layouts."""
    d = embed_dim
    per_block = num_blocks * n_sites * (3 * d * d + 3 * d)  # Linear(d, 3d) per site per block
    shared = n_sites * (3 * d * d + 3 * d) + num_blocks * n_sites * 3 * d  # shared maps + offsets
    return per_block - shared


def make_expert_inputs(attention_type, batch=2, n_img=7, n_state=5):
    torch.manual_seed(0)
    time_emb = torch.randn(batch, D)
    state = torch.randn(batch, n_state, D)
    if attention_type in ("self", "cross"):
        image_text = torch.randn(batch, n_img, D)
        return (image_text, state, time_emb), n_state
    # layerwise_cross
    hidden_states = [torch.randn(batch, n_img, VLM_DIM) for _ in range(BLOCKS + 1)]
    return (hidden_states, state, time_emb), n_state


@pytest.mark.parametrize("attention_type", EXPERTS)
def test_shared_param_count(attention_type):
    expert_cls, n_sites = EXPERTS[attention_type]
    expert_pb = expert_cls(make_config("per_block", attention_type))
    expert_sh = expert_cls(make_config("shared", attention_type))
    expected_savings = modulation_savings(D, BLOCKS, n_sites)
    assert count_params(expert_pb) - count_params(expert_sh) == expected_savings


def test_shared_param_count_full_size():
    # Winner shape: 2048 wide, 16 heads, 16 blocks, ff 4096. Meta device so no memory.
    with torch.device("meta"):
        expert_pb = CrossAttentionActionExpert(
            make_config("per_block", embed_dim=2048, num_heads=16, num_blocks=16, ff_dim=4096, device="meta")
        )
        expert_sh = CrossAttentionActionExpert(
            make_config("shared", embed_dim=2048, num_heads=16, num_blocks=16, ff_dim=4096, device="meta")
        )
    n_pb = count_params(expert_pb)
    n_sh = count_params(expert_sh)
    # Expert-only totals (flow wrapper projections/time MLPs not included)
    assert abs(n_pb - 1.624e9) < 0.01e9, f"per_block expert has {n_pb} params"
    assert abs(n_sh - 0.869e9) < 0.01e9, f"shared expert has {n_sh} params"
    assert n_pb - n_sh == modulation_savings(2048, 16, 4)  # 754,950,144 (~755M)


@pytest.mark.parametrize("attention_type", EXPERTS)
@pytest.mark.parametrize("mode", ["per_block", "shared", "shared_lora"])
def test_forward_shapes(attention_type, mode):
    expert_cls, _ = EXPERTS[attention_type]
    expert = expert_cls(make_config(mode, attention_type)).eval()
    inputs, n_state = make_expert_inputs(attention_type)
    out = expert(*inputs)
    # SimpleActionExpert returns all tokens; the cross variants return state tokens only
    assert out.shape[0] == 2 and out.shape[2] == D
    if attention_type != "self":
        assert out.shape[1] == n_state
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("attention_type", EXPERTS)
def test_shared_identity_at_init(attention_type):
    torch.manual_seed(0)
    expert_cls, _ = EXPERTS[attention_type]
    expert = expert_cls(make_config("shared", attention_type)).eval()
    time_emb = torch.randn(2, D)

    # Shared maps are zero-init, so base modulation is zero for any timestep
    base = {role: mod(time_emb) for role, mod in expert.shared_mod.items()}
    for role, value in base.items():
        assert torch.equal(value, torch.zeros_like(value)), role

    # Zero modulation + zero offsets -> gates are zero -> block is identity
    query = torch.randn(5, 2, D)
    kv = torch.randn(7, 2, D)
    block = expert.transformer_blocks[0]
    out = block(query, kv, base) if isinstance(block, CrossAttentionTransformerBlock) else block(query, base)
    assert torch.equal(out, query)


@pytest.mark.parametrize("attention_type", EXPERTS)
def test_shared_backward_through_gradient_checkpointing(attention_type):
    expert_cls, _ = EXPERTS[attention_type]
    cfg = make_config("shared", attention_type)
    cfg.enable_gradient_checkpointing = True
    expert = expert_cls(cfg).train()

    inputs, _ = make_expert_inputs(attention_type)
    context, state, time_emb = inputs
    state.requires_grad_(True)
    time_emb.requires_grad_(True)

    out = expert(context, state, time_emb)
    out.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert time_emb.grad is not None
    # Shared maps sit upstream of every block and must receive gradient
    for role, mod in expert.shared_mod.items():
        grad = mod[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all(), role


def test_adaptive_layer_norm_shared_offset_only():
    d = 16
    norm = AdaptiveLayerNorm(d, shared=True)
    assert not hasattr(norm, "modulation")
    assert norm.offset.shape == (3 * d,)
    assert torch.equal(norm.offset, torch.zeros(3 * d))


def test_adaptive_layer_norm_shared_lora_is_conditioned():
    torch.manual_seed(0)
    d = 16
    norm = AdaptiveLayerNorm(d, shared=True, lora_rank=4)
    x = torch.randn(3, 2, d)
    base = torch.zeros(2, 3 * d)
    out_a, gate_a = norm(x, (base, torch.zeros(2, d)))
    out_b, gate_b = norm(x, (base, torch.ones(2, d)))
    assert not hasattr(norm, "offset")
    assert norm.lora[1].out_features == 4
    assert norm.lora[2].out_features == 3 * d
    assert not torch.equal(out_a, out_b)
    assert not torch.equal(gate_a, gate_b)


@pytest.mark.parametrize("attention_type", EXPERTS)
def test_shared_lora_backward_through_gradient_checkpointing(attention_type):
    expert_cls, _ = EXPERTS[attention_type]
    cfg = make_config("shared_lora", attention_type)
    cfg.enable_gradient_checkpointing = True
    expert = expert_cls(cfg).train()
    context, state, time_emb = make_expert_inputs(attention_type)[0]
    state.requires_grad_(True)
    time_emb.requires_grad_(True)
    expert(context, state, time_emb).sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert time_emb.grad is not None and torch.isfinite(time_emb.grad).all()
    for name, param in expert.named_parameters():
        if ".lora." in name:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("attention_type", EXPERTS)
def test_gqa_forward_shapes(attention_type):
    expert_cls, _ = EXPERTS[attention_type]
    expert = expert_cls(make_config("shared", attention_type, gqa_groups=2)).eval()
    inputs, n_state = make_expert_inputs(attention_type)
    out = expert(*inputs)
    assert out.shape[0] == 2 and out.shape[2] == D
    if attention_type != "self":
        assert out.shape[1] == n_state


def test_gqa_reduces_attention_params():
    full = CrossAttentionActionExpert(make_config("shared", gqa_groups=1))
    gqa = CrossAttentionActionExpert(make_config("shared", gqa_groups=2))
    assert count_params(gqa) < count_params(full)
    block = gqa.transformer_blocks[0]
    assert isinstance(block.query_self_attention, GroupedQueryAttention)
    assert isinstance(block.cross_attention, GroupedQueryAttention)
    assert block.query_self_attention.num_kv_heads == HEADS // 2


def test_blocks_default_to_per_block():
    for block in (
        TransformerBlock(D, HEADS, FF, norm="adaptive", dropout_p=0.0),
        CrossAttentionTransformerBlock(D, HEADS, FF, norm="adaptive"),
    ):
        assert block.adaln_mode == "per_block"
        norm = block.pre_norm if hasattr(block, "pre_norm") else block.q_norm_1
        assert hasattr(norm, "modulation")


def test_config_validation():
    with pytest.raises(ValueError, match="adaln_mode"):
        make_config("bogus")
    with pytest.raises(ValueError, match="norm"):
        RhoConfig(device="cpu", norm="rms", attention_type="cross", adaln_mode="shared")
    with pytest.raises(ValueError, match="adaln_lora_rank"):
        make_config("shared_lora", adaln_lora_rank=0)
    with pytest.raises(ValueError, match="gqa_groups"):
        make_config("shared", gqa_groups=0)
    with pytest.raises(ValueError, match="divide"):
        make_config("shared", num_heads=6, gqa_groups=4)


def test_config_round_trip():
    cfg = make_config("shared_lora", adaln_lora_rank=4, gqa_groups=2)
    encoded = draccus.encode(cfg)
    assert encoded["adaln_mode"] == "shared_lora"
    assert encoded["adaln_lora_rank"] == 4
    assert encoded["gqa_groups"] == 2
    # draccus cannot decode feature_dict=None; drop it as eval config loading does
    encoded.pop("feature_dict")
    decoded = draccus.decode(PolicyConfig, encoded)
    assert decoded.adaln_mode == "shared_lora"
    assert decoded.adaln_lora_rank == 4
    assert decoded.gqa_groups == 2

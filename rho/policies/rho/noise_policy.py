"""Optional learned noise samplers for the flow policy.

The flow sampler integrates an ODE from an initial noise tensor. By default that tensor is Gaussian
(``FlowMatchingModel.sample_noise``), which is standard evaluation. A *noise policy* predicts it from
the observation instead, steering the policy without changing a single policy weight.

This is OFF by default. Set ``config.noise_policy`` to enable it; leave it ``None`` and the flow model
behaves exactly as before.

A noise policy is an ``nn.Module`` member of the flow model, so it is saved and restored with the
model's ``state_dict`` - one checkpoint carries both the policy and its sampler.

Contract
    forward(embed, mask, state, shape) -> (B, chunk, action_dim) noise
      embed : (B, S, D) prefix hidden state from the policy's own VLM forward
      mask  : (B, S)    validity mask for that prefix
      state : (B, T, D_state) the padded observation state the sampler already receives
      shape : the noise shape the Gaussian path would have produced
"""

from __future__ import annotations

import torch
import torch.nn as nn

NOISE_REGISTRY: dict[str, type[nn.Module]] = {}


def register_noise_policy(name: str):
    def deco(cls):
        NOISE_REGISTRY[name] = cls
        return cls

    return deco


def build_noise_policy(name: str | None, **kwargs) -> nn.Module | None:
    """Return None when no noise policy is configured (the default, Gaussian)."""
    if not name:
        return None
    if name not in NOISE_REGISTRY:
        raise KeyError(f"unknown noise_policy {name!r}; registered: {sorted(NOISE_REGISTRY)}")
    return NOISE_REGISTRY[name](**kwargs)


def masked_mean(embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a prefix hidden state over valid tokens -> (B, D)."""
    m = mask.to(embed.dtype).unsqueeze(-1)
    return (embed * m).sum(1) / m.sum(1).clamp(min=1)


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(1024, 1024, 1024), magnitude=None):
        super().__init__()
        self.magnitude = magnitude
        layers, d = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        w = self.net(x)
        return torch.tanh(w) * self.magnitude if self.magnitude is not None else w


class _VLMNoiseStudent(nn.Module):
    """Mean-pooled prefix embedding + proprio -> flattened noise."""

    def __init__(
        self,
        emb_dim=2048,
        state_dim=8,
        hidden_dims=(1024, 1024, 1024),
        magnitude=3.0,
        noise_steps=16,
        noise_dim=32,
        dropout=0.0,
    ):
        super().__init__()
        self.state_dim, self.noise_steps, self.noise_dim = state_dim, noise_steps, noise_dim
        self.register_buffer("emb_mean", torch.zeros(emb_dim))
        self.register_buffer("emb_std", torch.ones(emb_dim))
        self.drop = nn.Dropout(dropout)
        self.head = _MLP(emb_dim + state_dim, noise_steps * noise_dim, tuple(hidden_dims), magnitude)

    def forward(self, emb, state8):
        x = torch.cat([(emb.float() - self.emb_mean) / self.emb_std, state8.float()], dim=-1)
        return self.head(self.drop(x)).view(-1, self.noise_steps, self.noise_dim)


@register_noise_policy("vlm_residual")
class VLMResidualNoisePolicy(nn.Module):
    """noise = base(obs) + head(obs), both mean-pooled-prefix MLPs; the base stays frozen.

    The residual form is deliberate: the frozen base anchors the output on observations the correction
    data never covered, which matters most on long-horizon tasks. A single network predicting the
    noise directly is markedly less robust there.
    """

    def __init__(
        self,
        emb_dim=2048,
        state_dim=8,
        hidden_dims=(1024, 1024, 1024),
        base_magnitude=3.0,
        head_magnitude=1.5,
        noise_steps=16,
        noise_dim=32,
        scale=1.0,
    ):
        super().__init__()

        def mk(mag):
            return _VLMNoiseStudent(emb_dim, state_dim, hidden_dims, mag, noise_steps, noise_dim)

        self.base, self.head, self.scale = mk(base_magnitude), mk(head_magnitude), scale
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, embed, mask, state, shape):
        emb = masked_mean(embed, mask).float()
        s = state[:, -1] if state.ndim == 3 else state
        s8 = s.float()[:, : self.base.state_dim]
        with torch.no_grad():
            z = self.base(emb, s8)
        return (z + self.head(emb, s8)) * self.scale


@register_noise_policy("vlm_direct")
class VLMNoisePolicy(nn.Module):
    """noise = head(obs). A single network, no frozen anchor.

    This is what online adaptation trains from scratch on a new task: there is no
    prior noise policy to anchor on, so the residual form has nothing to put in its
    base. Where a base does exist -- a checkpoint already finetuned with a noise
    policy -- prefer ``vlm_residual``, whose frozen base keeps the output sane on
    observations the correction data never covered.
    """

    def __init__(
        self,
        emb_dim=2048,
        state_dim=8,
        hidden_dims=(1024, 1024, 1024),
        magnitude=3.0,
        noise_steps=16,
        noise_dim=32,
        scale=1.0,
    ):
        super().__init__()
        self.head = _VLMNoiseStudent(emb_dim, state_dim, hidden_dims, magnitude, noise_steps, noise_dim)
        self.scale = scale

    def forward(self, embed, mask, state, shape):
        emb = masked_mean(embed, mask).float()
        s = state[:, -1] if state.ndim == 3 else state
        return self.head(emb, s.float()[:, : self.head.state_dim]) * self.scale

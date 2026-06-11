"""Deterministic noise policy for FlowDAgger.

Unlike NoiseActor (TanhNormal, trained via SAC), this is a plain MLP that
emits a single point estimate of the noise vector. Trained via MSE against
expert noise w*, where w* is obtained by inverting human-intervention
actions through the base flow ODE (see rho.hil.noise_inverse_map).

Ported from:
  - hil_dsrl/hil_dsrl/train_dagger_noise.py:48     (PyTorch reference)
  - hil_dsrl_pi0/.../flowdagger/steering_policy.py (JAX reference, deterministic
                                                    via dist.mode())
"""

from collections.abc import Sequence

import torch
import torch.nn as nn


class DeterministicNoisePolicy(nn.Module):
    """MLP that predicts a noise vector w in R^noise_dim from an obs embedding.

    Args:
        obs_dim: observation embedding dimension (output of ObsEncoder).
        noise_dim: total flattened noise dimension
            (= noise_action_steps * noise_action_dim).
        hidden_dims: MLP hidden sizes.
        use_layer_norm: insert LayerNorm after each hidden linear layer.
        activation: hidden activation (Tanh in both references).
        magnitude: if not None, output is squashed to [-magnitude, magnitude]
            via a final tanh + scale. None disables the squash (matches the
            PyTorch reference; supervision keeps w in range).
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        use_layer_norm: bool = True,
        activation: type = nn.Tanh,
        magnitude: float | None = None,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.magnitude = magnitude

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, noise_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_emb: (B, obs_dim) observation embedding.

        Returns:
            (B, noise_dim) predicted noise vector.
        """
        w = self.net(obs_emb)
        if self.magnitude is not None:
            w = torch.tanh(w) * self.magnitude
        return w

    @torch.no_grad()
    def get_action(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass; named to mirror NoiseActor.get_action."""
        return self.forward(obs_emb)

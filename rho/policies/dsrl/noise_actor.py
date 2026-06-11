"""Noise-space actor for DSRL-SAC.

Predicts noise vectors z that steer the frozen base policy:
  action = base_policy.sample_actions(obs, noise=z)

Ported from dsrl_pi0 JAX implementation to PyTorch.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class TanhNormal:
    """Tanh-squashed Normal distribution scaled to [-magnitude, magnitude].

    Provides reparameterized sampling with correct log-probability computation
    accounting for the tanh Jacobian correction.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, magnitude: float = 2.0):
        self.mean = mean
        self.log_std = log_std
        self.std = log_std.exp()
        self.magnitude = magnitude
        self._normal = Normal(mean, self.std)

    def sample_and_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample with log probability.

        Returns:
            sample: (B, dim) in [-magnitude, magnitude]
            log_prob: (B,) scalar log probability per sample
        """
        # Reparameterized sample from base Normal
        z = self._normal.rsample()

        # Tanh squash
        tanh_z = torch.tanh(z)

        # Scale to [-magnitude, magnitude]
        sample = tanh_z * self.magnitude

        # Log probability with tanh Jacobian correction
        # log p(a) = log p(z) - sum(log(1 - tanh(z)^2)) - sum(log(magnitude))
        log_prob = self._normal.log_prob(z)  # (B, dim)
        # Numerically stable: log(1 - tanh^2) = 2 * (log(2) - z - softplus(-2*z))
        log_prob = log_prob - 2.0 * (torch.log(torch.tensor(2.0, device=z.device)) - z - F.softplus(-2.0 * z))
        log_prob = log_prob - torch.log(torch.tensor(self.magnitude, device=z.device))
        log_prob = log_prob.sum(dim=-1)  # (B,)

        return sample, log_prob

    def mode(self) -> torch.Tensor:
        """Deterministic action: tanh(mean) * magnitude."""
        return torch.tanh(self.mean) * self.magnitude


class NoiseActor(nn.Module):
    """MLP actor that outputs noise vectors for DSRL.

    Input: obs_embedding (B, obs_dim)
    Output: TanhNormal distribution over noise_dim = n_action_steps * noise_action_dim

    Args:
        obs_dim: observation embedding dimension
        noise_dim: total noise dimension (flattened)
        hidden_dims: MLP hidden layer sizes
        magnitude: noise magnitude bound
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        magnitude: float = 2.0,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.magnitude = magnitude

        # MLP backbone
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(in_dim, noise_dim)
        self.log_std_head = nn.Linear(in_dim, noise_dim)

        # Initialize output heads with small weights
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs_emb: torch.Tensor) -> TanhNormal:
        """
        Args:
            obs_emb: (B, obs_dim) observation embedding

        Returns:
            TanhNormal distribution over (B, noise_dim)
        """
        h = self.backbone(obs_emb)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return TanhNormal(mean, log_std, self.magnitude)

    def get_action(
        self, obs_emb: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get noise action from observation embedding.

        Args:
            obs_emb: (B, obs_dim) observation embedding
            deterministic: if True, use mode (no sampling)

        Returns:
            noise: (B, noise_dim) noise vector
            log_prob: (B,) log probability (None if deterministic)
        """
        dist = self.forward(obs_emb)
        if deterministic:
            return dist.mode(), None
        noise, log_prob = dist.sample_and_log_prob()
        return noise, log_prob

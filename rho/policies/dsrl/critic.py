"""Critic networks for DSRL-SAC.

Q-network ensemble that evaluates (obs_embedding, noise) pairs.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q-network: MLP mapping (obs_emb, noise) -> scalar Q-value.

    Uses LayerNorm after the first hidden layer for training stability.

    Args:
        obs_dim: observation embedding dimension
        noise_dim: noise vector dimension
        hidden_dims: MLP hidden layer sizes
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim + noise_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            if i == 0:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_emb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_emb: (B, obs_dim)
            noise: (B, noise_dim)

        Returns:
            (B,) Q-values
        """
        x = torch.cat([obs_emb, noise], dim=-1)
        return self.net(x).squeeze(-1)


class CriticEnsemble(nn.Module):
    """Ensemble of independent Q-networks.

    Args:
        obs_dim: observation embedding dimension
        noise_dim: noise vector dimension
        hidden_dims: MLP hidden layer sizes
        num_critics: number of Q-networks in ensemble
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        num_critics: int = 10,
    ):
        super().__init__()
        self.critics = nn.ModuleList([QNetwork(obs_dim, noise_dim, hidden_dims) for _ in range(num_critics)])
        self.num_critics = num_critics

    def forward(self, obs_emb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from all critics.

        Args:
            obs_emb: (B, obs_dim)
            noise: (B, noise_dim)

        Returns:
            (num_critics, B) Q-values
        """
        qs = torch.stack([c(obs_emb, noise) for c in self.critics], dim=0)
        return qs

    def q_min(self, obs_emb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compute min Q-value across ensemble.

        Args:
            obs_emb: (B, obs_dim)
            noise: (B, noise_dim)

        Returns:
            (B,) min Q-values
        """
        return self.forward(obs_emb, noise).min(dim=0).values

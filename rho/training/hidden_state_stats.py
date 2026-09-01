"""Lightweight scalar summaries of the VLM hidden state we feed into the
action expert. Surfaces the magnitude / spread of the extracted layer so we
can see drift over training without having to dump tensors."""

from __future__ import annotations

import torch


@torch.no_grad()
def hidden_state_stats(h: torch.Tensor, prefix: str = "hidden_state") -> dict[str, float]:
    """Return a flat dict of scalar stats for a hidden-state tensor.

    Args:
        h: Tensor shaped (B, S, D) or (B, S, ...). Computed in fp32 for stable
           reduction regardless of input dtype.
        prefix: Wandb key prefix.

    Returns:
        Dict with mean, std, abs_mean, rms, l2_norm_per_token, max_abs.
    """
    h32 = h.detach().float()
    # Per-token L2 over the trailing feature dim, then averaged across batch + seq.
    per_token_l2 = h32.flatten(0, -2).norm(dim=-1) if h32.ndim >= 2 else h32.abs()
    return {
        f"{prefix}/mean": float(h32.mean().item()),
        f"{prefix}/std": float(h32.std().item()),
        f"{prefix}/abs_mean": float(h32.abs().mean().item()),
        f"{prefix}/rms": float(h32.pow(2).mean().sqrt().item()),
        f"{prefix}/l2_norm_per_token": float(per_token_l2.mean().item()),
        f"{prefix}/max_abs": float(h32.abs().max().item()),
    }

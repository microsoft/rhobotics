"""Online adaptation: improve a policy by steering its sampling noise, not its weights.

The policy is frozen. A small noise policy learns to predict the initial noise the flow
sampler integrates, which is enough to change what the policy does without touching a single
policy weight -- so the prior is preserved exactly. The noise policy is a submodule of the
flow model, so it trains in place and is saved by the normal checkpoint path: the result is an
ordinary checkpoint that evaluates like any other.

The loop, per episode:

  1. Roll out. At each query point the noise policy proposes the noise; the frozen policy
     denoises it into an action chunk.
  2. An expert takes over on a schedule (``InterventionSchedule``) and drives to the end of
     the episode.
  3. Every action actually executed over a query period -- expert or policy -- is collected
     and the whole chunk is inverted back to noise (``rho.online.noise_inversion``).
  4. Those noise targets train the noise policy with an MSE loss.

Two details that matter more than they look:

* **The whole chunk is inverted, not just the expert's steps.** A chunk that straddles a
  takeover has policy actions then expert actions; inverting only the expert tail and padding
  the rest teaches the policy to freeze partway through a chunk. Collect every executed step.
* **Unreliable inversions are dropped.** ``perstep_fp`` reports its round-trip error; a chunk
  whose recovered noise does not decode back to the actions it came from is not a usable
  supervision target, so it is discarded rather than averaged in.

The loop is environment-agnostic: callers supply an env and an expert
(see ``environments/metaworld/online_adapt.py``).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from rho.online.noise_inversion import perstep_fp_noise_map

logger = logging.getLogger(__name__)


@dataclass
class InterventionSchedule:
    """Randomly timed, early-biased expert takeover.

    Per episode, with probability ``beta`` (decayed linearly over episodes), a takeover step
    is drawn uniformly from ``[takeover_min, takeover_max]``; once triggered the expert keeps
    control to the end of the episode.

    Deliberately NOT failure-triggered. Intervening only after things go wrong teaches
    recovery from bad states; intervening at an arbitrary point teaches the policy to avoid
    reaching them.
    """

    beta_start: float = 1.0
    beta_end: float = 0.1
    decay_episodes: int = 2000
    takeover_min: int = 0
    takeover_max: int = 75
    seed: int = 0
    force_immediate: bool = False
    _rng: np.random.RandomState = field(init=False, repr=False)
    _episodes: int = field(default=0, init=False, repr=False)
    _takeover: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.RandomState(self.seed)

    @property
    def beta(self) -> float:
        frac = min(1.0, self._episodes / max(self.decay_episodes, 1))
        return self.beta_start + (self.beta_end - self.beta_start) * frac

    def reset(self) -> None:
        if self.force_immediate:
            self._takeover = 0
        elif self._rng.random() < self.beta:
            self._takeover = self._rng.randint(self.takeover_min, self.takeover_max + 1)
        else:
            self._takeover = None
        self._episodes += 1

    def intervening_at(self, step: int) -> bool:
        return self._takeover is not None and step >= self._takeover


class NoiseTargetBuffer:
    """(observation embedding, noise target) pairs."""

    def __init__(self, capacity: int = 100_000):
        self.obs: deque = deque(maxlen=capacity)
        self.w: deque = deque(maxlen=capacity)

    def add(self, obs_emb: np.ndarray, w: np.ndarray) -> None:
        self.obs.append(np.asarray(obs_emb, dtype=np.float32))
        self.w.append(np.asarray(w, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, len(self.obs), size=min(batch_size, len(self.obs)))
        o = torch.from_numpy(np.stack([self.obs[i] for i in idx])).to(device)
        w = torch.from_numpy(np.stack([self.w[i] for i in idx])).to(device)
        return o, w


@dataclass
class AdaptationConfig:
    bc_lr: float = 1e-4
    bc_batch_size: int = 256
    bc_steps_per_episode: int = 100
    max_steps: int = 20_000
    seed_expert_episodes: int = 10
    query_freq: int = 8
    max_timesteps: int = 200
    fp_per_step: int = 5
    inversion_mse_threshold: float = 1e-3
    eval_episodes: int = 30
    eval_interval: int = 2000
    seed: int = 42


def summarize_inversions(diags: list[dict]) -> dict:
    """Aggregate inversion health.

    ``rt_*`` says whether the inversion solved at all; ``w_*`` says whether the target it
    produced is representable by the noise policy's bounded output. These fail independently:
    an inversion can round-trip perfectly and still be unreachable because it lands outside
    ``tanh(.) * magnitude``, in which case the loss floors out and no amount of training helps.
    """
    if not diags:
        return {}
    return {
        "n": len(diags),
        "dropped": int(sum(1 for d in diags if not d["kept"])),
        "rt_mean": float(np.mean([d["rt"] for d in diags])),
        "rt_max": float(np.max([d["rt"] for d in diags])),
        "w_mean": float(np.mean([d["w_mean"] for d in diags])),
        "w_p99": float(np.mean([d["w_p99"] for d in diags])),
        "w_max": float(np.max([d["w_max"] for d in diags])),
        "over_bound": float(np.mean([d["over"] for d in diags])),
    }


class OnlineAdapter:
    """Frozen policy + the noise policy being trained over it."""

    def __init__(self, policy_interface, policy, cfg: AdaptationConfig, device="cuda"):
        self.pi = policy_interface
        self.policy = policy
        self.flow_model = getattr(getattr(policy, "model", policy), "flow_model", policy)
        self.cfg = cfg
        self.device = device
        self.chunk = self.flow_model.config.chunk_size
        self.noise_dim = self.flow_model.config.max_action_dim

        # The noise policy is the flow model's own submodule, so it trains in place and is
        # saved by the normal checkpoint path -- nothing to fold in afterwards.
        self.noise_policy = getattr(self.flow_model, "noise_policy", None)
        if self.noise_policy is None:
            raise ValueError(
                "the policy has no noise_policy submodule; set config.noise_policy "
                "(e.g. 'vlm_direct') before building the policy"
            )
        for p in self.policy.parameters():
            p.requires_grad_(False)
        for p in self.noise_policy.parameters():
            p.requires_grad_(True)
        self.noise_policy.train()
        self.opt = torch.optim.Adam(self.noise_policy.parameters(), lr=cfg.bc_lr)

    # -- observation plumbing -------------------------------------------------

    def _prepare(self, obs: dict, action=None):
        o = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        if action is not None:
            o["action"] = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        o = self.pi.process_observation(o, process_action=action is not None)
        b = self.policy.consolidate_images(o)
        image, _ = self.policy.prepare_image(b)
        return (
            image,
            self.policy.prepare_prompt(b),
            self.policy.prepare_state(b),
            self.policy.prepare_action(b) if action is not None else None,
        )

    @torch.no_grad()
    def embed(self, obs: dict):
        """Return ``(prefix, mask, state)`` -- the conditioning the noise policy consumes.

        This is the policy's OWN prefix forward, the same one the sampler would run, so the
        noise policy sees exactly what it will see at serving time.
        """
        image, prompt, state, _ = self._prepare(obs)
        embed, mask = self.flow_model.get_image_text_hidden_state(image, prompt, image_mask=None)
        return embed, mask, state

    @torch.no_grad()
    def embed_for_buffer(self, obs: dict) -> np.ndarray:
        """Pooled conditioning stored as the BC input: ``[masked_mean(prefix), state]``.

        Pooling here rather than storing the full prefix keeps the buffer small, and is
        lossless for this noise policy because it mean-pools its input anyway.
        """
        embed, mask, state = self.embed(obs)
        m = mask.to(embed.dtype).unsqueeze(-1)
        pooled = (embed * m).sum(1) / m.sum(1).clamp(min=1)
        s = state[:, -1] if state.ndim == 3 else state
        s = s.float()[:, : self.noise_policy.head.state_dim]
        return torch.cat([pooled.float(), s], dim=-1).cpu().numpy()[0]

    @torch.no_grad()
    def propose_noise(self, obs: dict) -> np.ndarray:
        embed, mask, state = self.embed(obs)
        shape = (state.shape[0], self.chunk, self.noise_dim)
        return self.noise_policy(embed, mask, state, shape).float().cpu().numpy()[0]

    @torch.no_grad()
    def act(self, obs: dict, noise: np.ndarray) -> np.ndarray:
        o = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        return self.pi.get_action_chunk(o, noise=noise[None]).float().cpu().numpy()[0]

    @torch.no_grad()
    def invert(self, obs: dict, executed: np.ndarray):
        """Executed action chunk -> noise target, with health diagnostics."""
        image, prompt, state, a_target = self._prepare(obs, action=executed)
        precomputed = self.flow_model.get_image_text_hidden_state(image, prompt, image_mask=None)
        w, err = perstep_fp_noise_map(
            self.flow_model, state, a_target, precomputed, fp_per_step=self.cfg.fp_per_step
        )
        w_np = w.float().cpu().numpy()[0]
        a = np.abs(w_np)
        # magnitude lives on the student's MLP (noise_policy.head.head), not the student.
        # Reaching only one level deep silently yields None -> inf -> "nothing is over bound".
        bound = getattr(getattr(getattr(self.noise_policy, "head", None), "head", None),
                        "magnitude", None)
        bound = float("inf") if bound is None else float(bound)
        return w_np, {
            "rt": float(err.float().cpu().numpy().reshape(-1)[0]),
            "w_mean": float(a.mean()),
            "w_p99": float(np.percentile(a, 99)),
            "w_max": float(a.max()),
            "over": float((a > bound).mean()),
        }

    # -- training -------------------------------------------------------------

    def calibrate_embedding_stats(self, buf: NoiseTargetBuffer) -> dict:
        """Set the noise policy's ``emb_mean`` / ``emb_std`` from collected embeddings.

        ``_VLMNoiseStudent`` normalizes its input as ``(emb - emb_mean) / emb_std``, and those
        buffers initialise to 0 and 1. Rho's prefix hidden states have a per-dimension scale far
        from unit (std of order 10), so leaving them uncalibrated feeds the trunk inputs an order
        of magnitude too large and the BC loss plateaus regardless of learning rate. Call this
        once, after the seed episodes, before training.
        """
        head = self.noise_policy.head
        emb_dim = head.head.net[0].in_features - head.state_dim
        obs = np.stack(list(buf.obs))[:, :emb_dim]
        mean = torch.from_numpy(obs.mean(0)).to(self.device)
        std = torch.from_numpy(obs.std(0)).clamp(min=1e-3).to(self.device)
        with torch.no_grad():
            head.emb_mean.copy_(mean)
            head.emb_std.copy_(std)
        return {"n": int(obs.shape[0]),
                "emb_mean_abs": float(mean.abs().mean()),
                "emb_std_mean": float(std.mean())}

    def bc_step(self, buf: NoiseTargetBuffer) -> float:
        o, w = buf.sample(self.cfg.bc_batch_size, self.device)
        # the buffer stores [pooled prefix, state]; the noise policy pools internally, so feed
        # the pooled vector back as a length-1 sequence with a full mask.
        emb_dim = o.shape[-1] - self.noise_policy.head.state_dim
        pooled, state = o[:, :emb_dim], o[:, emb_dim:]
        pred = self.noise_policy(
            pooled.unsqueeze(1), torch.ones(o.shape[0], 1, device=o.device), state, None
        )
        loss = torch.nn.functional.mse_loss(pred.reshape(o.shape[0], -1), w.reshape(o.shape[0], -1))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

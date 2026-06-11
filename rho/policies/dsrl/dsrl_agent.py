"""DSRL-SAC Agent: combines encoder, actor, critic, and temperature for SAC training.

The agent predicts noise vectors z that steer a frozen flow-matching base policy.
Training follows standard SAC with critic ensemble, automatic temperature tuning,
and data augmentation (random crop + color jitter).
"""

import copy
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from rho.policies.dsrl.augmentations import augment_batch
from rho.policies.dsrl.critic import CriticEnsemble
from rho.policies.dsrl.dsrl_config import DSRLConfig
from rho.policies.dsrl.encoder import ObsEncoder, VLMObsEncoder
from rho.policies.dsrl.noise_actor import NoiseActor

logger = logging.getLogger(__name__)


class DSRLSACAgent:
    """DSRL-SAC agent with encoder, noise actor, critic ensemble, and temperature.

    Implements the full SAC training loop operating in noise space:
    - Encode observations with a lightweight CNN
    - Actor predicts noise z for the base policy
    - Critic evaluates (obs_emb, z) pairs
    - Temperature auto-tunes to maintain target entropy

    Args:
        config: DSRLConfig with all hyperparameters
        device: torch device for training
    """

    def __init__(self, config: DSRLConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        noise_dim = config.noise_dim
        obs_dim = config.obs_dim

        # Encoder
        if config.encoder_type == "vlm":
            self.encoder = VLMObsEncoder(
                latent_dim=config.latent_dim,
                state_dim=config.state_dim if config.include_state else 0,
            ).to(device)
        else:
            in_channels = 3 * config.num_cameras
            self.encoder = ObsEncoder(
                encoder_type=config.encoder_type,
                in_channels=in_channels,
                image_size=config.image_size,
                latent_dim=config.latent_dim,
                state_dim=config.state_dim if config.include_state else 0,
                norm_type=config.encoder_norm,
                use_spatial_softmax=config.use_spatial_softmax,
                softmax_temperature=config.softmax_temperature,
            ).to(device)

        # Actor
        self.actor = NoiseActor(
            obs_dim=obs_dim,
            noise_dim=noise_dim,
            hidden_dims=config.hidden_dims,
            magnitude=config.noise_magnitude,
        ).to(device)

        # Critic ensemble + target
        self.critic = CriticEnsemble(
            obs_dim=obs_dim,
            noise_dim=noise_dim,
            hidden_dims=config.hidden_dims,
            num_critics=config.num_critics,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # Temperature (learnable log_alpha)
        self.log_alpha = torch.tensor(
            [float(config.init_temperature)], device=device, dtype=torch.float32
        ).log()
        self.log_alpha.requires_grad_(True)

        # Target entropy
        if config.target_entropy == "auto":
            self.target_entropy = -noise_dim / 2.0
        else:
            self.target_entropy = float(config.target_entropy)

        # Optimizers
        # Actor and encoder share an optimizer (encoder trains end-to-end with actor)
        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": config.actor_lr},
                {"params": self.encoder.parameters(), "lr": config.encoder_lr},
            ]
        )

        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.critic.parameters(), "lr": config.critic_lr},
                {"params": self.encoder.parameters(), "lr": config.encoder_lr},
            ]
        )

        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config.temp_lr)

        # Training state
        self.train_step = 0

        logger.info(
            f"DSRLSACAgent initialized: noise_dim={noise_dim}, obs_dim={obs_dim}, "
            f"encoder={config.encoder_type}, critics={config.num_critics}, "
            f"target_entropy={self.target_entropy:.1f}"
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature value."""
        return self.log_alpha.exp()

    def encode_obs(self, images: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Encode raw images (+ optional state) to observation embedding.

        Args:
            images: (B, C, H, W) float tensor in [0, 1]
            state: (B, state_dim) optional robot state

        Returns:
            (B, obs_dim) observation embedding
        """
        return self.encoder(images, state=state)

    @torch.no_grad()
    def predict_noise(
        self,
        images: torch.Tensor,
        state: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Full inference: encode obs -> actor -> noise vector.

        Args:
            images: (B, C, H, W) float tensor in [0, 1]
            state: (B, state_dim) optional
            deterministic: use mode instead of sampling

        Returns:
            (B, n_action_steps, noise_action_dim) noise vector
        """
        self.encoder.eval()
        self.actor.eval()
        obs_emb = self.encode_obs(images, state)
        noise, _ = self.actor.get_action(obs_emb, deterministic=deterministic)
        # Reshape from flat to (B, n_action_steps, noise_action_dim)
        B = noise.shape[0]
        noise = noise.view(B, self.config.noise_action_steps, self.config.noise_action_dim)
        return noise

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Full SAC update step with UTD ratio.

        Performs utd_ratio critic updates, then 1 actor + 1 temperature update.

        Args:
            batch: dict from replay buffer sample

        Returns:
            Dict of training metrics
        """
        self.encoder.train()
        self.actor.train()
        self.critic.train()

        cfg = self.config
        info = {}

        # Augment images
        images, next_images = augment_batch(
            batch["images"],
            batch["next_images"],
            crop_padding=cfg.random_crop_padding,
            use_color_jitter=cfg.color_jitter,
            aug_next=cfg.aug_next,
        )

        state = batch.get("state")
        next_state = batch.get("next_state")
        noise = batch["noise"]
        reward = batch["reward"]
        discount = batch["discount"]
        mask = batch["mask"]

        # Encode observations (for critic updates, detach encoder for efficiency)
        # We encode once and reuse for all UTD critic updates
        with torch.no_grad():
            obs_emb_frozen = self.encode_obs(images, state)
            next_obs_emb_frozen = self.encode_obs(next_images, next_state)

        # Critic updates (UTD ratio)
        for _i in range(cfg.utd_ratio):
            critic_info = self._update_critic(
                obs_emb_frozen, noise, reward, next_obs_emb_frozen, mask, discount
            )
            self._soft_update_target()

        info.update(critic_info)

        # Actor update (1x) — encoder participates in actor gradient
        obs_emb = self.encode_obs(images, state)
        actor_info, log_prob = self._update_actor(obs_emb)
        info.update(actor_info)

        # Temperature update (1x)
        temp_info = self._update_temperature(log_prob)
        info.update(temp_info)

        # Noise, encoder, and reward stats
        with torch.no_grad():
            info["noise_norm"] = noise.norm(dim=-1).mean().item()
            info["noise_abs_mean"] = noise.abs().mean().item()
            info["reward_mean"] = reward.mean().item()

            # Per-dimension noise utilization
            noise_dim_std = noise.std(dim=0)
            info["noise_dim_std"] = noise_dim_std.mean().item()
            info["noise_dim_utilization"] = (noise_dim_std > 0.01).float().mean().item()

            # Per-step noise structure
            w = noise.view(-1, cfg.noise_action_steps, cfg.noise_action_dim)
            step_norms = w.norm(dim=-1)  # (B, steps)
            info["noise_step_std"] = step_norms.std(dim=-1).mean().item()

            # Encoder representation quality
            emb_std = obs_emb.std(dim=0)
            info["obs_emb_norm"] = obs_emb.norm(dim=-1).mean().item()
            info["obs_emb_std"] = emb_std.mean().item()
            info["obs_emb_dead_dims"] = (emb_std < 1e-4).sum().item()
            if obs_emb.shape[0] >= 2:
                info["obs_emb_cosine_sim"] = (
                    F.cosine_similarity(obs_emb[::2], obs_emb[1::2], dim=-1).mean().item()
                )

        self.train_step += 1
        return info

    def _update_critic(
        self,
        obs_emb: torch.Tensor,
        noise: torch.Tensor,
        reward: torch.Tensor,
        next_obs_emb: torch.Tensor,
        mask: torch.Tensor,
        discount: torch.Tensor,
    ) -> dict[str, float]:
        """SAC critic update with clipped double-Q target.

        Args:
            obs_emb: (B, obs_dim) current observation embeddings
            noise: (B, noise_dim) noise vectors taken
            reward: (B,) rewards
            next_obs_emb: (B, obs_dim) next observation embeddings
            mask: (B,) Bellman backup mask — 0.0 at episode end or takeover boundary
            discount: (B,) discount factors

        Returns:
            Dict of critic metrics
        """
        with torch.no_grad():
            # Sample next noise from current actor
            next_dist = self.actor(next_obs_emb)
            next_noise, next_log_prob = next_dist.sample_and_log_prob()

            # Target Q-value
            target_qs = self.target_critic(next_obs_emb, next_noise)  # (num_critics, B)
            if self.config.critic_reduction == "mean":
                target_q = target_qs.mean(dim=0)  # (B,)
            else:
                target_q = target_qs.min(dim=0).values  # (B,)

            if self.config.backup_entropy:
                target_q = target_q - self.alpha.detach() * next_log_prob

            # mask subsumes (1-done): 0.0 at episode end AND at takeover boundary
            target_q = reward + discount * mask * target_q

        # Critic loss: MSE over ensemble
        qs = self.critic(obs_emb, noise)  # (num_critics, B)
        td_errors = (qs.mean(dim=0) - target_q).abs()
        critic_loss = ((qs - target_q.unsqueeze(0)) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float("inf"))
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": qs.mean().item(),
            "q_max": qs.max().item(),
            "q_min": qs.min().item(),
            "q_spread": (qs.max(dim=0).values - qs.min(dim=0).values).mean().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_std": target_q.std().item(),
            "td_error_mean": td_errors.mean().item(),
            "td_error_max": td_errors.max().item(),
            "mask_mean": mask.mean().item(),
            "reward_std": reward.std().item(),
            "reward_min": reward.min().item(),
            "reward_max": reward.max().item(),
            "critic_grad_norm": critic_grad_norm.item(),
            # Next-state critic target diagnostics
            "next_q_pi": target_qs.min(dim=0).values.mean().item(),
            "next_log_probs": next_log_prob.mean().item(),
            "target_actor_entropy": -next_log_prob.mean().item(),
            "next_actions_mean": next_noise.mean().item(),
            "next_actions_std": next_noise.std().item(),
            "next_actions_min": next_noise.min().item(),
            "next_actions_max": next_noise.max().item(),
        }

    def _update_actor(self, obs_emb: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
        """SAC actor update: maximize Q - alpha * log_prob.

        Args:
            obs_emb: (B, obs_dim) observation embeddings (with grad through encoder)

        Returns:
            Tuple of (metrics dict, log_prob tensor for temperature update)
        """
        dist = self.actor(obs_emb)
        noise, log_prob = dist.sample_and_log_prob()

        # Q-value of actor's chosen noise
        qs = self.critic(obs_emb.detach(), noise)  # (num_critics, B)
        q = qs.mean(dim=0) if self.config.critic_reduction == "mean" else qs.min(dim=0).values

        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float("inf"))
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), float("inf"))
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "entropy": -log_prob.mean().item(),
            "q_pi": q.mean().item(),
            "actor_noise_norm": noise.detach().norm(dim=-1).mean().item(),
            "q_ensemble_std": qs.std(dim=0).mean().item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "encoder_grad_norm": encoder_grad_norm.item(),
            "actor_mean_norm": dist.mean.norm(dim=-1).mean().item(),
            "actor_mean_abs": dist.mean.abs().mean().item(),
            "actor_log_std_mean": dist.log_std.mean().item(),
            "actor_log_std_min": dist.log_std.min().item(),
            "actor_log_std_max": dist.log_std.max().item(),
        }, log_prob.detach()

    def _update_temperature(self, log_prob: torch.Tensor) -> dict[str, float]:
        """SAC temperature: maintain target entropy.

        Args:
            log_prob: (B,) log probabilities from actor

        Returns:
            Dict of temperature metrics
        """
        temp_loss = -(self.log_alpha * (log_prob.mean() + self.target_entropy).detach()).mean()

        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.temp_optimizer.step()

        return {
            "temperature": self.alpha.item(),
            "temperature_loss": temp_loss.item(),
        }

    def _soft_update_target(self):
        """Polyak averaging: target = tau * current + (1-tau) * target."""
        tau = self.config.tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters(), strict=False):
            tp.data.lerp_(p.data, tau)

    def get_policy_params(self) -> dict[str, torch.Tensor]:
        """Get actor + encoder state dicts for publishing to robot.

        Returns:
            Dict with 'actor' and 'encoder' state dicts (on CPU).
        """
        return {
            "actor": {k: v.cpu() for k, v in self.actor.state_dict().items()},
            "encoder": {k: v.cpu() for k, v in self.encoder.state_dict().items()},
        }

    def load_policy_params(self, params: dict[str, torch.Tensor]):
        """Load actor + encoder params (e.g., from hot-reload).

        Args:
            params: Dict with 'actor' and 'encoder' state dicts
        """
        if "actor" in params:
            self.actor.load_state_dict(params["actor"])
        if "encoder" in params:
            self.encoder.load_state_dict(params["encoder"])

    def save_checkpoint(self, path: str):
        """Save full agent checkpoint.

        Args:
            path: file path to save checkpoint
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "temp_optimizer": self.temp_optimizer.state_dict(),
                "train_step": self.train_step,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load full agent checkpoint.

        Args:
            path: file path to load checkpoint from
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.temp_optimizer.load_state_dict(ckpt["temp_optimizer"])
        self.train_step = ckpt["train_step"]
        logger.info(f"Loaded checkpoint from {path} (step {self.train_step})")

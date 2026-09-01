"""
DSRL Policy: Inference-side wrapper for Diffusion Steering via RL.

Combines a trained noise actor π^W with a frozen base policy (phi4mm / pi0)
accessed through a policy server.

Inference pipeline (chunk-noise approach, matching hil_dsrl_pi0 reference):
    1. Encode s → obs_emb   (ObsEncoder: CNN + spatial softmax + bottleneck + state)
    2. π^W(obs_emb) → w     (NoiseActor: flat noise_dim = n_action_steps * noise_action_dim)
    3. Reshape w → (n_action_steps, noise_action_dim)
    4. Tile last step → (chunk_size, noise_action_dim)
    5. Call policy server → action a ∈ R^{chunk_size × action_dim}  (π_dp(s, w))

Uses the same encoder/actor architecture as DSRLSACAgent so that trained
weights can be loaded directly. π^W weights are hot-swapped in real-time
via ParamSubscriber (ZMQ SUB) whenever the DSRLTrainer publishes an update.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from rho.hil.param_subscriber import ParamSubscriber
from rho.policies.base import PolicyConfig, PreTrainedPolicy
from rho.policies.dsrl.encoder import ObsEncoder
from rho.policies.dsrl.noise_actor import NoiseActor

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@PolicyConfig.register_subclass("dsrl")
@dataclass
class DSRLPolicyConfig(PolicyConfig):
    """
    Configuration for the DSRL inference-side policy wrapper.

    Network hyperparameters must match the DSRLConfig used during training
    so that trained weights can be loaded.
    """

    # ── Checkpoint ────────────────────────────────────────────────────────────
    noise_actor_checkpoint: str = ""  # Path to DSRLTrainer checkpoint .pt

    # ── Base policy (π_dp, held in-process) ────────────────────────────────
    base_policy_name: str = "rho"

    # ── Problem dimensions ────────────────────────────────────────────────────
    noise_action_steps: int = 16  # Query freq: actor predicts this many steps
    noise_action_dim: int = 32  # Per-step noise dimension
    action_chunk_size: int = 32  # Full chunk size for π_dp (tile last step)
    noise_magnitude: float = 1.0  # b_W: noise clamped to [-b_W, b_W]

    # ── Encoder (must match DSRLConfig / DSRLSACAgent) ────────────────────────
    encoder_type: str = "small"  # "small" or "resnet34"
    encoder_norm: str = "group"
    latent_dim: int = 50
    use_spatial_softmax: bool = True
    softmax_temperature: float = -1.0
    image_size: int = 128
    num_cameras: int = 2
    state_dim: int = 7
    include_state: bool = True

    # ── Actor (must match DSRLConfig / DSRLSACAgent) ──────────────────────────
    hidden_dims: tuple[int, ...] = (256, 256, 256)

    # ── Live weight updates from DSRLTrainer ──────────────────────────────────
    subscribe_to_trainer: bool = True
    trainer_host: str = "localhost"
    param_sub_port: int = 5556

    # ── Camera keys (must list image keys in Transition.obs) ─────────────────
    image_keys: list[str] = field(default_factory=list)

    @property
    def noise_dim(self) -> int:
        """Total flattened noise dimension (actor output size)."""
        return self.noise_action_steps * self.noise_action_dim

    @property
    def obs_dim(self) -> int:
        """Observation embedding dimension."""
        dim = self.latent_dim
        if self.include_state:
            dim += self.state_dim
        return dim


# ── Policy ────────────────────────────────────────────────────────────────────


class DSRLPolicy(PreTrainedPolicy):
    """
    DSRL inference-side policy using chunk-noise approach.

    Uses the same ObsEncoder + NoiseActor architecture as DSRLSACAgent
    so that trained weights can be loaded directly.
    """

    config_class = DSRLPolicyConfig
    name = "dsrl"

    def __init__(
        self,
        config: DSRLPolicyConfig,
        base_policy: PreTrainedPolicy,
        dataset_stats=None,
    ):
        super().__init__(config)
        self.config = config
        self.device = config.device
        self.image_keys: list[str] = config.image_keys
        self.base_policy = base_policy

        # ── Encoder (same as DSRLSACAgent) ────────────────────────────────
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
        ).to(self.device)

        # ── Actor (same as DSRLSACAgent) ──────────────────────────────────
        self.actor = NoiseActor(
            obs_dim=config.obs_dim,
            noise_dim=config.noise_dim,
            hidden_dims=config.hidden_dims,
            magnitude=config.noise_magnitude,
        ).to(self.device)

        # ── Load checkpoint ──────────────────────────────────────────────
        if config.noise_actor_checkpoint:
            self._load_checkpoint(config.noise_actor_checkpoint)

        # ── Live parameter subscriber ────────────────────────────────────
        self._param_sub: ParamSubscriber | None = None
        if config.subscribe_to_trainer:
            self._param_sub = ParamSubscriber(host=config.trainer_host, port=config.param_sub_port)
            self._param_sub.set_callback(self._on_params_update)
            self._param_sub.start()
            logger.info(
                f"[DSRLPolicy] Subscribed to trainer at {config.trainer_host}:{config.param_sub_port}"
            )

        self.eval()
        logger.info(
            f"[DSRLPolicy] Initialized: noise_dim={config.noise_dim} "
            f"({config.noise_action_steps}x{config.noise_action_dim}), "
            f"chunk_size={config.action_chunk_size}, b_W={config.noise_magnitude}"
        )

    # ── Checkpoint loading ─────────────────────────────────────────────────────

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
        logger.info(f"[DSRLPolicy] Loaded checkpoint <- {path}")

    # ── Live weight update callback ────────────────────────────────────────────

    def _on_params_update(self, params: dict) -> None:
        """Hot-swap π^W weights published by DSRLTrainer."""
        if "encoder" in params:
            self.encoder.load_state_dict({k: v.to(self.device) for k, v in params["encoder"].items()})
        if "actor" in params:
            self.actor.load_state_dict({k: v.to(self.device) for k, v in params["actor"].items()})
        logger.debug("[DSRLPolicy] π^W weights updated from trainer.")

    # ── Observation encoding ───────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_obs(self, obs: dict) -> torch.Tensor:
        """Encode obs to (1, obs_dim) embedding for the noise actor.

        PolicyInterface produces (B, T, C, 448, 448) tensors for the base
        rhoalpha model; this wrapper's encoder is single-shot and expects
        (B, 3*num_cameras, image_size, image_size). So take the latest
        timestep, resize, and concatenate across cameras.
        """
        target_size = self.config.image_size
        cam_tensors = []
        for key in self.image_keys:
            img = obs[key]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.copy()).float()
            if img.ndim == 5:
                img = img[:, -1]
            elif img.ndim == 4:
                pass
            elif img.ndim == 3 and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif img.ndim == 3:
                img = img.unsqueeze(0)
            else:
                raise ValueError(f"obs[{key}] unexpected shape {tuple(img.shape)}")

            img = img.to(self.device).float()
            if img.max() > 1.0 + 1e-3:
                img = img / 255.0
            if img.shape[-1] != target_size or img.shape[-2] != target_size:
                img = torch.nn.functional.interpolate(
                    img,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            cam_tensors.append(img)

        images = torch.cat(cam_tensors, dim=1)

        state = None
        if self.config.include_state:
            state = obs.get("observation.state", obs.get("state"))
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state.copy()).float()
            if torch.is_tensor(state):
                if state.ndim == 3:
                    state = state[:, -1]
                elif state.ndim == 1:
                    state = state.unsqueeze(0)
                state = state.to(self.device).float()[:, : self.config.state_dim]

        return self.encoder(images, state=state)

    # ── Action sampling ────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict,
        noise: Tensor | None = None,  # ignored; π^W generates noise internally
    ) -> dict:
        """
        Generate a steered action chunk via DSRL.

        Chunk-noise approach (matching hil_dsrl_pi0 reference):
        1. Actor outputs flat noise (noise_action_steps * noise_action_dim)
        2. Reshape to (noise_action_steps, noise_action_dim)
        3. Tile last step to fill (action_chunk_size, noise_action_dim)
        4. Feed to π_dp via policy server

        Returns:
            {"actions": (1, C, action_dim) tensor, "tactile_action": None}
        """
        self.eval()
        cfg = self.config

        embedding = self._encode_obs(batch)  # (1, obs_dim)

        # Actor: (1, noise_dim) flat
        noise_flat, _ = self.actor.get_action(embedding, deterministic=False)
        noise_np = noise_flat.squeeze(0).cpu().numpy()  # (noise_dim,)

        # Reshape to (noise_action_steps, noise_action_dim)
        w_chunk = noise_np.reshape(cfg.noise_action_steps, cfg.noise_action_dim)

        # Stash the chunk noise in obs for replay (travels with Transition)
        batch["__dsrl_noise_used__"] = noise_np

        # Tile last step to fill action_chunk_size (handled below by feeding
        # the tiled w_tiled directly to the in-process base policy).
        remaining = cfg.action_chunk_size - cfg.noise_action_steps
        if remaining > 0:
            last_step = w_chunk[-1:]  # (1, noise_action_dim)
            w_tiled = np.concatenate(
                [w_chunk, np.repeat(last_step, remaining, axis=0)], axis=0
            )  # (action_chunk_size, noise_action_dim)
        else:
            w_tiled = w_chunk[: cfg.action_chunk_size]

        # Feed tiled noise to the in-process base policy.
        noise_t = torch.from_numpy(w_tiled).to(self.device).unsqueeze(0)
        base_out = self.base_policy.sample_actions(batch, noise=noise_t)
        return {
            "actions": base_out["actions"],
            "tactile_action": base_out.get("tactile_action"),
        }

    def forward(self, batch: dict) -> tuple:
        raise NotImplementedError(
            "DSRLPolicy is an inference-only wrapper. RL training is handled by DSRLTrainer."
        )

    def compute_loss(self, batch: dict) -> tuple:
        raise NotImplementedError("DSRLPolicy doesn't compute loss; DSRLTrainer does.")

    def get_optim_params(self) -> dict:
        # Trainer owns its own optimizer. Wrapper exposes no train-time params.
        return {}

    def select_action(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        return self.sample_actions(batch, noise=noise)["actions"]

    def reset(self) -> None:
        pass

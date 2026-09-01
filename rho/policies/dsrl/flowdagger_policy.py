"""FlowDAggerPolicy: deterministic-noise inference-side wrapper.

Companion to DSRLPolicy. Differences:
  - Actor is a DeterministicNoisePolicy (single MLP head, no TanhNormal sampling).
  - No "stochastic exploration" -- given an obs, the same noise is emitted.
  - Weights are trained via supervised MSE in FlowDAggerTrainer, not SAC.

Everything downstream (base policy server round-trip, ParamSubscriber hot
reload, chunk-noise tiling) is identical to DSRLPolicy.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from rho.hil.param_subscriber import ParamSubscriber
from rho.policies.base import PolicyConfig, PreTrainedPolicy
from rho.policies.dsrl.encoder import ObsEncoder
from rho.policies.dsrl.noise_policy import DeterministicNoisePolicy

logger = logging.getLogger(__name__)


@PolicyConfig.register_subclass("flowdagger")
@dataclass
class FlowDAggerPolicyConfig(PolicyConfig):
    """Inference-side config for FlowDAgger. Mirrors DSRLPolicyConfig."""

    # ── Checkpoint ────────────────────────────────────────────────────────────
    noise_policy_checkpoint: str = ""

    # ── Base policy (the frozen flow model held in-process) ──────────────────
    base_policy_name: str = "rho"

    # ── Problem dimensions (must match the frozen base policy) ───────────────
    # Base policy expects initial_noise of shape (chunk_size, max_action_dim).
    # The actor outputs noise_action_steps * noise_action_dim flat scalars;
    # we reshape to (noise_action_steps, noise_action_dim) and tile the last
    # row to fill action_chunk_size when noise_action_steps < action_chunk_size.
    noise_action_steps: int = 16
    noise_action_dim: int = 32
    action_chunk_size: int = 16
    # If not None, output is squashed to [-magnitude, magnitude] via final tanh.
    # 3.0 matches the action_magnitude used in Mike's metaworld flowdagger runs;
    # the inverter clamps w* to [-b_W, b_W] (default b_W=2.0) so a slightly
    # larger inference bound gives the actor headroom.
    noise_magnitude: float | None = 3.0

    # ── Encoder (must match the encoder used to train the noise policy) ─────
    encoder_type: str = "small"  # "small" or "resnet34" or "vlm" (VLMObsEncoder)
    encoder_norm: str = "group"
    latent_dim: int = 256  # matches metaworld flowdagger
    use_spatial_softmax: bool = True
    softmax_temperature: float = -1.0
    image_size: int = 128
    num_cameras: int = 2
    state_dim: int = 7
    include_state: bool = True

    # ── Actor MLP ─────────────────────────────────────────────────────────────
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    use_layer_norm: bool = True

    # ── Live weight updates from FlowDAggerTrainer ────────────────────────────
    subscribe_to_trainer: bool = True
    trainer_host: str = "localhost"
    param_sub_port: int = 5556

    # ── Camera keys (must list image keys in Transition.obs) ─────────────────
    image_keys: list[str] = field(default_factory=list)

    @property
    def noise_dim(self) -> int:
        return self.noise_action_steps * self.noise_action_dim

    @property
    def obs_dim(self) -> int:
        dim = self.latent_dim
        if self.include_state:
            dim += self.state_dim
        return dim


class FlowDAggerPolicy(PreTrainedPolicy):
    """FlowDAgger inference-side policy. Deterministic noise prediction.

    Wraps an in-process base flow policy: sample_actions encodes obs, predicts
    a noise vector, and seeds the base policy's flow ODE with it. The
    FlowDAggerTrainer shares the same base policy for inverse_noise_map and
    pushes updated actor/encoder weights via ParamSubscriber.
    """

    config_class = FlowDAggerPolicyConfig
    name = "flowdagger"

    def __init__(
        self,
        config: FlowDAggerPolicyConfig,
        base_policy: PreTrainedPolicy,
        dataset_stats=None,
    ):
        super().__init__(config)
        self.config = config
        self.device = config.device
        self.image_keys: list[str] = config.image_keys
        self.base_policy = base_policy

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

        self.actor = DeterministicNoisePolicy(
            obs_dim=config.obs_dim,
            noise_dim=config.noise_dim,
            hidden_dims=config.hidden_dims,
            use_layer_norm=config.use_layer_norm,
            magnitude=config.noise_magnitude,
        ).to(self.device)

        if config.noise_policy_checkpoint:
            self._load_checkpoint(config.noise_policy_checkpoint)

        self._param_sub: ParamSubscriber | None = None
        if config.subscribe_to_trainer:
            self._param_sub = ParamSubscriber(host=config.trainer_host, port=config.param_sub_port)
            self._param_sub.set_callback(self._on_params_update)
            self._param_sub.start()
            logger.info(
                f"[FlowDAggerPolicy] Subscribed to trainer at {config.trainer_host}:{config.param_sub_port}"
            )

        self.eval()
        logger.info(
            f"[FlowDAggerPolicy] Initialized: noise_dim={config.noise_dim} "
            f"({config.noise_action_steps}x{config.noise_action_dim}), "
            f"chunk_size={config.action_chunk_size}, magnitude={config.noise_magnitude}"
        )

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
        logger.info(f"[FlowDAggerPolicy] Loaded checkpoint <- {path}")

    def _on_params_update(self, params: dict) -> None:
        """Hot-swap noise-policy weights pushed by the trainer."""
        if "encoder" in params:
            self.encoder.load_state_dict({k: v.to(self.device) for k, v in params["encoder"].items()})
        if "actor" in params:
            self.actor.load_state_dict({k: v.to(self.device) for k, v in params["actor"].items()})
        logger.debug("[FlowDAggerPolicy] noise-policy weights updated from trainer.")

    # ── Observation encoding ──────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_obs(self, obs: dict) -> torch.Tensor:
        # One-shot: log what the encoder actually receives. This is post
        # PolicyInterface.process_observation (keys remapped, transforms run),
        # so it tells us whether image_keys are right and what the encoder
        # sees at run time.
        from rho.policies.dsrl._obs_dump import dump_obs_once

        dump_obs_once(
            "encoder_input",
            obs,
            image_keys=self.image_keys,
            extra_fields={
                "device": str(self.device),
                "config.image_keys": list(self.image_keys),
                "config.num_cameras": self.config.num_cameras,
                "config.image_size": self.config.image_size,
                "config.state_dim": self.config.state_dim,
                "config.include_state": self.config.include_state,
            },
        )

        target_size = self.config.image_size
        cam_tensors = []
        for key in self.image_keys:
            img = obs[key]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.copy()).float()
            # PolicyInterface produces (B, T, C, H, W) for the base rhoalpha
            # model; our encoder is single-shot so take the most recent step.
            # Other layouts (4D batched, 3D unbatched HWC/CHW) handled below.
            if img.ndim == 5:
                img = img[:, -1]  # (B, C, H, W)
            elif img.ndim == 4:
                pass  # (B, C, H, W)
            elif img.ndim == 3 and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> (1,C,H,W)
            elif img.ndim == 3:
                img = img.unsqueeze(0)  # (1, C, H, W)
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

        # Concat cameras along channel dim: (B, 3*num_cameras, H, W)
        images = torch.cat(cam_tensors, dim=1)

        state = None
        if self.config.include_state:
            state = obs.get("observation.state", obs.get("state"))
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state.copy()).float()
            if torch.is_tensor(state):
                # Strip time dim if any: (B, T, D) -> (B, D); (D,) -> (1, D)
                if state.ndim == 3:
                    state = state[:, -1]
                elif state.ndim == 1:
                    state = state.unsqueeze(0)
                state = state.to(self.device).float()[:, : self.config.state_dim]

        return self.encoder(images, state=state)

    # ── Action sampling ───────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict,
        noise: Tensor | None = None,  # ignored; noise comes from the actor
    ) -> dict:
        self.eval()
        cfg = self.config

        embedding = self._encode_obs(batch)
        noise_flat = self.actor.get_action(embedding)
        noise_np = noise_flat.squeeze(0).cpu().numpy()

        w_chunk = noise_np.reshape(cfg.noise_action_steps, cfg.noise_action_dim)

        batch["__dsrl_noise_used__"] = noise_np

        remaining = cfg.action_chunk_size - cfg.noise_action_steps
        if remaining > 0:
            last_step = w_chunk[-1:]
            w_tiled = np.concatenate([w_chunk, np.repeat(last_step, remaining, axis=0)], axis=0)
        else:
            w_tiled = w_chunk[: cfg.action_chunk_size]

        # Feed the predicted noise straight to the in-process base flow model.
        noise_t = torch.from_numpy(w_tiled).to(self.device).unsqueeze(0)
        base_out = self.base_policy.sample_actions(batch, noise=noise_t)
        action_tensor = base_out["actions"]
        return {"actions": action_tensor, "tactile_action": base_out.get("tactile_action")}

    def forward(self, batch: dict) -> tuple:
        raise NotImplementedError(
            "FlowDAggerPolicy is an inference-only wrapper. DAgger training is handled by FlowDAggerTrainer."
        )

    def compute_loss(self, batch: dict) -> tuple:
        raise NotImplementedError("FlowDAggerPolicy doesn't compute loss; FlowDAggerTrainer does.")

    def get_optim_params(self) -> dict:
        # Trainer owns its own optimizer over encoder+actor. The wrapper
        # exposes no train-time params.
        return {}

    def select_action(self, batch: dict, noise: Tensor | None = None) -> Tensor:
        # Delegate to sample_actions and return the action chunk tensor.
        return self.sample_actions(batch, noise=noise)["actions"]

    def reset(self) -> None:
        pass

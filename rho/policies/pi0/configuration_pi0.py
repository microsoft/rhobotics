import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES_PREFIX
from rho.common.constants import OBSERVATION_STATE as OBS_STATE
from rho.common.types import FeatureType, PolicyFeature
from rho.models.optimizer import AdamWConfig
from rho.models.schedule import CosineDecayWithWarmupSchedulerConfig
from rho.policies.base import PolicyConfig

logger = logging.getLogger(__name__)


@PolicyConfig.register_subclass("pi0")
@dataclass
class PI0Config(PolicyConfig):
    name: str = "pi0"

    # OpenPI core knobs
    pi05: bool = False
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # OpenPI defaults
    action_dim: int = 32
    action_horizon: int = 50

    # Data interface
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    empty_cameras: int = 0

    # Flow matching / sampling
    num_inference_steps: int = 10
    # Alias for `num_inference_steps` exposed as `num_steps` via property below.
    # rho.hil.noise_inverse_map reads `flow_model.config.num_steps`; rhoalpha
    # names this attribute `num_steps`, so pi0 follows suit through the alias.
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Runtime toggles
    enable_gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    require_transformers_replace: bool = True

    # Tokenizer
    tokenizer_max_length: int = 48
    tokenizer_model_path: str | Path | None = (
        None  # /data/dean/models/paligemma_models/paligemma_tokenizer.model
    )

    # Training presets (match other policies)
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        allowed_dtypes = (torch.bfloat16, torch.float32)
        if not isinstance(self.dtype, torch.dtype) or self.dtype not in allowed_dtypes:
            allowed_str = ", ".join(d.__repr__() for d in allowed_dtypes)
            raise ValueError(
                f"PI0Config.dtype must be one of [{allowed_str}], got {self.dtype!r}. "
                "Set e.g. dtype: bfloat16 or dtype: float32 in your config."
            )

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})"
            )

        # Keep horizons consistent with OpenPI expectations
        self.action_horizon = self.chunk_size
        self.action_dim = self.max_action_dim

        if self.pi05 and self.tokenizer_max_length == 48:
            # In OpenPI, pi05 defaults to max_token_len=200; we map that to tokenizer_max_length.
            self.tokenizer_max_length = 200

        if self.device == "cpu" and self.dtype == torch.bfloat16:
            # CPU bfloat16 is frequently unsupported in practice.
            raise ValueError("PI0Config.dtype cannot be bfloat16 on CPU.")

        if self.optimizer is None:
            self.optimizer = AdamWConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = CosineDecayWithWarmupSchedulerConfig(
                peak_lr=self.optimizer_lr,
                decay_lr=self.scheduler_decay_lr,
                num_warmup_steps=self.scheduler_warmup_steps,
                num_decay_steps=self.scheduler_decay_steps,
            )

    def validate_features(self) -> None:
        # Ensure at least one image feature (OpenPI PI0 expects 3 views; we can pad missing).
        if len(self.image_features) == 0 and self.empty_cameras == 0:
            raise ValueError(
                "PI0 requires at least one visual observation. "
                "Provide image features or set empty_cameras to pad."
            )

        # State/action must exist.
        if self.robot_state_feature is None:
            raise ValueError(f"Missing required feature {OBS_STATE} in feature_dict")
        if self.action_feature is None:
            raise ValueError(f"Missing required feature {ACTION} in feature_dict")

        state_dim = self.robot_state_feature.shape[-1]
        action_dim = self.action_feature.shape[-1]
        if state_dim > self.max_state_dim:
            raise ValueError(f"State dim ({state_dim}) exceeds max_state_dim ({self.max_state_dim}).")
        if action_dim > self.max_action_dim:
            raise ValueError(f"Action dim ({action_dim}) exceeds max_action_dim ({self.max_action_dim}).")

        for i in range(int(self.empty_cameras)):
            key = f"{OBS_IMAGES_PREFIX}.empty_camera_{i}"
            self.feature_dict[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return list(range(-self.n_obs_steps + 1, 1))

    @property
    def action_delta_indices(self) -> list[int] | None:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def num_steps(self) -> int:
        """Alias for `num_inference_steps` to match rhoalpha's naming.

        Used by rho.hil.noise_inverse_map's perstep_fp / euler_reverse paths,
        which read `flow_model.config.num_steps`.
        """
        return self.num_inference_steps

    @property
    def delta_indices_dict(self) -> dict[str, list | None]:
        delta_indices: dict[str, list | None] = {}
        for key in self.feature_dict or {}:
            if key.startswith(ACTION) and self.action_delta_indices is not None:
                delta_indices[key] = self.action_delta_indices
            if key.startswith("observation") and self.observation_delta_indices is not None:
                delta_indices[key] = self.observation_delta_indices
        return delta_indices

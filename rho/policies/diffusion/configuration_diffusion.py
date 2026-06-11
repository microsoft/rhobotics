import logging
from dataclasses import dataclass

from rho.common.constants import ACTION, OBSERVATION_PREFIX
from rho.models.optimizer import AdamConfig
from rho.models.schedule import DiffuserSchedulerConfig
from rho.policies.base import PolicyConfig

logger = logging.getLogger(__name__)


@PolicyConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PolicyConfig):
    """
    Configuration class for DiffusionPolicy.

    This provides the essential parameters needed for the DiffusionPolicy to work
    with basic functionality. Based on the original Diffusion Policy paper.
    """

    name: str = "diffusion"  # Name of the policy

    # Temporal structure - core parameters for diffusion policy
    n_obs_steps: int = 2  # Number of observation steps to look back
    horizon: int = 16  # Number of action steps to predict
    n_action_steps: int = 8  # Number of action steps to execute

    # Vision backbone configuration
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Diffusion U-Net architecture
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise scheduler configuration
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    # Inference settings
    num_inference_steps: int | None = None

    # Training settings
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    @property
    def chunk_size(self) -> int:
        """Alias for horizon, for compatibility with the common policy interface."""
        return self.horizon

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(-self.n_obs_steps + 1, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def delta_indices_dict(self) -> dict[str, list | None]:
        """Build delta indices dict from feature_dict, matching RhoAlphaConfig behavior."""
        delta_indices = {}
        if self.feature_dict is not None:
            for key in self.feature_dict:
                if key.startswith(ACTION) and self.action_delta_indices is not None:
                    delta_indices[key] = self.action_delta_indices
                if key.startswith(OBSERVATION_PREFIX) and self.observation_delta_indices is not None:
                    delta_indices[key] = self.observation_delta_indices
        logger.debug(f"Delta indices: {delta_indices}")
        return delta_indices

    def __post_init__(self):
        logger.debug("DiffusionConfig __postinit__")
        if self.optimizer is None:
            self.optimizer = AdamConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = DiffuserSchedulerConfig(
                name=self.scheduler_name,
                num_warmup_steps=self.scheduler_warmup_steps,
            )
        else:
            self.scheduler_name = self.lr_scheduler.type
            self.scheduler_warmup_steps = getattr(
                self.lr_scheduler, "num_warmup_steps", self.scheduler_warmup_steps
            )

    def validate_features(self) -> None:
        """Validate the feature configuration"""
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def reward_delta_indices(self) -> None:
        return None

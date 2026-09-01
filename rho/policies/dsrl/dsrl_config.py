"""Configuration for DSRL agent (SAC and NA variants)."""

from dataclasses import dataclass


@dataclass
class DSRLConfig:
    """Configuration for DSRL (Diffusion Steering via RL).

    Supports two algorithm variants:
      - "sac": Standard SAC in noise space (Q^W only, no policy server calls).
      - "na":  Noise-Aliased (Algorithm 1) — Q^A + Q^W distillation, requires
               calling π_dp during training via policy server.
    """

    # Algorithm variant
    algorithm: str = "sac"  # "sac" or "na"

    # Noise space (derived from base policy config)
    noise_action_steps: int = 8
    noise_action_dim: int = 32
    noise_magnitude: float = 1.0

    # Encoder
    encoder_type: str = "small"  # "small", "resnet34", or "vlm"
    encoder_norm: str = "group"  # "group", "batch", "layer"
    latent_dim: int = 50
    use_spatial_softmax: bool = True
    softmax_temperature: float = -1.0  # -1 = learnable

    # Image settings
    image_keys: tuple[str, ...] = ("agentview",)
    image_size: int = 128
    num_cameras: int = 1

    # State
    state_dim: int = 0
    include_state: bool = False

    # SAC hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    encoder_lr: float = 3e-4
    discount: float = 0.999
    tau: float = 0.005
    init_temperature: float = 1.0
    target_entropy: str = "auto"  # "auto" = -noise_dim / 2
    backup_entropy: bool = True

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    num_critics: int = 10
    critic_reduction: str = "min"  # "min" (clipped double-Q) or "mean"

    # NA-specific: Q^A critic settings
    action_dim: int = 32  # Per-step action dimension (NA: Q^A value_dim)
    action_chunk_size: int = 50  # Action chunk length C (NA: for flattening actions)
    n_action_critics: int = 2  # Q^A ensemble size (NA only)

    # NA-specific: policy server for calling π_dp during training
    policy_server_url: str = "ws://localhost:8765"

    # Training
    batch_size: int = 256
    utd_ratio: int = 20
    start_training_after: int = 100

    # Data augmentation
    random_crop_padding: int = 4
    color_jitter: bool = True
    aug_next: bool = True

    # Replay buffer
    replay_buffer_capacity: int = 100_000

    # Query frequency
    query_freq: int = 8

    # Rewards (default: -1/0 sparse, matching reference implementation)
    reward_type: str = "sparse"  # "sparse" or "intervention"
    step_reward: float = -1.0
    success_reward: float = 0.0
    intervention_reward: float = -0.5
    reward_takeover_penalty: float = 0.0  # one-time penalty on first takeover step
    reward_expert_bonus: float = 0.0  # per-step bonus during expert control
    intervention_stop_reward: bool = False  # set mask=0 at takeover boundary (Bellman backup cutting)

    # Inverse noise map (for interventions)
    inversion_method: str = "adam"  # "adam", "euler_reverse", "hybrid"
    inverse_map_restarts: int = 3
    inverse_map_steps: int = 20
    inverse_map_lr: float = 0.01
    hybrid_batch_size: int = 8  # mini-batch size for hybrid Adam refinement at episode end

    # Publishing & logging
    publish_interval: int = 50
    log_interval: int = 100
    checkpoint_interval: int = 1000
    log_dir: str = "outputs/dsrl_trainer"

    # Wandb logging
    wandb_enabled: bool = False
    wandb_project: str = "dsrl-sac"
    wandb_entity: str = ""
    wandb_name: str = ""  # Run name (auto-generated if empty)
    wandb_group: str = ""  # Group name for related runs

    # Resume from previous training run
    resume_checkpoint: str = ""  # Path to agent checkpoint (.pt)
    resume_buffer: str = ""  # Path to replay buffer (.npz)

    # Base policy bookkeeping (informational, used by serve_hil)
    base_policy_name: str = "rho"
    base_policy_checkpoint: str = ""

    @property
    def noise_dim(self) -> int:
        """Total flattened noise dimension (noise_action_steps * noise_action_dim)."""
        return self.noise_action_steps * self.noise_action_dim

    @property
    def w_single_dim(self) -> int:
        """Per-step noise dimension for w_single training (paper: train on R^d, tile across chunk)."""
        return self.noise_action_dim

    @property
    def action_value_dim(self) -> int:
        """Flattened action dimension for Q^A critics (NA mode)."""
        return self.action_chunk_size * self.action_dim

    @property
    def obs_dim(self) -> int:
        """Observation embedding dimension fed to actor/critic."""
        dim = self.latent_dim
        if self.include_state:
            dim += self.state_dim
        return dim

    @property
    def effective_discount(self) -> float:
        """Discount adjusted for query frequency."""
        return self.discount**self.query_freq

    def to_policy_config(self):
        """Derive a matching DSRLPolicyConfig for the in-process inference wrapper.

        Imported lazily to avoid a cycle on rho.policies import.
        """
        from rho.policies.dsrl.dsrl_policy import DSRLPolicyConfig

        return DSRLPolicyConfig(
            noise_action_steps=self.noise_action_steps,
            noise_action_dim=self.noise_action_dim,
            action_chunk_size=self.action_chunk_size,
            noise_magnitude=self.noise_magnitude,
            encoder_type=self.encoder_type,
            encoder_norm=self.encoder_norm,
            latent_dim=self.latent_dim,
            use_spatial_softmax=self.use_spatial_softmax,
            softmax_temperature=self.softmax_temperature,
            image_size=self.image_size,
            num_cameras=self.num_cameras,
            state_dim=self.state_dim,
            include_state=self.include_state,
            hidden_dims=self.hidden_dims,
            image_keys=list(self.image_keys),
            subscribe_to_trainer=True,
            trainer_host="localhost",
        )

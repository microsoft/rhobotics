"""Config for FlowDAggerTrainer + FlowDAggerPolicy.

Defaults match Mike's metaworld flowdagger run:
    -m examples.launch_flowdagger
        --inversion_method perstep_fp
        --action_magnitude 3.0
        --latent_dim 256
        --hidden_dims 256 256 256
        --bc_lr 1e-4
        --bc_steps_per_episode 100
        --intervention_mode beta_decay   (β handled robot-side, not here)
        --resize_image 128
        --max_steps 60000

Knobs we deliberately don't bring in:
- --noise_basis_k:    low-rank basis option specific to pi05 setup; not yet ported.
- --query_freq:       robot-side (how often a new noise vector is queried) -- lives
                      in rho_infra HIL agent / action chunk handling, not here.
- --beta_*:           DAgger schedule for synthetic experts; with spacemouse HIL,
                      the human decides when to intervene so we just train on
                      whatever intervention transitions arrive.
"""

from dataclasses import dataclass, field


@dataclass
class FlowDAggerConfig:
    # ── Where the trainer logs go ────────────────────────────────────────────
    log_dir: str = "logs/flowdagger"

    # ── Checkpointing ────────────────────────────────────────────────────────
    # Periodic save every N MSE updates (mirrors DSRLTrainer's pattern).
    # 0 disables periodic saves; the on-stop final save still fires.
    checkpoint_interval: int = 1000
    # Optional path to a checkpoint saved by a prior run; loads encoder/actor/
    # optimizer + counters before training starts so a killed run can resume.
    # The buffer is NOT persisted (it can be 10GB+); expect a brief warm-up
    # while the new run refills enough interventions to hit
    # min_interventions_to_start before BC kicks back in.
    resume_from: str | None = None

    # ── Base policy linkage ──────────────────────────────────────────────────
    base_policy_name: str = "rhoalpha"
    # The trainer needs the loaded base policy in-process for inverse_noise_map.
    # The robot-facing policy server URL is used only if the trainer needs to
    # query the server (not used by flowdagger currently).

    # ── Inversion (perstep_fp is the favourite per Mike) ──────────────────────
    inversion_method: str = "perstep_fp"  # "perstep_fp" | "fixed_point" | "hybrid" | "euler_reverse" | "adam"
    inverse_map_steps: int = (
        3  # = fp_per_step for perstep_fp; = refine_steps for fixed_point; = adam steps for adam
    )
    inverse_map_lr: float = 0.01
    inverse_map_restarts: int = 1
    inverse_map_b_W: float = 3.0  # clamp |w*| during adam inversion; matches noise_magnitude below
    pre_norm_scale: float = 1.0  # multiply expert actions by this before inversion (mirror flowdagger flag)
    # Batch multiple intervention transitions into one inverse_noise_map
    # call. The VLM forward + Euler ODE iterations dominate inversion cost;
    # running many at once amortizes GPU launch + memory traffic. Mirrors
    # the JAX ref's _stack_observations + jnp.concatenate(targets) pattern.
    # 32 is comfortable on an H100-class GPU; bump to 64 if you have memory,
    # drop to 8-16 for smaller cards.
    inversion_batch_size: int = 32
    # Force-flush pending interventions after this many seconds if the batch
    # hasn't filled up (low intervention rate -> avoid stalling forever).
    inversion_flush_timeout_s: float = 5.0

    # ── Noise / chunk dimensions (must match the base policy) ────────────────
    noise_action_steps: int = 16  # tokens of noise the actor predicts per query
    noise_action_dim: int = 32  # per-token noise dim; rhoalpha max_action_dim
    action_chunk_size: int = 16  # base policy chunk_size (tile if larger)
    noise_magnitude: float = 3.0  # final tanh scale on actor output

    # ── Encoder (must match FlowDAggerPolicy on the inference side) ──────────
    encoder_type: str = "small"
    encoder_norm: str = "group"
    latent_dim: int = 256
    use_spatial_softmax: bool = True
    softmax_temperature: float = -1.0
    image_size: int = 128
    num_cameras: int = 2
    state_dim: int = 7
    include_state: bool = True
    # Canonical post-remap image keys. Used at BOTH inference (the noise
    # encoder reads these from PolicyInterface-processed obs) and training
    # (the trainer runs each transition through env + observation_mapping +
    # input_transforms before extracting images). No more wire-side aliases.
    image_keys: list[str] = field(default_factory=list)

    # ── Actor MLP (must match FlowDAggerPolicy) ───────────────────────────────
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    use_layer_norm: bool = True

    # When True, the encoder is set to eval() with all params frozen and
    # excluded from the BC optimizer. Right call for pretrained encoders
    # (resnet, dinov2, siglip, vlm) where we don't have enough intervention
    # data to fine-tune. Leave False for the small-CNN default which trains
    # from scratch alongside the noise actor.
    freeze_encoder: bool = False

    # ── DAgger BC training ───────────────────────────────────────────────────
    bc_lr: float = 1e-4
    bc_batch_size: int = 256
    bc_steps_per_update: int = 100  # MSE grad steps fired per update cycle
    update_every_n_interventions: int = 64
    min_interventions_to_start: int = 32
    # Buffer stores post-normalize float32 images at image_size, so each
    # transition is ~400KB for 2 cams at 128. 25k -> ~10GB; tune as needed.
    # This is the intervention buffer; the autonomous buffer has its own cap.
    buffer_capacity: int = 25_000
    weight_decay: float = 0.0
    max_grad_norm: float = 10.0

    # ── Dual-buffer episode-success gating ───────────────────────────────────
    # Only commit transitions from episodes that the operator marked success=True.
    # Failed episodes (including success=None and e-stop-killed episodes) are
    # dropped entirely. Within a successful episode, intervened transitions go
    # to the intervention buffer; non-intervened ("implicitly approved") go to
    # the autonomous buffer as an anti-drift anchor against correction-only overfit.
    intervention_sample_ratio: float = 0.5  # fraction of each BC batch drawn from intervention buffer
    autonomous_buffer_capacity: int = (
        5_000  # smaller than intervention buffer: autonomous goes stale faster as the policy moves
    )
    autonomous_subsample_every: int = (
        4  # keep every Nth autonomous transition; bounds memory + commit cost on long episodes
    )
    # "sampled_noise" uses transition.noise directly (cheap). "invert_action"
    # inverts transition.action through the base flow (expensive but principled).
    autonomous_target: str = "sampled_noise"
    # log-once when a terminal transition arrives with success=None (likely
    # missing operator annotation)
    warn_on_missing_episode_signal: bool = True

    # ── Param publishing ─────────────────────────────────────────────────────
    # Each "update cycle" = bc_steps_per_update MSE steps. After every cycle
    # we publish the freshly updated {encoder, actor} state dicts.
    publish_after_each_update_cycle: bool = True

    # ── Stop conditions ──────────────────────────────────────────────────────
    max_training_steps: int = 60_000  # cap on total MSE grad steps
    max_episodes: int = -1  # -1 = unbounded

    # ── Logging ──────────────────────────────────────────────────────────────
    log_interval: int = 100  # MSE steps between stdout logs
    wandb_enabled: bool = False
    wandb_project: str = "flowdagger"
    wandb_group: str = ""
    wandb_name: str = ""

    @property
    def noise_dim(self) -> int:
        return self.noise_action_steps * self.noise_action_dim

    @property
    def obs_dim(self) -> int:
        d = self.latent_dim
        if self.include_state:
            d += self.state_dim
        return d

    def to_policy_config(self):
        """Derive a matching FlowDAggerPolicyConfig from this trainer config.

        Same arch knobs on both sides so a single yaml block configures
        the in-process inference wrapper and the trainer consistently.
        Imported lazily to avoid a cycle on rho.policies import.
        """
        from rho.policies.dsrl.flowdagger_policy import FlowDAggerPolicyConfig

        return FlowDAggerPolicyConfig(
            base_policy_name=self.base_policy_name,
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
            use_layer_norm=self.use_layer_norm,
            image_keys=list(self.image_keys),
            subscribe_to_trainer=True,
            trainer_host="localhost",
            # param_sub_port left to FlowDAggerPolicyConfig default (5556);
            # FlowDAggerTrainer publishes there by default too.
        )

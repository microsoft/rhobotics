from dataclasses import dataclass, field
from pathlib import Path

from rho.common.constants import ACTION, OBSERVATION_PREFIX
from rho.models.optimizer import AdamWConfig
from rho.models.schedule import CosineDecayWithWarmupSchedulerConfig
from rho.policies.base import PolicyConfig


@PolicyConfig.register_subclass("rho")
@dataclass
class RhoConfig(PolicyConfig):
    """Configuration for the Rho flow-matching policy."""

    # ── Optional learned noise sampler (OFF by default: None == Gaussian) ──────
    # Name registered in rho.policies.rho.noise_policy.NOISE_REGISTRY, e.g. "vlm_residual".
    # When set, the flow model owns it as a submodule, so it is saved and restored with the
    # model state_dict - one checkpoint carries both the policy and its noise sampler.
    noise_policy: str | None = None
    noise_policy_kwargs: dict = field(default_factory=dict)

    name: str = "rho"
    pretrained_repo_id: str | None = "microsoft/rho-base"

    # Most robot policies expect 256x256 inputs. Set to None only when images
    # should retain their original size.
    resize_imgs_with_padding: tuple[int, int] | None = (256, 256)

    embed_dim: int = 2048
    num_heads: int = 16
    ff_dim: int = 4096
    norm: str = "adaptive"
    hidden_state_idx: int = 14
    num_blocks: int = 12
    max_seq_len: int = 2048 * 3

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 25
    max_state_dim: int = 32
    max_action_dim: int = 32

    num_steps: int = 10
    num_flow_samples: int = 8
    attention_implementation: str = "flash_attention_2"
    attention_type: str = "cross"
    adaln_mode: str = "shared"
    adaln_lora_rank: int = 256
    gqa_groups: int = 4
    cross_attention_dim: int | None = None
    vlm_layer_select: str = "last"
    shared_vlm_projection: bool = False
    kv_pos_encoding: bool = True

    freeze_vlm_backbone: bool = False
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = False
    train_expert_only: bool = True
    enable_gradient_checkpointing: bool = True

    optimizer_lr: float = 1e-4
    optimizer_lr_action_expert: float | None = None
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 240_000
    scheduler_decay_lr: float = 2.5e-6
    time_sampling_strategy: str = "beta"
    dropout_p: float = 0.02
    pos_emb_method: str = "sinusoidal"
    log_hidden_state_stats: bool = True

    @property
    def vlm_backend(self) -> str:
        return "phi5"

    @property
    def vlm_backbone_folder(self) -> str:
        return str(Path(__file__).resolve().parents[2] / "models" / "Phi-4-vision-5B")

    def __post_init__(self) -> None:
        if self.attention_type not in {"self", "cross", "layerwise_cross"}:
            raise ValueError(
                "attention_type must be one of {'self', 'cross', 'layerwise_cross'}, "
                f"got {self.attention_type!r}"
            )
        if self.adaln_mode not in {"per_block", "shared", "shared_lora"}:
            raise ValueError(
                f"adaln_mode must be one of {{'per_block', 'shared', 'shared_lora'}}, got {self.adaln_mode!r}"
            )
        if self.adaln_mode in {"shared", "shared_lora"} and self.norm != "adaptive":
            raise ValueError(f"adaln_mode={self.adaln_mode!r} requires norm='adaptive'")
        if self.adaln_lora_rank < 1:
            raise ValueError("adaln_lora_rank must be at least 1")
        if self.gqa_groups < 1 or self.num_heads % self.gqa_groups:
            raise ValueError("gqa_groups must be positive and divide num_heads")
        if self.embed_dim % self.num_heads:
            raise ValueError("num_heads must divide embed_dim")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.num_flow_samples < 1:
            raise ValueError("num_flow_samples must be at least 1")
        if not 0.0 <= self.dropout_p <= 1.0:
            raise ValueError("dropout_p must be between 0.0 and 1.0")

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

        if self.train_expert_only:
            self.freeze_vlm_backbone = True
            self.freeze_vision_encoder = True

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(-self.n_obs_steps + 1, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def delta_indices_dict(self) -> dict[str, list[int]]:
        return {
            key: self.action_delta_indices if key.startswith(ACTION) else self.observation_delta_indices
            for key in self.feature_dict
            if key.startswith(ACTION) or key.startswith(OBSERVATION_PREFIX)
        }

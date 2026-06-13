import logging
import os
from dataclasses import dataclass
from pathlib import Path

from rho.common.constants import ACTION, OBSERVATION_PREFIX, OBSERVATION_TACTILE
from rho.common.types import FeatureType, PolicyFeature, TrainingMode
from rho.models.optimizer import AdamWConfig
from rho.models.schedule import CosineDecayWithWarmupSchedulerConfig
from rho.policies.base import PolicyConfig

logger = logging.getLogger(__name__)

NEXT_REWARD_KEY = "next.reward"

# Phi4MM uses 448x448 images (matches SigLIP's native training resolution)
PHI4MM_IMAGE_SIZE = 448


# heavily based on configuartion_pi0.py
@PolicyConfig.register_subclass("rhoalpha")
@PolicyConfig.register_subclass("rhoalpha_tactile")
@PolicyConfig.register_subclass("phi4mm")
@PolicyConfig.register_subclass("phi4mm_tactile")
@dataclass
class RhoAlphaConfig(PolicyConfig):
    name: str = "rhoalpha"  # Name of the policy

    # VLM backend selection: "phi4mm", "phi5", "qwen25vl", or "qwen3vl"
    vlm_backend: str = "phi4mm"
    # Path/ID for the VLM backbone model (required for phi5, optional for phi4mm)
    vlm_backbone_folder: str | None = None
    # HuggingFace model name for Qwen backends (e.g. "Qwen/Qwen3-VL-4B-Instruct")
    vlm_model_name: str | None = None
    # Image resize dimensions for backends that need it (e.g. Qwen: (224, 224))
    resize_imgs_with_padding: tuple[int, int] | None = None
    # configs for the Phi4MMFlowMatching class
    embed_dim: int = 1024
    num_heads: int = 8
    ff_dim: int = 4096  # updated
    norm: str = "rms"  # ["adaptive", "rms"]
    # state_dim: int = 14 # TODO populate from the dataset

    # select which layer embedding from phi4mm to use
    # 0 is the after the encoders, but before the first transformer block
    # 1 is the first transformer block, 2 is the second transformer block, etc
    # 33 is the last transformer block, which is the same as the final embedding
    hidden_state_idx: int = 0  # which layer embedding to select [0,33]
    num_blocks: int = 32
    max_seq_len: int = 2048 * 3

    ### below are copied from PI0Config
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_tactile_dim: int = 36  # with temp (26), without temp (18) per arm
    max_action_dim: int = 32

    # Use HD (High Definition) multi-crop image processing
    # When True: Uses dynamic_hd with multiple crops (higher quality, more memory)
    # When False: Uses single-scale images only (faster, less memory, like phi4mm_fast)
    use_hd_transform: bool = True
    dynamic_hd: int = 36  # Number of crops when use_hd_transform=True (max 36 for Phi-4)

    # When use_hd_transform=False, whether to add separator tokens to match HD structure
    # - sub_GN: row separators (end of each row in patch grid) - adds 16 tokens
    # - glb_GN: global separator (marks image as "global", no sub-crops) - adds 1 token
    # Total: 256 + 16 + 1 = 273 tokens per image at 448x448 (vs 545 for full HD)
    add_separator_tokens: bool = True

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    # proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "flash_attention_2"  # or fa2, flex
    attention_type: str = "self"  # ["self", "cross", "layerwise_cross"]
    # Auto-populated from VLM backbone hidden size for layerwise_cross
    cross_attention_dim: int | None = None
    vlm_layer_select: str = (
        "last"  # ["first", "last", "stride"] — which VLM layers to use for layerwise_cross
    )
    shared_vlm_projection: bool = (
        False  # If True, use one shared projection for all layers instead of per-layer
    )
    kv_pos_encoding: bool = True  # If False, skip positional encoding on KV tokens (VLM has RoPE baked in)
    use_full_causal_mask: bool = False  # If True, use full causal instead of prefix-LM for FAST model

    # Finetuning settings
    freeze_vlm_backbone: bool = False
    freeze_vision_encoder: bool = True
    freeze_vision_projector: bool = False
    freeze_vision_transformer: bool = False
    freeze_language_model: bool = False

    train_expert_only: bool = True
    enable_gradient_checkpointing: bool = True
    # train_state_proj: bool = True # fixed to always to be true for now.

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 240_000
    scheduler_decay_lr: float = 2.5e-6

    drop_n_last_frames: int = 0

    # LoRA - Phi4MM's built-in LoRAs
    # Phi4MM loads vision LoRA (r=256) and audio LoRA (r=320) by default.
    # Audio components are always removed (saves ~600M params, not needed for robotics).
    # Vision LoRA: if False, stays active (trainable). If True, merged into base weights.
    merge_vision_lora: bool = True

    # LoRA - Your custom action LoRA
    use_action_lora: bool = False
    freeze_action_lora: bool = False  # LoRA contributes to forward but doesn't receive gradients
    action_lora_init_from_scratch: bool = True  # If True, initialize LoRA weights fresh (ignore checkpoint)
    action_lora_r: int = 32
    action_lora_alpha: int = 64
    action_lora_modules: list[str] = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    action_lora_dropout: float = 0.01
    action_lora_start: int = 0
    action_lora_end: int = 14

    # Flowmatching configs

    # ["beta", "beta_reverse", "uniform", "logit_normal"]
    # Time step sampling distribution for flow matching
    time_sampling_strategy: str = "beta"

    # TODO: Add EMA

    # ActionExpert Configs
    dropout_p: float = 0.1
    pos_emb_method: str = "sinusoidal"  # ["learned", "sinusoidal"]

    # Multi-head training support
    training_modes: list[TrainingMode] | None = (
        None  # List of training modes to support (defaults to [ROBOT_FLOWMATCH])
    )
    knowledge_insulation_alpha: float = 1.0  # Weight for flow loss in knowledge insulation (alpha in Eq. 4)

    # TACTILE ONLY CONFIGS
    # Mask to select a subset of the tactile signal before passing to the policy.
    # None: pass all values through. "anyskin": first 15 values. "anypressure": last 4 values.
    tactile_mask: str | None = None
    use_tactile_history: bool = True
    tactile_history_time: int = 2  # number of seconds of history to use
    tactile_history_length: int = 10  # number of tactile readings in the history
    robot_hz: int = 15  # robot frequency
    tactile_encoder: str = "linear"  # ["linear", "mlp_relu", "mlp_gelu", "mlp_silu"]
    use_tactile_head: bool = False
    tactile_head_loss: str = "mse"  # ["mse", "flow"]
    tactile_loss_beta: float = 0.1
    combine_action_tactile_head: bool = False  # this overrides use_tactile_head
    baku_weight_init: bool = False  # use the weight init from BAKU

    monkeypatch_siglip_encoder: bool = False  # use the siglip encoder from monkeypatch

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        _valid_backends = ("phi4mm", "phi5", "qwen25vl", "qwen3vl")
        if self.vlm_backend not in _valid_backends:
            raise ValueError(f"vlm_backend must be one of {_valid_backends}, got '{self.vlm_backend}'")

        # For phi5, default vlm_backbone_folder to the bundled model directory
        if self.vlm_backend == "phi5":
            if self.vlm_backbone_folder is None:
                default_model_dir = Path(__file__).resolve().parents[3] / "rho" / "models" / "Phi-4-vision-5B"
                self.vlm_backbone_folder = str(default_model_dir)
                logger.info(f"vlm_backbone_folder not set; defaulting to '{self.vlm_backbone_folder}'")
            folder = Path(self.vlm_backbone_folder).expanduser()
            if not folder.exists():
                # Fallback to VLM_BACKBONE_FOLDER env var
                env_val = os.environ.get("VLM_BACKBONE_FOLDER", "").strip()
                if env_val:
                    env_folder = Path(env_val).expanduser()
                    if env_folder.exists():
                        logger.info(
                            f"vlm_backbone_folder '{self.vlm_backbone_folder}' not found, "
                            f"falling back to VLM_BACKBONE_FOLDER env var: '{env_folder}'"
                        )
                        self.vlm_backbone_folder = str(env_folder)
                    else:
                        logger.warning(
                            f"vlm_backbone_folder '{self.vlm_backbone_folder}' not found "
                            f"and VLM_BACKBONE_FOLDER env var path '{env_folder}' also does not exist."
                        )
                else:
                    logger.warning(
                        f"vlm_backbone_folder '{self.vlm_backbone_folder}' does not exist. "
                        "Set VLM_BACKBONE_FOLDER env var to provide a fallback path, or "
                        "create a symlink to the model directory."
                    )

        # Map freeze_vision_model (Qwen naming) to freeze_vision_encoder (canonical)
        if hasattr(self, "freeze_vision_model") and self.freeze_vision_model:
            self.freeze_vision_encoder = True

        # Set default training modes if not specified
        if self.training_modes is None:
            self.training_modes = [TrainingMode.ROBOT_FLOWMATCH]

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. "
                "It is not ported yet in LeRobot."
            )

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
        else:
            self.scheduler_name = self.lr_scheduler.type
            self.scheduler_warmup_steps = getattr(
                self.lr_scheduler, "num_warmup_steps", self.scheduler_warmup_steps
            )
            if isinstance(self.lr_scheduler, CosineDecayWithWarmupSchedulerConfig):
                self.scheduler_decay_steps = getattr(
                    self.lr_scheduler, "num_decay_steps", self.scheduler_decay_steps
                )
                self.scheduler_decay_lr = getattr(self.lr_scheduler, "decay_lr", self.scheduler_decay_lr)

        if self.train_expert_only:
            self.freeze_vlm_backbone = True
            self.freeze_vision_encoder = True

        if self.freeze_vision_encoder:
            self.freeze_vision_projector = True
            self.freeze_vision_transformer = True

        if self.use_tactile_history:
            assert self.tactile_history_time * self.robot_hz % self.tactile_history_length == 0, (
                "tactile_history_time multiplied by robot_hz (tactile frequency) must be "
                "divisible by tactile_history_length"
            )
            self.reshaped_tactile_dim = self.max_tactile_dim * self.tactile_history_length
            logger.info(f"Using tactile history. Reshaped tactile dim: {self.reshaped_tactile_dim}")
        else:
            self.reshaped_tactile_dim = self.max_tactile_dim

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #   raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    @property
    def tactile_observation_delta_indices(self) -> None:
        if self.use_tactile_history:
            num_timesteps = self.tactile_history_time * self.robot_hz
            step = num_timesteps // self.tactile_history_length

            # create and reverse list
            return list(range(0, -1 * num_timesteps, -1 * step))[::-1]
        return None

    @property
    def observation_delta_indices(self) -> None:
        return list(range(-self.n_obs_steps + 1, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def delta_indices_dict(self) -> dict[str, list | None]:
        delta_indices = {}
        for key in self.feature_dict:
            if key == NEXT_REWARD_KEY and self.reward_delta_indices is not None:
                delta_indices[key] = self.reward_delta_indices
            if key.startswith(ACTION) and self.action_delta_indices is not None:
                delta_indices[key] = self.action_delta_indices
            if key.startswith(OBSERVATION_PREFIX) and self.observation_delta_indices is not None:
                delta_indices[key] = self.observation_delta_indices
            if key == OBSERVATION_TACTILE and self.tactile_observation_delta_indices is not None:
                delta_indices[key] = self.tactile_observation_delta_indices
        logger.debug(f"Delta indices: {delta_indices}")
        return delta_indices

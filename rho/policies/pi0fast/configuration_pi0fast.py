#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from rho.common.types import FeatureType, PolicyFeature
from rho.models.optimizer import AdamWConfig
from rho.models.schedule import CosineDecayWithWarmupSchedulerConfig
from rho.policies.base import PolicyConfig


@PolicyConfig.register_subclass("pi0fast")
@dataclass
class PI0FASTConfig(PolicyConfig):
    name: str = "pi0fast"  # Name of the policy

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 10
    n_action_steps: int = 5

    # Normalization mapping - keeping the structure from LeRobot
    # Note: May need to adapt this to rho's normalization approach
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "QUANTILE",
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    max_decoding_steps: int = 256
    fast_skip_tokens: int = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
    max_input_seq_len: int = 256  # 512

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # Adding this from phi4mm config

    # Frozen parameters
    freeze_vision_encoder: bool = True
    freeze_lm_head: bool = True
    freeze_vision_projector: bool = False  # Adding from phi4mm pattern
    freeze_vision_transformer: bool = False  # Adding from phi4mm pattern

    # Training presets
    train_expert_only: bool = True  # Adding from phi4mm pattern
    enable_gradient_checkpointing: bool = True  # Adding from phi4mm pattern

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5

    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    checkpoint_path: str = None

    padding_side: str = "right"

    precision: str = "bfloat16"
    grad_clip_norm: float = 1

    # Allows padding/truncation of generated action tokens during detokenization to ensure decoding.
    # In the original version, tensors of 0s were generated if shapes didn't match for stable decoding.
    relaxed_action_decoding: bool = True

    drop_n_last_frames: int = 0  # Adding from phi4mm pattern

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. It is not ported yet."
            )

        # Initialize optimizer if not provided
        if self.optimizer is None:
            self.optimizer = AdamWConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )

        # Initialize scheduler if not provided
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

    def validate_features(self) -> None:
        """Validate and add empty camera features if needed."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    @property
    def observation_delta_indices(self) -> None:
        """PI0FAST doesn't use observation deltas."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """Return indices for action deltas."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """PI0FAST doesn't use reward deltas."""
        return None

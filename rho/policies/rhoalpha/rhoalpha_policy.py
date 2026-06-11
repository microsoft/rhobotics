"""
Phi4MM Policy for interfacing with rho infrastructure.

This module provides the policy class that:
- Wraps RhoAlphaRoboticsModel for all robotics training types
- Prepares batches for model input (images, prompts, states, actions)
- Manages action queues for environment rollouts
- Handles image consolidation and padding
- Routes forward/sample_actions calls to the underlying model
"""

import logging
from collections import deque

import numpy as np
import torch
from lerobot.policies.utils import populate_queues
from torch import Tensor

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_ENVIRONMENT_STATE as OBS_ENV
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES
from rho.common.constants import OBSERVATION_LANG as OBS_TASK
from rho.common.constants import OBSERVATION_STATE as OBS_ROBOT
from rho.common.transforms import ResizeWithPadding, build_key_padding_transform
from rho.common.types import TrainingMode
from rho.policies.base import PreTrainedPolicy
from rho.policies.rhoalpha.backbone import create_backbone_adapter
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_robotics import RhoAlphaRoboticsModel

logger = logging.getLogger(__name__)

OBS_IMAGES_IS_PAD = f"{OBS_IMAGES}_is_pad"


class RhoAlphaPolicy(PreTrainedPolicy):
    """
    Phi4MM Policy class for interfacing with LeRobot.

    This class wraps RhoAlphaRoboticsModel, which can handle multiple training modes
    (flow matching, autoregressive, knowledge insulation, etc.) and routes to the
    appropriate model based on config.training_modes and batch["training_mode"].

    Handles:
    - Data preparation and batching
    - Image consolidation and preprocessing
    - Action queue management for rollouts
    - Routing to underlying model's forward/sample_actions
    """

    config_class = RhoAlphaConfig
    name = "rhoalpha"

    def _remap_backwards_compatible_keys(self, state_dict: dict) -> dict:
        """Remap legacy phi4mm checkpoint keys to new format.

        The phi4mm refactor changed the model structure:
        - Old: Phi4MMPolicy.model = Phi4MMFlowMatching (directly has vlm_backbone, action_expert, etc.)
        - New: Phi4MMPolicy.model = Phi4MMRoboticsModel.flow_model = Phi4MMFlowMatchingModel

        Old keys: model.vlm_backbone.*, model.action_expert.*, etc.
        New keys: model.flow_model.vlm_backbone.*, model.flow_model.action_expert.*, etc.

        Note: Tactile checkpoints (Phi4MMTactilePolicy) still use the old structure directly
        and should NOT be remapped.
        """
        has_old_vlm_format = any(k.startswith("model.vlm_backbone.") for k in state_dict)
        has_new_format = any(k.startswith("model.flow_model.") for k in state_dict)
        has_tactile = any(k.startswith("model.tactile_projector") for k in state_dict)

        if not has_old_vlm_format or has_new_format or has_tactile:
            return state_dict

        logger.info("Detected legacy phi4mm checkpoint format, remapping keys...")

        legacy_prefixes = [
            "model.vlm_backbone.",
            "model.vlm_projector.",
            "model.state_projector.",
            "model.action_in_proj.",
            "model.action_head.",
            "model.action_time_mlp_in.",
            "model.action_time_mlp_out.",
            "model.time_mlp_in.",
            "model.time_mlp_out.",
            "model.action_expert.",
        ]

        remapped_dict = {}
        remapped_count = 0

        for key, value in state_dict.items():
            new_key = key
            for prefix in legacy_prefixes:
                if key.startswith(prefix):
                    new_key = key.replace("model.", "model.flow_model.", 1)
                    remapped_count += 1
                    break
            remapped_dict[new_key] = value

        logger.info(f"Remapped {remapped_count} keys from legacy format")
        return remapped_dict

    def __init__(
        self,
        config: RhoAlphaConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration (with training_modes list)
            dataset_stats: Dataset statistics for normalization
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.device = config.device

        # Instantiate unified robotics model that handles all training types
        self.model = RhoAlphaRoboticsModel(config)
        self.model.print_freezing_status()

        # Track whether the backbone was created without pretrained weights.
        # If so, load_from_pretrained() must be called before forward().
        model_backend = getattr(self.model.flow_model, "_backend", None)
        self._has_uninitialized_backbone = model_backend is not None and getattr(
            model_backend, "_backbone_has_uninitialized_weights", False
        )

        # Backend adapter for prompt formatting (lightweight, no model weights)
        self._backend = create_backbone_adapter(config)

        self.n_action_steps = config.n_action_steps

        # Queues for rollout
        self._queues = None

        self.input_padding_transform = build_key_padding_transform(
            self.config.feature_dict,
            self.config.delta_indices_dict,
        )

        self._resize_transform = None
        if self.config.resize_imgs_with_padding is not None:
            from rho.common.transforms import ResizeWithPadding

            h, w = self.config.resize_imgs_with_padding
            self._resize_transform = ResizeWithPadding(h, w)

        self.reset()

    def reset(self):
        """Reset action queues (called when environment resets)."""
        self._queues = {
            OBS_ROBOT: deque(maxlen=self.config.n_obs_steps),
            OBS_TASK: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
            self._queues[OBS_IMAGES_IS_PAD] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV] = deque(maxlen=self.config.n_obs_steps)

    def get_optim_params(self) -> dict:
        """Get parameters for optimizer."""
        return self.parameters()

    def consolidate_images(self, batch):
        """
        Consolidate all image features into a single OBS_IMAGES key.

        Handles both 4D (batch, C, H, W) and 5D (batch, 1, C, H, W) image inputs.
        """
        num_img_features = len(self.config.image_features)
        if num_img_features > 0:
            first_key = list(self.config.image_features.keys())[0]
            if batch[first_key].ndim == 4:
                # Images: (batch_size, 3, H, W)
                batch[OBS_IMAGES] = torch.cat(
                    [batch[key].unsqueeze(1) for key in self.config.image_features],
                    dim=1,
                )
            elif batch[first_key].ndim == 5:
                # Images: (batch_size, 1, 3, H, W)
                batch[OBS_IMAGES] = torch.cat([batch[key] for key in self.config.image_features], dim=1)

            # Consolidate padding masks
            if f"{first_key}_is_pad" in batch:
                batch[OBS_IMAGES_IS_PAD] = torch.cat(
                    [batch[f"{key}_is_pad"].unsqueeze(1) for key in self.config.image_features],
                    dim=1,
                )
            else:
                # No padding info available - assume all valid
                batch[OBS_IMAGES_IS_PAD] = torch.zeros(
                    batch[OBS_IMAGES].shape[0],
                    batch[OBS_IMAGES].shape[1],
                    1,
                    dtype=torch.long,
                    device=batch[OBS_IMAGES].device,
                )
        else:
            raise ValueError(
                f"No image features found in configuration. Expected mappings with {OBS_IMAGES} prefix."
            )
        return batch

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        Select a single action for environment execution.

        Manages action queue - only calls select_actions when queue is empty.
        """
        self.eval()
        batch_size = 1
        if isinstance(batch[OBS_TASK], (list, tuple, torch.Tensor, np.ndarray)):
            batch_size = len(batch[OBS_TASK])
        batch = self.input_padding_transform(batch, batch_size=batch_size, device=self.device)
        batch = dict(batch)
        batch = self.consolidate_images(batch)

        # Populate queues
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Refill action queue if depleted
        if len(self._queues["action"]) == 0:
            batch = {
                OBS_IMAGES: torch.cat(list(self._queues[OBS_IMAGES]), dim=1),
                OBS_IMAGES_IS_PAD: torch.cat(list(self._queues[OBS_IMAGES_IS_PAD]), dim=1),
                OBS_ROBOT: torch.stack(list(self._queues[OBS_ROBOT]), dim=1),
                OBS_TASK: list(self._queues[OBS_TASK][0]),
            }

            image, image_mask = self.prepare_image(batch)
            state = self.prepare_state(batch).to(self.device)
            prompt = self.prepare_prompt(batch)

            actions = self.model.sample_actions(image, prompt, state, noise=noise, image_mask=image_mask)

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, : self.config.n_action_steps, :original_action_dim]

            # Queue has shape (n_action_steps, batch_size, *)
            self._queues["action"].extend(actions.transpose(0, 1))

        return self._queues["action"].popleft()

    @torch.no_grad
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        Sample a chunk_size actions given environment observations.

        Args:
            batch: Dictionary with observations and task prompts

        Returns:
            dict: {
                "actions": Tensor of shape (B, chunk_size, action_dim)
                    where chunk_size and action_dim are determined by the policy configuration.
                "tactile_action": None
            }

        Notes:
            - All tensors should be on the same device (e.g., cuda).
            - Image stacking and batch formatting are handled internally.
            - The returned action chunk contains all steps for the current inference.
            - State and tactile inputs will be padded to their configured max dimensions
              (cfg.max_state_dim, cfg.max_tactile_dim).
        """
        self.eval()

        orig_batch = batch
        batch = dict(batch)
        batch = self.consolidate_images(batch)

        image, _ = self.prepare_image(batch)
        prompt = self.prepare_prompt(batch)
        state = self.prepare_state(batch)

        actions = self.model.sample_actions(image, prompt, state, noise=noise)

        # Stash the initial noise used by the flow model so DSRL can record it
        # on the resulting Transition. Writes to orig_batch (caller's dict) so
        # the noise is visible after sample_actions returns.
        flow_model = getattr(self.model, "flow_model", self.model)
        if hasattr(flow_model, "_last_initial_noise"):
            orig_batch["__dsrl_noise_used__"] = flow_model._last_initial_noise

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return {"actions": actions, "tactile_action": None}

    @torch.no_grad
    def sample_actions_rtc(
        self,
        batch: dict[str, Tensor],
        inference_delay: int,
        execution_horizon: int,
        prev_actions: Tensor | None = None,
        noise: Tensor | None = None,
        beta: float = 40.0,
    ) -> Tensor:
        # first time inferencing, so just return default actions
        if prev_actions is None:
            return self.sample_actions(batch, noise=noise)

        self.eval()

        batch = dict(batch)
        # consolidate all the image_features into one key: OBS_IMAGES
        batch = self.consolidate_images(batch)

        image, _ = self.prepare_image(batch)  # assumes `observation.images`
        prompt = self.prepare_prompt(batch)  # assumes "task" key is present in the batch
        state = self.prepare_state(batch)  # assumes 'observation.state' key is present in the batch

        actions = self.model.sample_actions_rtc(
            image,
            prompt,
            state,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            prev_actions=prev_actions,
            noise=noise,
            beta=beta,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return {"actions": actions, "tactile_action": None}

    def load_from_pretrained(self, checkpoint_path):
        super().load_from_pretrained(checkpoint_path)
        self._has_uninitialized_backbone = False

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Training forward pass.

        Args:
            batch: Training batch with observations, actions, etc.

        Returns:
            Tuple of (loss, loss_dict)
        """
        if self._has_uninitialized_backbone:
            raise RuntimeError(
                "VLM backbone has uninitialized (random) weights because no model weight "
                "files were found in vlm_backbone_folder. Call load_from_pretrained() with "
                "a checkpoint, or point vlm_backbone_folder to a directory containing "
                ".safetensors files."
            )
        batch = dict(batch)
        batch = self.consolidate_images(batch)

        image, image_mask = self.prepare_image(batch)
        prompt = self.prepare_prompt(batch)
        state = self.prepare_state(batch)
        action = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad").to(self.device) if "action_is_pad" in batch else None

        # Extract training_mode if present
        training_mode = batch.get("training_mode", TrainingMode.ROBOT_FLOWMATCH.value)
        # Handle list, string, or enum
        if isinstance(training_mode, list):
            training_mode = training_mode[0] if training_mode else TrainingMode.ROBOT_FLOWMATCH.value
        if isinstance(training_mode, str):
            training_mode = TrainingMode(training_mode)
        elif not isinstance(training_mode, TrainingMode):
            training_mode = TrainingMode.ROBOT_FLOWMATCH

        # Route to model forward
        loss_dict = {}
        losses = self.model.forward(
            image,
            prompt,
            state,
            action,
            noise=noise,
            time=time,
            image_mask=image_mask,
            training_mode=training_mode,
        )

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]

        loss = losses.mean()
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Compute loss (calls forward)."""
        return self.forward(batch)

    def prepare_image(self, batch):
        """
        Prepare images for Phi4MM model.

        Returns:
            images: List[List[torch.Tensor]] - batch_size x num_images
            image_mask: List[torch.Tensor] - masking info per batch element
        """
        img_key = OBS_IMAGES
        img_pad_key = OBS_IMAGES_IS_PAD
        batch_size, num_images = batch[img_key].shape[:2]

        resize = None
        if self.config.resize_imgs_with_padding is not None:
            h, w = self.config.resize_imgs_with_padding
            resize = ResizeWithPadding(height=h, width=w)
            if not hasattr(self, "_logged_resize"):
                self._logged_resize = True

        images = []
        image_mask = []
        for batch_idx in range(batch_size):
            images_per_sample = []
            masking_per_sample = []
            for image_idx in range(num_images):
                img_tensor = batch[img_key][batch_idx][image_idx]

                if self._resize_transform is not None:
                    img_tensor = self._resize_transform(img_tensor)

                # Ensure tensor is in [0, 1] range
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0

                if resize is not None:
                    img_tensor = resize(img_tensor)

                images_per_sample.append(img_tensor)

                if img_pad_key in batch:
                    image_is_pad = batch[img_pad_key][batch_idx][image_idx]
                    # Reverse padding logic: 1 for valid, 0 for padding
                    masking_per_sample.append((image_is_pad == 0).int())
                else:
                    masking_per_sample.append(torch.tensor([1], dtype=torch.int32))

            image_mask.append(torch.cat(masking_per_sample).to(device=self.device))
            images.append(images_per_sample)

        return images, image_mask

    def prepare_prompt(self, batch):
        """Prepare text prompts — delegates to the VLM backend adapter."""
        return self._backend.prepare_prompt(batch)

    def prepare_state(self, batch):  # from modeling_pi0.py
        """Pad state"""

        if len(batch[OBS_ROBOT].shape) == 2:
            # If the state is a single value, we need to expand it to match the batch size
            batch[OBS_ROBOT] = batch[OBS_ROBOT].unsqueeze(1)
        state = self.pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action to max_action_dim."""
        actions = self.pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def pad_vector(self, vector, new_dim):
        """
        Pad vector to new dimension.

        Can be (batch_size, sequence_length, features_dimension)
        or (batch_size, features_dimension)
        """
        if vector.shape[-1] == new_dim:
            return vector
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector
        return new_vector

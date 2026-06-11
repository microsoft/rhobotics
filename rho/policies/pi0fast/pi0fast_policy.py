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

"""
π0+FAST: Efficient Action Tokenization for Vision-Language-Action Models

[Paper](https://huggingface.co/papers/2501.09747)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.
Disclaimer: It is not expected to perform as well as the original implementation.
"""

import logging
from collections import deque

import torch
from torch import Tensor

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES
from rho.common.constants import OBSERVATION_LANG as OBS_TASK
from rho.common.constants import OBSERVATION_STATE as OBS_STATE
from rho.policies.base import PreTrainedPolicy, populate_queues
from rho.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from rho.policies.pi0fast.pi0fast_models import PI0FAST

logger = logging.getLogger(__name__)


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class PI0FASTPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FAST tokenizer and model to train and run inference within rho."""

    config_class = PI0FASTConfig
    name = "pi0fast"

    def __init__(
        self,
        config: PI0FASTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.device = config.device if hasattr(config, "device") else "cuda"

        self.model = PI0FAST(config)
        self.model.to(self.device)

        self.n_action_steps = config.n_action_steps

        # queues are populated during rollout of the policy, they contain the n latest
        # observations and actions
        self._queues = None

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            OBS_TASK: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the from_pretrained method to display important disclaimer."""
        logger.warning(
            "DISCLAIMER: The PI0FAST model is ported from JAX by the Hugging Face team. "
            "It is not expected to perform as well as the original implementation. "
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        return super().from_pretrained(*args, **kwargs)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_batch_for_model(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare the batch for the model by consolidating image features."""
        # Consolidate all the image_features into one key: OBS_IMAGES
        num_img_features = len(self.config.image_features)
        batch = dict(batch)

        if num_img_features > 0:
            first_key = list(self.config.image_features.keys())[0]
            if batch[first_key].ndim == 4:
                # image features that come in as (batch_size, 3, H, W) (e.g. pusht, LIBERO_resim)
                batch[OBS_IMAGES] = torch.cat(
                    [batch[key].unsqueeze(1) for key in self.config.image_features],
                    dim=1,
                )
            elif batch[first_key].ndim == 5:
                # image features that come in as (batch_size, 1, 3, H, W) (e.g. aloha, LIBERO_noops)
                batch[OBS_IMAGES] = torch.cat([batch[key] for key in self.config.image_features], dim=1)
        else:
            # If no image features, create empty placeholder
            batch[OBS_IMAGES] = torch.empty(
                (batch[OBS_STATE].shape[0], 0, 3, 224, 224),
                device=batch[OBS_STATE].device,
                dtype=torch.float32,
            )

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare batch
        batch = self.prepare_batch_for_model(batch)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        # Generate actions
        actions = self.model.generate_actions(batch)

        # Trim to configured action steps
        actions = actions[:, : self.config.n_action_steps]

        # Get original action dimension
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `predict_action_chunk` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling the model when the
        queue is empty.
        """
        self.eval()

        # Prepare batch
        batch = self.prepare_batch_for_model(batch)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        # Note: It's important that this happens after preparing the batch
        self._queues = populate_queues(self._queues, batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            # Prepare stacked batch from queues
            queued_batch = {}

            if OBS_IMAGES in self._queues:
                queued_batch[OBS_IMAGES] = torch.cat(list(self._queues[OBS_IMAGES]), dim=1)
            else:
                # Create empty image tensor if no images
                queued_batch[OBS_IMAGES] = torch.empty(
                    (batch[OBS_STATE].shape[0], 0, 3, 224, 224),
                    device=batch[OBS_STATE].device,
                    dtype=torch.float32,
                )

            # PI0FAST uses n_obs_steps=1, so we just take the last (only) observation
            # Stack creates (batch_size, n_obs_steps, state_dim), but we need (batch_size, state_dim)
            stacked_state = torch.stack(list(self._queues[OBS_STATE]), dim=1)
            queued_batch[OBS_STATE] = stacked_state[:, -1]  # Take the last observation
            queued_batch[OBS_TASK] = list(self._queues[OBS_TASK][0])  # list of strings

            # Generate actions
            actions = self.model.generate_actions(queued_batch)

            actions = actions[:, : self.config.n_action_steps]

            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.generate_actions` returns a (batch_size, n_action_steps, action_dim) tensor,
            # but the queue effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss."""

        # Prepare batch
        batch = self.prepare_batch_for_model(batch)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        loss_dict = self.model.forward(batch)

        return loss_dict["loss"], loss_dict

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Compute the loss for training."""
        return self.forward(batch)

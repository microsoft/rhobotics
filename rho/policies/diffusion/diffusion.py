import logging
from collections import deque

import torch
from torch import Tensor

from rho.common.constants import ACTION, OBSERVATION_ENVIRONMENT_STATE, OBSERVATION_IMAGE, OBSERVATION_STATE
from rho.policies.base import PreTrainedPolicy, populate_queues
from rho.policies.diffusion.configuration_diffusion import DiffusionConfig
from rho.policies.diffusion.modeling_diffusion import DiffusionModel

logger = logging.getLogger(__name__)


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
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
        # config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest
        # observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBSERVATION_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBSERVATION_IMAGE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBSERVATION_ENVIRONMENT_STATE] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_image_batch(
        self, batch: dict[str, Tensor], ensure_seq_dim: bool = False
    ) -> dict[str, Tensor]:
        """Stack per-camera image features into a single OBSERVATION_IMAGE tensor.

        Handles HWC -> CHW conversion when needed. Stacks cameras along dim=-4
        (the camera dimension, before C, H, W).

        Args:
            batch: Dictionary of observation tensors.
            ensure_seq_dim: If True, ensures the resulting image tensor has shape
                (B, S, N, C, H, W) — i.e. 6D — by inserting a sequence dim of 1
                when it is missing. Use this for training (forward) and direct
                inference (sample_actions) where _prepare_global_conditioning
                expects 6D input. Leave False for select_action where the
                observation queue handles sequencing.

        Returns a shallow copy of *batch* with `OBSERVATION_IMAGE` set.
        """
        if not self.config.image_features:
            logger.warning("IMAGE FEATURE NOT FOUND")
            return batch

        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original

        # convert images to CHW format if they are in HWC format
        image_tensors = []
        for key in self.config.image_features:
            img = batch[key]
            if img.ndim >= 3 and img.shape[-1] == 3:
                if img.ndim == 3:  # Single image: H, W, C -> C, H, W
                    img = img.permute(2, 0, 1)
                elif img.ndim == 4:  # Batch of images: B, H, W, C -> B, C, H, W
                    img = img.permute(0, 3, 1, 2)
                elif img.ndim == 5:  # Batch+seq: B, S, H, W, C -> B, S, C, H, W
                    img = img.permute(0, 1, 4, 2, 3)
            image_tensors.append(img)

        # Stack cameras along dim=-4 (before C, H, W):
        #   3D (C,H,W)     -> 4D (N,C,H,W)
        #   4D (B,C,H,W)   -> 5D (B,N,C,H,W)
        #   5D (B,S,C,H,W) -> 6D (B,S,N,C,H,W)
        batch[OBSERVATION_IMAGE] = torch.stack(image_tensors, dim=-4)

        if ensure_seq_dim:
            img = batch[OBSERVATION_IMAGE]
            if img.ndim == 4:  # (N,C,H,W) -> (1,1,N,C,H,W): add batch + seq
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.ndim == 5:  # (B,N,C,H,W) -> (B,1,N,C,H,W): add seq
                img = img.unsqueeze(1)
            # ndim == 6: already (B,S,N,C,H,W)
            batch[OBSERVATION_IMAGE] = img

        return batch

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_image_batch(batch)  # no seq dim — queue handles it

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    # def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
    #    """Run the batch through the model and compute the loss for training or validation."""
    #    if self.config.image_features:
    #        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
    #        batch["observation.images"] = torch.stack(
    #            [batch[key] for key in self.config.image_features], dim=-4
    #        )
    #    else:
    #        print("IMAGE FEATURE NOT FOUND")
    #    actions = self.diffusion.generate_actions(batch)
    #
    #    return actions, None

    def sample_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self._prepare_image_batch(batch, ensure_seq_dim=True)
        actions = self.diffusion.generate_actions(batch)

        return {"actions": actions}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Compute the loss for training. This is what the training script expects."""
        batch = self._prepare_image_batch(batch, ensure_seq_dim=True)
        loss = self.diffusion.compute_loss(batch)
        return loss, None

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return self.forward(batch)

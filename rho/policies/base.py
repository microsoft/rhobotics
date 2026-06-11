import abc
import logging
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
from draccus import ChoiceRegistry
from torch import Tensor, nn

from rho.common.constants import (
    ACTION,
    OBSERVATION_ENVIRONMENT_STATE,
    OBSERVATION_IMAGE,
    OBSERVATION_PREFIX,
    OBSERVATION_STATE,
    OBSERVATION_TACTILE,
)
from rho.common.types import PolicyFeature
from rho.models.optimizer import OptimizerConfig
from rho.models.schedule import LRSchedulerConfig

logger = logging.getLogger(__name__)


@draccus.encode.register(torch.dtype)
def torch_dtype_encoder(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]  # Gets "bfloat16" from "torch.bfloat16"


@draccus.decode.register(torch.dtype)
def torch_dtype_decoder(dtype_str: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(f"Invalid torch dtype: {dtype_str}") from None


@dataclass
class PolicyConfig(ChoiceRegistry):
    """
    Minimal configuration class for policy models.
    """

    name: str = None
    feature_dict: dict[str, PolicyFeature] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    lr_scheduler: LRSchedulerConfig = None
    optimizer: OptimizerConfig = None

    @property
    def robot_state_feature(self):
        """Get the robot state feature from feature_dict"""
        return self.feature_dict.get(OBSERVATION_STATE)

    @property
    def action_feature(self):
        """Get the action feature from feature_dict"""
        return self.feature_dict.get(ACTION)

    @property
    def env_state_feature(self):
        """Get the environment state feature from feature_dict"""
        return self.feature_dict.get(OBSERVATION_ENVIRONMENT_STATE)

    @property
    def image_features(self):
        """Get all image features from feature_dict"""
        return {k: v for k, v in self.feature_dict.items() if k.startswith(OBSERVATION_IMAGE)}

    @property
    def tactile_features(self):
        """Get all tactile features from feature_dict"""
        return {k: v for k, v in self.feature_dict.items() if k.startswith(OBSERVATION_TACTILE)}

    @property
    def input_features(self):
        """Get input features (observations)"""
        return {k: v for k, v in self.feature_dict.items() if k.startswith(OBSERVATION_PREFIX)}

    @property
    def output_features(self):
        """Get output features (actions)"""
        return {k: v for k, v in self.feature_dict.items() if k == ACTION}

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return [0]

    def get_optimizer_preset(self) -> OptimizerConfig | None:
        return self.optimizer

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return self.lr_scheduler


class PreTrainedPolicy(nn.Module, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PolicyConfig, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.device = config.device if hasattr(config, "device") else "cpu"

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the
        optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other
                information. Apart from the loss which is a Tensor, all other
                items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other
                information. Apart from the loss which is a Tensor, all other
                items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return one action to run in the environment (potentially in
        batch mode).

        When the model uses a history of observations, or outputs a sequence
        of actions, this method deals with caching.
        """
        raise NotImplementedError

    def _remap_backwards_compatible_keys(self, state_dict: dict) -> dict:
        """Remap checkpoint keys for backwards compatibility.

        Override in subclasses that need to handle legacy checkpoint formats.
        The base implementation is a no-op.

        Args:
            state_dict: The loaded state dictionary

        Returns:
            Remapped state dictionary (or original if no remapping needed)
        """
        return state_dict

    def load_from_pretrained(self, checkpoint_path):
        """
        Load model weights from a pretrained checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        # Handle different checkpoint formats
        if "policy_state_dict" in checkpoint:
            # Training checkpoint format
            state_dict = checkpoint["policy_state_dict"]
        elif "state_dict" in checkpoint:
            # Alternative checkpoint format
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the entire checkpoint is the state dict
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Handle backward compatibility for legacy checkpoint formats
        state_dict = self._remap_backwards_compatible_keys(state_dict)

        self.load_state_dict(state_dict, strict=True)

        # Clear the checkpoint from memory
        del checkpoint, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Hook for subclasses to do post-load setup (e.g., attach LoRA adapters)
        self._post_checkpoint_load()

        # Move model to the correct device after loading weights
        if hasattr(self, "device"):
            self.to(self.device)
            logger.info(f"Pretrained weights loaded and moved to {self.device}")
        else:
            logger.info("Pretrained weights loaded successfully")

    def _post_checkpoint_load(self):
        """Hook called after checkpoint loading. Override in subclasses for post-load setup."""
        pass


def populate_queues(queues, batch, exclude_keys=None):
    """
    Populate observation/action queues with data from a batch.

    Args:
        queues: Dictionary of deques to populate
        batch: Dictionary containing the batch data
        exclude_keys: Optional list of keys to exclude from population

    Returns:
        The updated queues dictionary
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues

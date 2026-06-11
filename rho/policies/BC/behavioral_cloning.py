import logging
from dataclasses import dataclass

import torch

from rho.common.types import FeatureType
from rho.models.common import MLP, BasicCNN, GELUTanh, TransformerBlock
from rho.policies.base import PolicyConfig, PreTrainedPolicy

logger = logging.getLogger(__name__)


@PolicyConfig.register_subclass("behavioral_cloning")
@dataclass
class BehavioralCloningPolicyConfig(PolicyConfig):
    """
    Configuration class for BehavioralCloningPolicy.
    """

    name: str = "behavioral_cloning"  # Name of the policy

    # CNN encoder configuration
    encoder_dim: int = 128  # Output dimension from CNN encoder
    cnn_hidden_dim: int = 256  # Hidden dimension in CNN fully connected layers

    # MLP configuration
    mlp_hidden_dim: int = 64  # Hidden dimension in MLP layers

    # Architecture configuration
    conv_channels: list[int] = None  # Convolutional channel sizes
    dropout_rate: float = 0.1  # Dropout rate for regularization
    use_transformer: bool = False  # Use transformer blocks in action head

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128]  # Default conv channels


class BehavioralCloningPolicy(PreTrainedPolicy):
    """
    A policy that implements behavioral cloning.
    This policy is trained to mimic the actions of an expert based on
    observations.
    """

    def __init__(self, config: BehavioralCloningPolicyConfig):
        """
        Initialize the BehavioralCloningPolicy with a BasicCNN model.

        Args:
            config: PolicyConfig containing feature_dict and other config
        """
        # Must call super().__init__() first for PyTorch nn.Module
        super().__init__(config)

        # Get dimensions from feature_dict
        if config.feature_dict is None:
            raise ValueError("PolicyConfig must have feature_dict populated")

        # Extract input channels from observation features
        input_channels = 3  # Default to RGB
        num_actions = 2  # Default for PushT (x, y coordinates)

        # Find image observation to get channels
        for key, feature in config.feature_dict.items():
            if (
                key.startswith("observation")
                and feature.type == FeatureType.VISUAL
                and len(feature.shape) == 3
            ):
                # Assuming CHW format for images
                input_channels = feature.shape[0]
                break

        # Find action feature to get action dimension
        for key, feature in config.feature_dict.items():
            if key == "action" and feature.type == FeatureType.ACTION:
                num_actions = feature.shape[0] if len(feature.shape) > 0 else 1
                break

        if config.use_transformer:
            # Use transformer blocks with final linear layer for actions
            action_head = torch.nn.Sequential(
                TransformerBlock(embed_dim=config.encoder_dim, num_heads=8, ff_dim=config.encoder_dim * 4),
                TransformerBlock(embed_dim=config.encoder_dim, num_heads=8, ff_dim=config.encoder_dim * 4),
                torch.nn.Linear(config.encoder_dim, num_actions),
            )
        else:
            # Use MLP for action head if not using transformer
            action_head = MLP(
                in_features=config.encoder_dim,
                hidden_features=config.mlp_hidden_dim,
                out_features=num_actions,
                act_layer=GELUTanh,
                drop=config.dropout_rate,
            )
        # Initialize the BasicCNN model and append the action head
        self.model = torch.nn.Sequential(
            BasicCNN(
                input_channels=input_channels,
                num_actions=config.encoder_dim,
                conv_channels=config.conv_channels,
                fc_hidden=config.cnn_hidden_dim,
            ),
            action_head,
        )

        logger.info("Initialized BehavioralCloningPolicy with:")
        logger.info(f"  Input channels: {input_channels}")
        logger.info(f"  Number of actions: {num_actions}")
        logger.debug(f"  Model: {self.model}")

    def compute_loss(self, batch) -> tuple[torch.Tensor, dict | None]:
        """
        Compute the loss for behavioral cloning.
        The loss is typically the mean squared error between predicted
        actions and expert actions.

        Args:
            batch: Dictionary containing observations and actions

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other
                information. Apart from the loss which is a Tensor, all other
                items should be logging-friendly, native Python types.
        """
        predicted_actions = self.forward(batch)
        target_actions = batch["action"]
        loss = ((predicted_actions - target_actions) ** 2).mean()
        return loss, None

    def select_action(self, inputs):
        """
        Select an action based on the observations.
        """
        with torch.no_grad():
            return self(inputs)

    def reset(self):
        """
        Reset the policy state.
        This is useful if the policy maintains any internal state that
        needs to be cleared.
        """
        # Placeholder implementation; replace with actual reset logic if needed
        pass

    def get_optim_params(self):
        """
        Returns the parameters for the optimizer.
        This can include learning rate, weight decay, etc.
        """
        return self.model.parameters()

    def forward(self, batch):
        """
        Forward pass through the model.
        """
        observations = batch["observation.image"]
        return self.model(observations)

"""
Tests for the Phi4MM policy implementation.

These tests are resource-intensive and require GPU access.
"""

import pytest
import torch

from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets.data_config import DataConfig
from rho.policies import make_policy
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_policy import RhoAlphaPolicy
from tests.utils import DEVICE


@pytest.fixture(scope="class")
def rhoalpha_features():
    """Create features compatible with Phi4MM policy."""
    return {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 64, 64),  # Phi4MM expects 64x64 images
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(2,),  # Simple 2D state for testing
        ),
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(2,),  # Simple 2D action for testing
        ),
    }


@pytest.fixture(scope="class")
def rhoalpha_feature_config(rhoalpha_features):
    """Create a DataConfig for Phi4MM."""
    normalization_mapping = {
        "observation.image": NormalizationMode.IDENTITY,
        "observation.state": NormalizationMode.MEAN_STD,
        "action": NormalizationMode.MEAN_STD,
    }

    # Create mock stats
    stats = {
        "observation.state": {"mean": torch.tensor([0.0, 0.0]), "std": torch.tensor([1.0, 1.0])},
        "action": {"mean": torch.tensor([0.0, 0.0]), "std": torch.tensor([1.0, 1.0])},
    }

    return DataConfig(features=rhoalpha_features, normalization_mapping=normalization_mapping, stats=stats)


@pytest.fixture(scope="class")
def rhoalpha_policy_config(rhoalpha_feature_config):
    """Create a RhoAlphaConfig for testing."""
    config = RhoAlphaConfig(
        feature_dict=rhoalpha_feature_config.features,
        device=DEVICE,
        n_action_steps=8,  # Should be 8 for Phi4MM
    )

    return config


@pytest.fixture(scope="class")
def rhoalpha_policy(rhoalpha_policy_config):
    """Create a Phi4MM policy instance that can be reused across tests in a class."""
    policy = make_policy(rhoalpha_policy_config)
    policy.eval()  # Set to evaluation mode for testing
    return policy


@pytest.fixture
def rhoalpha_dummy_batch(rhoalpha_policy_config, rhoalpha_policy):
    """Create a dummy batch for testing Phi4MM policy."""

    def _create_batch(batch_size=1, include_actions=False):
        device = rhoalpha_policy.device
        dummy_batch = {}

        for key, feature in rhoalpha_policy_config.feature_dict.items():
            if "image" in key:
                # Image features - use 64x64 as expected by Phi4MM
                dummy_batch[key] = torch.randn(batch_size, *feature.shape, device=device)
            elif "state" in key:
                # State features
                dummy_batch[key] = torch.randn(batch_size, feature.shape[-1], device=device)
            elif key == "action" and include_actions:
                # Action features for computing loss - shape should be (batch, n_action_steps, n_actions)
                # Using n_action_steps from config (not chunk_size)
                chunk_size = rhoalpha_policy.config.chunk_size  # Should be 8
                n_actions = feature.shape[-1]
                dummy_batch[key] = torch.randn(batch_size, chunk_size, n_actions, device=device)

        # Add task prompts based on batch size
        if batch_size == 1:
            dummy_batch["task"] = ["Pick up the object and place it in the box"]
        elif batch_size == 2:
            dummy_batch["task"] = [
                "Pick up the blue object and move it",
                "Grasp the item and place it carefully",
            ]
        elif batch_size == 3:
            dummy_batch["task"] = [
                "Pick up the red cube",
                "Move the object to the left",
                "Place the item in the container",
            ]
        else:
            # Default fallback for other batch sizes
            dummy_batch["task"] = [f"Task {i + 1}: Manipulate the object" for i in range(batch_size)]

        return dummy_batch

    return _create_batch


class TestRhoAlphaConfig:
    """Test the RhoAlphaConfig configuration class."""

    def test_rhoalpha_config_initialization(self, rhoalpha_feature_config):
        """Test that RhoAlphaConfig can be initialized with default values."""
        config = RhoAlphaConfig(
            name="rhoalpha",
            feature_dict=rhoalpha_feature_config.features,
        )

        assert config.name == "rhoalpha"
        assert config.embed_dim == 1024  # Default value
        assert config.num_heads == 8
        assert config.ff_dim == 4096  # Default is 4096
        assert config.hidden_state_idx == 0
        assert config.num_blocks == 32
        assert config.max_seq_len == 6144
        assert config.device == "cuda"

        # Input/output structure
        assert config.n_obs_steps == 1
        assert config.chunk_size == 50
        assert config.n_action_steps == 50

        # Dimensions
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32


class TestRhoAlphaPolicyFunctionality:
    """Test Phi4MM policy functionality including inference and device handling."""

    @pytest.mark.gpu
    @pytest.mark.resource_intensive
    def test_make_policy_creates_rhoalpha_policy(self, rhoalpha_policy):
        """Test that make_policy can create a Phi4MM policy instance."""

        # Use the pre-created policy
        policy = rhoalpha_policy

        # Verify policy type and basic properties
        assert isinstance(policy, RhoAlphaPolicy)
        assert policy.name == "rhoalpha"
        assert policy.device == DEVICE

        # Check that policy inherits from PreTrainedPolicy and has required methods
        assert hasattr(policy, "forward")
        assert hasattr(policy, "select_action")
        assert hasattr(policy, "reset")
        assert hasattr(policy, "config_class")

        # Check inheritance
        assert isinstance(policy, torch.nn.Module)

        # Check basic PyTorch module functionality
        assert hasattr(policy, "train")
        assert hasattr(policy, "eval")
        assert hasattr(policy, "parameters")
        assert hasattr(policy, "state_dict")

    @pytest.mark.gpu
    @pytest.mark.resource_intensive
    def test_rhoalpha_select_action(self, rhoalpha_policy, rhoalpha_policy_config, rhoalpha_dummy_batch):
        """Test that Phi4MM policy can perform select_action."""
        policy = rhoalpha_policy

        # Reset policy to clear any existing queues
        policy.reset()

        # Create dummy batch using the fixture (without actions for select_action)
        batch_size = 1
        dummy_batch = rhoalpha_dummy_batch(batch_size=batch_size, include_actions=False)

        # Ensure no action key is present (select_action should only get observations)
        assert "action" not in dummy_batch

        # Forward pass should work without errors
        with torch.no_grad():
            output = policy.select_action(dummy_batch)
        print(f"Output shape: {output.shape}")

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size

        # Check that output has the expected action dimension
        expected_action_dim = rhoalpha_policy_config.feature_dict["action"].shape[-1]
        assert output.shape[-1] == expected_action_dim

    @pytest.mark.gpu
    @pytest.mark.resource_intensive
    def test_rhoalpha_select_action_batch(
        self, rhoalpha_policy, rhoalpha_policy_config, rhoalpha_dummy_batch
    ):
        """Test that Phi4MM policy can handle batched inputs."""
        policy = rhoalpha_policy

        # Reset policy to clear any existing queues
        policy.reset()

        # Test with larger batch size using the fixture (without actions for select_action)
        batch_size = 3
        dummy_batch = rhoalpha_dummy_batch(batch_size=batch_size, include_actions=False)

        # Ensure no action key is present (select_action should only get observations)
        assert "action" not in dummy_batch

        with torch.no_grad():
            output = policy.select_action(dummy_batch)

        assert output.shape[0] == batch_size
        expected_action_dim = rhoalpha_policy_config.feature_dict["action"].shape[-1]
        assert output.shape[-1] == expected_action_dim

    @pytest.mark.gpu
    @pytest.mark.resource_intensive
    def test_rhoalpha_reset_functionality(self, rhoalpha_policy):
        """Test the reset functionality of Phi4MM policy."""
        policy = rhoalpha_policy

        # Reset should not raise an error
        policy.reset()

        # Reset should be callable multiple times
        policy.reset()
        policy.reset()

    @pytest.mark.gpu
    @pytest.mark.resource_intensive
    def test_rhoalpha_compute_loss(self, rhoalpha_policy, rhoalpha_policy_config, rhoalpha_dummy_batch):
        """Test the compute_loss functionality of Phi4MM policy."""
        policy = rhoalpha_policy
        policy.train()  # Set to training mode for loss computation

        # Create dummy batch with actions included using the fixture
        batch_size = 2
        dummy_batch = rhoalpha_dummy_batch(batch_size=batch_size, include_actions=True)

        # Verify action shape is correct: (batch, n_action_steps, n_actions)
        assert "action" in dummy_batch

        # Expand the state dimension to account for policy.config.n_obs_steps
        dummy_batch["observation.state"] = (
            dummy_batch["observation.state"].unsqueeze(1).expand(-1, rhoalpha_policy_config.n_obs_steps, -1)
        )

        # Compute loss should work without errors
        loss, loss_dict = policy.compute_loss(dummy_batch)

        # Check that loss is returned
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss_dict is not None

        assert loss.requires_grad  # Should be differentiable
        assert loss.dim() == 0  # Should be a scalar
        assert loss.item() >= 0  # Loss should be non-negative


class TestRhoAlphaPolicyRegistry:
    """Test Phi4MM policy integration with the policy registry."""

    def test_rhoalpha_in_policy_registry(self):
        """Test that phi4mm is properly registered in the policy registry."""
        from rho.policies import POLICY_REGISTRY, list_available_policies

        # Check registry contains phi4mm
        assert "phi4mm" in POLICY_REGISTRY

        # Check list_available_policies includes phi4mm
        available_policies = list_available_policies()
        assert "phi4mm" in available_policies

        # Check that the registry returns the correct class
        phi4mm_class = POLICY_REGISTRY["phi4mm"]
        assert phi4mm_class is not None

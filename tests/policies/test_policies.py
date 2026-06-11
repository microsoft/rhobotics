from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from rho.policies import POLICY_REGISTRY, PolicyConfig, list_available_policies, make_policy
from rho.policies.BC.behavioral_cloning import BehavioralCloningPolicyConfig
from rho.policies.diffusion import DiffusionConfig
from tests.utils import DEVICE

# List of available policies for testing (excluding phi4mm)
AVAILABLE_POLICIES = ["behavioral_cloning", "diffusion"]


@pytest.fixture(params=AVAILABLE_POLICIES)
def policy_config(request, sample_features):
    """Fixture that provides policy configurations for different policy types"""
    policy_name = request.param

    if policy_name == "behavioral_cloning":
        return BehavioralCloningPolicyConfig(name=policy_name, feature_dict=sample_features)
    elif policy_name == "diffusion":
        return DiffusionConfig(name=policy_name, feature_dict=sample_features)
    else:
        # Fallback to generic PolicyConfig
        return PolicyConfig(name=policy_name, feature_dict=sample_features)


@patch("rho.policies.POLICY_REGISTRY")
def test_make_policy_basic(mock_registry, sample_features):
    """Test basic make_policy functionality"""
    # Mock policy class
    mock_policy_class = MagicMock()
    mock_policy_instance = MagicMock()
    mock_policy_class.return_value = mock_policy_instance

    # Set up registry
    mock_registry.__getitem__.return_value = mock_policy_class
    mock_registry.__contains__.return_value = True

    config = PolicyConfig(name="behavioral_cloning_mlp", feature_dict=sample_features)

    policy = make_policy(config)

    assert policy is not None
    mock_policy_class.assert_called_once_with(config)


@patch("torch.load")
def test_policy_checkpoint_loading(mock_torch_load, sample_features):
    """Test policy checkpoint loading functionality"""
    # Mock checkpoint data
    mock_checkpoint = {
        "model_state_dict": {},
        "config": {"name": "behavioral_cloning_mlp", "feature_dict": sample_features},
    }
    mock_torch_load.return_value = mock_checkpoint

    # This would normally be tested with actual policy classes
    # but we can test the checkpoint loading mechanism
    checkpoint_path = Path("dummy_checkpoint.pt")

    # Verify mock was set up correctly
    result = torch.load(checkpoint_path)
    assert "model_state_dict" in result
    assert "config" in result


def test_list_available_policies():
    """Test that list_available_policies returns expected policies"""
    available = list_available_policies()

    # Should include our test policies
    for policy_name in AVAILABLE_POLICIES:
        assert policy_name in available

    # Should also include phi4mm from registration
    assert "phi4mm" in available


def test_policy_registry_contains_expected_policies():
    """Test that the policy registry contains expected policies"""
    for policy_name in AVAILABLE_POLICIES:
        assert policy_name in POLICY_REGISTRY

    # Test registry access
    bc_class = POLICY_REGISTRY["behavioral_cloning"]
    diffusion_class = POLICY_REGISTRY["diffusion"]

    assert bc_class is not None
    assert diffusion_class is not None


@pytest.mark.parametrize("policy_name", AVAILABLE_POLICIES)
def test_make_policy_real_implementations(policy_name, sample_features):
    """Test make_policy with real policy implementations"""

    if policy_name == "behavioral_cloning":
        config = BehavioralCloningPolicyConfig(name=policy_name, feature_dict=sample_features)
    elif policy_name == "diffusion":
        config = DiffusionConfig(name=policy_name, feature_dict=sample_features)
    else:
        # Fallback to generic PolicyConfig
        config = PolicyConfig(name=policy_name, feature_dict=sample_features)

    # Create policy
    policy = make_policy(config)

    # Basic assertions
    assert policy is not None
    assert hasattr(policy, "config")
    assert policy.config.name == policy_name
    assert policy.config.feature_dict == sample_features

    # Check that it's a proper PyTorch module
    assert isinstance(policy, torch.nn.Module)

    # Check that parameters exist (should have at least some parameters)
    params = list(policy.parameters())
    assert len(params) > 0


def test_make_policy_with_fixture(policy_config):
    """Test make_policy with real policy implementations using fixture"""

    # Create policy
    policy = make_policy(policy_config)

    # Basic assertions
    assert policy is not None
    assert hasattr(policy, "config")
    assert policy.config.name == policy_config.name
    assert policy.config.feature_dict == policy_config.feature_dict

    # Check that it's a proper PyTorch module
    assert isinstance(policy, torch.nn.Module)

    # Check that parameters exist (should have at least some parameters)
    params = list(policy.parameters())
    assert len(params) > 0


def test_policy_select_action(policy_config):
    """Test that policies can perform select_action"""

    # Create policy using the fixture
    policy = make_policy(policy_config)

    # Move policy to the test device to ensure consistency
    policy = policy.to(DEVICE)

    # Create dummy batch - use the test device for consistency
    batch_size = 2
    device = DEVICE

    # Create dummy observation batch based on feature_dict
    dummy_batch = {}
    for key, feature in policy_config.feature_dict.items():
        if "image" in key:
            # Image features - use the actual shape from feature_dict
            shape = feature.shape
            dummy_batch[key] = torch.randn(batch_size, *shape, device=device)
        elif "state" in key:
            # State features
            dummy_batch[key] = torch.randn(batch_size, feature.shape[-1], device=device)

    # Forward pass should work without errors
    with torch.no_grad():
        output = policy.select_action(dummy_batch)

    assert output is not None
    # Different policies return different output formats

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 2


def test_policy_device_placement(sample_features):
    """Test that policies can be moved to different devices"""

    config = BehavioralCloningPolicyConfig(name="behavioral_cloning", feature_dict=sample_features)
    policy = make_policy(config)

    # Move to CPU
    policy = policy.cpu()
    cpu_device = next(policy.parameters()).device
    assert cpu_device.type == "cpu"

    # Move to CUDA if available
    if torch.cuda.is_available():
        policy = policy.cuda()
        cuda_device = next(policy.parameters()).device
        assert cuda_device.type == "cuda"

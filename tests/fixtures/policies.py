import pytest
import torch

from rho.policies import PolicyConfig


@pytest.fixture
def dummy_policy_config(sample_features):
    """Create a dummy policy configuration for testing"""
    return PolicyConfig(name="behavioral_cloning_mlp", feature_dict=sample_features)


@pytest.fixture
def dummy_bc_policy_config(sample_features):
    """Create a dummy behavioral cloning policy configuration for testing"""
    try:
        from rho.policies.BC.behavioral_cloning import BehavioralCloningPolicyConfig

        return BehavioralCloningPolicyConfig(
            name="behavioral_cloning_mlp",
            feature_dict=sample_features,
            n_obs_steps=1,
            n_action_steps=1,
            input_shapes={"observation.image": [3, 96, 96], "observation.state": [2]},
            output_shapes={"action": [2]},
            input_normalization_modes={"observation.image": "mean_std", "observation.state": "mean_std"},
            output_normalization_modes={"action": "mean_std"},
        )
    except ImportError:
        pytest.skip("BehavioralCloningPolicyConfig not available")


@pytest.fixture
def dummy_optimizer():
    """Create a dummy optimizer for testing"""
    model = torch.nn.Linear(10, 1)
    return torch.optim.Adam(model.parameters(), lr=1e-4)


@pytest.fixture
def dummy_scheduler(dummy_optimizer):
    """Create a dummy scheduler for testing"""
    return torch.optim.lr_scheduler.StepLR(dummy_optimizer, step_size=100, gamma=0.9)

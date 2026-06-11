import pytest
import torch

from rho.environment.env import EnvironmentConfig, GymEnvironmentConfig


class DummyPolicy:
    """A dummy policy that always returns [0, 0] actions for testing."""

    def __init__(self, device="cpu"):
        self.device = device

    def select_action(self, obs):
        """Always return a zero action."""
        # Determine batch size from observation
        if isinstance(obs, dict):
            # Get batch size from any observation tensor
            for _, value in obs.items():
                if hasattr(value, "shape"):
                    batch_size = value.shape[0]
                    break
            else:
                batch_size = 1
        else:
            batch_size = 1

        # Return action as torch tensor to match expected policy behavior
        if batch_size == 1:
            action = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)
        else:
            # For vectorized environments, return actions for each environment
            action = torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device)
        return action

    def sample_actions(self, obs):
        """Return zero actions in expected format for evaluate_policy."""
        action = self.select_action(obs)
        # Add action_horizon dimension: (batch, action_horizon, action_dim)
        if action.dim() == 1:  # noqa: SIM108
            action = action.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        else:
            action = action.unsqueeze(1)  # (batch, 1, 2)
        return {"actions": action}

    def reset(self):
        """Reset policy state (dummy implementation)."""
        pass


@pytest.fixture
def pusht_env_config():
    """Create a GymEnvironmentConfig for PushT."""
    return GymEnvironmentConfig(
        env_name="gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=200,
        n_envs=1,
        seed=42,
    )


@pytest.fixture
def pusht_vectorized_env_config():
    """Create a vectorized GymEnvironmentConfig for PushT."""
    return GymEnvironmentConfig(
        env_name="gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=200,
        n_envs=4,
        seed=42,
    )


@pytest.fixture
def dummy_policy():
    """Create a dummy policy for testing."""
    return DummyPolicy()


@pytest.fixture
def dummy_env_config():
    """Create a basic EnvironmentConfig for testing."""
    return EnvironmentConfig(name="test_env")

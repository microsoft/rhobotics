import pytest
import torch

from tests.utils import DEVICE


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--all", action="store_true", default=False, help="Run all tests including resource-intensive ones"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "resource_intensive: mark test as resource-intensive (requires --all flag)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip resource-intensive tests unless --all flag is provided."""
    if config.getoption("--all"):
        # --all flag provided, run all tests
        return

    skip_resource_intensive = pytest.mark.skip(reason="need --all option to run resource-intensive tests")
    for item in items:
        if "resource_intensive" in item.keywords:
            item.add_marker(skip_resource_intensive)


# Import fixture modules as plugins
pytest_plugins = [
    "tests.fixtures.features",
    "tests.fixtures.datasets",
    "tests.fixtures.policies",
]


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


@pytest.fixture
def device():
    """Return the test device"""
    return DEVICE


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary directory for config files"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_checkpoint_path(tmp_path):
    """Create a dummy checkpoint file for testing"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"

    # Create a dummy checkpoint
    dummy_state = {"model_state_dict": {}, "optimizer_state_dict": {}, "step": 100, "loss": 0.5}
    torch.save(dummy_state, checkpoint_path)
    return checkpoint_path


# Environment fixtures (moved from tests.fixtures.env to avoid early import)
@pytest.fixture
def pusht_env_config():
    """Create a GymEnvironmentConfig for PushT."""
    from rho.environment.env import GymEnvironmentConfig

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
    from rho.environment.env import GymEnvironmentConfig

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
    from tests.fixtures.env import DummyPolicy

    return DummyPolicy()


@pytest.fixture
def dummy_env_config():
    """Create a basic EnvironmentConfig for testing."""
    from rho.environment import EnvironmentConfig

    return EnvironmentConfig(name="test_env")

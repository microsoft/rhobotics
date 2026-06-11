"""
Basic availability tests for the rho package.

These tests check that all the main modules and classes can be imported successfully.
"""

import pytest


def test_rho_import():
    """Test that rho package can be imported"""
    import rho

    assert rho is not None


def test_common_imports():
    """Test that common module components can be imported"""
    import rho.common.constants
    from rho.common.wandb_logging import WandBConfig, WandBLogger
    from rho.environment import EnvironmentConfig, GymEnvironment

    assert EnvironmentConfig is not None
    assert GymEnvironment is not None
    assert WandBConfig is not None
    assert WandBLogger is not None
    assert rho.common.constants is not None


def test_datasets_imports():
    """Test that datasets module components can be imported"""
    from rho.datasets import make_dataloader
    from rho.datasets.lerobot_dataset import LeRobotDatasetConfig

    assert LeRobotDatasetConfig is not None
    assert make_dataloader is not None


def test_policies_imports():
    """Test that policies module components can be imported"""
    from rho.policies import PolicyConfig, make_policy

    assert PolicyConfig is not None
    assert make_policy is not None


def test_training_imports():
    """Test that training module components can be imported"""
    from rho.eval.eval_config import EvalConfig, SimEvalConfig
    from rho.training.train import TrainConfig
    from rho.training.train_utils import TrainLogger, load_training_state, save_checkpoint

    assert TrainConfig is not None
    assert EvalConfig is not None
    assert SimEvalConfig is not None
    assert TrainLogger is not None
    assert save_checkpoint is not None
    assert load_training_state is not None


def test_env_imports():
    """Test that utils module components can be imported"""
    from rho.environment import ENV_REGISTRY, make_environment, register_environment

    assert make_environment is not None
    assert register_environment is not None
    assert ENV_REGISTRY is not None


def test_models_imports():
    """Test that models module can be imported"""
    try:
        import rho.models

        # Models might have specific dependencies, so we just test import
        assert rho.models is not None
    except ImportError:
        pytest.skip("Models module dependencies not available")


@pytest.mark.parametrize(
    "module_name", ["rho.common", "rho.datasets", "rho.policies", "rho.training", "rho.utils"]
)
def test_module_imports(module_name):
    """Test that each main module can be imported"""
    import importlib

    module = importlib.import_module(module_name)
    assert module is not None


def test_torch_imports():
    """Test that torch is available (required dependency)"""
    import torch

    assert torch is not None
    assert hasattr(torch, "cuda")


def test_basic_functionality(sample_features):
    """Test basic functionality works"""
    from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
    from rho.environment import EnvironmentConfig
    from rho.policies import PolicyConfig

    # Test creating configurations
    env_config = EnvironmentConfig()
    dataset_config = LeRobotDatasetConfig()
    policy_config = PolicyConfig(name="behavioral_cloning_mlp", feature_dict=sample_features)

    assert env_config.name is not None
    assert dataset_config is not None
    assert policy_config.name is not None

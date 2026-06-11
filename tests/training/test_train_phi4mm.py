from pathlib import Path

import pytest
import torch

from rho.common.wandb_logging import WandBConfig
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_policy import RhoAlphaPolicy
from rho.training.train import TrainConfig, train


@pytest.fixture
def rhoalpha_config(sample_features):
    """Create a RhoAlphaConfig for testing"""
    return RhoAlphaConfig(
        name="rhoalpha",
        feature_dict=sample_features,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_obs_steps=1,
        n_action_steps=1,
        max_state_dim=5,  # match dummy batch
        max_action_dim=2,  # match dummy batch
        enable_gradient_checkpointing=True,  # Memory optimization (default is True)
        chunk_size=10,  # Smaller chunk size for testing to reduce memory usage
    )


@pytest.fixture
def rhoalpha_policy(rhoalpha_config):
    """Create a RhoAlphaPolicy for testing"""
    return RhoAlphaPolicy(config=rhoalpha_config)


@pytest.fixture
def train_config(dataset_config_fixture, rhoalpha_config):
    """Create a TrainConfig for testing"""
    return TrainConfig(
        wandb=WandBConfig(enabled=False),
        dataset=dataset_config_fixture,
        validation_dataset=None,
        policy=rhoalpha_config,
        pretrained_checkpoint=None,
        resume=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=4,
        learning_rate=1e-4,
        steps=10,  # keep short for test
        output_dir="test_outputs/phi4mm_train",
        save_checkpoint_every=5,
        logging_interval=2,
        validation_interval=5,
        eval_interval=5,
        eval_batch_size=4,
        eval_num_episodes=1,
        record_videos=False,
    )


class TestTrainRhoAlpha:
    """Test the train() function with RhoAlphaConfig and policy"""

    @pytest.mark.resource_intensive
    def test_train_rhoalpha(self, train_config):
        # Remove output dir if exists
        import shutil

        out_dir = Path(train_config.output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)

        # Ensure the checkpoint folder parent exists
        # (the train function should do this, but let's be safe)
        train_config.checkpoint_folder.parent.mkdir(parents=True, exist_ok=True)

        # Run train
        train(train_config)

        # Check that checkpoint and config files were created
        assert train_config.checkpoint_folder.exists()
        ckpt_files = list(train_config.checkpoint_folder.glob("checkpoint_step_*.pt"))
        assert len(ckpt_files) > 0
        config_file = train_config.checkpoint_folder.parent / "train_config.json"
        assert config_file.exists()

        # Clear GPU memory after training to avoid OOM when loading checkpoint
        import torch

        torch.cuda.empty_cache()

        # Optionally, check that the checkpoint contains expected keys
        # Load on CPU to avoid GPU memory issues
        checkpoint = torch.load(ckpt_files[0], weights_only=False, map_location="cpu")
        assert "policy_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "metrics" in checkpoint
        assert "step" in checkpoint

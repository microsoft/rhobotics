from pathlib import Path
from unittest.mock import patch

from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.policies import PolicyConfig
from rho.training.train import TrainConfig


def test_train_config_post_init(sample_features):
    """Test TrainConfig post-initialization logic"""
    dataset_config = LeRobotDatasetConfig(repo_id="lerobot/pusht")
    config = TrainConfig(
        dataset=dataset_config,
        batch_size=64,  # This should override dataset batch_size
        policy=PolicyConfig(name="test_policy", feature_dict=sample_features),
    )

    # batch_size should be applied to dataset config
    assert config.dataset.batch_size == 64


@patch("torch.load")
def test_train_config_with_pretrained_checkpoint(mock_torch_load):
    """Test TrainConfig with pretrained checkpoint"""
    # Mock checkpoint file
    mock_torch_load.return_value = {"model_state_dict": {}, "optimizer_state_dict": {}, "step": 1000}

    # Create a temporary checkpoint file for testing
    checkpoint_path = Path("test_checkpoint.pt")

    # Only mock exists() for the checkpoint path, not globally
    # (a global mock breaks lerobot internals that check for optional files)
    _original_exists = Path.exists

    def _selective_exists(self):
        if self.name == checkpoint_path.name:
            return True
        return _original_exists(self)

    dataset_config = LeRobotDatasetConfig(repo_id="lerobot/pusht")

    with patch.object(Path, "exists", _selective_exists):
        config = TrainConfig(dataset=dataset_config, resume=True, pretrained_checkpoint=str(checkpoint_path))

    assert config.pretrained_checkpoint == str(checkpoint_path)

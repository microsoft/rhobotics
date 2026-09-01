from pathlib import Path
from unittest.mock import patch

import draccus
import pytest

from rho.common.serialization import serialize_to_dict
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.policies import PolicyConfig
from rho.policies.rho import RhoConfig
from rho.training.train import TrainConfig, initialize_checkpoint_folder, resolve_training_checkpoint


def test_train_cli_accepts_nested_dataset_overrides(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
dataset:
  repo_id: lerobot/pusht
validation_dataset:
  repo_id: lerobot/pusht
policy:
  type: behavioral_cloning
"""
    )

    config = draccus.parse(
        TrainConfig,
        config_path=config_path,
        args=[
            f"--dataset.root_dir={tmp_path / 'train-data'}",
            "--dataset.streaming=true",
            "--dataset.max_episodes=12",
            f"--validation_dataset.root_dir={tmp_path / 'validation-data'}",
        ],
    )

    assert isinstance(config.dataset, LeRobotDatasetConfig)
    assert config.dataset.root_dir == str(tmp_path / "train-data")
    assert config.dataset.streaming is True
    assert config.dataset.max_episodes == 12
    assert isinstance(config.validation_dataset, LeRobotDatasetConfig)
    assert config.validation_dataset.root_dir == str(tmp_path / "validation-data")


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


def test_timestamped_checkpoint_folder_is_initialized_once(tmp_path, sample_features):
    dataset_config = LeRobotDatasetConfig(repo_id="lerobot/pusht")
    config = TrainConfig(
        dataset=dataset_config,
        policy=PolicyConfig(name="test_policy", feature_dict=sample_features),
        output_dir=str(tmp_path),
    )

    assert config.checkpoint_folder is None

    first = initialize_checkpoint_folder(config, timestamp="0825_103000")
    second = initialize_checkpoint_folder(config, timestamp="0825_103001")

    assert first == tmp_path / "0825_103000" / "checkpoints"
    assert second == first
    assert [path.name for path in tmp_path.iterdir()] == ["0825_103000"]


def test_named_checkpoint_folder_is_created(tmp_path, sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=PolicyConfig(name="test_policy", feature_dict=sample_features),
        output_dir=str(tmp_path),
        run_name="named-run",
    )

    checkpoint_folder = initialize_checkpoint_folder(config)

    assert checkpoint_folder == tmp_path / "named-run" / "checkpoints"
    assert checkpoint_folder.is_dir()


def test_hub_resume_uses_output_folder_instead_of_cache(tmp_path, sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=PolicyConfig(name="test_policy", feature_dict=sample_features),
        pretrained_checkpoint="organization/checkpoint",
        resume=True,
        output_dir=str(tmp_path),
    )
    config.resolved_checkpoint = Path("/cache/models--organization--checkpoint/snapshots/revision")

    checkpoint_folder = initialize_checkpoint_folder(config, timestamp="0825_103000")

    assert checkpoint_folder == tmp_path / "0825_103000" / "checkpoints"
    assert checkpoint_folder.is_dir()


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
    assert config.resolved_checkpoint is None

    with patch("rho.training.train.resolve_checkpoint", return_value=checkpoint_path) as resolve:
        config.resolved_checkpoint = resolve_training_checkpoint(config)

    assert config.resolved_checkpoint == checkpoint_path
    resolve.assert_called_once_with(
        str(checkpoint_path),
        revision=None,
        cache_dir=None,
        include_training_state=True,
    )


def test_training_uses_policy_default_repo_without_mutating_request(sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=RhoConfig(feature_dict=sample_features),
    )
    downloaded = Path("/cache/rho-open-debug-v0")

    with patch("rho.training.train.resolve_checkpoint", return_value=downloaded) as resolve:
        config.resolved_checkpoint = resolve_training_checkpoint(config)

    assert config.pretrained_checkpoint is None
    assert config.resolved_checkpoint == downloaded
    resolve.assert_called_once_with(
        "microsoft/rho-base",
        revision=None,
        cache_dir=None,
        include_training_state=False,
    )


def test_explicit_checkpoint_overrides_policy_default(sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=RhoConfig(feature_dict=sample_features),
        pretrained_checkpoint="microsoft/explicit-checkpoint",
        checkpoint_revision="revision",
        checkpoint_cache_dir="/cache",
    )
    downloaded = Path("/cache/explicit-checkpoint")

    with patch("rho.training.train.resolve_checkpoint", return_value=downloaded) as resolve:
        assert resolve_training_checkpoint(config) == downloaded

    resolve.assert_called_once_with(
        "microsoft/explicit-checkpoint",
        revision="revision",
        cache_dir="/cache",
        include_training_state=False,
    )


def test_named_run_checkpoint_wins_without_download(tmp_path, sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=RhoConfig(feature_dict=sample_features),
        output_dir=str(tmp_path),
        run_name="existing-run",
    )
    existing = config.checkpoint_folder / "checkpoint_step_0000100"

    with (
        patch("rho.training.train.find_latest_checkpoint", return_value=existing),
        patch("rho.training.train.resolve_checkpoint") as resolve,
    ):
        assert resolve_training_checkpoint(config) == existing

    assert config.resume
    assert config.pretrained_checkpoint is None
    resolve.assert_not_called()


def test_resume_does_not_fall_back_to_policy_default(sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=RhoConfig(feature_dict=sample_features),
        resume=True,
    )

    with pytest.raises(ValueError, match="explicit pretrained_checkpoint"):
        resolve_training_checkpoint(config)


def test_resolved_checkpoint_is_runtime_only(sample_features):
    config = TrainConfig(
        dataset=LeRobotDatasetConfig(repo_id="lerobot/pusht"),
        policy=RhoConfig(feature_dict=sample_features),
        pretrained_checkpoint="microsoft/requested",
    )
    config.resolved_checkpoint = Path("/cache/snapshots/resolved")

    encoded = serialize_to_dict(config)

    assert encoded["pretrained_checkpoint"] == "microsoft/requested"
    assert "resolved_checkpoint" not in encoded

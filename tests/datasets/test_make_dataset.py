"""
Integration tests for make_dataset function using a realistic dummy dataset.
"""

import json

import torch

from rho.common.constants import ACTION, OBSERVATION_IMAGE, OBSERVATION_STATE
from rho.datasets import make_dataloader, make_dataset
from rho.datasets.lerobot_dataset import TransformedLeRobotDataset


class TestMakeDatasetIntegration:
    """Integration tests for make_dataset using a realistic dummy dataset."""

    def test_create_dummy_dataset_structure(self, dummy_dataset_fixture):
        """Test that our dummy dataset fixture creates the expected structure."""
        dataset_root = dummy_dataset_fixture

        # Verify directory structure
        assert (dataset_root / "meta").exists()
        assert (dataset_root / "data").exists()

        # Verify metadata files
        assert (dataset_root / "meta" / "info.json").exists()
        assert (dataset_root / "meta" / "stats.json").exists()
        assert (dataset_root / "meta" / "tasks.parquet").exists()

        # Verify info.json content
        with open(dataset_root / "meta" / "info.json") as f:
            info = json.load(f)

        assert info["total_episodes"] == 2
        assert info["total_frames"] == 40
        assert "observation.image" in info["features"]
        assert "observation.state" in info["features"]
        assert "action" in info["features"]

    def test_tlds_with_dummy_dataset(self, dummy_dataset_fixture):
        """Test that make_dataset can create a TransformedLeRobotDataset with our dummy dataset."""

        dataset_root = dummy_dataset_fixture
        # The root parameter should be the parent directory, not the dataset directory itself
        # LeRobotDataset expects to find the dataset at root/repo_id

        # This should work without throwing exceptions
        dataset = TransformedLeRobotDataset(repo_id="DUMMY_DATASET", root=dataset_root, data_transforms=None)

        # Verify we get the expected type
        assert isinstance(dataset, TransformedLeRobotDataset)
        assert dataset.repo_id == "DUMMY_DATASET"
        assert dataset.data_transforms is None

    def test_make_dataset_with_fixtures(
        self, dataset_config_fixture, feature_config_fixture, policy_config_fixture
    ):
        """Test make_dataset using the config fixtures."""

        # This should work without throwing exceptions and return a proper dataset
        dataset = make_dataset(dataset_config_fixture, policy_config_fixture)

        # Verify we get the expected type
        assert isinstance(dataset, TransformedLeRobotDataset)
        assert dataset.repo_id == "DUMMY_DATASET"

        # Verify the dataset has the expected length (should have 20 frames total)
        assert len(dataset) == 40

        # Test that we can get a sample from the dataset
        sample = dataset[0]

        # Verify the sample has the expected keys
        assert OBSERVATION_IMAGE in sample
        assert OBSERVATION_STATE in sample
        assert ACTION in sample

        # Verify the shapes match what we expect
        assert sample[OBSERVATION_IMAGE].shape == torch.Size([1, 3, 84, 84])
        assert sample[OBSERVATION_STATE].shape == torch.Size([1, 4])
        assert sample[ACTION].shape == torch.Size([1, 2])

        # Verify the data types
        assert sample[OBSERVATION_IMAGE].dtype == torch.float32
        assert sample[OBSERVATION_STATE].dtype == torch.float32
        assert sample[ACTION].dtype == torch.float32

    def test_make_dataloader_with_fixtures(
        self, dataset_config_fixture, feature_config_fixture, policy_config_fixture
    ):
        """Test make_dataloader using the config fixtures."""

        # This should work without throwing exceptions and return a proper dataloader
        dataloader, sampler = make_dataloader(dataset_config_fixture, policy_config_fixture)

        # Verify we get the expected type
        assert hasattr(dataloader, "batch_size"), "DataLoader should have batch_size attribute"
        assert hasattr(dataloader, "dataset"), "DataLoader should have dataset attribute"

        # Verify the dataloader configuration
        assert dataloader.batch_size == dataset_config_fixture.batch_size
        assert dataloader.num_workers == dataset_config_fixture.num_workers
        assert not dataloader.drop_last

        # Verify we can iterate through the dataloader
        batch = next(iter(dataloader))

        # Verify the batch has the expected keys
        assert OBSERVATION_IMAGE in batch
        assert OBSERVATION_STATE in batch
        assert ACTION in batch

        # Verify the batch shapes (should have batch dimension)
        expected_batch_size = dataset_config_fixture.batch_size
        assert batch[OBSERVATION_IMAGE].shape == torch.Size([expected_batch_size, 1, 3, 84, 84])
        assert batch[OBSERVATION_STATE].shape == torch.Size([expected_batch_size, 1, 4])
        assert batch[ACTION].shape == torch.Size([expected_batch_size, 1, 2])

        # Verify the data types
        assert batch[OBSERVATION_IMAGE].dtype == torch.float32
        assert batch[OBSERVATION_STATE].dtype == torch.float32
        assert batch[ACTION].dtype == torch.float32

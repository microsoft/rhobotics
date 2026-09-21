import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.datasets.multi_dataset import CombinedDataset, MultiDatasetConfig, WeightedDatasetConfig


@pytest.fixture(scope="session")
def temp_datasets_root():
    """Create a temporary directory for test datasets."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def mock_dataset_config():
    """Configuration for mock datasets."""
    return mock_dataset_config_bare()


def mock_dataset_config_bare():
    """Configuration for mock datasets."""
    fps = 20  # same as LIBERO dataset
    image_shape = (64, 64, 3)  # Reduced from 256x256 to 64x64 for faster tests
    image_frame_config = {
        "dtype": "image",
        "shape": image_shape,
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": float(fps),
            "video.height": image_shape[0],
            "video.width": image_shape[1],
            "video.channels": image_shape[2],
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    features = {
        "observation.images.image": image_frame_config,
        "observation.images.wrist_image": image_frame_config,
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        },
    }
    return {"fps": fps, "features": features, "image_shape": image_shape}


def _feature_spec(action_dim: int, state_dim: int | None = None) -> dict[str, PolicyFeature]:
    state_dim = state_dim or action_dim
    return {
        OBSERVATION_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,)),
    }


def _stats_for_features(features: dict[str, PolicyFeature]) -> dict:
    return {
        key: {
            "mean": torch.zeros(feature.shape),
            "std": torch.ones(feature.shape),
            "min": torch.zeros(feature.shape),
            "max": torch.ones(feature.shape),
        }
        for key, feature in features.items()
    }


def _lerobot_cfg_with_features(features: dict[str, PolicyFeature]) -> LeRobotDatasetConfig:
    return LeRobotDatasetConfig(
        repo_id="test/dataset",
        features=features,
        stats=_stats_for_features(features),
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
    )


class TestMultiDatasetConfig:
    def test_explicit_parent_features_define_transformed_features(self):
        child_features = _feature_spec(action_dim=7, state_dim=8)
        parent_features = _feature_spec(action_dim=20, state_dim=20)

        cfg = MultiDatasetConfig(
            datasets=[WeightedDatasetConfig(dataset=_lerobot_cfg_with_features(child_features))],
            features=parent_features,
        )

        assert cfg.transformed_feature_dict[ACTION].shape == (20,)
        assert cfg.transformed_feature_dict[OBSERVATION_STATE].shape == (20,)

    def test_heterogeneous_child_features_require_parent_features(self):
        with pytest.raises(ValueError, match="heterogeneous transformed_features"):
            MultiDatasetConfig(
                datasets=[
                    WeightedDatasetConfig(dataset=_lerobot_cfg_with_features(_feature_spec(action_dim=7))),
                    WeightedDatasetConfig(dataset=_lerobot_cfg_with_features(_feature_spec(action_dim=8))),
                ]
            )


@pytest.fixture(scope="session")
def mock_datasets(temp_datasets_root, mock_dataset_config):
    """Create mock datasets for testing."""
    dataset_names = ["test_dataset1", "test_dataset2", "test_dataset3"]
    dataset_lengths = [2, 4, 6]

    return create_and_populate_mock_datasets(
        dataset_names=dataset_names,
        dataset_lengths=dataset_lengths,
        root_dir=temp_datasets_root,
        config=mock_dataset_config,
        random_seed=42,
    )


def create_and_populate_mock_datasets(
    dataset_names: list[str],
    dataset_lengths: list[int],
    root_dir: str,
    config: dict,
    random_seed: int = 42,
) -> list[LeRobotDataset]:
    """Create and populate mock datasets for testing."""
    if len(dataset_names) != len(dataset_lengths):
        raise ValueError("dataset_names and dataset_lengths must have the same length")

    # Set random seed for reproducible tests
    np.random.seed(random_seed)

    # Create datasets
    mock_datasets = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(root_dir, dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        mock_datasets.append(
            LeRobotDataset.create(
                repo_id=dataset_name,
                fps=config["fps"],
                features=config["features"],
                root=dataset_path,
            )
        )

    # Populate datasets
    features = mock_datasets[0].meta.info["features"]
    feature_keys = list(features.keys())
    ignored_features = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    feature_keys = [k for k in feature_keys if k not in ignored_features]

    mock_tasks = [
        "move the mock into the task",
        "test the task with the mock",
        "extrapolate the task without a mock",
        "pass the mocky to the left hand side",
    ]

    for i, dataset in enumerate(mock_datasets):
        # Use different random parameters for each dataset to ensure they're distinct
        mean = np.random.uniform(-2.0, 2.0)
        std = np.random.uniform(0.5, 2.0)
        state_action_stats = {
            "mean": mean,
            "std": std,
            "min": mean - 2 * std,
            "max": mean + 2 * std,
        }
        image_stats = {
            "mean": np.random.uniform(100.0, 200.0),
            "std": np.random.uniform(25.0, 100.0),
            "min": 0,
            "max": 255,
        }

        num_frames = np.random.randint(10, 20)  # Reduced from 20-100 to 10-20
        for _ in range(dataset_lengths[i]):
            task = np.random.choice(mock_tasks)
            for _ in range(num_frames):
                frame = {}
                for feature in feature_keys:
                    dtype = features[feature]["dtype"]
                    if dtype == "video" or dtype == "image":
                        image_shape = features[feature]["shape"]
                        mock_image = np.random.normal(
                            loc=image_stats["mean"], scale=image_stats["std"], size=image_shape
                        ).astype(np.uint8)
                        mock_image = np.clip(mock_image, image_stats["min"], image_stats["max"])
                        frame[feature] = mock_image
                    elif dtype == "float32":
                        shape = features[feature]["shape"]
                        mock_data = np.random.normal(
                            loc=state_action_stats["mean"], scale=state_action_stats["std"], size=shape
                        ).astype(np.float32)
                        mock_data = np.clip(mock_data, state_action_stats["min"], state_action_stats["max"])
                        frame[feature] = mock_data
                frame["task"] = task
                dataset.add_frame(frame)
            dataset.save_episode()
        # Finalize to ensure proper metadata is saved
        dataset.finalize()

    # dataset needs to be loaded from disk so `episode_data_index` is set
    # otherwise it will be None and the sampler will fail
    loaded_datasets = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(root_dir, dataset_name)
        loaded_dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=dataset_path,
        )
        loaded_datasets.append(loaded_dataset)

    return loaded_datasets


class TestCombinedDataset:
    """Test suite for CombinedDataset."""

    def test_dataset_length(self, mock_datasets):
        """Test that the multi-dataset length equals the sum of individual dataset lengths."""
        multi_dataset = CombinedDataset(
            datasets=mock_datasets,
        )

        expected_length = sum(len(dataset) for dataset in mock_datasets)
        assert len(multi_dataset) == expected_length

    def test_cumulative_lengths(self, mock_datasets):
        """Test that cumulative lengths are calculated correctly."""
        multi_dataset = CombinedDataset(
            datasets=mock_datasets,
        )

        expected_cumulative = []
        cumulative_sum = 0
        for dataset in mock_datasets:
            cumulative_sum += len(dataset)
            expected_cumulative.append(cumulative_sum)

        assert multi_dataset.cumulative_lengths == expected_cumulative

    def test_item_access_integrity(self, mock_datasets):
        """Test that items are accessed correctly across datasets with integrity preserved."""
        multi_dataset = CombinedDataset(
            datasets=mock_datasets,
        )

        # Test samples from each dataset
        for mock_dataset_idx, mock_dataset in enumerate(mock_datasets):
            # Test a few samples from each dataset
            test_indices = [0, len(mock_dataset) // 2, len(mock_dataset) - 1]

            for idx in test_indices:
                # Calculate the global index in the multi-dataset
                if mock_dataset_idx > 0:
                    multi_idx = multi_dataset.cumulative_lengths[mock_dataset_idx - 1] + idx
                else:
                    multi_idx = idx

                if multi_idx % 30 == 0:
                    sample = multi_dataset[multi_idx]
                    original_sample = mock_dataset[idx]
                    dataset_idx = sample["dataset_index"]

                    # Verify dataset index is correct
                    assert dataset_idx == mock_dataset_idx, (
                        f"Dataset index mismatch: expected {mock_dataset_idx}, got {dataset_idx}"
                    )

                    # Verify data integrity
                    assert sample["task"] == original_sample["task"], (
                        f"Task mismatch at dataset {mock_dataset_idx}, index {idx}"
                    )

                    np.testing.assert_array_equal(
                        sample["observation.state"],
                        original_sample["observation.state"],
                        err_msg=f"State data mismatch at dataset {mock_dataset_idx}, index {idx}",
                    )

                    np.testing.assert_array_equal(
                        sample["action"],
                        original_sample["action"],
                        err_msg=f"Action data mismatch at dataset {mock_dataset_idx}, index {idx}",
                    )

                    assert (
                        sample["observation.images.image"].shape
                        == original_sample["observation.images.image"].shape
                    ), f"Image shape mismatch at dataset {mock_dataset_idx}, index {idx}"

                    assert (
                        sample["observation.images.wrist_image"].shape
                        == original_sample["observation.images.wrist_image"].shape
                    ), f"Wrist image shape mismatch at dataset {mock_dataset_idx}, index {idx}"

    def test_boundary_indices(self, mock_datasets):
        """Test access at dataset boundaries."""
        multi_dataset = CombinedDataset(
            datasets=mock_datasets,
        )

        # Test first item
        sample = multi_dataset[0]
        assert sample["dataset_index"] == 0

        # Test last item
        last_idx = len(multi_dataset) - 1
        sample = multi_dataset[last_idx]
        assert sample["dataset_index"] == len(mock_datasets) - 1

        # Test boundary between datasets
        for i in range(len(mock_datasets) - 1):
            boundary_idx = multi_dataset.cumulative_lengths[i] - 1
            sample = multi_dataset[boundary_idx]
            assert sample["dataset_index"] == i

            next_sample = multi_dataset[boundary_idx + 1]
            assert next_sample["dataset_index"] == i + 1

    def test_out_of_bounds_access(self, mock_datasets):
        """Test that out-of-bounds access raises appropriate errors."""
        multi_dataset = CombinedDataset(
            datasets=mock_datasets,
        )

        # Out of bounds access raises RuntimeError after exhausting retries
        with pytest.raises(RuntimeError, match="Failed to get valid sample"):
            multi_dataset[len(multi_dataset)]

        with pytest.raises(RuntimeError, match="Failed to get valid sample"):
            multi_dataset[-1 - len(multi_dataset)]

    def test_empty_dataset_list(self):
        """Test initialization with empty dataset list."""
        multi_dataset = CombinedDataset(datasets=[])
        assert len(multi_dataset) == 0
        assert multi_dataset.cumulative_lengths == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

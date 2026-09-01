#!/usr/bin/env python

import json
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rho.common.constants import ACTION, OBSERVATION_IMAGE, OBSERVATION_STATE
from rho.common.types import NormalizationMode, PolicyFeature
from rho.datasets.data_config import BaseDatasetConfig as DataConfig
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.policies.base import PolicyConfig


@pytest.fixture
def dummy_dataset_config(tmp_path):
    """Create a dummy dataset configuration for testing"""
    # Create a temporary directory that actually exists
    dataset_root = tmp_path / "dummy_dataset"
    dataset_root.mkdir(exist_ok=True)

    return LeRobotDatasetConfig(
        repo_id="dummy/dataset",
        root_dir=str(dataset_root),
        batch_size=4,
        num_workers=0,
        use_imagenet_stats=False,
        features=None,
        observation_mapping=None,
        observation_whitelist=None,
        transform_mapping=None,
        prefetch_factor=2,
    )


@pytest.fixture
def dummy_feature_dict(sample_features):
    """Create dummy feature dictionary for testing"""
    return sample_features


@pytest.fixture
def feature_config_fixture():
    """Create a DataConfig that matches the dummy dataset features."""
    # Define the same features as in dummy_dataset_fixture
    features = {
        OBSERVATION_IMAGE: PolicyFeature(
            shape=(3, 84, 84),  # channels, height, width for policy
            type="VISUAL",  # Specify type for clarity
        ),
        OBSERVATION_STATE: PolicyFeature(shape=(4,), type="STATE"),
        ACTION: PolicyFeature(shape=(2,), type="ACTION"),
    }

    # Define normalization mapping
    normalization_mapping = {
        OBSERVATION_IMAGE: NormalizationMode.IDENTITY,
        OBSERVATION_STATE: NormalizationMode.MEAN_STD,
        ACTION: NormalizationMode.MEAN_STD,
    }

    # Create dummy stats for normalization
    stats = {
        OBSERVATION_IMAGE: {
            "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
            "mean": np.array([[[0.5]], [[0.5]], [[0.5]]]),
            "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
            "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),
        },
        OBSERVATION_STATE: {
            "max": np.array([1.0, 1.0, 1.0, 1.0]),
            "mean": np.array([0.0, 0.0, 0.0, 0.0]),
            "min": np.array([-1.0, -1.0, -1.0, -1.0]),
            "std": np.array([0.5, 0.5, 0.5, 0.5]),
        },
        ACTION: {
            "max": np.array([1.0, 1.0]),
            "mean": np.array([0.0, 0.0]),
            "min": np.array([-1.0, -1.0]),
            "std": np.array([0.3, 0.3]),
        },
    }

    return DataConfig(normalization_mapping=normalization_mapping, features=features, stats=stats)


@pytest.fixture
def mock_dataset_metadata(tmp_path, sample_features):
    """Create mock dataset metadata for testing"""
    features = sample_features

    # Create mock dataset structure
    dataset_dir = tmp_path / "dummy_dataset"
    dataset_dir.mkdir()

    # Create mock meta files
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir()

    return {
        "path": str(dataset_dir),
        "features": features,
        "total_episodes": 10,
        "total_frames": 1000,
        "fps": 30,
    }


@pytest.fixture
def mock_dataset_structure(tmp_path):
    """Create a mock dataset structure for testing without HuggingFace dependencies."""
    dataset_root = tmp_path / "mock_dataset"
    dataset_root.mkdir(exist_ok=True)

    # Create meta directory
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Create a minimal info.json
    info_content = {
        "codebase_version": "v2.0",
        "data_path": "data",
        "fps": 10,
        "splits": {"train": "0:100"},
        "total_episodes": 100,
        "total_frames": 1000,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info_content, f)

    # Create a minimal stats.json
    stats_content = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.1, 0.1, 0.1]},
        "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        "action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats_content, f)

    # Create data directory
    data_dir = dataset_root / "data"
    data_dir.mkdir(exist_ok=True)

    return dataset_root


@pytest.fixture
def dummy_dataset_fixture(tmp_path):
    """
    Create a realistic dummy dataset using TransformedLeRobotDataset.create() API.
    This is the proper way to create datasets programmatically.
    """
    repo_id = "DUMMY_DATASET"
    root = str(tmp_path / "datasets")  # Create subdirectory so it doesn't exist yet

    # Define features that the dataset will have
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (84, 84, 3),  # height, width, channels
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (4,), "names": ["x", "y", "vx", "vy"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["action_x", "action_y"]},
    }

    mock_tasks = [
        "move the mock into the task",
        "test the task with the mock",
        "extrapolate the task without a mock",
        "pass the mocky to the left hand side",
    ]

    # Create empty dataset metadata using the create() method
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=10, root=root, robot_type="dummy_robot", features=features, use_videos=False
    )

    # LeRobot v3 does not currently support this multi-episode construction path.
    for episode_idx in range(2):
        task_name = mock_tasks[episode_idx % len(mock_tasks)]

        # Add 10 frames per episode
        for _ in range(20):
            # Create dummy image (84x84x3)
            dummy_image = np.random.randint(0, 256, size=(84, 84, 3), dtype=np.uint8)
            dummy_image_pil = PIL.Image.fromarray(dummy_image)

            # Create dummy state and action
            dummy_state = np.random.randn(4).astype(np.float32)
            dummy_action = np.random.randn(2).astype(np.float32)

            frame = {
                "observation.image": dummy_image_pil,
                "observation.state": dummy_state,
                "action": dummy_action,
                "task": task_name,
            }

            dataset.add_frame(frame)

        # Save the episode
        dataset.save_episode()

    return Path(root)


@pytest.fixture
def dataset_config_fixture(dummy_dataset_fixture, feature_config_fixture):
    """Create a LeRobotDatasetConfig that works with the dummy dataset."""
    dataset_root = dummy_dataset_fixture

    return LeRobotDatasetConfig(
        repo_id="DUMMY_DATASET",
        root_dir=dataset_root,
        batch_size=8,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        features=feature_config_fixture.features,
        stats=feature_config_fixture.stats,
        normalization_mapping={
            OBSERVATION_IMAGE: NormalizationMode.IDENTITY,
            OBSERVATION_STATE: NormalizationMode.MEAN_STD,
            ACTION: NormalizationMode.MEAN_STD,
        },
    )


@pytest.fixture
def policy_config_fixture(feature_config_fixture):
    """Create a PolicyConfig that works with the dummy dataset features."""
    return PolicyConfig(
        name="dummy_policy",
        feature_dict=feature_config_fixture.features,
        device="cpu",  # Use CPU for testing
    )

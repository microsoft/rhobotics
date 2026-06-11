import numpy as np
import pytest

from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets.data_config import DataConfig


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }


@pytest.fixture
def pusht_features():
    """
    Matches the actual PushT environment specification:
        - 2D state (x, y position)
        - 2D action (x, y action)
        - RGB image (3, 96, 96)
    """
    return {
        "observation.image": PolicyFeature(
            shape=(3, 96, 96),  # channels, height, width
            type="VISUAL",
        ),
        "observation.state": PolicyFeature(
            shape=(2,),  # x, y position
            type="STATE",
        ),
        "action": PolicyFeature(
            shape=(2,),  # x, y action
            type="ACTION",
        ),
    }


@pytest.fixture
def sample_normalization_mapping():
    """Create sample normalization mapping for testing."""
    return {
        "observation.image": NormalizationMode.IDENTITY,
        "observation.state": NormalizationMode.MEAN_STD,
        "action": NormalizationMode.MEAN_STD,
    }


@pytest.fixture
def sample_stats():
    """Create sample stats for testing."""
    return {
        "observation.state": {"mean": [0.5, 0.3], "std": [0.2, 0.1]},
        "action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }


@pytest.fixture
def pusht_stats():
    """Create sample stats for testing."""
    return {
        "observation.image": {
            "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
            "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
            "mean": np.array([[[0.5]], [[0.5]], [[0.5]]]),
            "std": np.array([[[0.2]], [[0.2]], [[0.2]]]),
        },
        "observation.state": {
            "min": np.array([-1.0, -1.0]),
            "max": np.array([1.0, 1.0]),
            "mean": np.array([0.0, 0.0]),
            "std": np.array([0.3, 0.3]),
        },
        "action": {
            "min": np.array([-1.0, -1.0]),
            "max": np.array([1.0, 1.0]),
            "mean": np.array([0.0, 0.0]),
            "std": np.array([0.3, 0.3]),
        },
    }


@pytest.fixture
def simple_feature_config(sample_features, sample_normalization_mapping, sample_stats):
    """Create a simple DataConfig for testing (96x96 images, 2D state)."""
    return DataConfig(
        normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
    )


@pytest.fixture
def pusht_feature_config(pusht_features, pusht_stats):
    """Create a DataConfig for PushT environment testing."""

    normalization_mapping = {
        "observation.image": NormalizationMode.IDENTITY,  # As per PushT spec
        "observation.state": NormalizationMode.MEAN_STD,
        "action": NormalizationMode.MEAN_STD,
    }

    return DataConfig(
        features=pusht_features,
        normalization_mapping=normalization_mapping,
        stats=pusht_stats,
        chunk_size=1,  # Required for evaluate_policy
    )

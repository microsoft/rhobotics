"""Tests for rho.datasets module."""

import torch

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.transforms import DeltaActions
from rho.common.types import ActionType, FeatureType, NormalizationMode, PolicyFeature
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig, resolve_action_delta_indices_for_dataset


def make_mock_features():
    """Create mock features dict with proper PolicyFeature objects."""
    return {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }


def make_mock_stats():
    """Create mock stats dict."""
    return {
        "observation.image": {"mean": torch.zeros(3), "std": torch.ones(3)},
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }


def make_delta_actions_transform_mapping():
    """Create a transform mapping with DeltaActions for action chunk modes."""
    return {
        ACTION: [
            DeltaActions(
                action_key=ACTION,
                state_key=OBSERVATION_STATE,
                action_type=ActionType.POSITION,
                relative_to_state=True,
                post_norm=False,
            )
        ]
    }


class TestActionTimeHorizon:
    def test_resolves_per_dataset_action_delta_indices_from_fps(self):
        class Policy:
            chunk_size = 50
            action_delta_indices = list(range(50))

        class Metadata:
            fps = 15

        assert resolve_action_delta_indices_for_dataset(
            Policy(),
            Metadata(),
            action_time_horizon_s=1.0,
            min_action_chunk_size=8,
        ) == list(range(15))

    def test_clips_action_delta_indices_to_min_and_policy_chunk(self):
        class Policy:
            chunk_size = 50
            action_delta_indices = list(range(50))

        class LowFpsMetadata:
            fps = 5

        class HighFpsMetadata:
            fps = 100

        assert resolve_action_delta_indices_for_dataset(
            Policy(),
            LowFpsMetadata(),
            action_time_horizon_s=1.0,
            min_action_chunk_size=8,
        ) == list(range(8))
        assert resolve_action_delta_indices_for_dataset(
            Policy(),
            HighFpsMetadata(),
            action_time_horizon_s=1.0,
            min_action_chunk_size=8,
        ) == list(range(50))


class TestLeRobotDatasetConfig:
    """Test the LeRobotDatasetConfig dataclass."""

    def test_init_with_features_and_stats(self):
        """Test LeRobotDatasetConfig initialization with provided features and stats."""
        features = make_mock_features()
        stats = make_mock_stats()

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
        )

        assert config.repo_id == "test/dataset"
        assert config.batch_size == 64
        assert config.features == features
        assert config.stats == stats

    def test_get_transforms_returns_callable(self):
        """Test that get_transforms returns a callable transform function."""
        features = make_mock_features()
        stats = make_mock_stats()

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
        )

        transform = config.get_transforms()
        assert callable(transform)

        # Apply the transform to a test batch
        test_batch = {
            "observation.image": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7),
            "action": torch.randn(1, 16, 7),  # action chunk
        }
        result = transform(test_batch)
        assert "observation.image" in result
        assert "observation.state" in result
        assert "action" in result

    def test_remap_features(self):
        """Test the remap_features method with observation_mapping."""
        features = make_mock_features()
        stats = make_mock_stats()

        observation_mapping = {"old.obs": "new.obs"}

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
            observation_mapping=observation_mapping,
        )

        test_batch = {
            "old.obs": torch.tensor([1, 2, 3]),
            "unchanged": torch.tensor([4, 5, 6]),
        }

        result = config.remap_features(test_batch)

        assert "new.obs" in result
        assert "old.obs" not in result
        assert "unchanged" in result

    def test_remap_features_carries_is_pad(self):
        """`{feature}_is_pad` masks should follow their feature through remap/whitelist."""
        features = make_mock_features()
        stats = make_mock_stats()

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
            observation_mapping={"old.obs": "new.obs"},
            observation_whitelist=["new.obs"],
        )

        test_batch = {
            "old.obs": torch.tensor([[1, 2, 3]]),
            "old.obs_is_pad": torch.tensor([[False, True]]),
            "dropped.obs": torch.tensor([[7, 8, 9]]),
            "dropped.obs_is_pad": torch.tensor([[True]]),
        }

        result = config.remap_features(test_batch)

        # Feature and its pad mask are both renamed and kept.
        assert "new.obs" in result
        assert "new.obs_is_pad" in result
        assert torch.equal(result["new.obs_is_pad"], torch.tensor([[False, True]]))
        # Non-whitelisted feature and its pad mask are both dropped.
        assert "dropped.obs" not in result
        assert "dropped.obs_is_pad" not in result
        assert "old.obs_is_pad" not in result

    def test_get_input_transform_normalizes_data(self):
        """Test that get_input_transform creates a normalization transform."""
        features = make_mock_features()
        stats = make_mock_stats()

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
            normalization_mapping={
                "observation.state": NormalizationMode.MEAN_STD,
                "action": NormalizationMode.MEAN_STD,
                "observation.image": NormalizationMode.IDENTITY,
            },
        )

        input_transform = config.get_input_transform()
        assert callable(input_transform)

        # Apply transform to test batch
        test_batch = {
            "observation.state": torch.randn(1, 7),
            "action": torch.randn(1, 16, 7),
        }
        result = input_transform(test_batch)
        assert "observation.state" in result
        assert "action" in result


class TestNeedsDeltaActions:
    """Test the needs_delta_actions functionality."""

    def test_needs_delta_actions_returns_false_for_standard_modes(self):
        """Test that needs_delta_actions returns False for standard normalization modes."""
        features = make_mock_features()
        stats = make_mock_stats()

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
            normalization_mapping={
                FeatureType.ACTION: NormalizationMode.MEAN_STD,
            },
        )

        assert config.needs_delta_actions() is False

    def test_needs_delta_actions_returns_true_for_actionchunk_modes(self):
        """Test that needs_delta_actions returns True for action chunk modes."""
        features = make_mock_features()
        stats = make_mock_stats()
        transform_mapping = make_delta_actions_transform_mapping()

        actionchunk_modes = [
            NormalizationMode.ACTIONCHUNK_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_QUANTILE,
            NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE,
        ]

        for mode in actionchunk_modes:
            config = LeRobotDatasetConfig(
                repo_id="test/dataset",
                features=features,
                stats=stats,
                normalization_mapping={
                    FeatureType.ACTION: mode,
                },
                transform_mapping=transform_mapping,
            )
            assert config.needs_delta_actions() is True, f"Expected True for {mode}"

    def test_needs_perdim_delta_actions(self):
        """Test needs_perdim_delta_actions returns True for perdim modes."""
        features = make_mock_features()
        stats = make_mock_stats()
        transform_mapping = make_delta_actions_transform_mapping()

        perdim_modes = [
            NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE,
        ]

        for mode in perdim_modes:
            config = LeRobotDatasetConfig(
                repo_id="test/dataset",
                features=features,
                stats=stats,
                normalization_mapping={
                    FeatureType.ACTION: mode,
                },
                transform_mapping=transform_mapping,
            )
            assert config.needs_perdim_delta_actions() is True, f"Expected True for {mode}"

    def test_needs_perdim_delta_actions_false_for_non_perdim(self):
        """Test needs_perdim_delta_actions returns False for non-perdim action chunk modes."""
        features = make_mock_features()
        stats = make_mock_stats()
        transform_mapping = make_delta_actions_transform_mapping()

        config = LeRobotDatasetConfig(
            repo_id="test/dataset",
            features=features,
            stats=stats,
            normalization_mapping={
                FeatureType.ACTION: NormalizationMode.ACTIONCHUNK_MEAN_STD,
            },
            transform_mapping=transform_mapping,
        )

        assert config.needs_perdim_delta_actions() is False

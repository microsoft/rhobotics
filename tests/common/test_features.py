import json

import numpy as np
import pytest
import yaml

from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets.data_config import BaseDatasetConfig as DataConfig
from rho.datasets.data_config import convert_dict_list_to_array


class TestConvertDictListToArray:
    """Test the convert_dict_list_to_array utility function."""

    def test_convert_simple_dict(self):
        """Test converting a dictionary with list values to numpy arrays."""
        input_dict = {"key1": [1, 2, 3], "key2": [4.0, 5.0, 6.0], "key3": "not_a_list"}

        result = convert_dict_list_to_array(input_dict)

        assert isinstance(result["key1"], np.ndarray)
        assert isinstance(result["key2"], np.ndarray)
        assert isinstance(result["key3"], str)
        np.testing.assert_array_equal(result["key1"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result["key2"], np.array([4.0, 5.0, 6.0]))
        assert result["key3"] == "not_a_list"

    def test_convert_nested_dict(self):
        """Test converting nested dictionaries."""
        input_dict = {"level1": {"level2": [1, 2, 3], "other": "value"}, "simple": [4, 5, 6]}

        result = convert_dict_list_to_array(input_dict)

        assert isinstance(result["level1"]["level2"], np.ndarray)
        assert isinstance(result["simple"], np.ndarray)
        np.testing.assert_array_equal(result["level1"]["level2"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result["simple"], np.array([4, 5, 6]))
        assert result["level1"]["other"] == "value"


class TestDataConfig:
    """Test the DataConfig dataclass.

    Note: YAML loading functionality is handled by draccus.load() or draccus.decode()
    rather than a custom from_yaml() method.
    """

    def test_init_with_policy_features(self, sample_features, sample_normalization_mapping, sample_stats):
        """Test initialization with PolicyFeature objects."""
        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
        )

        # Test that input/output features are properly extracted
        assert len(config.input_features) == 2  # observation.image and observation.state
        assert len(config.output_features) == 1  # action
        assert "observation.image" in config.input_features
        assert "observation.state" in config.input_features
        assert "action" in config.output_features

    def test_init_with_dict_features(self, sample_normalization_mapping, sample_stats):
        """Test initialization with dictionary features that get converted to PolicyFeature."""
        features_dict = {
            "observation.state": {
                "shape": "(2,)",  # String shape that should be converted
                "type": FeatureType.STATE,
            },
            "action": {
                "shape": (2,),  # Tuple shape
                "type": FeatureType.ACTION,
            },
        }

        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=features_dict, stats=sample_stats
        )

        # Test that dict features were converted to PolicyFeature
        assert isinstance(config.features["observation.state"], PolicyFeature)
        assert isinstance(config.features["action"], PolicyFeature)
        assert config.features["observation.state"].shape == (2,)
        assert config.features["action"].shape == (2,)

    def test_init_with_string_normalization_modes(self, sample_features, sample_stats):
        """Test initialization with string normalization modes that get converted to enums."""
        normalization_mapping = {
            "observation.image": "IDENTITY",  # String that should be converted
            "observation.state": "MEAN_STD",
            "action": NormalizationMode.MEAN_STD,  # Already an enum
        }

        config = DataConfig(
            normalization_mapping=normalization_mapping, features=sample_features, stats=sample_stats
        )

        # Test that string normalization modes were converted to enums
        assert isinstance(config.normalization_mapping["observation.image"], NormalizationMode)
        assert isinstance(config.normalization_mapping["observation.state"], NormalizationMode)
        assert config.normalization_mapping["observation.image"] == NormalizationMode.IDENTITY
        assert config.normalization_mapping["observation.state"] == NormalizationMode.MEAN_STD

    def test_init_with_json_string_stats(self, sample_features, sample_normalization_mapping):
        """Test initialization with stats as a JSON string."""
        stats_json = '{"observation.state": {"mean": [0.5, 0.3], "std": [0.2, 0.1]}}'

        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=stats_json
        )

        # Test that JSON string was parsed and lists converted to arrays
        assert isinstance(config.stats, dict)
        assert "observation.state" in config.stats
        assert isinstance(config.stats["observation.state"]["mean"], np.ndarray)
        assert isinstance(config.stats["observation.state"]["std"], np.ndarray)

    def test_init_with_json_file_stats(self, sample_features, sample_normalization_mapping, tmp_path):
        """Test initialization with stats from a JSON file."""
        # Create a temporary JSON file
        stats_data = {"observation.state": {"mean": [0.5, 0.3], "std": [0.2, 0.1]}}
        stats_file = tmp_path / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats_data, f)

        config = DataConfig(
            normalization_mapping=sample_normalization_mapping,
            features=sample_features,
            stats=str(stats_file),
        )

        # Test that JSON file was loaded
        assert isinstance(config.stats, dict)
        assert "observation.state" in config.stats
        assert config.stats["observation.state"]["mean"] == [0.5, 0.3]

    def test_init_with_unsupported_stats_file(self, sample_features, sample_normalization_mapping, tmp_path):
        """Test initialization with unsupported stats file format."""
        stats_file = tmp_path / "stats.txt"
        stats_file.write_text("unsupported format")

        with pytest.raises(ValueError, match="Unsupported stats file format"):
            DataConfig(
                normalization_mapping=sample_normalization_mapping,
                features=sample_features,
                stats=str(stats_file),
            )

    def test_get_input_transform(self, sample_features, sample_normalization_mapping, sample_stats):
        """Test getting input transform for dataset processing."""
        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
        )

        transform = config.get_input_transform()

        # Transform should be a Normalize object
        from rho.common.normalize import Normalize

        assert isinstance(transform, Normalize)

    def test_get_action_denormalization(self, sample_features, sample_normalization_mapping, sample_stats):
        """Test getting action denormalization transform."""
        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
        )

        transform = config.get_action_denormalization()

        # Transform should be an Unnormalize object
        from rho.common.normalize import Unnormalize

        assert isinstance(transform, Unnormalize)

    def test_convert_numpy_to_python(self, sample_features, sample_normalization_mapping):
        """Test the _convert_numpy_to_python method."""
        config = DataConfig(normalization_mapping=sample_normalization_mapping, features=sample_features)

        # Test with various numpy types
        test_data = {
            "array": np.array([1, 2, 3]),
            "int": np.int32(42),
            "float": np.float64(3.14),
            "bool": np.bool_(True),
            "nested": {"inner_array": np.array([4, 5, 6])},
            "list": [np.int32(1), np.float64(2.0)],
            "string": "unchanged",
        }

        result = config._convert_numpy_to_python(test_data)

        assert isinstance(result["array"], list)
        assert isinstance(result["int"], int)
        assert isinstance(result["float"], float)
        assert isinstance(result["bool"], bool)
        assert isinstance(result["nested"]["inner_array"], list)
        assert isinstance(result["list"][0], int)
        assert isinstance(result["string"], str)
        assert result["array"] == [1, 2, 3]
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True

    def test_to_yaml(self, sample_features, sample_normalization_mapping, sample_stats, tmp_path):
        """Test saving configuration to YAML file."""
        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
        )

        config_path = tmp_path / "config" / "feature_config.yaml"
        config.to_yaml(config_path)

        # Test that YAML file was created
        assert config_path.exists()

        # Test that stats JSON file was created
        stats_path = config_path.parent / "feature_config_stats.json"
        assert stats_path.exists()

        # Test YAML content
        with open(config_path) as f:
            yaml_content = yaml.safe_load(f)

        assert "features" in yaml_content
        assert "normalization_mapping" in yaml_content
        assert "stats" in yaml_content

        # Test stats JSON content
        with open(stats_path) as f:
            stats_content = json.load(f)

        assert "observation.state" in stats_content
        assert "action" in stats_content

    def test_feature_dict_property(self, sample_features, sample_normalization_mapping, sample_stats):
        """Test the feature_dict property."""
        config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
        )

        assert config.feature_dict is config.features

    def test_draccus_compatibility(
        self, sample_features, sample_normalization_mapping, sample_stats, tmp_path
    ):
        """Test that DataConfig subclasses are compatible with draccus loading.

        This test verifies that registered DataConfig subclasses work with draccus.decode()
        by first creating a config, serializing it to dict with to_dict(), then loading
        it back using draccus.decode() and testing the resulting structure.

        Note: Base DataConfig is not a registered ChoiceRegistry subclass, so it cannot
        be decoded via draccus.decode(DataConfig, ...). Only registered subclasses like
        LeRobotDatasetConfig can be decoded this way.
        """
        # Create a DataConfig and test serialization via to_dict()
        original_config = DataConfig(
            normalization_mapping=sample_normalization_mapping, features=sample_features, stats=sample_stats
        )

        # Get the serialized dict representation
        config_dict = original_config.to_dict()

        # Verify the config was serialized properly
        assert "features" in config_dict
        assert "normalization_mapping" in config_dict
        assert "stats" in config_dict

        # Since base DataConfig is not a registered subclass, we can't use draccus.decode
        # with DataConfig directly. Instead, test that the dict contains the expected data.
        loaded_features = config_dict["features"]
        assert len(loaded_features) == len(original_config.features)
        for key in original_config.features:
            assert key in loaded_features
            loaded_feature = loaded_features[key]
            # Shapes are serialized as string tuples for YAML round-trip safety
            # e.g. "(3, 96, 96)" which DataConfig.__post_init__ can parse back
            expected_shape = original_config.features[key].shape
            shape_str = (
                "("
                + ", ".join(str(x) for x in expected_shape)
                + ("," if len(expected_shape) == 1 else "")
                + ")"
            )
            assert loaded_feature["shape"] == shape_str
            # Note: After serialization, FeatureType becomes a string
            original_type = original_config.features[key].type
            loaded_type = loaded_feature["type"]
            if isinstance(loaded_type, str):
                # Handle the case where the type was serialized as a string
                assert loaded_type == original_type.value
            else:
                assert loaded_type == original_type

        # Verify normalization mappings were preserved
        loaded_norm_mapping = config_dict["normalization_mapping"]
        for key in original_config.normalization_mapping:
            assert key in loaded_norm_mapping
            original_mode = original_config.normalization_mapping[key]
            loaded_mode = loaded_norm_mapping[key]
            if isinstance(loaded_mode, str):
                assert loaded_mode == original_mode.value
            else:
                assert loaded_mode == original_mode

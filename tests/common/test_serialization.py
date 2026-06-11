"""Tests for rho.common.serialization module."""

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from rho.common.serialization import (
    fixup_feature_shapes,
    make_yaml_safe,
    serialize_to_dict,
    serialize_to_json,
    serialize_to_json_string,
    serialize_to_yaml,
    serialize_to_yaml_string,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


class Priority(enum.Enum):
    LOW = 1
    HIGH = 2


@dataclass
class SimpleConfig:
    name: str = "default"
    value: int = 42


@dataclass
class ConfigWithToDict:
    """A dataclass with a custom to_dict that excludes `secret`."""

    name: str = "visible"
    secret: str = "hidden"

    def to_dict(self) -> dict:
        return {"name": self.name}


@dataclass
class ParentConfig:
    child: ConfigWithToDict = field(default_factory=ConfigWithToDict)
    extra: str = "hello"


@dataclass
class ConfigWithExcludedFields:
    """Mimics MultiDatasetConfig — has both persisted and computed fields."""

    datasets: list = field(default_factory=list)
    computed_weights: list = field(default_factory=list)
    computed_features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import fields as dc_fields

        _exclude = {"computed_weights", "computed_features"}
        return {
            f.name: serialize_to_dict(getattr(self, f.name))
            for f in dc_fields(self)
            if f.name not in _exclude
        }


# ---------------------------------------------------------------------------
# Tests: serialize_to_dict
# ---------------------------------------------------------------------------


class TestSerializeToDict:
    def test_primitives(self):
        assert serialize_to_dict(None) is None
        assert serialize_to_dict(True) is True
        assert serialize_to_dict(42) == 42
        assert serialize_to_dict(3.14) == 3.14
        assert serialize_to_dict("hello") == "hello"

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        assert serialize_to_dict(arr) == [1, 2, 3]

    def test_numpy_scalars(self):
        assert serialize_to_dict(np.bool_(True)) is True
        assert serialize_to_dict(np.int64(42)) == 42
        assert serialize_to_dict(np.float32(3.14)) == pytest.approx(3.14, abs=1e-5)

    def test_torch_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        assert serialize_to_dict(t) == [1.0, 2.0, 3.0]

    def test_torch_dtype(self):
        assert serialize_to_dict(torch.bfloat16) == "bfloat16"
        assert serialize_to_dict(torch.float32) == "float32"

    def test_path(self):
        p = Path("/some/path")
        assert serialize_to_dict(p) == "/some/path"

    def test_enum_uses_name(self):
        assert serialize_to_dict(Color.RED) == "RED"
        assert serialize_to_dict(Priority.HIGH) == "HIGH"

    def test_dict(self):
        d = {"a": np.int64(1), "b": Path("/x")}
        result = serialize_to_dict(d)
        assert result == {"a": 1, "b": "/x"}

    def test_list_and_tuple(self):
        assert serialize_to_dict([1, np.float64(2.0)]) == [1, 2.0]
        assert serialize_to_dict((1, 2)) == [1, 2]

    def test_simple_dataclass(self):
        cfg = SimpleConfig(name="test", value=10)
        result = serialize_to_dict(cfg)
        assert result == {"name": "test", "value": 10}

    def test_dataclass_with_to_dict(self):
        """to_dict() should be called and 'secret' excluded."""
        cfg = ConfigWithToDict(name="visible", secret="hidden")
        result = serialize_to_dict(cfg)
        assert result == {"name": "visible"}
        assert "secret" not in result

    def test_nested_dataclass_calls_to_dict(self):
        """serialize_to_dict on a parent should call child's to_dict."""
        parent = ParentConfig(child=ConfigWithToDict(name="v", secret="s"), extra="hi")
        result = serialize_to_dict(parent)
        # child should use its to_dict (no secret)
        assert result["child"] == {"name": "v"}
        assert result["extra"] == "hi"

    def test_dataclass_with_excluded_fields(self):
        """Mimics MultiDatasetConfig pattern."""
        cfg = ConfigWithExcludedFields(
            datasets=["a", "b"],
            computed_weights=[0.5, 0.5],
            computed_features={"x": 1},
        )
        result = serialize_to_dict(cfg)
        assert result == {"datasets": ["a", "b"]}
        assert "computed_weights" not in result
        assert "computed_features" not in result

    def test_type_objects(self):
        assert serialize_to_dict(int) == "int"
        assert serialize_to_dict(str) == "str"


# ---------------------------------------------------------------------------
# Tests: make_yaml_safe
# ---------------------------------------------------------------------------


class TestMakeYamlSafe:
    def test_passthrough_primitives(self):
        assert make_yaml_safe(42) == 42
        assert make_yaml_safe("hello") == "hello"
        assert make_yaml_safe(None) is None

    def test_enum_to_value(self):
        assert make_yaml_safe(Color.RED) == "red"
        assert make_yaml_safe(Priority.HIGH) == 2

    def test_path_to_string(self):
        assert make_yaml_safe(Path("/a/b")) == "/a/b"

    def test_strip_none_from_dict(self):
        d = {"a": 1, "b": None, "c": "hello"}
        result = make_yaml_safe(d, strip_none=True)
        assert result == {"a": 1, "c": "hello"}

    def test_no_strip_none(self):
        d = {"a": 1, "b": None}
        result = make_yaml_safe(d, strip_none=False)
        assert result == {"a": 1, "b": None}

    def test_nested_strip_none(self):
        d = {"outer": {"a": 1, "b": None}, "c": None}
        result = make_yaml_safe(d, strip_none=True)
        assert result == {"outer": {"a": 1}}

    def test_list_passthrough(self):
        assert make_yaml_safe([1, 2, Path("/x")]) == [1, 2, "/x"]


# ---------------------------------------------------------------------------
# Tests: fixup_feature_shapes
# ---------------------------------------------------------------------------


class TestFixupFeatureShapes:
    def test_list_to_string(self):
        features = {"obs": {"shape": [3, 224, 224], "type": "image"}}
        fixup_feature_shapes(features)
        assert features["obs"]["shape"] == "(3, 224, 224)"

    def test_already_string(self):
        features = {"obs": {"shape": "(3, 224, 224)", "type": "image"}}
        fixup_feature_shapes(features)
        assert features["obs"]["shape"] == "(3, 224, 224)"

    def test_no_shape_key(self):
        features = {"obs": {"type": "image"}}
        fixup_feature_shapes(features)
        assert features == {"obs": {"type": "image"}}

    def test_non_dict_value(self):
        """Non-dict feature values should be ignored."""
        features = {"obs": "some_string"}
        fixup_feature_shapes(features)
        assert features == {"obs": "some_string"}


# ---------------------------------------------------------------------------
# Tests: file writers
# ---------------------------------------------------------------------------


class TestFileWriters:
    def test_serialize_to_json(self, tmp_path):
        cfg = SimpleConfig(name="test", value=5)
        path = tmp_path / "out.json"
        serialize_to_json(cfg, path)
        with open(path) as f:
            data = json.load(f)
        assert data == {"name": "test", "value": 5}

    def test_serialize_to_yaml(self, tmp_path):
        import yaml

        cfg = SimpleConfig(name="test", value=5)
        path = tmp_path / "out.yaml"
        serialize_to_yaml(cfg, path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data == {"name": "test", "value": 5}

    def test_serialize_to_json_string(self):
        cfg = SimpleConfig(name="test", value=5)
        s = serialize_to_json_string(cfg)
        data = json.loads(s)
        assert data == {"name": "test", "value": 5}

    def test_serialize_to_yaml_string(self):
        import yaml

        cfg = SimpleConfig(name="test", value=5)
        s = serialize_to_yaml_string(cfg)
        data = yaml.safe_load(s)
        assert data == {"name": "test", "value": 5}


# ---------------------------------------------------------------------------
# Tests: DataConfig.to_dict integration
# ---------------------------------------------------------------------------


class TestDataConfigToDict:
    def test_dataconfig_excludes_runtime_fields(self):
        """DataConfig.to_dict() should not include non-field runtime attributes."""
        from rho.common.types import NormalizationMode, PolicyFeature
        from rho.datasets.data_config import DataConfig

        # Create a minimal DataConfig
        cfg = DataConfig(
            features={
                "observation.image": PolicyFeature(type="VISUAL", shape=(3, 224, 224)),
            },
            normalization_mapping={"observation.image": NormalizationMode.IDENTITY},
        )
        result = cfg.to_dict()

        # Should have Features with shapes as strings (round-trip safe)
        assert isinstance(result["features"]["observation.image"]["shape"], str)
        assert "transformed_features" not in result or result.get("transformed_features") is None

    def test_feature_shapes_are_strings(self):
        """Feature shapes should be converted to string format in to_dict output."""
        from rho.common.types import PolicyFeature
        from rho.datasets.data_config import DataConfig

        cfg = DataConfig(
            features={
                "obs": PolicyFeature(type="VISUAL", shape=(3, 64, 64)),
                "act": PolicyFeature(type="ACTION", shape=(7,)),
            },
        )
        result = cfg.to_dict()
        assert result["features"]["obs"]["shape"] == "(3, 64, 64)"
        assert result["features"]["act"]["shape"] == "(7,)"

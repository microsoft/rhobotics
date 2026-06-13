"""Tests for train_config.json parsing and config loading in rho.eval.eval_config."""

import json

import pytest

from rho.eval.eval_config import (
    _normalize_dataset_dict,
    _normalize_policy_dict,
    find_dataset_by_root_dir,
    load_configs_from_train_config,
)

# =============================================================================
# _normalize_policy_dict
# =============================================================================


class TestNormalizePolicyDict:
    """Tests for the _normalize_policy_dict helper."""

    def test_type_field_preserved(self):
        d = {"type": "diffusion", "chunk_size": 16}
        result = _normalize_policy_dict(d)
        assert result["type"] == "diffusion"

    def test_name_falls_back_to_type(self):
        d = {"name": "diffusion", "chunk_size": 16}
        result = _normalize_policy_dict(d)
        assert result["type"] == "diffusion"

    def test_lr_scheduler_name_renamed_to_type(self):
        d = {"type": "diffusion", "lr_scheduler": {"name": "cosine", "warmup_steps": 100}}
        result = _normalize_policy_dict(d)
        assert result["lr_scheduler"]["type"] == "cosine"
        assert "name" not in result["lr_scheduler"]

    def test_missing_type_and_name_raises(self):
        d = {"chunk_size": 16}
        with pytest.raises(ValueError, match="missing both"):
            _normalize_policy_dict(d)


# =============================================================================
# _normalize_dataset_dict
# =============================================================================


class TestNormalizeDatasetDict:
    """Tests for the _normalize_dataset_dict helper."""

    def test_strips_none_values(self):
        d = {"batch_size": 32, "stats": None, "features": None}
        result = _normalize_dataset_dict(d)
        assert "stats" not in result
        assert "features" not in result
        assert result["batch_size"] == 32

    def test_uppercases_action_type(self):
        d = {"action_type": "ee_6d_pos"}
        result = _normalize_dataset_dict(d)
        assert result["action_type"] == "EE_6D_POS"

    def test_already_uppercase_action_type_unchanged(self):
        d = {"action_type": "POSITION"}
        result = _normalize_dataset_dict(d)
        assert result["action_type"] == "POSITION"

    def test_uppercases_transform_mapping_action_type(self):
        d = {"transform_mapping": {"action": [{"type": "delta_actions", "action_type": "ee_6d_pos"}]}}
        result = _normalize_dataset_dict(d)
        assert result["transform_mapping"]["action"][0]["action_type"] == "EE_6D_POS"

    def test_infers_action_type_from_transform_mapping(self):
        d = {"transform_mapping": {"action": [{"type": "delta_actions", "action_type": "POSITION"}]}}
        result = _normalize_dataset_dict(d)
        assert result["action_type"] == "POSITION"

    def test_does_not_infer_action_type_from_non_delta_transform(self):
        """Only delta_actions / convert_to_6d_actions should trigger inference."""
        d = {"transform_mapping": {"action": [{"type": "some_other_transform", "action_type": "POSITION"}]}}
        result = _normalize_dataset_dict(d)
        assert "action_type" not in result

    def test_infers_type_lerobot_from_repo_id(self):
        d = {"repo_id": "lerobot/aloha_sim_insertion"}
        result = _normalize_dataset_dict(d)
        assert result["type"] == "lerobot"

    def test_infers_type_lerobot_from_root_dir(self):
        d = {"root_dir": "/data/my_dataset"}
        result = _normalize_dataset_dict(d)
        assert result["type"] == "lerobot"

    def test_does_not_overwrite_existing_type(self):
        d = {"repo_id": "lerobot/aloha", "type": "custom"}
        result = _normalize_dataset_dict(d)
        assert result["type"] == "custom"

    def test_old_nested_features_migration(self):
        d = {
            "features": {
                "normalization_mapping": {"action": "min_max"},
                "features": {"action": {"dtype": "float32"}},
                "stats": {"action": {"mean": [0.0]}},
                "clip_values": {"action": (-1, 1)},
                "chunk_size": 16,
            }
        }
        result = _normalize_dataset_dict(d)
        assert result["normalization_mapping"] == {"action": "min_max"}
        assert result["features"] == {"action": {"dtype": "float32"}}
        assert result["stats"] == {"action": {"mean": [0.0]}}
        assert result["clip_values"] == {"action": (-1, 1)}
        assert result["chunk_size"] == 16

    def test_multidataset_requires_root_dir(self):
        d = {"datasets": [{"dataset": {"root_dir": "/data/a"}}]}
        result = _normalize_dataset_dict(d)
        assert result is None

    def test_multidataset_selects_matching(self):
        d = {
            "datasets": [
                {"dataset": {"root_dir": "/data/a", "batch_size": 16}},
                {"dataset": {"root_dir": "/data/b", "batch_size": 32}},
            ]
        }
        result = _normalize_dataset_dict(d, dataset_root_dir="/data/b")
        assert result["batch_size"] == 32

    def test_multidataset_no_match_raises(self):
        d = {"datasets": [{"dataset": {"root_dir": "/data/a"}}]}
        result = _normalize_dataset_dict(d, dataset_root_dir="/data/missing")
        assert result is None


# =============================================================================
# find_dataset_by_root_dir
# =============================================================================


class TestFindDatasetByRootDir:
    """Tests for find_dataset_by_root_dir."""

    def test_leaf_match(self):
        d = {"root_dir": "/data/a", "batch_size": 1}
        assert find_dataset_by_root_dir(d, "/data/a") == d

    def test_leaf_no_match(self):
        d = {"root_dir": "/data/a"}
        assert find_dataset_by_root_dir(d, "/data/b") is None

    def test_dataset_cfgs_match(self):
        d = {
            "dataset_cfgs": [
                {"root_dir": "/data/a", "batch_size": 1},
                {"root_dir": "/data/b", "batch_size": 2},
            ]
        }
        result = find_dataset_by_root_dir(d, "/data/b")
        assert result["batch_size"] == 2

    def test_nested_datasets_match(self):
        d = {
            "datasets": [
                {"dataset": {"root_dir": "/data/x", "batch_size": 10}},
                {"dataset": {"root_dir": "/data/y", "batch_size": 20}},
            ]
        }
        result = find_dataset_by_root_dir(d, "/data/y")
        assert result["batch_size"] == 20

    def test_returns_none_when_no_match(self):
        d = {"datasets": [{"dataset": {"root_dir": "/data/x"}}]}
        assert find_dataset_by_root_dir(d, "/data/z") is None


# =============================================================================
# load_configs_from_train_config  (integration with real JSON on disk)
# =============================================================================


class TestLoadConfigsFromTrainConfig:
    """Tests for load_configs_from_train_config using temp JSON files."""

    def test_returns_none_none_for_missing_file(self, tmp_path):
        policy, dataset = load_configs_from_train_config(tmp_path / "nonexistent.json")
        assert policy is None
        assert dataset is None

    def test_loads_policy_config(self, tmp_path):
        config = {
            "policy": {"type": "diffusion", "name": "diffusion"},
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, dataset = load_configs_from_train_config(config_path)
        assert policy is not None
        assert policy.name == "diffusion"
        assert dataset is None

    def test_loads_dataset_config(self, tmp_path):
        config = {
            "dataset": {"batch_size": 128},
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, dataset = load_configs_from_train_config(config_path)
        assert policy is None
        assert dataset is not None
        assert dataset.batch_size == 128

    def test_loads_both(self, tmp_path):
        config = {
            "policy": {"type": "diffusion", "name": "diffusion"},
            "dataset": {"batch_size": 64},
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, dataset = load_configs_from_train_config(config_path)
        assert policy is not None
        assert dataset is not None
        assert dataset.batch_size == 64

    def test_applies_backward_compat_to_policy(self, tmp_path):
        """name→type fallback should be applied."""
        config = {
            "policy": {
                "name": "diffusion",
            },
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, _ = load_configs_from_train_config(config_path)
        assert policy is not None
        assert policy.name == "diffusion"

    def test_applies_backward_compat_to_dataset(self, tmp_path):
        """action_type uppercasing should be applied."""
        config = {
            "dataset": {"action_type": "ee_6d_pos"},
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        _, dataset = load_configs_from_train_config(config_path)
        assert dataset is not None
        assert dataset.action_type.value == "EE_6D_POS"

    def test_multidataset_selection(self, tmp_path):
        dir_a = tmp_path / "data_a"
        dir_b = tmp_path / "data_b"
        dir_a.mkdir()
        dir_b.mkdir()

        config = {
            "dataset": {
                "datasets": [
                    {"dataset": {"root_dir": str(dir_a), "batch_size": 16}},
                    {"dataset": {"root_dir": str(dir_b), "batch_size": 32}},
                ]
            },
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        _, dataset = load_configs_from_train_config(config_path, dataset_root_dir=str(dir_b))
        assert dataset is not None
        assert dataset.batch_size == 32

"""Tests for train_config.json parsing and config loading in rho.eval.eval_config."""

import json
from types import SimpleNamespace

import pytest

from rho.eval.eval import (
    _initialize_default_checkpoint,
    _make_multi_eval_config,
    _make_policy_interface_config,
)
from rho.eval.eval_config import (
    _normalize_dataset_dict,
    _normalize_policy_dict,
    find_dataset_by_root_dir,
    load_configs_from_train_config,
    load_policy_config_from_json,
)


def test_sim_eval_wires_rtc_policy_interface_config():
    dataset = object()
    policy = object()
    cfg = SimpleNamespace(
        dataset=dataset,
        device="cpu",
        eval_mode="rtc",
        inference_delay=9,
        execution_horizon=16,
        beta=15,
        guidance_schedule="constant",
    )

    result = _make_policy_interface_config(cfg, policy)

    assert result.data_config is dataset
    assert result.policy is policy
    assert result.execution_horizon == 16
    assert result.guidance_schedule == "constant"


def test_eval_uses_policy_default_checkpoint():
    class Config:
        pretrained_checkpoint = None
        policy = SimpleNamespace(pretrained_repo_id="organization/rho-model")

        def __init__(self):
            self.post_init_source = None

        def __post_init__(self):
            self.post_init_source = self.pretrained_checkpoint

    cfg = Config()

    _initialize_default_checkpoint(cfg)

    assert cfg.post_init_source == "organization/rho-model"


def test_eval_requires_checkpoint_source():
    cfg = SimpleNamespace(
        pretrained_checkpoint=None,
        policy=SimpleNamespace(pretrained_repo_id=None),
    )

    with pytest.raises(ValueError, match="No pretrained checkpoint"):
        _initialize_default_checkpoint(cfg)


def test_multi_eval_preserves_dataset_and_policy(monkeypatch, tmp_path):
    dataset = object()
    policy = object()
    environment = object()
    parent = SimpleNamespace(
        eval_num_episodes=4,
        record_videos=False,
        device="cpu",
        seed=42,
        dataset=object(),
        policy=object(),
        dataset_root_dir=None,
        eval_mode="standard",
        inference_delay=6,
        execution_horizon=None,
        beta=10,
        guidance_schedule="paper",
    )
    entry = SimpleNamespace(
        environment=environment,
        dataset=dataset,
        policy=policy,
    )
    captured = {}

    def capture_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("rho.eval.eval.SimEvalConfig", capture_config)

    result = _make_multi_eval_config(
        parent,
        entry,
        tmp_path / "checkpoint",
        tmp_path / "output",
        "libero_spatial",
    )

    assert result.dataset is dataset
    assert result.policy is policy
    assert result.environment is environment
    assert captured["output_dir"].endswith("libero_spatial")


# =============================================================================
# _normalize_policy_dict
# =============================================================================


class TestNormalizePolicyDict:
    """Tests for the _normalize_policy_dict helper."""

    def test_type_field_preserved(self):
        d = {"type": "rho", "chunk_size": 16}
        result = _normalize_policy_dict(d)
        assert result["type"] == "rho"

    def test_name_falls_back_to_type(self):
        d = {"name": "rho", "chunk_size": 16}
        result = _normalize_policy_dict(d)
        assert result["type"] == "rho"

    def test_lr_scheduler_type_migrated_from_legacy_name(self):
        d = {"type": "rho", "lr_scheduler": {"name": "constant"}}
        result = _normalize_policy_dict(d)
        assert result["lr_scheduler"]["type"] == "constant"
        assert "name" not in result["lr_scheduler"]

    def test_diffusers_algorithm_name_is_not_treated_as_registry_type(self):
        d = {"type": "rho", "lr_scheduler": {"name": "cosine", "num_warmup_steps": 100}}
        result = _normalize_policy_dict(d)
        assert result["lr_scheduler"] == {
            "type": "diffuser",
            "schedule_name": "cosine",
            "num_warmup_steps": 100,
        }

    def test_diffusers_algorithm_name_migrates_when_type_is_present(self):
        d = {
            "type": "rho",
            "lr_scheduler": {
                "type": "diffuser",
                "name": "linear",
                "num_warmup_steps": 100,
            },
        }
        result = _normalize_policy_dict(d)
        assert result["lr_scheduler"]["schedule_name"] == "linear"
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
            "policy": {"type": "rho", "name": "rho"},
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, dataset = load_configs_from_train_config(config_path)
        assert policy is not None
        assert policy.name == "rho"
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
            "policy": {"type": "rho", "name": "rho"},
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
                "name": "rho",
            },
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, _ = load_configs_from_train_config(config_path)
        assert policy is not None
        assert policy.name == "rho"

    def test_ignores_unknown_policy_fields(self, tmp_path, caplog):
        """Stale train_config.json policy fields should warn instead of failing decode."""
        config = {
            "policy": {
                "type": "rho",
                "name": "rho",
                "old_field_a": True,
                "old_field_b": 12,
            },
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy, _ = load_configs_from_train_config(config_path)

        assert policy is not None
        assert policy.name == "rho"
        assert "old_field_a" in caplog.text
        assert "old_field_b" in caplog.text
        assert "Ignoring unsupported fields from train_config.json for policy" in caplog.text

    def test_policy_only_loader_ignores_unknown_policy_fields(self, tmp_path, caplog):
        """The policy-only loader used by auto-finetune should get the same compatibility."""
        config = {
            "policy": {
                "type": "rho",
                "name": "rho",
                "removed_policy_field": True,
            },
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        policy = load_policy_config_from_json(config_path)

        assert policy is not None
        assert policy.name == "rho"
        assert "removed_policy_field" in caplog.text

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

    def test_ignores_unknown_dataset_fields(self, tmp_path, caplog):
        """Stale train_config.json dataset fields should warn instead of failing decode."""
        config = {
            "dataset": {
                "batch_size": 128,
                "removed_dataset_field": "legacy",
            },
        }
        config_path = tmp_path / "train_config.json"
        config_path.write_text(json.dumps(config))

        _, dataset = load_configs_from_train_config(config_path)

        assert dataset is not None
        assert dataset.batch_size == 128
        assert "removed_dataset_field" in caplog.text

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

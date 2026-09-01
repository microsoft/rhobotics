from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
from torch import nn

from rho.checkpoints import (
    DATA_CONFIG_FILE,
    FEATURES_FILE,
    MANIFEST_FILE,
    POLICY_CONFIG_FILE,
    STATS_FILE,
    TRAINING_STATE_FILE,
    checkpoint_lease,
    is_checkpoint_bundle,
    load_bundle_metadata,
    load_bundle_state_dict,
    load_bundle_training_state,
    load_bundle_weights,
    load_manifest,
    resolve_checkpoint,
    save_checkpoint_bundle,
    validate_checkpoint_bundle,
)
from rho.common.types import FeatureType, PolicyFeature
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.eval.eval_config import load_configs_from_checkpoint
from rho.policies.rho.configuration_rho import RhoConfig
from rho.training.train_utils import cleanup_old_checkpoints, find_latest_checkpoint, save_checkpoint


@dataclass
class _PolicyConfig:
    name: str = "tiny"
    feature_dict: dict | None = None
    vlm_backbone_folder: str | None = None


@dataclass
class _DataConfig:
    root_dir: str
    stats: dict
    features: dict


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _PolicyConfig(feature_dict={"observation.state": {"shape": [2], "type": "STATE"}})
        self.config.vlm_backbone_folder = "/machine/specific/model"
        self.linear = nn.Linear(2, 2)


class _TiedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _PolicyConfig()
        self.input = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, 2, bias=False)
        self.output.weight = self.input.weight


def test_bundle_round_trip_separates_portable_artifacts(tmp_path):
    policy = _TinyPolicy()
    original_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    data_config = _DataConfig(
        root_dir="/machine/specific/data",
        stats={"observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}},
        features={"observation.state": {"shape": [2], "type": "STATE"}},
    )
    checkpoint = tmp_path / "checkpoint_step_0000010"

    save_checkpoint_bundle(
        policy,
        checkpoint,
        step=10,
        training_state={"step": 10, "optimizer_state_dict": {}},
        data_config=data_config,
        max_shard_size=100,
    )

    assert is_checkpoint_bundle(checkpoint)
    for filename in (
        MANIFEST_FILE,
        POLICY_CONFIG_FILE,
        FEATURES_FILE,
        DATA_CONFIG_FILE,
        STATS_FILE,
        TRAINING_STATE_FILE,
    ):
        assert (checkpoint / filename).is_file()
    assert list(checkpoint.glob("*.safetensors"))

    metadata = load_bundle_metadata(checkpoint)
    assert "feature_dict" not in metadata["policy"]
    assert "vlm_backbone_folder" not in metadata["policy"]
    assert metadata["features"] == policy.config.feature_dict
    assert "root_dir" not in metadata["data_config"]
    assert metadata["stats"] == data_config.stats

    for parameter in policy.parameters():
        parameter.data.zero_()
    load_bundle_weights(policy, checkpoint)
    for key, value in policy.state_dict().items():
        assert torch.equal(value, original_state[key])

    assert load_bundle_training_state(checkpoint)["step"] == 10
    timings = load_manifest(checkpoint)["save_timings_seconds"]
    assert timings["model_weights"] >= 0
    assert timings["training_state"] >= 0


def test_bundle_restores_shared_tensor_aliases(tmp_path):
    policy = _TiedPolicy()
    original_state = {key: value.clone() for key, value in policy.state_dict().items()}
    checkpoint = tmp_path / "checkpoint_step_0000010"

    save_checkpoint_bundle(policy, checkpoint, step=10)

    state_dict = load_bundle_state_dict(checkpoint)
    assert state_dict.keys() == original_state.keys()
    for key, value in state_dict.items():
        assert torch.equal(value, original_state[key])

    policy.input.weight.data.zero_()
    load_bundle_weights(policy, checkpoint)
    assert torch.equal(policy.input.weight, original_state["input.weight"])
    assert policy.input.weight is policy.output.weight


def test_find_latest_uses_highest_completed_bundle(tmp_path):
    policy = _TinyPolicy()
    save_checkpoint_bundle(policy, tmp_path / "checkpoint_step_0000010", step=10)
    save_checkpoint_bundle(policy, tmp_path / "checkpoint_step_0000030", step=30)
    incomplete = tmp_path / "checkpoint_step_0000040"
    incomplete.mkdir()
    (incomplete / "model.safetensors").touch()

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint_step_0000030"


def test_find_latest_skips_corrupt_higher_step_bundle(tmp_path):
    policy = _TinyPolicy()
    valid = tmp_path / "checkpoint_step_0000010"
    corrupt = tmp_path / "checkpoint_step_0000020"
    save_checkpoint_bundle(policy, valid, step=10)
    save_checkpoint_bundle(policy, corrupt, step=20)
    next(corrupt.glob("*.safetensors")).write_bytes(b"")

    assert find_latest_checkpoint(tmp_path) == valid


def test_cleanup_preserves_old_checkpoint_when_current_is_corrupt(tmp_path):
    policy = _TinyPolicy()
    old = tmp_path / "checkpoint_step_0000100"
    current = tmp_path / "checkpoint_step_0000200"
    save_checkpoint_bundle(policy, old, step=100)
    save_checkpoint_bundle(policy, current, step=200)
    next(current.glob("*.safetensors")).write_bytes(b"")

    with pytest.raises(ValueError):
        cleanup_old_checkpoints(tmp_path, current_step=200, keep_checkpoint_interval=200)

    assert old.exists()
    assert current.exists()


def test_cleanup_preserves_checkpoint_while_reader_holds_lease(tmp_path):
    policy = _TinyPolicy()
    old = tmp_path / "checkpoint_step_0000100"
    current = tmp_path / "checkpoint_step_0000200"
    save_checkpoint_bundle(policy, old, step=100)
    save_checkpoint_bundle(policy, current, step=200)

    with checkpoint_lease(old):
        cleanup_old_checkpoints(tmp_path, current_step=200, keep_checkpoint_interval=200)

    assert old.exists()
    assert current.exists()


def test_training_state_pruning_skips_active_reader(tmp_path):
    policy = _TinyPolicy()
    checkpoint = tmp_path / "checkpoint_step_0000100"
    save_checkpoint_bundle(
        policy,
        checkpoint,
        step=100,
        training_state={"step": 100},
    )

    with checkpoint_lease(checkpoint):
        from rho.checkpoints import remove_bundle_training_state

        assert not remove_bundle_training_state(checkpoint)

    assert (checkpoint / TRAINING_STATE_FILE).exists()
    assert load_bundle_training_state(checkpoint)["step"] == 100


def test_validation_rejects_missing_or_empty_artifacts(tmp_path):
    policy = _TinyPolicy()
    checkpoint = tmp_path / "checkpoint_step_0000010"
    save_checkpoint_bundle(
        policy,
        checkpoint,
        step=10,
        training_state={"step": 10},
    )
    (checkpoint / TRAINING_STATE_FILE).write_bytes(b"")

    with pytest.raises(ValueError):
        validate_checkpoint_bundle(checkpoint)


def test_training_state_retention_keeps_current_and_interval_steps(tmp_path):
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters())

    for step in (100, 200, 300):
        save_checkpoint(
            policy,
            optimizer,
            step,
            {"loss": 1.0},
            tmp_path,
            keep_training_state_interval=200,
        )

    checkpoint_100 = tmp_path / "checkpoint_step_0000100"
    checkpoint_200 = tmp_path / "checkpoint_step_0000200"
    checkpoint_300 = tmp_path / "checkpoint_step_0000300"
    assert load_manifest(checkpoint_100)["artifacts"]["training_state"] is None
    assert not (checkpoint_100 / TRAINING_STATE_FILE).exists()
    assert load_bundle_training_state(checkpoint_200) is not None
    assert load_bundle_training_state(checkpoint_300) is not None


def test_bundle_policy_and_data_configs_round_trip(tmp_path):
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    stats = {
        "observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "action": {"mean": torch.zeros(2), "std": torch.ones(2)},
    }
    policy = _TinyPolicy()
    policy.config = RhoConfig(feature_dict=features, device="cpu")
    data_config = LeRobotDatasetConfig(
        repo_id="test/dataset",
        root_dir=tmp_path,
        features=features,
        stats=stats,
    )
    checkpoint = tmp_path / "checkpoint_step_0000010"
    save_checkpoint_bundle(policy, checkpoint, step=10, data_config=data_config)

    loaded_policy, loaded_data_config = load_configs_from_checkpoint(checkpoint)

    assert isinstance(loaded_policy, RhoConfig)
    assert loaded_policy.feature_dict == features
    assert isinstance(loaded_data_config, LeRobotDatasetConfig)
    assert loaded_data_config.root_dir is None
    assert torch.equal(
        torch.as_tensor(loaded_data_config.stats["action"]["mean"]),
        torch.zeros(2),
    )


def test_huggingface_resolver_downloads_only_portable_artifacts(tmp_path):
    policy = _TinyPolicy()
    bundle = tmp_path / "checkpoint_step_0000010"
    save_checkpoint_bundle(policy, bundle, step=10)

    with patch("rho.checkpoints.snapshot_download", return_value=str(bundle)) as download:
        resolved = resolve_checkpoint(
            "technology-and-research/rho",
            revision="test-revision",
            cache_dir=tmp_path / "cache",
        )

    assert resolved == bundle
    allow_patterns = download.call_args.kwargs["allow_patterns"]
    assert TRAINING_STATE_FILE not in allow_patterns
    assert "model*.safetensors" in allow_patterns
    assert download.call_args.kwargs["revision"] == "test-revision"


def test_huggingface_resolver_can_request_trusted_training_state(tmp_path):
    policy = _TinyPolicy()
    bundle = tmp_path / "checkpoint_step_0000010"
    save_checkpoint_bundle(
        policy,
        bundle,
        step=10,
        training_state={"step": 10},
    )

    with patch("rho.checkpoints.snapshot_download", return_value=str(bundle)) as download:
        resolve_checkpoint("technology-and-research/rho", include_training_state=True)

    assert TRAINING_STATE_FILE in download.call_args.kwargs["allow_patterns"]


def test_local_resolver_does_not_call_huggingface(tmp_path):
    policy = _TinyPolicy()
    bundle = tmp_path / "checkpoint_step_0000010"
    save_checkpoint_bundle(policy, bundle, step=10)

    with patch("rho.checkpoints.snapshot_download") as download:
        assert resolve_checkpoint(bundle) == bundle

    download.assert_not_called()

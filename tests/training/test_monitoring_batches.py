"""Isolation and replay contracts for distributed monitoring batches."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets import make_dataloader
from rho.datasets.lerobot_dataset import EpisodeAwareSampler, LeRobotDatasetConfig
from rho.training.distributed_monitoring import LookaheadBatchIterator, make_local_monitoring_loader
from tests.datasets.test_dataset_splits import split_dataset as _split_dataset
from tests.training.test_distributed_monitoring import _DataConfig


@pytest.mark.parametrize("requested", [0, 1, 3, 7, 13])
def test_repeated_peek_preserves_batches_and_accumulation_boundaries(monkeypatch, requested):
    monkeypatch.setattr("rho.training.distributed_monitoring.broadcast_object_list", lambda *a, **kw: None)
    loader = SimpleNamespace(end_of_dataloader=False, remainder=0)
    fetched = []

    def source():
        for index in range(100):
            loader.end_of_dataloader = index % 3 == 2
            loader.remainder = index // 3
            fetched.append(index)
            yield {"action": torch.full((index % 2 + 1, 2, 1), index), "index": index}

    buffered = LookaheadBatchIterator(source(), loader)
    accelerator = SimpleNamespace(is_main_process=True)
    for expected in range(9):
        batch = next(buffered)
        assert batch["index"] == expected
        assert loader.end_of_dataloader is (expected % 3 == 2)
        assert loader.remainder == expected // 3
        before = (loader.end_of_dataloader, loader.remainder)
        first = buffered.peek(requested, accelerator)
        fetched_count = len(fetched)
        second = buffered.peek(requested, accelerator)
        assert [id(b) for b in first] == [id(b) for b in second]
        assert len(fetched) == fetched_count
        assert (loader.end_of_dataloader, loader.remainder) == before
        assert sum(b["action"].shape[0] for b in first) >= requested
    assert fetched == list(range(len(fetched)))


def test_nonmain_uses_main_decisions_instead_of_its_own_batch_sizes(monkeypatch):
    decisions = iter([True, True, False])

    def broadcast(flag, **kwargs):
        flag[0] = next(decisions)

    monkeypatch.setattr("rho.training.distributed_monitoring.broadcast_object_list", broadcast)
    batches = [{"action": torch.ones(10, 1), "index": i} for i in range(4)]
    iterator = LookaheadBatchIterator(iter(batches), [])
    result = iterator.peek(1, SimpleNamespace(is_main_process=False))
    assert len(result) == 2  # This worker already exceeds the limit after its first batch.
    assert [next(iterator)["index"] for _ in range(4)] == [0, 1, 2, 3]


def test_failed_peek_restores_current_loader_state(monkeypatch):
    monkeypatch.setattr("rho.training.distributed_monitoring.broadcast_object_list", lambda *a, **kw: None)
    loader = SimpleNamespace(end_of_dataloader=False, remainder=0)

    def source():
        yield {"action": torch.ones(1, 1)}
        loader.end_of_dataloader = True
        raise RuntimeError("test loader failure")

    buffered = LookaheadBatchIterator(source(), loader)
    next(buffered)
    with pytest.raises(RuntimeError, match="test loader failure"):
        buffered.peek(1, SimpleNamespace(is_main_process=True))
    assert loader.end_of_dataloader is False


class _CappedConfig(_DataConfig):
    def make_sampler(self, dataset, policy_cfg=None):
        return EpisodeAwareSampler([0], [4], shuffle=True, seed=self.seed)


@pytest.mark.parametrize("config_type", [_DataConfig, _CappedConfig])
def test_local_loader_does_not_advance_training_sampler_or_rng(config_type):
    config = config_type()
    training, sampler = make_dataloader(config, device="cpu", deterministic=True)
    iterator = iter(training)
    next(iterator)
    before_rng = training.generator.get_state().clone()
    before_sampler = deepcopy(sampler.save_state()) if sampler is not None else None
    monitor = make_local_monitoring_loader(config, None, "cpu")
    assert monitor.dataset is not training.dataset
    assert monitor.generator is not training.generator
    assert monitor.sampler is not training.sampler
    assert monitor.num_workers == 0
    list(monitor)
    list(monitor)
    assert torch.equal(training.generator.get_state(), before_rng)
    if sampler is not None:
        assert sampler.save_state() == before_sampler


def test_local_loader_keeps_original_config_unchanged():
    config = _DataConfig(num_workers=4)
    monitor = make_local_monitoring_loader(config, None, "cpu")
    assert monitor.num_workers == 0
    assert config.num_workers == 4


def test_negative_peek_limit_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        LookaheadBatchIterator(iter([]), []).peek(-1, SimpleNamespace(is_main_process=True))


@pytest.fixture(scope="module")
def monitoring_dataset(tmp_path_factory):
    return _split_dataset.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("cap", [None, 1])
def test_real_lerobot_monitoring_loader_preserves_split_and_sampler_state(monitoring_dataset, cap):
    config = LeRobotDatasetConfig(
        repo_id="test/local-splits",
        root_dir=str(monitoring_dataset),
        split_name="validation",
        max_episodes=cap,
        num_workers=0,
        batch_size=2,
        features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "action": PolicyFeature(FeatureType.ACTION, (2,)),
        },
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
    )
    training, sampler = make_dataloader(config, device="cpu", deterministic=True)
    iterator = iter(training)
    next(iterator)
    before_rng = training.generator.get_state().clone()
    before_sampler = deepcopy(sampler.save_state()) if sampler is not None else None
    monitor = make_local_monitoring_loader(config, None, "cpu")
    episodes = [int(ep) for batch in monitor for ep in batch["episode_index"]]
    assert sorted(episodes) == ([2] * 4 + ([3] * 5 if cap is None else []))
    assert torch.equal(training.generator.get_state(), before_rng)
    if sampler is not None:
        assert sampler.save_state() == before_sampler

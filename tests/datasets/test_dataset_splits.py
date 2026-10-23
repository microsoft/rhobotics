"""Episode-level split selection against small, local LeRobot datasets."""

import json
from collections import Counter

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets.lerobot_dataset import (
    LeRobotDatasetConfig,
    TransformedLeRobotDataset,
    TransformedStreamingLeRobotDataset,
)


@pytest.fixture(scope="module")
def split_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("split_data") / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="test/local-splits",
        fps=10,
        root=root,
        use_videos=False,
        robot_type="test",
        features={
            OBSERVATION_STATE: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            ACTION: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        },
    )
    for episode, length in enumerate([2, 3, 4, 5]):
        for frame in range(length):
            value = np.array([episode * 100 + frame, episode * 100 + frame + 0.5], dtype=np.float32)
            dataset.add_frame({OBSERVATION_STATE: value.copy(), ACTION: value.copy(), "task": "test"})
        dataset.save_episode()
    dataset.finalize()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": "0:2", "validation": "2:4", "all": "0:4"}
    info_path.write_text(json.dumps(info))
    return root


def _load(root, streaming=False, **kwargs):
    if streaming:
        return TransformedStreamingLeRobotDataset(
            repo_id="test/local-splits", root=root, buffer_size=2, shuffle=False, **kwargs
        )
    return TransformedLeRobotDataset(repo_id="test/local-splits", root=root, download_videos=False, **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
def test_named_splits_select_disjoint_episodes(split_dataset, streaming):
    train = _load(split_dataset, streaming, split_name="train")
    validation = _load(split_dataset, streaming, split_name="validation")
    train_ids = Counter(int(sample["episode_index"]) for sample in train)
    validation_ids = Counter(int(sample["episode_index"]) for sample in validation)

    assert train_ids == {0: 2, 1: 3}
    assert validation_ids == {2: 4, 3: 5}
    assert train_ids.keys().isdisjoint(validation_ids)
    assert train.num_frames == 5
    assert validation.num_frames == 9
    assert train.num_episodes == validation.num_episodes == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_unknown_split_is_rejected(split_dataset, streaming):
    with pytest.raises(ValueError, match="split.*validaton"):
        _load(split_dataset, streaming, split_name="validaton")


@pytest.mark.parametrize("streaming", [False, True])
def test_explicit_episodes_cannot_expand_named_split(split_dataset, streaming):
    dataset = _load(split_dataset, streaming, split_name="validation", episodes=[0, 3])
    assert Counter(int(sample["episode_index"]) for sample in dataset) == {3: 5}
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 5


@pytest.mark.parametrize("streaming", [False, True])
def test_empty_episode_intersection_is_rejected(split_dataset, streaming):
    with pytest.raises(ValueError, match="No episodes"):
        _load(split_dataset, streaming, split_name="validation", episodes=[0, 1])


def test_streaming_split_survives_blacklist_reload(split_dataset):
    dataset = _load(split_dataset, streaming=True, split_name="validation", observation_blacklist=[])
    assert Counter(int(sample["episode_index"]) for sample in dataset) == {2: 4, 3: 5}


@pytest.mark.parametrize("streaming", [False, True])
def test_split_preserves_temporal_windows(split_dataset, streaming):
    dataset = _load(
        split_dataset,
        streaming,
        split_name="validation",
        delta_timestamps={OBSERVATION_STATE: [-0.1, 0.0], ACTION: [0.0, 0.1]},
    )
    samples = {(int(s["episode_index"]), int(s["frame_index"])): s for s in dataset}
    assert {ep for ep, _ in samples} == {2, 3}
    first = samples[2, 0]
    second = samples[2, 1]
    last = samples[2, 3]
    torch.testing.assert_close(torch.as_tensor(second[OBSERVATION_STATE])[:, 0], torch.tensor([200.0, 201.0]))
    torch.testing.assert_close(torch.as_tensor(second[ACTION])[:, 0], torch.tensor([201.0, 202.0]))
    assert torch.as_tensor(first[f"{OBSERVATION_STATE}_is_pad"]).tolist() == [True, False]
    assert torch.as_tensor(last[f"{ACTION}_is_pad"]).tolist() == [False, True]


def test_max_episodes_uses_split_relative_frame_indices(split_dataset):
    config = LeRobotDatasetConfig(
        repo_id="test/local-splits",
        root_dir=str(split_dataset),
        split_name="validation",
        max_episodes=1,
        features={
            OBSERVATION_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
        },
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
    )
    dataset = config.make_dataset()
    sampler = config.make_sampler(dataset)
    assert len(sampler) == 4
    assert sorted(sampler) == [0, 1, 2, 3]
    assert {int(dataset[index]["episode_index"]) for index in sampler} == {2}


@pytest.mark.parametrize("streaming", [False, True])
def test_full_split_preserves_all_data(split_dataset, streaming):
    dataset = _load(split_dataset, streaming, split_name="all")
    assert Counter(int(sample["episode_index"]) for sample in dataset) == {0: 2, 1: 3, 2: 4, 3: 5}
    assert dataset.num_frames == 14
    assert dataset.num_episodes == 4


@pytest.mark.parametrize("interval", ["0:5", "-1:2", "2:2", "3:2", "0:4:2", "0:50%", "bad", None])
def test_invalid_split_ranges_fail_closed(interval):
    from types import SimpleNamespace

    from rho.datasets.lerobot_dataset import _resolve_split_episodes

    metadata = SimpleNamespace(total_episodes=4, info={"splits": {"validation": interval}})
    with pytest.raises(ValueError, match="split.*validation"):
        _resolve_split_episodes(metadata, "validation")


def test_missing_split_metadata_only_defaults_to_train():
    from types import SimpleNamespace

    from rho.datasets.lerobot_dataset import _resolve_split_episodes

    metadata = SimpleNamespace(total_episodes=4, info={})
    assert _resolve_split_episodes(metadata, "train") is None
    assert _resolve_split_episodes(metadata, "train", episodes=[3, 1, 3]) == [1, 3]
    with pytest.raises(ValueError, match="split.*validation"):
        _resolve_split_episodes(metadata, "validation")

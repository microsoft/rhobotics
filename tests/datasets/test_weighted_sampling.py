"""Weighted mixtures must preserve dataset and selected-episode identities."""

import json
from collections import Counter
from copy import deepcopy
from itertools import islice

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
from rho.datasets import make_dataloader
from rho.datasets.lerobot_dataset import EpisodeAwareSampler, LeRobotDatasetConfig
from rho.datasets.multi_dataset import (
    MultiDatasetConfig,
    MultiDatasetWeightedSampler,
    ShuffleType,
    WeightedDatasetConfig,
)


@pytest.fixture(scope="module")
def mixture_roots(tmp_path_factory):
    roots = []
    for child, lengths in enumerate(((3, 6, 9), (6, 9, 3), (4, 8, 12))):
        root = tmp_path_factory.mktemp(f"mixture_{child}") / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=f"test/weighted-{child}",
            root=root,
            fps=10,
            use_videos=False,
            robot_type="test",
            features={
                key: {"dtype": "float32", "shape": (2,), "names": ["dataset", "frame"]}
                for key in (OBSERVATION_STATE, ACTION)
            },
        )
        absolute_frame = 0
        for length in lengths:
            for _ in range(length):
                value = np.array([child, absolute_frame], dtype=np.float32)
                dataset.add_frame(
                    {OBSERVATION_STATE: value.copy(), ACTION: value.copy(), "task": f"child-{child}"}
                )
                absolute_frame += 1
            dataset.save_episode()
        dataset.finalize()
        path = root / "meta/info.json"
        info = json.loads(path.read_text())
        info["splits"] = {"train": "0:2", "validation": "2:3", "all": "0:3"}
        path.write_text(json.dumps(info))
        roots.append(root)
    return roots


def _config(roots, *, weights=(1, 1), caps=(None, None), split="all", shuffle=ShuffleType.FRAME):
    features = {
        OBSERVATION_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    children = [
        LeRobotDatasetConfig(
            repo_id=f"test/weighted-{child}",
            root_dir=str(root),
            split_name=split,
            max_episodes=cap,
            features=deepcopy(features),
            normalization_mapping=dict.fromkeys(FeatureType, NormalizationMode.IDENTITY),
            num_workers=0,
            seed=10 + child,
        )
        for child, (root, cap) in enumerate(zip(roots[: len(weights)], caps, strict=True))
    ]
    return MultiDatasetConfig(
        datasets=[
            WeightedDatasetConfig(child, weight) for child, weight in zip(children, weights, strict=True)
        ],
        features=deepcopy(features),
        num_workers=0,
        batch_size=2,
        shuffle_type=shuffle,
        seed=42,
    )


@pytest.mark.parametrize("weights,expected", [((9, 1), {0: 18, 1: 2}), ((1, 9), {0: 2, 1: 18})])
def test_uncapped_loader_honors_relative_weights(mixture_roots, weights, expected):
    loader, sampler = make_dataloader(_config(mixture_roots, weights=weights), device="cpu")
    assert sampler is not None
    assert len(sampler) == sum(expected.values())
    counts = Counter()
    for batch in islice(loader, len(loader)):
        counts.update(batch["dataset_index"].tolist())
        assert batch[ACTION][:, 0].tolist() == batch["dataset_index"].float().tolist()
    assert counts == expected


@pytest.mark.parametrize("caps", [(1, 1), (1, None), (None, 1)])
@pytest.mark.parametrize("split", ["all", "train", "validation"])
def test_caps_keep_dataset_and_episode_identity(mixture_roots, caps, split):
    cfg = _config(mixture_roots, caps=caps, split=split)
    dataset = cfg.make_dataset()
    sampler = cfg.make_sampler(dataset)
    expected_lengths = {"all": (18, 18), "train": (9, 15), "validation": (9, 3)}[split]
    eligible = [
        ((3, 6)[child] if split != "validation" else expected_lengths[child])
        if cap is not None
        else expected_lengths[child]
        for child, cap in enumerate(caps)
    ]
    counts = Counter()
    for index in islice(sampler, len(sampler)):
        sample = dataset[index]
        child = sample["dataset_index"]
        counts[child] += 1
        assert sample[ACTION][0].item() == child
        allowed = {2} if split == "validation" else ({0} if caps[child] else {0, 1, 2})
        if split == "train":
            allowed &= {0, 1}
        assert int(sample["episode_index"]) in allowed
    assert counts == {0: min(eligible), 1: min(eligible)}
    assert sampler.cumulative_episode_indices == [0, expected_lengths[0], sum(expected_lengths)]


def test_selected_split_uses_its_actual_dataset_lengths(mixture_roots):
    cfg = _config(mixture_roots, split="validation")
    dataset = cfg.make_dataset()
    sampler = cfg.make_sampler(dataset)
    assert sampler is not None
    indices = list(islice(sampler, len(sampler)))
    assert Counter(dataset[i]["dataset_index"] for i in indices) == {0: 3, 1: 3}
    assert {int(dataset[i]["episode_index"]) for i in indices} == {2}
    assert sampler.cumulative_episode_indices == [0, 9, 12]


def test_contributions_match_eligible_sampler_allocations(mixture_roots):
    cfg = _config(mixture_roots, caps=(1, None), weights=(1, 2), split="train")
    dataset = cfg.make_dataset()
    sampler = cfg.make_sampler(dataset)
    contributions = cfg.get_contributions()
    assert [child["length"] for child in contributions["child_contributions"]] == [3, 15]
    assert [child["weighted_length"] for child in contributions["child_contributions"]] == [3, 6]
    assert contributions["weighted_length"] == len(sampler) == 9


@pytest.mark.parametrize("shuffle", [False, True])
def test_partial_quotas_do_not_repeat_the_same_child_prefix(shuffle):
    sampler = MultiDatasetWeightedSampler(
        [EpisodeAwareSampler([0], [2], shuffle=shuffle), EpisodeAwareSampler([0], [8], shuffle=shuffle)],
        sample_weights=[1, 1],
        shuffle_type=ShuffleType.NONE,
        seed=42,
    )
    indices = list(islice(sampler, 4 * len(sampler)))
    assert sorted(index - 2 for index in indices if index >= 2) == list(range(8))


@pytest.mark.parametrize("shuffle", [ShuffleType.FRAME, ShuffleType.NONE])
@pytest.mark.parametrize("consumed", [0, 1, 5, 6, 7, 17, 18, 19])
def test_factory_sampler_resumes_across_allocation_boundaries(mixture_roots, shuffle, consumed):
    def make_sampler():
        cfg = _config(mixture_roots, caps=(1, None), shuffle=shuffle)
        return cfg.make_sampler(cfg.make_dataset())

    original = make_sampler()
    iterator = iter(original)
    list(islice(iterator, consumed))
    state = deepcopy(original.save_state())
    expected = list(islice(iterator, 30))
    restored = make_sampler()
    restored.load_state(state)
    assert list(islice(restored, 30)) == expected


@pytest.mark.parametrize("weights", [[0, 1], [1, 0]])
def test_zero_weight_children_are_not_sampled(mixture_roots, weights):
    cfg = _config(mixture_roots, weights=weights)
    dataset = cfg.make_dataset()
    sampler = cfg.make_sampler(dataset)
    assert sampler is not None
    assert len(sampler) == 18
    assert {dataset[i]["dataset_index"] for i in islice(sampler, 36)} == {weights.index(1)}


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weight"):
        MultiDatasetWeightedSampler([EpisodeAwareSampler([0], [3]), EpisodeAwareSampler([0], [3])], weights)


def test_positive_weight_empty_child_is_rejected():
    with pytest.raises(ValueError, match="empty|eligible"):
        MultiDatasetWeightedSampler([EpisodeAwareSampler([], []), EpisodeAwareSampler([0], [3])])


def test_unequal_dataset_lengths_keep_requested_proportions(mixture_roots):
    cfg = _config(mixture_roots, weights=(1, 2, 1), caps=(None, None, None))
    dataset = cfg.make_dataset()
    sampler = cfg.make_sampler(dataset)
    assert sampler is not None
    assert sampler.cumulative_episode_indices == [0, 18, 36, 60]
    counts = Counter()
    for i in islice(sampler, len(sampler)):
        sample = dataset[i]
        child = sample["dataset_index"]
        assert sample[ACTION][0].item() == child
        counts[child] += 1
    assert counts == {0: 9, 1: 18, 2: 9}


def test_nested_weights_survive_flattening(mixture_roots):
    inner = _config(mixture_roots, weights=(9, 1))
    third = _config(mixture_roots, weights=(1, 1, 1), caps=(None, None, None)).dataset_cfgs[2]
    cfg = MultiDatasetConfig(
        datasets=[WeightedDatasetConfig(inner, 10), WeightedDatasetConfig(third, 5)],
        num_workers=0,
    )
    dataset = cfg.make_dataset()
    sampler = cfg.make_sampler(dataset)
    assert sampler is not None
    assert Counter(dataset[i]["dataset_index"] for i in islice(sampler, len(sampler))) == {0: 18, 1: 2, 2: 10}


def test_restricted_local_indices_use_explicit_full_lengths():
    sampler = MultiDatasetWeightedSampler(
        [EpisodeAwareSampler([4], [6]), EpisodeAwareSampler([2], [4])],
        dataset_lengths=[10, 6],
        shuffle_type=ShuffleType.NONE,
    )
    assert sorted(islice(sampler, len(sampler))) == [4, 5, 12, 13]


def test_disabled_empty_sampler_preserves_following_dataset_offset():
    sampler = MultiDatasetWeightedSampler(
        [EpisodeAwareSampler([], []), EpisodeAwareSampler([0], [4])],
        sample_weights=[0, 1],
        dataset_lengths=[5, 4],
    )
    assert sorted(islice(sampler, len(sampler))) == [5, 6, 7, 8]


@pytest.mark.parametrize("lengths", [[3], [3, -1], [3, 2.5]])
def test_invalid_dataset_lengths_are_rejected(lengths):
    with pytest.raises(ValueError, match="dataset_lengths"):
        MultiDatasetWeightedSampler(
            [EpisodeAwareSampler([0], [3]), EpisodeAwareSampler([0], [3])],
            dataset_lengths=lengths,
        )


def test_sampling_sizes_do_not_leak_into_serialized_config(mixture_roots):
    cfg = _config(mixture_roots, caps=(1, None))
    before = cfg.to_dict()
    cfg.make_sampler(cfg.make_dataset())
    assert cfg.to_dict() == before

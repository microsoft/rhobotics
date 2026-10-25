"""Exercise Rho's map-style reader through real local dataset construction."""

import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rho.datasets.lerobot_dataset import TransformedLeRobotDataset

HEAD = "observation.images.head"
WRIST = "observation.images.wrist"
IMAGE = "observation.images.debug"
STATE = "observation.state"
EXTRA = "observation.extra"


@pytest.fixture(scope="module")
def local_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("reader") / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="test/local-reader",
        root=root,
        fps=10,
        use_videos=False,
        features={
            STATE: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            EXTRA: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            IMAGE: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channels"]},
        },
        robot_type="test",
    )
    for episode in range(2):
        for frame in range(3):
            value = np.array([episode * 10 + frame, 0.5], dtype=np.float32)
            dataset.add_frame(
                {
                    STATE: value.copy(),
                    EXTRA: value.copy(),
                    "action": value.copy(),
                    IMAGE: np.zeros((8, 8, 3), dtype=np.uint8),
                    "task": "test",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    # Declare two video streams without encoding MP4s. Tests isolate the decoder
    # boundary while exercising the real metadata, Parquet reader and indexing.
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": "0:1", "validation": "1:2", "all": "0:2"}
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    for key in (HEAD, WRIST):
        info["features"][key] = {
            "dtype": "video",
            "shape": [8, 8, 3],
            "names": ["height", "width", "channels"],
        }
        path = root / f"videos/{key}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.touch()
    info_path.write_text(json.dumps(info))
    for path in (root / "meta/episodes").glob("*/*.parquet"):
        table = pq.read_table(path)
        for key in (HEAD, WRIST):
            for field, value in (
                ("chunk_index", 0),
                ("file_index", 0),
                ("from_timestamp", 0.0),
                ("to_timestamp", 0.3),
            ):
                table = table.append_column(f"videos/{key}/{field}", pa.array([value] * len(table)))
        pq.write_table(table, path)
    return root


@pytest.fixture
def decode_calls(monkeypatch):
    calls = []

    def decode(path, timestamps, *args, **kwargs):
        calls.append((Path(path), timestamps))
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return torch.zeros(len(timestamps), 3, 8, 8)

    monkeypatch.setattr("lerobot.datasets.dataset_reader.decode_video_frames", decode)
    return calls


def _load(root, **kwargs):
    return TransformedLeRobotDataset(repo_id="test/local-reader", root=root, **kwargs)


@pytest.mark.parametrize("blacklist", [[HEAD], [HEAD, IMAGE, EXTRA]])
@pytest.mark.parametrize("split", ["all", "validation"])
def test_blacklist_applies_before_decode_and_transforms(local_dataset, decode_calls, blacklist, split):
    delta_timestamps = {STATE: [-0.1, 0.0], "action": [0.0, 0.1], HEAD: [0.0], EXTRA: [0.0]}
    original = delta_timestamps.copy()
    transform = Mock(side_effect=lambda sample: sample)
    dataset = _load(
        local_dataset,
        split_name=split,
        observation_blacklist=blacklist,
        delta_timestamps=delta_timestamps,
        data_transforms=transform,
    )
    sample = dataset[1]
    for key in blacklist:
        assert key not in dataset.features
        assert key not in dataset.hf_dataset.column_names
        assert key not in sample
        assert f"{key}_is_pad" not in sample
        assert key not in dataset.delta_timestamps
        assert key not in dataset.reader.delta_indices
        assert key not in transform.call_args.args[0]
    assert WRIST in sample
    assert all(HEAD not in str(path) for path, _ in decode_calls)
    assert delta_timestamps == original
    episode = 1 if split == "validation" else 0
    assert int(sample["episode_index"]) == episode
    torch.testing.assert_close(sample[STATE][:, 0], torch.tensor([episode * 10.0, episode * 10.0 + 1]))
    torch.testing.assert_close(sample["action"][:, 0], torch.tensor([episode * 10.0 + 1, episode * 10.0 + 2]))
    assert all(
        key in json.loads((local_dataset / "meta/info.json").read_text())["features"] for key in blacklist
    )
    assert all(key in _load(local_dataset).features for key in blacklist)


def test_blacklisted_image_is_not_decoded(local_dataset, decode_calls, monkeypatch):
    def fail_decode(*args, **kwargs):
        raise AssertionError("blacklisted image decoded")

    monkeypatch.setattr(datasets.Image, "decode_example", fail_decode)
    dataset = _load(local_dataset, observation_blacklist=[IMAGE])
    assert IMAGE not in dataset[0]


def test_blacklisted_delta_is_removed_before_validation(local_dataset, decode_calls):
    # This offset is not a frame multiple at 10 Hz, but its camera is excluded.
    dataset = _load(local_dataset, observation_blacklist=[HEAD], delta_timestamps={HEAD: [0.15]})
    assert HEAD not in dataset[0]


@pytest.mark.parametrize("blacklist", [None, [], ["observation.does_not_exist"]])
def test_unfiltered_features_remain_available(local_dataset, decode_calls, blacklist):
    dataset = _load(local_dataset, observation_blacklist=blacklist)
    sample = dataset[0]
    assert {HEAD, WRIST, IMAGE, STATE, EXTRA, "action"}.issubset(sample)
    assert {HEAD, WRIST}.issubset(dataset.meta.video_keys)
    assert {path.parts[-3] for path, _ in decode_calls} == {HEAD, WRIST}


def test_constructor_uses_lightweight_cache_check(local_dataset, monkeypatch):
    original_exists = Path.exists
    original_try_load = DatasetReader.try_load
    cache_checks = []
    video_checks = []

    def try_load(reader):
        check = Mock(wraps=reader._check_cached_episodes_sufficient)
        reader._check_cached_episodes_sufficient = check
        cache_checks.append(check)
        return original_try_load(reader)

    def exists(path):
        if path.suffix == ".mp4":
            video_checks.append(path)
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(DatasetReader, "try_load", try_load)
    dataset = _load(local_dataset, split_name="all")
    assert len(dataset) == 6
    assert len(cache_checks) == 1
    cache_checks[0].assert_called_once_with()
    assert video_checks == []
    assert (
        type(dataset.reader)._check_cached_episodes_sufficient
        is not DatasetReader._check_cached_episodes_sufficient
    )


def test_missing_video_fails_at_decode_not_construction(local_dataset, tmp_path, decode_calls, monkeypatch):
    root = tmp_path / "missing-videos"
    shutil.copytree(local_dataset, root)
    shutil.rmtree(root / "videos")
    download = Mock(side_effect=AssertionError("unexpected Hub download"))
    monkeypatch.setattr(LeRobotDataset, "_download", download)
    dataset = _load(root)
    download.assert_not_called()
    with pytest.raises(FileNotFoundError, match="videos"):
        dataset[0]


def test_blacklisted_missing_videos_are_not_needed(local_dataset, tmp_path, monkeypatch):
    root = tmp_path / "blacklisted-videos"
    shutil.copytree(local_dataset, root)
    shutil.rmtree(root / "videos")
    download = Mock(side_effect=AssertionError("unexpected Hub download"))
    monkeypatch.setattr(LeRobotDataset, "_download", download)
    dataset = _load(root, observation_blacklist=[HEAD, WRIST])
    assert STATE in dataset[0]
    download.assert_not_called()


def test_recreated_reader_preserves_blacklist(local_dataset, decode_calls):
    dataset = _load(local_dataset, observation_blacklist=[HEAD, IMAGE, EXTRA])
    dataset.reader = None
    assert {HEAD, IMAGE, EXTRA}.isdisjoint(dataset[0])
    assert (
        type(dataset.reader)._check_cached_episodes_sufficient
        is not DatasetReader._check_cached_episodes_sufficient
    )


def test_native_lerobot_reader_is_unchanged(local_dataset):
    dataset = LeRobotDataset(repo_id="test/local-reader", root=local_dataset)
    assert type(dataset.reader) is DatasetReader
    assert {HEAD, WRIST, IMAGE, EXTRA}.issubset(dataset.features)


@pytest.mark.parametrize("missing", ["all_parquet", "selected_episode"])
def test_missing_parquet_still_uses_download_path(local_dataset, tmp_path, monkeypatch, missing):
    root = tmp_path / "incomplete-data"
    shutil.copytree(local_dataset, root)
    if missing == "all_parquet":
        shutil.rmtree(root / "data")
    else:
        for path in (root / "data").glob("*/*.parquet"):
            table = pq.read_table(path)
            table = table.filter(pa.compute.equal(table["episode_index"], 0))
            pq.write_table(table, path)
    download = Mock(side_effect=RuntimeError("download requested"))
    monkeypatch.setattr(LeRobotDataset, "_download", download)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.get_safe_version", lambda repo, revision: revision)
    with pytest.raises(RuntimeError, match="download requested"):
        _load(root, split_name="all")
    download.assert_called_once_with(True)


def test_cached_selected_episode_does_not_require_other_episodes(local_dataset, tmp_path, decode_calls):
    root = tmp_path / "selected-data"
    shutil.copytree(local_dataset, root)
    for path in (root / "data").glob("*/*.parquet"):
        table = pq.read_table(path)
        table = table.filter(pa.compute.equal(table["episode_index"], 1))
        pq.write_table(table, path)
    dataset = _load(root, split_name="validation", observation_blacklist=[HEAD])
    assert len(dataset) == 3
    assert int(dataset[0]["episode_index"]) == 1
    assert HEAD not in dataset[0]


def test_factory_applies_blacklist(local_dataset, decode_calls):
    from rho.common.types import FeatureType, NormalizationMode, PolicyFeature
    from rho.datasets.lerobot_dataset import LeRobotDatasetConfig

    config = LeRobotDatasetConfig(
        repo_id="test/local-reader",
        root_dir=str(local_dataset),
        observation_blacklist=[HEAD, IMAGE, EXTRA],
        features={
            STATE: PolicyFeature(FeatureType.STATE, (2,)),
            "action": PolicyFeature(FeatureType.ACTION, (2,)),
            WRIST: PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        normalization_mapping=dict.fromkeys(FeatureType, NormalizationMode.IDENTITY),
    )
    dataset = config.make_dataset()
    assert {HEAD, IMAGE, EXTRA}.isdisjoint(dataset.features)
    assert {HEAD, IMAGE, EXTRA}.isdisjoint(dataset[0])
    assert WRIST in dataset[0]


def test_explicit_refresh_still_downloads(local_dataset, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata_download = Mock()
    download = Mock()
    monkeypatch.setattr(LeRobotDatasetMetadata, "_pull_from_repo", metadata_download)
    monkeypatch.setattr(LeRobotDataset, "_download", download)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.get_safe_version", lambda repo, revision: revision)
    monkeypatch.setattr("lerobot.datasets.dataset_metadata.get_safe_version", lambda repo, revision: revision)
    dataset = _load(local_dataset, force_cache_sync=True, observation_blacklist=[HEAD])
    assert len(dataset) == 3
    assert HEAD not in dataset.features
    assert metadata_download.called
    download.assert_called_once_with(True)

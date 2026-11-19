"""Regression tests for composing ordinary and chunked Parquet statistics."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from rho.utils.recompute_lerobot_stats_parquet import compute_chunked_stats_from_parquet, main


@pytest.fixture
def parquet_dataset(tmp_path):
    dataset_path = tmp_path / "dataset"
    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "data").mkdir()
    features = {
        "action": {"dtype": "float32", "shape": [1]},
        "observation.state": {"dtype": "float32", "shape": [1]},
        "observation.images.camera": {"dtype": "video", "shape": [3, 8, 8]},
    }
    (dataset_path / "meta" / "info.json").write_text(json.dumps({"features": features}))
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "action": [[0.0], [10.0], [20.0]],
            "observation.state": [[-1.0], [9.0], [19.0]],
        }
    ).to_parquet(dataset_path / "data" / "episode.parquet", index=False)
    # Leave out min/max so the exact-quantile path is used for all output locations.
    stale_stats = {
        "action": {
            "mean": [-44.0],
            "std": [44.0],
            "q01": [-88.0],
            "q99": [88.0],
            "q01_chunk": [-77.0],
            "q99_chunk": [77.0],
            "mean_chunk2": [[-66.0], [-66.0]],
        },
        "observation.state": {"q01": [-55.0], "q99": [55.0]},
        "observation.images.camera": {"mean": [[[0.5]], [[0.4]], [[0.3]]]},
    }
    (dataset_path / "meta" / "stats.json").write_text(json.dumps(stale_stats))
    return dataset_path, stale_stats


def _run_stats_cli(monkeypatch, dataset_path, stats_type, output_path=None):
    args = [
        "recompute_lerobot_stats_parquet.py",
        "--dataset_path",
        str(dataset_path),
        "--stats_type",
        stats_type,
        "--chunk_sizes",
        "1",
        "2",
    ]
    if output_path is not None:
        args.extend(["--output_path", str(output_path)])
    monkeypatch.setattr(sys, "argv", args)
    main()
    result_path = output_path if output_path is not None else dataset_path / "meta" / "stats.json"
    return json.loads(result_path.read_text())


@pytest.mark.parametrize("output_location", ["new", "existing", "in_place"])
def test_both_modes_keep_fresh_quantiles(parquet_dataset, tmp_path, monkeypatch, output_location):
    dataset_path, stale_stats = parquet_dataset
    metadata_path = dataset_path / "meta" / "stats.json"
    metadata_before = metadata_path.read_bytes()
    quantiles = _run_stats_cli(monkeypatch, dataset_path, "quantile", tmp_path / "quantiles.json")

    output_path = None if output_location == "in_place" else tmp_path / "combined.json"
    if output_location == "existing":
        output_path.write_text(json.dumps(stale_stats))
    combined = _run_stats_cli(monkeypatch, dataset_path, "both", output_path)

    for key, expected in [("action", [0.2, 19.8]), ("observation.state", [-0.8, 18.8])]:
        for name, value in zip(("q01", "q99"), expected, strict=True):
            assert combined[key][name] == quantiles[key][name]
            np.testing.assert_allclose(combined[key][name], [value])

    if output_location == "new":
        for key in ("action", "observation.state"):
            for name in ("mean", "std", "min", "max"):
                assert combined[key][name] == quantiles[key][name]
    else:
        for name in ("mean", "std"):
            assert combined["action"][name] == stale_stats["action"][name]

    # Requested chunk statistics must also replace inherited values.
    action = combined["action"]
    for name in ("mean_chunk", "min_chunk", "max_chunk", "q01_chunk", "q99_chunk"):
        np.testing.assert_allclose(action[name], [1.0])
    np.testing.assert_allclose(action["std_chunk"], [0.0])
    np.testing.assert_allclose(action["mean_chunk2"], [[1.0], [23.0 / 3.0]])
    np.testing.assert_allclose(action["q01_chunk2"], [[1.0], [1.2]])
    np.testing.assert_allclose(action["q99_chunk2"], [[1.0], [11.0]])
    assert combined["observation.images.camera"] == stale_stats["observation.images.camera"]
    if output_path is not None:
        assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("existing_output", [False, True])
def test_chunk_mode_preserves_custom_output_stats(parquet_dataset, tmp_path, monkeypatch, existing_output):
    dataset_path, stale_stats = parquet_dataset
    metadata_path = dataset_path / "meta" / "stats.json"
    metadata_before = metadata_path.read_bytes()
    output_path = tmp_path / "chunks.json"
    destination_stats = {
        "action": {
            "q01": [-2.0],
            "q99": [2.0],
            "count": [123],
            "mean_chunk": [-66.0],
            "mean_chunk3": [[-7.0], [-8.0], [-9.0]],
        },
        "observation.environment": {"mean": [42.0]},
    }
    if existing_output:
        output_path.write_text(json.dumps(destination_stats))

    result = _run_stats_cli(monkeypatch, dataset_path, "chunk", output_path)

    expected_action = destination_stats["action"] if existing_output else stale_stats["action"]
    for name in ("q01", "q99"):
        assert result["action"][name] == expected_action[name]
    if existing_output:
        for name in ("count", "mean_chunk3"):
            assert result["action"][name] == destination_stats["action"][name]
        assert result["observation.environment"] == destination_stats["observation.environment"]
    assert result["observation.images.camera"] == stale_stats["observation.images.camera"]
    assert result["observation.state"] == stale_stats["observation.state"]
    np.testing.assert_allclose(result["action"]["mean_chunk"], [1.0])
    np.testing.assert_allclose(result["action"]["mean_chunk2"], [[1.0], [23.0 / 3.0]])
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("base_stats", [{}, {"action": {"q01": [-3.0], "mean_chunk": [-99.0]}}])
def test_explicit_chunk_base_stats_override_output_without_mutation(parquet_dataset, tmp_path, base_stats):
    dataset_path, stale_stats = parquet_dataset
    output_path = tmp_path / "chunks.json"
    output_path.write_text(json.dumps({"action": {"q01": [-999.0]}, "output.only": {"mean": [7.0]}}))
    output_before = output_path.read_bytes()
    base_before = json.dumps(base_stats)

    result = compute_chunked_stats_from_parquet(
        dataset_path=dataset_path,
        action_keys=["action"],
        action_to_state_mapping={"action": "observation.state"},
        action_to_type_mapping={"action": "position"},
        chunk_sizes=[1],
        output_path=output_path,
        base_stats=base_stats,
    )

    expected = base_stats.get("action", stale_stats["action"])
    assert result["action"]["q01"] == expected["q01"]
    np.testing.assert_allclose(result["action"]["mean_chunk"], [1.0])
    assert "output.only" not in result
    assert json.dumps(base_stats) == base_before
    assert output_path.read_bytes() == output_before

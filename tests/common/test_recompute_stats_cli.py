"""Exercise the public statistics command with real Parquet files."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("compute_mode", ["single_pass", "two_pass", "quantile_only"])
@pytest.mark.parametrize("custom_output", [False, True], ids=["in_place", "custom"])
def test_combined_cli_matches_separate_modes(tmp_path, compute_mode, custom_output):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data").mkdir()
    features = {
        "action.joints": {"dtype": "float32", "shape": [1]},
        "observation.joints": {"dtype": "float32", "shape": [1]},
        "observation.images.camera": {"dtype": "video", "shape": [3, 8, 8]},
    }
    (dataset / "meta" / "info.json").write_text(json.dumps({"features": features}))
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "action.joints": [[0.0], [10.0], [20.0]],
            "observation.joints": [[-1.0], [9.0], [19.0]],
        }
    ).to_parquet(dataset / "data" / "episode.parquet", index=False)
    mapping = tmp_path / "action_mapping.yaml"
    mapping.write_text("action.joints:\n  state_key: observation.joints\n  action_type: POSITION\n")
    metadata = {
        "action.joints": {
            "q01": [-88.0],
            "q99": [88.0],
            "q02_chunk2": [[-77.0], [-77.0]],
            "mean_chunk3": [[100.0]] * 3,
            "calibration": [7.0],
        },
        "observation.joints": {"q01": [-55.0], "q99": [55.0]},
        "observation.images.camera": {"mean": [[[0.5]], [[0.4]], [[0.3]]]},
    }
    metadata_path = dataset / "meta" / "stats.json"
    metadata_bytes = json.dumps(metadata).encode()
    destination = {
        "action.joints": {
            **metadata["action.joints"],
            "mean_chunk3": [[90.0]] * 3,
            "calibration": [12.0],
        },
        "observation.joints": metadata["observation.joints"],
        "output.only": {"mean": [42.0]},
    }
    bounds = {"action.joints": {"min_chunk2": [[1.0], [1.0]], "max_chunk2": [[1.0], [11.0]]}}
    results = {}
    for stats_type in ("quantile", "chunk", "both"):
        # Every invocation starts from the same metadata and destination contents.
        metadata_path.write_bytes(metadata_bytes)
        output = tmp_path / f"{stats_type}.json" if custom_output else metadata_path
        if custom_output:
            output.write_text(json.dumps(destination))
        bounds_path = output.with_name(f"{output.stem}_first_pass_bounds{output.suffix}")
        if compute_mode == "quantile_only" and stats_type != "quantile":
            bounds_path.write_text(json.dumps(bounds))
        command = [
            sys.executable,
            "-m",
            "rho.utils.recompute_lerobot_stats_parquet",
            "--dataset_path",
            str(dataset),
            "--stats_type",
            stats_type,
            "--compute_mode",
            compute_mode,
            "--action_mapping",
            str(mapping),
            "--chunk_sizes",
            "2",
            "--chunk_quantiles",
            "1",
            "99",
            "2",
            "98",
        ]
        if custom_output:
            command.extend(["--output_path", str(output)])
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results[stats_type] = json.loads(output.read_text())
        if custom_output:
            assert metadata_path.read_bytes() == metadata_bytes
        if compute_mode == "quantile_only" and stats_type != "quantile":
            assert f"Loading precomputed bounds from {bounds_path}" in completed.stdout
        if compute_mode == "two_pass" and stats_type != "quantile":
            assert json.loads(bounds_path.read_text()) == bounds

    combined = results["both"]
    for key in ("action.joints", "observation.joints"):
        for field in ("q01", "q99"):
            assert combined[key][field] == results["quantile"][key][field]
            assert combined[key][field] != metadata[key][field]
    action = combined["action.joints"]
    for field in ("mean", "std", "min", "max", "q01", "q99", "q02", "q98"):
        np.testing.assert_allclose(
            action[f"{field}_chunk2"], results["chunk"]["action.joints"][f"{field}_chunk2"]
        )
    np.testing.assert_allclose(action["mean_chunk2"], [[1.0], [23.0 / 3.0]])
    np.testing.assert_allclose(action["min_chunk2"], [[1.0], [1.0]])
    np.testing.assert_allclose(action["max_chunk2"], [[1.0], [11.0]])
    assert action["q02_chunk2"] != metadata["action.joints"]["q02_chunk2"]
    preserved = destination if custom_output else metadata
    for field in ("mean_chunk3", "calibration"):
        assert action[field] == preserved["action.joints"][field]
        assert results["chunk"]["action.joints"][field] == preserved["action.joints"][field]
    assert combined["observation.images.camera"] == metadata["observation.images.camera"]
    if custom_output:
        assert combined["output.only"] == destination["output.only"]
        assert results["chunk"]["output.only"] == destination["output.only"]

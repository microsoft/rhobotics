"""
Recompute LeRobot dataset statistics directly from parquet files.

This script reads parquet files directly instead of using the dataset's dataloader,
which avoids loading image/video data and is much faster for computing statistics.

Supports computing:
- Quantile statistics (q01/q99) for all numeric keys
- Chunked delta action statistics (mean, std, min, max, q02, q98)
- Both types of statistics

Usage:
    # Compute only quantile stats (q01/q99) - updates meta/stats.json in place
    python rho/utils/recompute_lerobot_stats_parquet.py \
        --dataset_path /data/simran/agibot_test/task_362 \
        --stats_type quantile

    # Compute only chunked delta action stats
    python rho/utils/recompute_lerobot_stats_parquet.py \
        --dataset_path /data/simran/agibot_test/task_362 \
        --stats_type chunk \
        --action_mapping config/datasets/agibot/action_mapping.yaml \
        --output_path config/datasets/agibot/task_362

    # Compute both quantile and chunked stats
    python rho/utils/recompute_lerobot_stats_parquet.py \
        --dataset_path /data/simran/agibot_test/task_362 \
        --stats_type both \
        --action_mapping config/datasets/agibot/action_mapping.yaml \
        --output_path config/datasets/agibot/task_362
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from rho.common.transforms import DeltaActions
from rho.utils.normalize import RunningStats, RunningStatsChunked

# =============================================================================
# Common Utilities
# =============================================================================


def get_parquet_files(dataset_path: Path) -> list[Path]:
    """
    Get list of parquet files from LeRobot dataset.

    Args:
        dataset_path: Path to the LeRobot dataset root directory.

    Returns:
        List of parquet file paths.
    """
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def load_feature_info(dataset_path: Path) -> dict[str, dict]:
    """
    Load feature information from meta/info.json.

    Args:
        dataset_path: Path to the LeRobot dataset root directory.

    Returns:
        Dict mapping feature names to their info (dtype, shape, etc.).
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"Warning: No info.json found at {info_path}")
        return {}

    with open(info_path) as f:
        info = json.load(f)

    return info.get("features", {})


def load_dataset_stats(dataset_path: Path) -> dict:
    """Load existing stats from dataset metadata (stats.json)."""
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        print(f"Loading existing stats from {stats_path}")
        with open(stats_path) as f:
            return json.load(f)

    print("Warning: No existing stats found in dataset metadata")
    return {}


# =============================================================================
# Quantile Statistics Utilities
# =============================================================================


def get_numeric_keys_from_features(
    feature_info: dict[str, dict],
) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    """
    Get list of numeric column keys and their shapes from feature info.

    Args:
        feature_info: Dict mapping feature names to their info.

    Returns:
        Tuple of (list of numeric key names, dict mapping keys to their shapes).
    """
    numeric_keys = []
    key_shapes: dict[str, tuple[int, ...]] = {}

    # Columns to skip (metadata columns and video/image columns)
    skip_columns = {"episode_index", "frame_index", "index", "timestamp", "task_index"}
    skip_dtypes = {"video", "image"}

    for key, info in feature_info.items():
        if key in skip_columns:
            continue

        dtype = info.get("dtype", "")
        if dtype in skip_dtypes:
            continue

        # Check if it's a numeric dtype
        if dtype in ("float32", "float64", "int32", "int64", "float", "int"):
            numeric_keys.append(key)
            shape = info.get("shape", [1])
            key_shapes[key] = tuple(shape)

    return numeric_keys, key_shapes


def recursive_to_numpy(item) -> np.ndarray:
    """
    Recursively convert a nested list/array to a flattened numpy array.

    Handles cases where parquet stores data as numpy object arrays containing
    inner numpy arrays (e.g., array([array([...]), array([...])], dtype=object)).

    Args:
        item: A scalar, list, or nested list/array of numeric values.

    Returns:
        Flattened numpy array of float32.
    """
    if isinstance(item, np.ndarray):
        # Check if it's an object array (contains nested arrays)
        if item.dtype == object:
            # Recursively process each element
            arrays = [recursive_to_numpy(x) for x in item]
            return np.concatenate(arrays)
        else:
            return item.astype(np.float32).flatten()
    elif isinstance(item, (list, tuple)):
        # Recursively convert each element and concatenate
        arrays = [recursive_to_numpy(x) for x in item]
        return np.concatenate(arrays)
    else:
        # Scalar value
        return np.array([item], dtype=np.float32)


def load_parquet_for_quantile(
    parquet_path: Path,
    keys: list[str],
    key_shapes: dict[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    """
    Load data from a single parquet file into numpy arrays for quantile computation.

    Flattens multi-dimensional arrays to 2D (batch, flattened_dims) for RunningStats.

    Args:
        parquet_path: Path to the parquet file.
        keys: List of column keys to load.
        key_shapes: Dict mapping keys to their original shapes.

    Returns:
        Dict mapping column names to numpy arrays of shape (batch, flattened_dims).
    """
    df = pd.read_parquet(parquet_path)

    data_dict: dict[str, np.ndarray] = {}
    for col in keys:
        if col in df.columns:
            col_data = df[col].tolist()

            # Get expected shape and compute flat dimension
            expected_shape = key_shapes.get(col, (1,))
            flat_dim = int(np.prod(expected_shape))
            batch_size = len(col_data)

            # Convert each element to numpy array using recursive flattening
            arrays = []
            for item in col_data:
                item_arr = recursive_to_numpy(item)
                arrays.append(item_arr)

            arr = np.stack(arrays, axis=0)

            # Ensure shape is (batch, flat_dim)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.shape[1] != flat_dim:
                # Reshape if needed
                arr = arr.reshape(batch_size, flat_dim)

            data_dict[col] = arr

    return data_dict


def to_nested_list(arr: np.ndarray, shape: tuple[int, ...]) -> list:
    """
    Convert a flattened numpy array to a nested Python list with the given shape.

    Args:
        arr: Flattened numpy array.
        shape: Target shape for the nested list.

    Returns:
        Nested Python list with the specified shape.
    """
    if arr is None:
        return []

    # Reshape to target shape
    if len(shape) > 1:
        arr = arr.reshape(shape)

    # Recursively convert to nested Python lists
    def to_python_list(a):
        if isinstance(a, np.ndarray):
            return [to_python_list(x) for x in a]
        elif isinstance(a, (np.floating, np.integer)):
            return float(a)
        else:
            return a

    return to_python_list(arr)


def compute_quantile_stats_from_parquet(
    dataset_path: Path,
    compute_mode: str = "single_pass",
    output_path: Path | None = None,
) -> tuple[dict, Path]:
    """
    Compute q01/q99 statistics for all numeric keys directly from parquet files.

    Args:
        dataset_path: Path to LeRobot dataset.
        compute_mode: "single_pass" or "two_pass".
        output_path: Path to save output stats. If file exists, stats will be
                     appended/merged. Defaults to dataset_path/meta/stats.json.

    Returns:
        Tuple of (computed statistics dict, output path).
    """
    # Get parquet files
    print(f"\nLoading data from {dataset_path}")
    parquet_files = get_parquet_files(dataset_path)

    # Load feature info and get numeric keys with shapes
    feature_info = load_feature_info(dataset_path)
    numeric_keys, key_shapes = get_numeric_keys_from_features(feature_info)
    print(f"Found {len(numeric_keys)} numeric keys: {numeric_keys}")

    # Default output path to meta/stats.json if not provided
    if output_path is None:
        output_path = dataset_path / "meta" / "stats.json"

    # Load existing stats from output path if it exists
    existing_stats: dict | None = None
    precomputed_bounds: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    if output_path.exists():
        print(f"Loading existing stats from {output_path}")
        with open(output_path) as f:
            existing_stats = json.load(f)

        # Extract min/max bounds from existing stats
        precomputed_bounds = {}
        for key in numeric_keys:
            if key in existing_stats and "min" in existing_stats[key] and "max" in existing_stats[key]:
                # Flatten min/max arrays in case they're nested (like image stats)
                min_val = np.array(existing_stats[key]["min"]).flatten()
                max_val = np.array(existing_stats[key]["max"]).flatten()
                precomputed_bounds[key] = (min_val, max_val)
        print(f"Loaded precomputed bounds for {len(precomputed_bounds)} keys")

    print("\n" + "=" * 60)
    print("Computing q01/q99 statistics for all numeric keys")
    print("=" * 60)

    # Helper function to process all parquet files
    def process_all_parquet_files(
        stats_objects: dict[str, RunningStats | None],
        stage_name: str = "Processing",
    ) -> dict[str, RunningStats | None]:
        """Process all parquet files and update running stats objects."""
        for pf in tqdm(parquet_files, desc=f"{stage_name} parquet files"):
            try:
                data = load_parquet_for_quantile(pf, numeric_keys, key_shapes)
            except Exception as e:
                print(f"  Warning: Failed to load {pf}: {e}")
                continue

            for key in numeric_keys:
                if key not in data:
                    continue

                values = data[key]  # Already 2D from load_parquet_for_quantile

                # Get or initialize stats object
                stats_obj = stats_objects.get(key)
                if stats_obj is None:
                    if precomputed_bounds and key in precomputed_bounds:
                        known_min, known_max = precomputed_bounds[key]
                        stats_obj = RunningStats(known_min=known_min, known_max=known_max)
                    elif compute_mode == "two_pass" and stage_name == "First stage":
                        stats_obj = RunningStats(no_quantile=True)
                    else:
                        stats_obj = RunningStats()
                    stats_objects[key] = stats_obj

                # Update stats
                stats_obj.update(values)

        return stats_objects

    # Initialize stats objects
    stats_objects: dict[str, RunningStats | None] = dict.fromkeys(numeric_keys)

    if precomputed_bounds:
        # Quantile-only mode: use precomputed bounds
        print("\n--- Computing quantiles with precomputed min/max bounds ---")
        stats_objects = process_all_parquet_files(stats_objects, stage_name="Processing")

    elif compute_mode == "two_pass":
        # ===== FIRST PASS: Compute min/max bounds =====
        print("\n--- First pass: Computing min/max bounds (no quantile computation) ---")
        stats_objects = process_all_parquet_files(stats_objects, stage_name="First stage")

        # Extract min/max bounds from first pass
        computed_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in numeric_keys:
            stats_obj = stats_objects.get(key)
            if stats_obj is not None:
                stats = stats_obj.get_statistics()
                computed_bounds[key] = (stats.min, stats.max)
                print(
                    f"  {key}: min range [{stats.min.min():.4f}, {stats.min.max():.4f}], "
                    f"max range [{stats.max.min():.4f}, {stats.max.max():.4f}]"
                )

        # Save intermediate stats after first pass
        if output_path:
            first_pass_stats: dict[str, dict[str, list]] = {}
            for key, (known_min, known_max) in computed_bounds.items():
                first_pass_stats[key] = {
                    "min": known_min.tolist() if known_min.size > 1 else [float(known_min)],
                    "max": known_max.tolist() if known_max.size > 1 else [float(known_max)],
                }

            print(f"\nSaving first pass min/max bounds to: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(first_pass_stats, f, indent=2)

        # ===== SECOND PASS: Recompute with known bounds =====
        print("\n--- Second pass: Computing statistics with known bounds ---")

        # Re-initialize stats objects with known bounds
        stats_objects = {}
        for key in numeric_keys:
            if key in computed_bounds:
                known_min, known_max = computed_bounds[key]
                stats_objects[key] = RunningStats(known_min=known_min, known_max=known_max)
            else:
                stats_objects[key] = None

        stats_objects = process_all_parquet_files(stats_objects, stage_name="Second stage")

    else:
        # Single pass mode
        stats_objects = process_all_parquet_files(stats_objects, stage_name="Processing")

    # Extract final statistics
    output_stats: dict[str, dict[str, list]] = {}

    for key in numeric_keys:
        stats_obj = stats_objects.get(key)
        if stats_obj is None:
            print(f"  Warning: No data processed for {key}")
            continue

        stats = stats_obj.get_statistics()

        # Get original shape for this key
        original_shape = key_shapes.get(key, (1,))

        output_stats[key] = {
            "mean": to_nested_list(stats.mean, original_shape),
            "std": to_nested_list(stats.std, original_shape),
            "min": to_nested_list(stats.min, original_shape),
            "max": to_nested_list(stats.max, original_shape),
            "q01": to_nested_list(stats.q01, original_shape) if stats.q01 is not None else [],
            "q99": to_nested_list(stats.q99, original_shape) if stats.q99 is not None else [],
        }

        mean_val = stats.mean.mean() if stats.mean.size > 1 else float(stats.mean)
        std_val = stats.std.mean() if stats.std.size > 1 else float(stats.std)
        print(f"  {key}: mean={mean_val:.4f}, std={std_val:.4f}")

    print("\n" + "=" * 60)
    print("Quantile statistics computed successfully!")
    print("=" * 60)

    # Merge stats - start with existing stats if available
    combined_stats = {}

    if existing_stats:
        for key, stat_dict in existing_stats.items():
            combined_stats[key] = {}
            for stat_name, stat_value in stat_dict.items():
                if isinstance(stat_value, np.ndarray):
                    combined_stats[key][stat_name] = stat_value.tolist()
                else:
                    combined_stats[key][stat_name] = stat_value

    # Add/update with new stats (quantiles)
    for key, new_stats in output_stats.items():
        if key not in combined_stats:
            combined_stats[key] = {}
        # Only add q01/q99 to existing keys when we have existing stats
        if existing_stats:
            if "q01" in new_stats and new_stats["q01"]:
                combined_stats[key]["q01"] = new_stats["q01"]
            if "q99" in new_stats and new_stats["q99"]:
                combined_stats[key]["q99"] = new_stats["q99"]
        else:
            # No existing stats - add all computed stats
            combined_stats[key].update(new_stats)

    return combined_stats, output_path


# =============================================================================
# Chunked Statistics Utilities
# =============================================================================


def infer_action_type_from_key(action_key: str) -> str:
    """
    Infer action type based on action key naming conventions.

    Args:
        action_key: The action feature key string.

    Returns:
        Inferred action type string matching DeltaActions transform types:
        "joint_position", "ee_rpy_pos", "ee_quat_pos", or "ee_6d_pos"
    """
    if "ee_6d" in action_key:
        return "ee_6d_pos"
    elif "ee_quat" in action_key:
        return "ee_quat_pos"
    elif "ee_rpy" in action_key:
        return "ee_rpy_pos"
    elif "joint" in action_key:
        return "joint_position"
    else:
        return "joint_position"


def load_parquet_for_chunks(
    parquet_path: Path,
    all_keys: set[str],
) -> dict[str, torch.Tensor]:
    """
    Load data from a single parquet file into torch tensors for chunk computation.

    Args:
        parquet_path: Path to the parquet file.
        all_keys: Set of column keys to load.

    Returns:
        Dict mapping column names to tensors.
    """
    df = pd.read_parquet(parquet_path)

    tensor_dict: dict[str, torch.Tensor] = {}
    for col in all_keys:
        if col in df.columns:
            col_data = df[col].tolist()
            arr = np.array(col_data, dtype=np.float32)
            tensor_dict[col] = torch.from_numpy(arr)

    return tensor_dict


def build_action_chunks(
    data: dict[str, torch.Tensor],
    action_keys: list[str],
    state_keys: list[str],
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """
    Build action chunks from tensor data, organized by episode.

    Args:
        data: Dict mapping column names to tensors.
        action_keys: List of action keys to build chunks for.
        state_keys: List of corresponding state keys.
        chunk_size: Size of action chunks to build.

    Returns:
        Dict mapping keys to tensors:
            - Each action_key maps to tensor of shape (N, chunk_size, action_dim)
            - Each state_key maps to tensor of shape (N, state_dim)
    """
    episode_indices = data["episode_index"]

    # Use frame_index if available, otherwise use index
    frame_indices = data["frame_index"] if "frame_index" in data else data["index"]

    # Get unique episodes
    unique_episodes = torch.unique(episode_indices)

    # Initialize lists for each key
    chunks_dict: dict[str, list[torch.Tensor]] = {}
    for key in action_keys + state_keys:
        chunks_dict[key] = []

    for episode_idx in tqdm(unique_episodes, desc="Building chunks", leave=False):
        # Get mask for this episode
        mask = episode_indices == episode_idx
        episode_frames = frame_indices[mask]

        # Sort by frame index within episode
        sort_indices = torch.argsort(episode_frames)

        # Get sorted data for each key
        episode_data: dict[str, torch.Tensor] = {}
        for key in action_keys + state_keys:
            if key in data:
                episode_data[key] = data[key][mask][sort_indices]

        episode_len = len(episode_frames)

        # Create chunks for each valid starting position
        for start_idx in range(episode_len):
            end_idx = start_idx + chunk_size

            # Build action chunks for each action key
            for action_key in action_keys:
                assert action_key in episode_data, f"Action key '{action_key}' not found in episode data"
                episode_actions = episode_data[action_key]

                if end_idx <= episode_len:
                    # Full chunk available
                    action_chunk = episode_actions[start_idx:end_idx]
                else:
                    # Pad with last action
                    action_dim = episode_actions.shape[-1] if episode_actions.ndim > 1 else 1
                    action_chunk = torch.zeros((chunk_size, action_dim), dtype=episode_actions.dtype)
                    available = episode_len - start_idx
                    action_chunk[:available] = episode_actions[start_idx:]
                    if available > 0:
                        action_chunk[available:] = episode_actions[-1]

                chunks_dict[action_key].append(action_chunk)

            # Get state at start_idx for each state key
            for state_key in state_keys:
                assert state_key in episode_data, f"State key '{state_key}' not found in episode data"
                chunks_dict[state_key].append(episode_data[state_key][start_idx])

    # Stack all chunks into tensors
    result: dict[str, torch.Tensor] = {}
    for key, chunks_list in chunks_dict.items():
        if chunks_list:
            result[key] = torch.stack(chunks_list)

    if not result:
        raise ValueError("No valid action chunks built")

    return result


def apply_delta_transform(
    actions: torch.Tensor,
    states: torch.Tensor,
    action_key: str,
    state_key: str,
    action_type: str,
) -> torch.Tensor:
    """
    Apply DeltaActions transform to action chunks.

    Args:
        actions: Tensor of shape (N, chunk_size, action_dim).
        states: Tensor of shape (N, state_dim).
        action_key: Key for action data.
        state_key: Key for state data.
        action_type: Type of action for proper delta computation.

    Returns:
        Delta actions tensor of shape (N, chunk_size, action_dim).
    """
    # Ensure float tensors
    actions = actions.float()
    states = states.float()

    # Create a dummy data dict for the transform
    data = {
        state_key: states,
        action_key: actions,
    }

    # Apply DeltaActions transform
    delta_transform = DeltaActions(
        state_key=state_key,
        action_key=action_key,
        relative_to_state=True,
        post_norm=False,
        action_type=action_type,
    )

    transformed = delta_transform(data)
    return transformed[action_key]


def compute_chunked_stats_from_parquet(
    dataset_path: Path,
    action_keys: list[str],
    action_to_state_mapping: dict[str, str],
    action_to_type_mapping: dict[str, str],
    chunk_sizes: list[int] | None = None,
    compute_mode: str = "single_pass",
    output_path: Path | None = None,
) -> dict:
    """
    Compute chunked statistics directly from parquet files.

    Args:
        dataset_path: Path to LeRobot dataset.
        action_keys: List of action feature keys to compute stats for.
        action_to_state_mapping: Mapping from action keys to state keys.
        action_to_type_mapping: Mapping from action keys to action types.
        chunk_sizes: List of chunk sizes to compute stats for.
        compute_mode: "single_pass", "two_pass", or "quantile_only".
        output_path: Path to save output stats.

    Returns:
        Dict of computed statistics.
    """
    if chunk_sizes is None:
        chunk_sizes = [50]

    # Get unique state keys
    state_keys = list(set(action_to_state_mapping.values()))

    # Get parquet files
    print(f"\nLoading metadata from {dataset_path}")
    parquet_files = get_parquet_files(dataset_path)

    # Load metadata
    meta_path = dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {meta_path}")

    # Columns to load (action keys, state keys, and episode_index)
    all_keys = set(action_keys) | set(state_keys) | {"episode_index", "frame_index", "index"}

    # Load existing stats
    existing_stats = load_dataset_stats(dataset_path)

    # ===== Compute per-timestep chunked statistics for each chunk size =====
    print("\n" + "=" * 60)
    print(f"Computing per-timestep chunked statistics for chunk sizes: {chunk_sizes}")
    print("=" * 60)

    # Load precomputed bounds if using quantile_only mode
    precomputed_bounds = None
    if compute_mode == "quantile_only" and output_path:
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        if stats_path.exists():
            print(f"Loading precomputed bounds from {stats_path}")
            with open(stats_path) as f:
                existing_output = json.load(f)
            precomputed_bounds = {}
            for key in action_keys:
                if key in existing_output:
                    precomputed_bounds[key] = {}
                    for stat_name, stat_value in existing_output[key].items():
                        if "min_chunk" in stat_name or "max_chunk" in stat_name:
                            precomputed_bounds[key][stat_name] = np.array(stat_value)
        else:
            print(f"Warning: No existing stats file found at {stats_path} for quantile_only mode")

    # Helper function to process all parquet files and update stats objects
    def process_all_parquet_files(
        stats_objects: dict[str, dict[int, RunningStatsChunked | None]],
        pass_name: str = "Processing",
    ) -> dict[str, dict[int, RunningStatsChunked | None]]:
        """Process all parquet files and update running stats objects."""
        for pf in tqdm(parquet_files, desc=f"{pass_name} parquet files"):
            try:
                data = load_parquet_for_chunks(pf, all_keys)
            except Exception as e:
                print(f"  Warning: Failed to load {pf}: {e}")
                continue

            # Process each chunk_size, building chunks for all action keys at once
            for chunk_size in chunk_sizes:
                # Build action chunks for all action keys at once
                try:
                    chunks_batch = build_action_chunks(data, action_keys, state_keys, chunk_size)
                except ValueError:
                    continue

                # Process each action_key
                for action_key in action_keys:
                    state_key = action_to_state_mapping.get(action_key, "observation.state")
                    action_type = action_to_type_mapping.get(
                        action_key, infer_action_type_from_key(action_key)
                    )

                    assert action_key in chunks_batch, f"Action key '{action_key}' not found in chunks batch"
                    assert state_key in chunks_batch, f"State key '{state_key}' not found in chunks batch"

                    actions = chunks_batch[action_key]
                    states = chunks_batch[state_key]

                    # Apply delta transform
                    delta_actions = apply_delta_transform(actions, states, action_key, state_key, action_type)
                    delta_actions_np = delta_actions.numpy()

                    # Get or initialize stats object
                    stats_obj = stats_objects[action_key][chunk_size]
                    if stats_obj is None:
                        stats_obj = RunningStatsChunked()
                        stats_objects[action_key][chunk_size] = stats_obj

                    # Process in batches for memory efficiency
                    batch_size = 1024
                    for i in range(0, len(delta_actions_np), batch_size):
                        batch = delta_actions_np[i : i + batch_size]
                        stats_obj.update(batch)

        return stats_objects

    # Initialize running stats objects for each (action_key, chunk_size) combination
    # Structure: stats_objects[action_key][chunk_size] = RunningStatsChunked or None (lazy init)
    stats_objects: dict[str, dict[int, RunningStatsChunked | None]] = {}
    for action_key in action_keys:
        stats_objects[action_key] = {}
        for chunk_size in chunk_sizes:
            if compute_mode == "quantile_only" and precomputed_bounds:
                # Use precomputed bounds
                min_key = f"min_chunk{chunk_size}" if chunk_size > 1 else "min_chunk"
                max_key = f"max_chunk{chunk_size}" if chunk_size > 1 else "max_chunk"
                if action_key in precomputed_bounds and min_key in precomputed_bounds[action_key]:
                    stats_objects[action_key][chunk_size] = RunningStatsChunked(
                        known_min=precomputed_bounds[action_key][min_key],
                        known_max=precomputed_bounds[action_key][max_key],
                    )
                else:
                    stats_objects[action_key][chunk_size] = None  # Lazy init
            else:
                stats_objects[action_key][chunk_size] = None  # Lazy init

    if compute_mode == "two_pass":
        # ===== FIRST PASS: Compute min/max bounds =====
        print("\n--- First pass: Computing min/max bounds (no quantile computation) ---")

        # Re-initialize stats objects with no_quantile=True for memory efficiency
        for action_key in action_keys:
            for chunk_size in chunk_sizes:
                stats_objects[action_key][chunk_size] = RunningStatsChunked(no_quantile=True)

        stats_objects = process_all_parquet_files(stats_objects, pass_name="First pass")

        # Extract min/max bounds from first pass
        computed_bounds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for action_key in action_keys:
            computed_bounds[action_key] = {}
            for chunk_size in chunk_sizes:
                stats_obj = stats_objects[action_key][chunk_size]
                if stats_obj is not None:
                    stats_dict = stats_obj.get_statistics()
                    computed_bounds[action_key][chunk_size] = (
                        stats_dict["min"],
                        stats_dict["max"],
                    )
                    print(
                        f"  {action_key} (chunk_size={chunk_size}): "
                        f"min range [{stats_dict['min'].min():.4f}, {stats_dict['min'].max():.4f}], "
                        f"max range [{stats_dict['max'].min():.4f}, {stats_dict['max'].max():.4f}]"
                    )

        # Save intermediate stats after first pass (for quantile_only mode)
        if output_path:
            first_pass_stats: dict[str, dict[str, list]] = {}
            for action_key in action_keys:
                first_pass_stats[action_key] = {}
                for chunk_size in chunk_sizes:
                    if action_key in computed_bounds and chunk_size in computed_bounds[action_key]:
                        known_min, known_max = computed_bounds[action_key][chunk_size]
                        if chunk_size == 1:
                            to_append = ""
                            min_key = f"min_chunk{to_append}"
                            max_key = f"max_chunk{to_append}"
                            first_pass_stats[action_key][min_key] = known_min.squeeze(0).tolist()
                            first_pass_stats[action_key][max_key] = known_max.squeeze(0).tolist()
                        else:
                            to_append = f"{chunk_size}"
                            first_pass_stats[action_key][f"min_chunk{to_append}"] = known_min.tolist()
                            first_pass_stats[action_key][f"max_chunk{to_append}"] = known_max.tolist()

            first_pass_file = output_path
            print(f"\nSaving first pass min/max bounds to: {first_pass_file}")
            first_pass_file.parent.mkdir(parents=True, exist_ok=True)
            with open(first_pass_file, "w") as f:
                json.dump(first_pass_stats, f, indent=2)

        # ===== SECOND PASS: Recompute with known bounds =====
        print("\n--- Second pass: Computing statistics with known bounds ---")

        # Re-initialize stats objects with known bounds
        stats_objects = {}
        for action_key in action_keys:
            stats_objects[action_key] = {}
            for chunk_size in chunk_sizes:
                if action_key in computed_bounds and chunk_size in computed_bounds[action_key]:
                    known_min, known_max = computed_bounds[action_key][chunk_size]
                    stats_objects[action_key][chunk_size] = RunningStatsChunked(
                        known_min=known_min,
                        known_max=known_max,
                    )
                else:
                    stats_objects[action_key][chunk_size] = None  # Lazy init

        stats_objects = process_all_parquet_files(stats_objects, pass_name="Second pass")

    else:
        # Single pass or quantile_only mode
        stats_objects = process_all_parquet_files(stats_objects, pass_name="Processing")

    # Extract final statistics from all running stats objects
    chunked_stats_output: dict[str, dict[str, list]] = {}

    for action_key in action_keys:
        chunked_stats_output[action_key] = {}
        for chunk_size in chunk_sizes:
            stats_obj = stats_objects[action_key][chunk_size]
            if stats_obj is None:
                print(f"  Warning: No data processed for {action_key} chunk_size={chunk_size}")
                continue

            # Get final statistics
            stats_dict = stats_obj.get_statistics()

            if chunk_size == 1:
                to_append = ""
                for key in stats_dict:
                    stats_dict[key] = stats_dict[key].squeeze(0)
            else:
                to_append = f"{chunk_size}"

            chunked_stats_output[action_key][f"mean_chunk{to_append}"] = stats_dict["mean"].tolist()
            chunked_stats_output[action_key][f"std_chunk{to_append}"] = stats_dict["std"].tolist()
            chunked_stats_output[action_key][f"min_chunk{to_append}"] = stats_dict["min"].tolist()
            chunked_stats_output[action_key][f"max_chunk{to_append}"] = stats_dict["max"].tolist()
            chunked_stats_output[action_key][f"q02_chunk{to_append}"] = stats_dict["q02"].tolist()
            chunked_stats_output[action_key][f"q98_chunk{to_append}"] = stats_dict["q98"].tolist()

            mean_min, mean_max = stats_dict["mean"].min(), stats_dict["mean"].max()
            print(f"  {action_key} (chunk_size={chunk_size}): mean range [{mean_min:.4f}, {mean_max:.4f}]")

    print("\n" + "=" * 60)
    print("Chunked statistics computed successfully!")
    print("=" * 60)

    # Merge with existing stats
    combined_stats = {}

    # Add existing stats (convert numpy arrays to lists)
    for key, stat_dict in existing_stats.items():
        combined_stats[key] = {}
        for stat_name, stat_value in stat_dict.items():
            if isinstance(stat_value, np.ndarray):
                combined_stats[key][stat_name] = stat_value.tolist()
            else:
                combined_stats[key][stat_name] = stat_value

    # Add chunked stats
    for key, chunked_values in chunked_stats_output.items():
        if key not in combined_stats:
            combined_stats[key] = {}
        combined_stats[key].update(chunked_values)

    return combined_stats


def replace_stats_with_identity(indices_dict: dict[str, list[int]], stats: dict) -> dict:
    """
    Replace statistics for specified features with identity stats.

    Args:
        indices_dict: Dictionary mapping feature keys to list of indices to set to identity.
        stats: Statistics dictionary.

    Returns:
        Modified statistics dictionary.
    """
    for key, indices in indices_dict.items():
        if key not in stats:
            print(f"Warning: Key '{key}' not found in stats, skipping identity replacement")
            continue

        for stat_name in list(stats[key].keys()):
            stat_value = np.array(stats[key][stat_name])
            if stat_value.ndim == 1:
                for idx in indices:
                    if idx < len(stat_value):
                        if "min" in stat_name or "q01" in stat_name or "q02" in stat_name:
                            stat_value[idx] = -1.0
                        elif (
                            "max" in stat_name
                            or "q99" in stat_name
                            or "q98" in stat_name
                            or "std" in stat_name
                        ):
                            stat_value[idx] = 1.0
                        elif "mean" in stat_name:
                            stat_value[idx] = 0.0
            elif stat_value.ndim == 2:
                for idx in indices:
                    if idx < stat_value.shape[1]:
                        if "min" in stat_name or "q01" in stat_name or "q02" in stat_name:
                            stat_value[:, idx] = -1.0
                        elif (
                            "max" in stat_name
                            or "q99" in stat_name
                            or "q98" in stat_name
                            or "std" in stat_name
                        ):
                            stat_value[:, idx] = 1.0
                        elif "mean" in stat_name:
                            stat_value[:, idx] = 0.0

            stats[key][stat_name] = stat_value.tolist()

        print(f"Replaced stats for '{key}' at indices {indices} with identity values")

    return stats


def load_action_mapping(path: str | None) -> dict[str, dict[str, str]] | None:
    """
    Load action mapping from a YAML file.

    Expected format:
        action.joint_position:
            state_key: observation.joint_position
            action_type: joint_position
        action.ee_quat_pos:
            state_key: observation.ee_quat_pos
            action_type: ee_quat_pos

    Args:
        path: Path to the YAML file containing action mappings.

    Returns:
        Dictionary mapping action keys to their state_key and action_type,
        or None if no path provided.
    """
    if path is None:
        return None

    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Action mapping file not found: {path}")

    with open(mapping_path) as f:
        mapping = yaml.safe_load(f)

    # Validate the mapping structure
    for action_key, config in mapping.items():
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid mapping for '{action_key}': expected dict with 'state_key' and 'action_type'"
            )
        if "state_key" not in config:
            raise ValueError(f"Missing 'state_key' for action '{action_key}'")
        if "action_type" not in config:
            raise ValueError(f"Missing 'action_type' for action '{action_key}'")

    print(f"Loaded action mapping with {len(mapping)} entries from {path}")
    return mapping


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Recompute LeRobot dataset statistics directly from parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute only quantile stats (q01/q99)
    python rho/utils/recompute_lerobot_stats.py \\
        --dataset_path /data/dataset \\
        --stats_type quantile

    # Compute only chunked delta action stats
    python rho/utils/recompute_lerobot_stats.py \\
        --dataset_path /data/dataset \\
        --stats_type chunk \\
        --action_mapping config/datasets/action_mapping.yaml \\
        --output_path config/datasets/stats

    # Compute both quantile and chunked stats
    python rho/utils/recompute_lerobot_stats.py \\
        --dataset_path /data/dataset \\
        --stats_type both \\
        --action_mapping config/datasets/action_mapping.yaml \\
        --output_path config/datasets/stats
        """,
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the LeRobot dataset directory."
    )
    parser.add_argument(
        "--stats_type",
        type=str,
        default="both",
        choices=["quantile", "chunk", "both"],
        help="Type of statistics to compute: "
        "'quantile' computes q01/q99 for all numeric keys, "
        "'chunk' computes chunked delta action stats, "
        "'both' computes all statistics (default).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output stats JSON file. If not provided, "
        "uses meta/stats.json. If file exists, new stats will be merged/appended.",
    )
    parser.add_argument(
        "--action_mapping",
        type=str,
        default=None,
        help="Path to YAML file mapping action keys to state_key and action_type. "
        "Format: action_key: {state_key: ..., action_type: ...}. "
        "Required for chunk stats.",
    )
    parser.add_argument(
        "--action_keys",
        type=str,
        nargs="+",
        default=["action"],
        help="List of action keys to compute stats for (if action_mapping not provided).",
    )
    parser.add_argument(
        "--chunk_sizes",
        type=int,
        nargs="+",
        default=[50],
        help="List of chunk sizes to compute stats for (for chunk stats).",
    )
    parser.add_argument(
        "--indices_dict",
        type=str,
        default=None,
        help="JSON string or path to JSON/YAML file mapping feature keys to indices for "
        "identity stats (to be used for rotation stats).",
    )
    parser.add_argument(
        "--compute_mode",
        type=str,
        default="single_pass",
        choices=["single_pass", "two_pass", "quantile_only"],
        help="Computation mode: "
        "'single_pass' computes everything at once (default), "
        "'two_pass' does first pass for min/max then second pass for quantiles with fixed bins, "
        "'quantile_only' uses min/max from existing output stats file (for chunk stats).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path) if args.output_path else dataset_path / "meta" / "stats.json"
    print(f"Output path: {output_path}")

    combined_stats: dict = {}

    # Compute quantile stats if requested
    if args.stats_type in ("quantile", "both"):
        print("\n" + "=" * 70)
        print("COMPUTING QUANTILE STATISTICS")
        print("=" * 70)

        quantile_stats, quantile_output_path = compute_quantile_stats_from_parquet(
            dataset_path=dataset_path,
            compute_mode=args.compute_mode if args.compute_mode != "quantile_only" else "single_pass",
            output_path=output_path,
        )
        combined_stats.update(quantile_stats)

        # Save quantile stats so the chunk step loads the updated q01/q99
        quantile_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting quantile statistics to: {quantile_output_path}")
        with open(quantile_output_path, "w") as f:
            json.dump(combined_stats, f, indent=2)

    # Compute chunked stats if requested
    if args.stats_type in ("chunk", "both"):
        print("\n" + "=" * 70)
        print("COMPUTING CHUNKED STATISTICS")
        print("=" * 70)

        # Load action mapping
        action_mapping = load_action_mapping(args.action_mapping)

        # Build action_to_state_mapping and action_to_type_mapping
        if action_mapping:
            action_keys = list(action_mapping.keys())
            action_to_state_mapping = {k: v["state_key"] for k, v in action_mapping.items()}
            action_to_type_mapping = {k: v["action_type"] for k, v in action_mapping.items()}
        else:
            action_keys = args.action_keys
            action_to_state_mapping = dict.fromkeys(action_keys, "observation.state")
            action_to_type_mapping = {k: infer_action_type_from_key(k) for k in action_keys}

        print(f"Action keys: {action_keys}")
        for k in action_keys:
            print(f"  {k} -> state: {action_to_state_mapping[k]}, type: {action_to_type_mapping[k]}")

        chunk_stats = compute_chunked_stats_from_parquet(
            dataset_path=dataset_path,
            action_keys=action_keys,
            action_to_state_mapping=action_to_state_mapping,
            action_to_type_mapping=action_to_type_mapping,
            chunk_sizes=args.chunk_sizes,
            compute_mode=args.compute_mode,
            output_path=output_path,
        )

        # Merge chunk stats with combined stats
        for key, stat_dict in chunk_stats.items():
            if key not in combined_stats:
                combined_stats[key] = {}
            combined_stats[key].update(stat_dict)

    # Parse indices_dict and apply identity stats if provided
    if args.indices_dict:
        indices_path = Path(args.indices_dict)
        if indices_path.is_file():
            with open(indices_path) as f:
                indices_dict = yaml.safe_load(f) if indices_path.suffix in (".yaml", ".yml") else json.load(f)
        else:
            try:
                indices_dict = json.loads(args.indices_dict)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"--indices_dict must be a valid JSON string or path to a JSON/YAML file. Error: {e}"
                ) from e

        combined_stats = replace_stats_with_identity(indices_dict, combined_stats)

    # Save final output for chunk or both modes
    if args.stats_type in ("chunk", "both"):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting combined statistics to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(combined_stats, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()

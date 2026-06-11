import argparse
import json
from pathlib import Path

import draccus
import numpy as np
import torch
import yaml
from draccus.parsers.config_parsers import YAMLParser
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torch.utils.data import DataLoader
from tqdm import tqdm

from rho.common.transforms import DeltaActions
from rho.common.types import FeatureType, NormalizationMode
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.datasets.multi_dataset import MultiDatasetConfig
from rho.policies.base import PolicyConfig
from rho.utils.normalize import RunningStatsChunked


def infer_action_type_from_key(action_key: str) -> str:
    """
    Infer action type based on action key naming conventions.

    Args:
        action_key: The action feature key string.

    Returns:
        Inferred action type string matching DeltaActions transform types:
        "joint_position", "ee_rpy_pos", "ee_quat_pos", or "ee_6d_pos"
    """
    # Check for specific end effector types (order matters - check most specific first)
    if "ee_6d" in action_key:
        return "ee_6d_pos"
    elif "ee_quat" in action_key:
        return "ee_quat_pos"
    elif "ee_rpy" in action_key:
        return "ee_rpy_pos"
    # Check for joint actions (e.g., action.joint_position)
    elif "joint" in action_key:
        return "joint_position"
    else:
        # Default to joint_position for unknown types
        return "joint_position"


def compute_dataset_stats(
    cfg: LeRobotDatasetConfig,
    output_path: str,
    batch_size: int | None = None,
    action_mapping: dict[str, dict[str, str]] | None = None,
):
    """
    Compute dataset statistics including chunked stats for action features.

    Args:
        cfg: LeRobotDatasetConfig with dataset configuration
        output_path: Path to save the computed statistics
        batch_size: Batch size for data loading
        action_mapping: Optional mapping from action keys to their state_key and action_type.
                       Format: {action_key: {"state_key": ..., "action_type": ...}}
    """
    # Turn off normalization by setting all mappings to identity
    for name in cfg.normalization_mapping:
        cfg.normalization_mapping[name] = NormalizationMode("IDENTITY")

    # Turn off observation_mapping since stats use original dataset keys
    original_observation_mapping = cfg.observation_mapping
    cfg.observation_mapping = None

    # Load existing stats directly from dataset metadata
    try:
        ds_meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root_dir)
        original_stats = ds_meta.stats if hasattr(ds_meta, "stats") else {}
        print(f"Loaded {len(original_stats)} stat entries from dataset metadata")
    except Exception as e:
        print(f"Warning: Could not load stats from dataset metadata: {e}")
        original_stats = {}

    if not original_stats:
        raise ValueError(
            "No stats found in dataset metadata. Please ensure the dataset has pre-computed statistics."
        )

    # Create reverse mapping: feature keys -> original keys
    reverse_mapping = {}  # remapped -> original
    if original_observation_mapping is not None:
        reverse_mapping = {v: k for k, v in original_observation_mapping.items()}

    # Set cfg.features.stats with original stats for lookup
    cfg.stats = original_stats

    # Get action keys to compute from action_mapping if provided, otherwise infer from cfg.features
    if action_mapping:
        action_keys_to_compute = list(action_mapping.keys())
        print(f"Using action keys from action_mapping: {action_keys_to_compute}")
    else:
        # Collect all ACTION features from cfg.features and map to original keys
        action_keys_to_compute = []
        for feature_key, feature in cfg.features.items():
            if feature.type == FeatureType.ACTION:
                original_key = reverse_mapping.get(feature_key, feature_key)
                action_keys_to_compute.append(original_key)

    if not action_keys_to_compute:
        print("Warning: No ACTION features found in cfg.features, skipping chunked stats computation")
        return
    print(f"Computing chunked stats for (original) action keys: {action_keys_to_compute}")

    # Build mapping from action keys to their corresponding state keys and action types
    action_to_state_mapping = {}
    action_to_type_mapping = {}

    for action_key in action_keys_to_compute:
        if action_mapping and action_key in action_mapping:
            # Use provided mapping
            state_key = action_mapping[action_key]["state_key"]
            action_type = action_mapping[action_key]["action_type"]
        else:
            # Fallback to inference-based mapping
            if "tactile" in action_key:
                state_key = "observations.tactile"
            elif "tcp_forces" in action_key:
                state_key = "observations.tcp_forces"
            elif "ee_6d_pos" in action_key:
                state_key = "observation.ee_6d_pos"
            elif "ee_quat_pos" in action_key:
                state_key = "observation.ee_quat_pos"
            elif "joint_position" in action_key:
                state_key = "observation.joint_position"
            elif "eef_position" in action_key:
                state_key = "observations.eef_position"
            else:
                # Fallback to default state key
                state_key = reverse_mapping.get("observation.state", "observation.state")
            action_type = infer_action_type_from_key(action_key)

        assert state_key in ds_meta.features, f"Expected '{state_key}' in dataset features"

        action_to_state_mapping[action_key] = state_key
        action_to_type_mapping[action_key] = action_type
        print(f"  {action_key} -> state_key: {state_key}, action_type: {action_type}")

    print("\nComputing per-dimension-only chunked statistics...")
    perdim_stats = compute_perdim_chunked_stats(
        cfg,
        batch_size=batch_size,
        action_keys_to_compute=action_keys_to_compute,
        action_to_state_mapping=action_to_state_mapping,
        action_to_type_mapping=action_to_type_mapping,
    )

    # Store all chunked stats with original keys
    chunked_stats_output = {}

    # Add per-dimension stats to output
    for original_key, stat in perdim_stats.items():
        if original_key not in chunked_stats_output:
            chunked_stats_output[original_key] = {}

        # Add per-dimension stats with "_chunk" suffix (no chunk size number)
        chunked_stats_output[original_key]["mean_chunk"] = stat["mean"]
        chunked_stats_output[original_key]["std_chunk"] = stat["std"]
        chunked_stats_output[original_key]["min_chunk"] = stat["min"]
        chunked_stats_output[original_key]["max_chunk"] = stat["max"]
        chunked_stats_output[original_key]["q02_chunk"] = stat["q02"]
        chunked_stats_output[original_key]["q98_chunk"] = stat["q98"]

        print(f"Added {original_key}: mean/std/min/max/q02/q98_chunk (per-dimension only)")

    # Compute chunked statistics for multiple chunk sizes
    chunk_sizes = [16, 32, 50]
    print(f"\nComputing chunked statistics for chunk sizes: {chunk_sizes}")
    print(f"Action keys to compute (original keys): {action_keys_to_compute}")

    for chunk_size in chunk_sizes:
        chunked_stats = compute_chunked_stats(
            cfg,
            chunk_size=chunk_size,
            batch_size=batch_size,
            action_keys_to_compute=action_keys_to_compute,
            action_to_state_mapping=action_to_state_mapping,
            action_to_type_mapping=action_to_type_mapping,
        )

        # chunked_stats already uses original keys, so just add to output
        for original_key, stat in chunked_stats.items():
            # Ensure the action key exists in chunked_stats_output
            if original_key not in chunked_stats_output:
                chunked_stats_output[original_key] = {}

            # Add each statistic with chunk size suffix
            chunked_stats_output[original_key][f"mean_chunk{chunk_size}"] = stat["mean"]
            chunked_stats_output[original_key][f"std_chunk{chunk_size}"] = stat["std"]
            chunked_stats_output[original_key][f"min_chunk{chunk_size}"] = stat["min"]
            chunked_stats_output[original_key][f"max_chunk{chunk_size}"] = stat["max"]
            chunked_stats_output[original_key][f"q02_chunk{chunk_size}"] = stat["q02"]
            chunked_stats_output[original_key][f"q98_chunk{chunk_size}"] = stat["q98"]

            print(f"Added {original_key}: mean/std/min/max/q02/q98_chunk{chunk_size}")

    print("\n" + "=" * 60)
    print("All chunked statistics computed successfully!")
    print("=" * 60)

    # Merge chunked stats into original stats
    combined_stats = {}
    for key, stat_dict in original_stats.items():
        # Convert numpy arrays to lists for JSON serialization
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

    # Save combined stats as JSON file
    output_path_obj = Path(output_path)
    stats_path = output_path_obj.parent / f"{output_path_obj.stem}_stats.json"

    print(f"\nWriting combined statistics to: {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(combined_stats, f, indent=2)

    print("Combined statistics saved with original dataset keys")


def _process_batch(
    batch: dict,
    action_keys: list[str],
    chunk_size: int,
    stats_dict: dict,
    action_to_state_mapping: dict[str, str] | None = None,
    action_to_type_mapping: dict[str, str] | None = None,
) -> None:
    """
    Process a batch by applying DeltaActions transform and updating statistics.

    Args:
        batch: Raw batch from dataloader
        action_keys: List of action feature keys (original dataset keys) to process
        chunk_size: Expected chunk size for actions
        stats_dict: Dictionary mapping keys to RunningStatsChunked objects
        action_to_state_mapping: Mapping from action keys to their corresponding state keys
        action_to_type_mapping: Mapping from action keys to their action types
    """
    batch_torch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

    for key in action_keys:
        # Get the appropriate state key for this action
        state_key = (
            action_to_state_mapping.get(key, "observation.state")
            if action_to_state_mapping
            else "observation.state"
        )

        # Get the action type from mapping or infer from key
        action_type = (
            action_to_type_mapping.get(key, infer_action_type_from_key(key))
            if action_to_type_mapping
            else infer_action_type_from_key(key)
        )

        # Apply DeltaActions transform for this specific action key
        delta_transform = DeltaActions(
            state_key=state_key,
            action_key=key,
            relative_to_state=True,
            post_norm=False,
            action_type=action_type,
        )
        batch_transformed = delta_transform(batch_torch)

        values = batch_transformed[key].numpy()

        # Ensure we have 3D array (batch, time, dims)
        if values.ndim == 2:
            values = values[:, np.newaxis, :]
        elif values.ndim == 1:
            values = values[:, np.newaxis, np.newaxis]

        # Truncate or pad to chunk_size if needed
        if values.shape[1] != chunk_size:
            if values.shape[1] > chunk_size:
                values = values[:, :chunk_size, :]
            else:
                pad_width = ((0, 0), (0, chunk_size - values.shape[1]), (0, 0))
                values = np.pad(values, pad_width, mode="constant")

        stats_dict[key].update(values)


def compute_chunked_stats(
    cfg: LeRobotDatasetConfig,
    chunk_size: int = 16,
    batch_size: int | None = None,
    action_keys_to_compute: list[str] | None = None,
    action_to_state_mapping: dict[str, str] | None = None,
    action_to_type_mapping: dict[str, str] | None = None,
):
    """
    Compute statistics for action chunks with temporal structure preserved.

    Args:
        cfg: LeRobotDatasetConfig with dataset configuration
        chunk_size: Size of action chunks (default: 16)
        batch_size: Batch size for data loading
        action_keys_to_compute: List of action keys (ORIGINAL dataset keys) to compute stats for
        action_to_state_mapping: Mapping from action keys to their corresponding state keys (ORIGINAL keys)
        action_to_type_mapping: Mapping from action keys to their action types

    Returns:
        Dict mapping original action keys to chunked statistics
    """
    if action_keys_to_compute is None:
        action_keys_to_compute = ["action"]
    print(f"\n{'=' * 80}")
    print(f"Computing chunked statistics (chunk_size={chunk_size})")
    print(f"{'=' * 80}\n")

    # Note: normalization should already be turned off in compute_dataset_stats()

    policy_cfg = PolicyConfig()
    policy_cfg._action_delta_indices = list(range(chunk_size))
    type(policy_cfg).action_delta_indices = property(lambda self: self._action_delta_indices)
    print(f"Using action_delta_indices: {policy_cfg.action_delta_indices}")

    dataset = cfg.make_dataset(policy_cfg=policy_cfg)

    if batch_size is None:
        batch_size = cfg.batch_size

    # Use the same num_workers settings that work for compute_dataset_stats
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": None,
        "num_workers": 0,  # Set to 0 to avoid video decoding issues with multiprocessing
        "persistent_workers": False,
        "drop_last": False,
    }
    data_loader = DataLoader(dataset, **dataloader_kwargs)

    print(f"Computing chunked stats for original keys: {action_keys_to_compute}")

    # Get existing stats (should be original stats passed from compute_dataset_stats)
    existing_stats = cfg.stats if hasattr(cfg, "stats") and cfg.stats else None
    if existing_stats is None or not isinstance(existing_stats, dict):
        raise ValueError(
            "Cannot compute chunked stats: no stats available. "
            "Make sure compute_dataset_stats() has set cfg.stats with original stats."
        )

    # Validate that all requested keys exist in stats
    for key in action_keys_to_compute:
        if key not in existing_stats:
            raise ValueError(
                f"Cannot compute chunked stats for '{key}': no existing stats found. "
                f"Available keys: {list(existing_stats.keys())}"
            )

    # Initialize RunningStatsChunked instances (no longer need to pre-initialize with min/max)
    chunked_stats = {}
    print("\n--- Initializing RunningStatsChunked instances ---")
    for original_key in action_keys_to_compute:
        if original_key not in existing_stats:
            print(
                f"Warning: No existing stats found for '{original_key}', skipping chunked stats for this key"
            )
            continue

        # Create RunningStatsChunked instance (simplified - no pre-initialization needed)
        stats_obj = RunningStatsChunked()
        chunked_stats[original_key] = stats_obj
        print(f"  {original_key}: Initialized RunningStatsChunked")

    if not chunked_stats:
        print("No valid action keys with existing stats found, skipping chunked stats computation")
        return {}

        # ===== Single pass: Accumulate data and compute statistics =====
    print("\n--- Computing chunked statistics (accumulating data) ---")

    data_iter = iter(data_loader)
    max_fail_yield = 0.05
    fail_count = 0
    total_count = 0

    for _ in tqdm(range(len(data_loader)), desc="Accumulating data"):
        total_count += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except Exception as e:
            fail_count += 1
            print(f"\nError loading batch {total_count}: {type(e).__name__}: {e}")
            if fail_count / len(data_loader) > max_fail_yield:
                raise RuntimeError(
                    f"Failed to load {fail_count}/{total_count} batches ({fail_count / total_count:.2%}),"
                    f" exceeding threshold of {max_fail_yield:.2%}"
                ) from e  # Changed from 'from None' to 'from e' to preserve the original error
            continue

        _process_batch(
            batch,
            action_keys_to_compute,
            chunk_size,
            chunked_stats,
            action_to_state_mapping=action_to_state_mapping,
            action_to_type_mapping=action_to_type_mapping,
        )

    # Get final statistics
    print("\n--- Computing final statistics from accumulated data ---")
    result = {}
    for key, stats in chunked_stats.items():
        stats_dict = stats.get_statistics()
        # Convert numpy arrays to lists for JSON serialization
        result[key] = {
            "mean": stats_dict["mean"].tolist(),
            "std": stats_dict["std"].tolist(),
            "min": stats_dict["min"].tolist(),
            "max": stats_dict["max"].tolist(),
            "q02": stats_dict["q02"].tolist(),
            "q98": stats_dict["q98"].tolist(),
        }
        print(f"\n{key} chunked stats:")
        print(f"  Shape: {stats_dict['mean'].shape}")
        print(f"  Mean range: [{stats_dict['mean'].min():.4f}, {stats_dict['mean'].max():.4f}]")
        print(f"  Std range: [{stats_dict['std'].min():.4f}, {stats_dict['std'].max():.4f}]")
        print(f"  Q02 range: [{stats_dict['q02'].min():.4f}, {stats_dict['q02'].max():.4f}]")
        print(f"  Q98 range: [{stats_dict['q98'].min():.4f}, {stats_dict['q98'].max():.4f}]")

    return result


def compute_perdim_chunked_stats(
    cfg: LeRobotDatasetConfig,
    batch_size: int | None = None,
    action_keys_to_compute: list[str] | None = None,
    action_to_state_mapping: dict[str, str] | None = None,
    action_to_type_mapping: dict[str, str] | None = None,
):
    """
    Compute per-dimension statistics for action chunks (aggregated across time).

    This is different from compute_chunked_stats which computes per-dimension AND per-timestep.
    This function applies DeltaActions and then computes stats per dimension only.
    No chunk_size parameter needed since we flatten across time dimension.

    Args:
        cfg: LeRobotDatasetConfig with dataset configuration
        batch_size: Batch size for data loading
        action_keys_to_compute: List of action keys (ORIGINAL dataset keys) to compute stats for
        action_to_state_mapping: Mapping from action keys to their corresponding state keys
        action_to_type_mapping: Mapping from action keys to their action types
    """
    if action_keys_to_compute is None:
        action_keys_to_compute = ["action"]
    print(f"\n{'=' * 80}")
    print("Computing per-dimension chunked statistics (time-aggregated)")
    print(f"{'=' * 80}\n")

    # We don't need to set action_delta_indices since we're not using chunking
    # Just create a basic policy config
    policy_cfg = PolicyConfig()

    dataset = cfg.make_dataset(policy_cfg=policy_cfg)

    if batch_size is None:
        batch_size = cfg.batch_size

    # Use the same num_workers settings that work for compute_dataset_stats
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": None,
        "num_workers": 0,  # Set to 0 to avoid video decoding issues with multiprocessing
        "persistent_workers": False,
        "drop_last": False,
    }
    data_loader = DataLoader(dataset, **dataloader_kwargs)

    print(f"Computing per-dimension stats for original keys: {action_keys_to_compute}")

    # Get existing stats (should be original stats passed from compute_dataset_stats)
    existing_stats = cfg.stats if hasattr(cfg, "stats") and cfg.stats else None
    if existing_stats is None or not isinstance(existing_stats, dict):
        raise ValueError(
            "Cannot compute per-dimension stats: no stats available. "
            "Make sure compute_dataset_stats() has set cfg.stats with original stats."
        )

    # Validate that all requested keys exist in stats
    for key in action_keys_to_compute:
        if key not in existing_stats:
            raise ValueError(
                f"Cannot compute per-dimension stats for '{key}': no existing stats found. "
                f"Available keys: {list(existing_stats.keys())}"
            )

    # Collect all delta action data per dimension
    collected_data = {key: [] for key in action_keys_to_compute}

    print("\n--- Collecting delta action data for per-dimension stats ---")

    data_iter = iter(data_loader)
    max_fail_yield = 0.05
    fail_count = 0
    total_count = 0

    for _ in tqdm(range(len(data_loader)), desc="Collecting delta actions"):
        total_count += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except Exception as e:
            fail_count += 1
            print(f"\nError loading batch {total_count}: {type(e).__name__}: {e}")
            if fail_count / len(data_loader) > max_fail_yield:
                raise RuntimeError(
                    f"Failed to load {fail_count}/{total_count} batches ({fail_count / total_count:.2%}),"
                    f" exceeding threshold of {max_fail_yield:.2%}"
                ) from e
            continue

        # Process batch to get delta actions
        batch_torch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

        for key in action_keys_to_compute:
            # Get the appropriate state key for this action
            state_key = (
                action_to_state_mapping.get(key, "observation.state")
                if action_to_state_mapping
                else "observation.state"
            )

            # Get the action type from mapping or infer from key
            action_type = (
                action_to_type_mapping.get(key, infer_action_type_from_key(key))
                if action_to_type_mapping
                else infer_action_type_from_key(key)
            )

            # Apply DeltaActions transform for this specific action key
            delta_transform = DeltaActions(
                state_key=state_key,
                action_key=key,
                relative_to_state=True,
                post_norm=False,
                action_type=action_type,
            )
            batch_transformed = delta_transform(batch_torch)

            values = batch_transformed[key].numpy()

            # Ensure we have at least 2D array (batch, dims) or 3D (batch, time, dims)
            if values.ndim == 1:
                values = values[:, np.newaxis]  # (batch, dims)
            elif values.ndim == 2 and len(values.shape) == 2:
                # Already (batch, dims) - this is fine
                pass
            elif values.ndim == 2:
                # Might be (batch, time) - add dims
                values = values[:, :, np.newaxis]

            # Flatten across batch and time dimensions to get per-dimension stats
            # If 2D: (batch, dims) -> (batch, dims)
            # If 3D: (batch, time, dims) -> (batch*time, dims)
            flattened = values.reshape(-1, values.shape[-1]) if values.ndim == 3 else values

            collected_data[key].append(flattened)

    # Compute per-dimension statistics
    print("\n--- Computing final per-dimension statistics ---")
    result = {}
    for key, data_list in collected_data.items():
        if not data_list:
            continue

        # Concatenate all data: (total_samples, dims)
        all_data = np.concatenate(data_list, axis=0)

        # Compute statistics per dimension
        stats = {
            "mean": np.mean(all_data, axis=0).tolist(),  # Shape: (dims,)
            "std": np.std(all_data, axis=0).tolist(),
            "min": np.min(all_data, axis=0).tolist(),
            "max": np.max(all_data, axis=0).tolist(),
            "q02": np.percentile(all_data, 2, axis=0).tolist(),
            "q98": np.percentile(all_data, 98, axis=0).tolist(),
        }

        result[key] = stats
        print(f"\n{key} per-dimension stats:")
        print(f"  Data shape: {all_data.shape} -> output dims: ({len(stats['mean'])},)")
        print(f"  Mean range: [{min(stats['mean']):.4f}, {max(stats['mean']):.4f}]")
        print(f"  Std range: [{min(stats['std']):.4f}, {max(stats['std']):.4f}]")
        print(f"  Min range: [{min(stats['min']):.4f}, {max(stats['min']):.4f}]")
        print(f"  Max range: [{min(stats['max']):.4f}, {max(stats['max']):.4f}]")

    return result


def replace_stats_with_identity(indices_dict: dict[str, list[int]], output_path: str) -> dict:
    """
    Replace statistics for specified features with identity stats.

    Args:
        indices_dict: Dictionary mapping feature keys to list of indices to set to identity
        stats: Original statistics dictionary

    Returns:
        New statistics dictionary with specified indices set to identity
    """
    with open(Path(output_path)) as f:
        stats = json.load(f)

    new_stats = {}
    for key, stat in stats.items():
        if key in indices_dict:
            indices = indices_dict[key]
            new_stats[key] = {}

            for stat_key in stat:
                stat[stat_key] = np.array(stat[stat_key])
                if stat[stat_key].ndim == 1:
                    for idx in indices:
                        if "min" in stat_key or "q01" in stat_key or "q02" in stat_key:
                            stat[stat_key][idx] = -1.0
                        elif "max" in stat_key or "q99" in stat_key or "q98" in stat_key or "std" in stat_key:
                            stat[stat_key][idx] = 1.0
                        elif "mean" in stat_key:
                            stat[stat_key][idx] = 0.0
                elif stat[stat_key].ndim == 2:
                    for idx in indices:
                        if "min" in stat_key or "q01" in stat_key or "q02" in stat_key:
                            stat[stat_key][:, idx] = -1.0
                        elif "max" in stat_key or "q99" in stat_key or "q98" in stat_key or "std" in stat_key:
                            stat[stat_key][:, idx] = 1.0
                        elif "mean" in stat_key:
                            stat[stat_key][:, idx] = 0.0

                new_stats[key][stat_key] = stat[stat_key].tolist()
            print(f"Replaced stats for '{key}' at indices {indices} with identity values")
        else:
            new_stats[key] = stat

    with open(output_path, "w") as f:
        json.dump(new_stats, f, indent=2)


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


def main():
    parser = argparse.ArgumentParser(
        description="Recompute feature normalization stats for a dataset config."
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to dataset config YAML file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to new feature yaml file")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for data loading")
    parser.add_argument(
        "--action_mapping",
        type=str,
        default=None,
        help="Path to YAML file mapping action keys to state_key and action_type. "
        "Format: action_key: {state_key: ..., action_type: ...}",
    )
    parser.add_argument(
        "--indices_dict",
        type=str,
        default=None,
        help="JSON string or path to JSON/YAML file mapping feature keys to indices for "
        "identity stats (to be used for rotation stats). "
        "Example: '{\"action.ee_quat_pos\": [3, 4, 5, 6, 11, 12, 13, 14]}'",
    )
    args = parser.parse_args()

    # Load action mapping from YAML file if provided
    action_mapping = load_action_mapping(args.action_mapping)

    # Parse indices_dict from JSON string, JSON file, or YAML file
    indices_dict = None
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

    with open(args.config_path) as f:
        config_dict = YAMLParser.load_config(f)

    # Try to load as MultiDatasetConfig first
    cfg = None
    try:
        cfg = draccus.decode(MultiDatasetConfig, config_dict)
        print(f"Detected MultiDatasetConfig with {len(cfg.datasets)} datasets")

        # Process each dataset in the multi-dataset config

    except Exception as e:
        print(f"Not a MultiDatasetConfig, trying as LeRobotDatasetConfig: {e}")
        # Try as single LeRobotDatasetConfig
        try:
            cfg = draccus.decode(LeRobotDatasetConfig, config_dict)
        except Exception as e2:
            print(f"Failed to decode as LeRobotDatasetConfig: {e2}")
            raise e2

    if isinstance(cfg, MultiDatasetConfig):
        for i, weighted_dataset in enumerate(cfg.datasets):
            dataset_cfg = weighted_dataset.dataset
            output_name = f"dataset_{i}_{dataset_cfg.repo_id}"
            output_path = Path(args.output_path).parent / output_name

            print(f"Processing dataset {i + 1}/{len(cfg.datasets)}: {dataset_cfg.repo_id}")
            compute_dataset_stats(
                dataset_cfg, str(output_path), args.batch_size, action_mapping=action_mapping
            )
            if indices_dict:
                replace_stats_with_identity(indices_dict, str(output_path) + "_stats.json")
    else:
        print(f"Detected single LeRobotDatasetConfig: {cfg.repo_id}")
        print(f"Dataset path: {cfg.root_dir}")
        compute_dataset_stats(cfg, args.output_path, args.batch_size, action_mapping=action_mapping)
        if indices_dict:
            replace_stats_with_identity(indices_dict, args.output_path + "_stats.json")


if __name__ == "__main__":
    main()

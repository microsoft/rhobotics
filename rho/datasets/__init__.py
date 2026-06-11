import logging
from typing import TYPE_CHECKING

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from rho.datasets.lerobot_dataset import LeRobotDatasetConfig
from rho.datasets.multi_dataset import AlkuMultiDataset, MultiDatasetConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rho.policies.base import PolicyConfig


def print_dataset_contributions(contributions: dict, indent: int = 0, pct: float | None = None):
    """
    Pretty print dataset contributions recursively.

    Args:
        contributions: Dictionary with keys: name, length, weighted_length, child_contributions
        indent: Current indentation level for nested datasets
        pct: Percentage contribution of this dataset (passed from parent)
    """
    if not contributions:
        logger.info("No dataset contributions available")
        return

    prefix = "  " * indent
    name = contributions.get("name", "Unknown")
    length = contributions.get("length", 0)
    weighted_length = contributions.get("weighted_length", 0)
    child_contributions = contributions.get("child_contributions", [])

    if indent == 0:
        logger.info("\n" + "🎯 DATASET CONTRIBUTIONS" + "\n" + "=" * 80)

    # Print this dataset's contribution
    pct_str = f" ({pct:.1f}%)" if pct is not None else ""
    if weighted_length > 0:
        logger.info(f"{prefix}📦 {name}: {weighted_length:,}/{length:,} frames{pct_str}")
    else:
        logger.info(f"{prefix}📦 {name}: {length:,} frames{pct_str}")

    # Recursively print child contributions
    if child_contributions:
        total_weighted = sum(c.get("weighted_length", 0) for c in child_contributions)
        for child in child_contributions:
            child_weighted = child.get("weighted_length", 0)
            child_length = child.get("length", 0)
            child_name = child.get("name", "Unknown")
            child_pct = (child_weighted / total_weighted * 100) if total_weighted > 0 else 0

            child_prefix = "  " * (indent + 1)
            if child.get("child_contributions"):
                # Has nested children - recurse with percentage
                print_dataset_contributions(child, indent + 1, pct=child_pct)
            else:
                # Leaf node
                logger.info(
                    f"{child_prefix}- {child_name}: {child_weighted:,}/{child_length:,} ({child_pct:.1f}%)"
                )

    if indent == 0:
        logger.info("=" * 80)


def make_dataset(
    dataset_cfg: LeRobotDatasetConfig | MultiDatasetConfig, policy_cfg: "PolicyConfig" = None
) -> torch.utils.data.Dataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset."""

    return dataset_cfg.make_dataset(policy_cfg)


def make_sampler(
    config: LeRobotDatasetConfig | MultiDatasetConfig,
    dataset: LeRobotDataset | AlkuMultiDataset,
    policy_cfg: "PolicyConfig" = None,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the given dataset configuration.

    Args:
        config: LeRobotDatasetConfig | MultiDatasetConfig containing all dataset parameters
        policy_cfg: Optional PolicyConfig for additional configurations

    Returns:
        DataLoader: Configured PyTorch DataLoader for the dataset
    """

    return config.make_sampler(dataset, policy_cfg)


def make_dataloader(
    config: LeRobotDatasetConfig | MultiDatasetConfig, policy_cfg: "PolicyConfig" = None, device="cuda"
) -> DataLoader:
    """
    Create a PyTorch DataLoader for LeRobot datasets.

    Args:
        config: LeRobotDatasetConfig | MultiDatasetConfig containing all dataset parameters
        policy_cfg: Optional PolicyConfig for additional configurations
        device: Device to use for the DataLoader (default: "cuda")

    Returns:
        DataLoader: Configured PyTorch DataLoader for the dataset
    """
    # Create the dataset

    dataset = config.make_dataset(policy_cfg)

    sampler = config.make_sampler(dataset, policy_cfg)

    print_dataset_contributions(config.get_contributions())

    shuffle = sampler is None and not config.streaming

    # Create the dataloader
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": device != "cpu",  # Pin memory if not using CPU
        "persistent_workers": config.num_workers > 0,  # Keep workers alive
        "drop_last": False,  # Don't drop incomplete batches
    }
    if config.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

    dataloader: DataLoader = DataLoader(dataset, **dataloader_kwargs)

    return dataloader, sampler

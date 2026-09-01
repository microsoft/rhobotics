#!/usr/bin/env python3
"""
Dataset Check Utility

This script checks if a dataset can be loaded successfully with retry logic.
Useful for debugging dataset loading issues and validating configurations.
"""

import logging
import time

import draccus

from rho.datasets import make_dataloader
from rho.training.train import TrainConfig

logger = logging.getLogger(__name__)


def check_dataset_loading(cfg: TrainConfig, max_retries: int = 5, retry_delay: float = 1.0) -> bool:
    """
    Attempt to load the dataset with retry logic.

    Args:
        cfg: TrainConfig containing dataset configuration
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries

    Returns:
        bool: True if dataset loads successfully, False otherwise
    """
    repo_id = getattr(cfg.dataset, "repo_id", None)
    dataset_label = repo_id or type(cfg.dataset).__name__
    logger.info(f"Checking dataset loading for: {dataset_label}")
    logger.info(f"Dataset config: {cfg.dataset}")
    logger.info("-" * 60)

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}: Loading dataset...")

            # Attempt to create the dataloader
            training_dataloader, _ = make_dataloader(cfg.dataset, cfg.policy)

            # Try to get the first batch to ensure the dataset is actually loadable
            logger.info("  Dataloader created successfully")

            # Check if we can extract features
            underlying_dataset = training_dataloader.dataset
            logger.info(f"  Dataset type: {type(underlying_dataset)}")
            try:
                logger.info(f"  Dataset length: {len(underlying_dataset)}")
            except TypeError:
                logger.info("  Dataset length: unavailable for iterable dataset")

            if hasattr(underlying_dataset, "meta"):
                logger.info("  Dataset metadata available")
                logger.info(f"    - Features: {list(underlying_dataset.meta.features.keys())}")

            batch = next(iter(training_dataloader))
            logger.info("  Successfully loaded first batch")
            logger.info(
                f"    - Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}"
            )

            if isinstance(batch, dict):
                for key, value in batch.items():
                    if hasattr(value, "shape"):
                        logger.info(f"      {key}: {value.shape}")
                    else:
                        logger.info(f"      {key}: {type(value)}")

            logger.info(f"Dataset loading SUCCESSFUL on attempt {attempt + 1}")
            return True

        except Exception as e:
            logger.error(f"  Failed: {e!s}")

            if attempt < max_retries:
                logger.info(f"  Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Dataset loading FAILED after {max_retries + 1} attempts")
                logger.error(f"Final error: {e!s}")
                return False

    return False


@draccus.wrap()
def main(cfg: TrainConfig) -> None:
    """
    Main function to check dataset loading.

    Args:
        cfg: TrainConfig containing the dataset configuration to test
    """
    logger.info("=" * 60)
    logger.info("DATASET LOADING CHECK")
    logger.info("=" * 60)

    # Validate basic config
    if cfg.dataset is None:
        raise ValueError("No dataset configuration provided")

    repo_id = getattr(cfg.dataset, "repo_id", None)
    if repo_id is not None and not repo_id:
        raise ValueError("Dataset repo_id cannot be empty")

    # Set batch size if not provided
    if cfg.batch_size is not None:
        cfg.dataset.batch_size = cfg.batch_size
        logger.info(f"Override batch size: {cfg.batch_size}")

    # Attempt to load the dataset
    success = check_dataset_loading(cfg)

    logger.info("=" * 60)
    if success:
        logger.info("DATASET CHECK PASSED")
        logger.info("The dataset can be loaded successfully!")
    else:
        logger.error("DATASET CHECK FAILED")
        logger.error("There are issues with the dataset configuration or availability.")
    logger.info("=" * 60)

    if not success:
        raise RuntimeError("Dataset check failed")


if __name__ == "__main__":
    main()

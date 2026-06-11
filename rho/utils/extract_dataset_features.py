#!/usr/bin/env python3
"""
Extract dataset features and normalization statistics to create a FeatureConfig.

This script accepts either a LeRobotDatasetConfig or TrainConfig YAML file,
extracts the features and normalization statistics from the dataset,
and saves them to a FeatureConfig file.

Usage:
    python -m rho.utils.extract_dataset_features --config config.yaml --output-file features.yaml
    python -m rho.utils.extract_dataset_features --config train_config.yaml \
        --output-file features.yaml --dataset-only
"""

import argparse
import logging
from pathlib import Path

import draccus
from draccus.parsers.config_parsers import YAMLParser

from rho.datasets.lerobot_dataset import LeRobotDatasetConfig

logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: Path) -> LeRobotDatasetConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Either LeRobotDatasetConfig or TrainConfig depending on the file content

    Raises:
        ValueError: If the config type cannot be determined
    """

    with open(config_path) as f:
        config_dict = YAMLParser.load_config(f)

    # Try to determine config type from the structure
    if "dataset" in config_dict:
        config_dict = config_dict["dataset"]

    if "repo_id" in config_dict:
        # This looks like a LeRobotDatasetConfig
        return draccus.decode(LeRobotDatasetConfig, config_dict)
    else:
        raise ValueError(
            f"Cannot determine config type from {config_path}. "
            "Expected either Train/EvalConfig with a 'dataset' keys) "
            "or LeRobotDatasetConfig (with 'repo_id' key)"
        )


def main():
    parser = argparse.ArgumentParser(description="Extract dataset features and create FeatureConfig")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (LeRobotDatasetConfig or TrainConfig)",
    )
    parser.add_argument(
        "--output-file", type=Path, required=True, help="Path to output FeatureConfig YAML file"
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Force treating config as LeRobotDatasetConfig (for TrainConfig files)",
    )
    parser.add_argument(
        "--normalization-mapping", type=Path, help="Path to YAML file with custom normalization mapping"
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip computing normalization statistics (faster, but less accurate)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading configuration from: {args.config}")

    # Load the configuration

    dataset_config = load_config_from_yaml(args.config)

    # Extract FeatureConfig
    # It should be created automatically from the LeRobotDatasetConfig __post_init__
    feature_config = dataset_config.features

    # Save FeatureConfig to file
    logger.info(f"Saving FeatureConfig to: {args.output_file}")
    feature_config.to_yaml(args.output_file)

    logger.info("Feature extraction completed successfully!")
    logger.info(f"FeatureConfig saved to: {args.output_file}")
    if not args.skip_stats:
        stats_file = args.output_file.parent / f"{args.output_file.stem}_stats.json"
        logger.info(f"Normalization stats saved to: {stats_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Libero Training Script

This script sets up training for policies in the LIBERO benchmark environment.
It creates a LiberoEnvWrapper and passes it to the main training function
along with the training configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch.distributed as dist

logger = logging.getLogger(__name__)

from rho.common.wandb_logging import WandBConfig  # noqa: E402
from rho.datasets.lerobot_dataset import LeRobotDatasetConfig  # noqa: E402
from rho.policies import PolicyConfig  # noqa: E402

# Then import everything else
from rho.training.train import TrainConfig, train  # noqa: E402
from rho.training.train_accelerate import train as train_accelerate  # noqa: E402
from rho.utils import init_logging  # noqa: E402

# Initialize logging early (before draccus parsing) - will be reconfigured later with accelerator
# Uses ALKU_LOG_LEVEL env var if set, otherwise defaults to INFO
init_logging()


def is_accelerate_launch():
    """
    Detect if we're running under accelerate launch.

    Returns:
        bool: True if running under accelerate, False otherwise
    """
    # Method 1: Check environment variables that are truly accelerate-specific
    accelerate_env_vars = [
        "ACCELERATE_USE_CPU",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_NUM_PROCESSES",
    ]

    if any(var in os.environ for var in accelerate_env_vars):
        return True

    # Method 2: Check if WORLD_SIZE > 1 (multi-GPU distributed launch)
    # Note: LOCAL_RANK, RANK, WORLD_SIZE are set by cluster schedulers (e.g.
    # Volcano/Amulet) even for single-GPU jobs, so presence alone is not enough.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return True

    try:
        import psutil

        current_process = psutil.Process()
        parent = current_process.parent()
        if parent and "accelerate" in parent.name().lower():
            return True
    except (ImportError, AttributeError):
        pass

    return False


# Import Libero environment components (adjust path for Docker)

from env import LiberoEnvConfig  # noqa: E402


@dataclass
class LiberoTrainConfig(TrainConfig):
    """Training configuration specific to Libero environments."""

    # Override dataset defaults for Libero
    dataset: LeRobotDatasetConfig = field(
        default_factory=lambda: LeRobotDatasetConfig(
            repo_id="libero/libero_object", batch_size=32, num_workers=4
        )
    )

    # Override policy defaults for Libero
    policy: PolicyConfig = field(default_factory=lambda: PolicyConfig(name="DiffusionPolicy", device="cuda"))

    # Libero environment configuration
    env_config: Any = field(
        default_factory=lambda: LiberoEnvConfig(
            task_suite_name="libero_object",
            task_id=0,
            init_state_id=0,
            resolution=256,
            max_episode_steps=250,
            seed=42,
        )
    )

    # Override training defaults for Libero
    output_dir: str = "outputs/training_libero"
    steps: int = 50000
    save_checkpoint_every: int = 2000
    eval_interval: int = 5000
    record_videos: bool = True

    # WandB configuration for Libero
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            enabled=True,
            project="libero_training",
            username="msrx-eai",  # Use default entity
            group="libero_object",
            tags=["libero", "diffusion_policy", "robotics"],
        )
    )


if __name__ == "__main__":
    use_accelerate = is_accelerate_launch()
    logger.info(f"Accelerate session status is: {use_accelerate}")

    if use_accelerate:
        logger.info("Using accelerate training...")

        if "RANK" in os.environ and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=1),  # 60 minutes instead of default 10 minutes
            )
            logger.info("Manually initialized process group with 60-minute timeout")

        train_accelerate()
    else:
        logger.info("Using standard training...")
        train()

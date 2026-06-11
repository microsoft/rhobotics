#!/usr/bin/env python3
"""
Libero Evaluation Script

This script sets up evaluation for trained policies in the LIBERO benchmark environment.
It imports the necessary Libero components and uses the shared evaluation function.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import Libero environment components (this registers LiberoEnvConfig)
from env import LiberoEnvConfig  # noqa: E402

from rho.eval.eval import eval  # noqa: E402
from rho.eval.eval_config import SimEvalConfig  # noqa: E402


@dataclass
class LiberoEvalConfig(SimEvalConfig):
    """Evaluation configuration specific to Libero environments."""

    # Libero environment configuration
    environment: LiberoEnvConfig = field(
        default_factory=lambda: LiberoEnvConfig(
            task_suite_name="libero_object",
            task_id=0,
            init_state_id=0,
            resolution=256,
            max_episode_steps=250,
            seed=42,
        )
    )

    # Override eval defaults for Libero
    output_dir: str = "outputs/eval_libero"
    num_episodes: int = 10
    record_video: bool = True


def test_imports():
    """Test GPU-related imports and configurations."""
    import ctypes
    import os

    logger.info(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'not set')}")
    logger.info(f"DISPLAY={os.environ.get('DISPLAY', 'not set')}")

    try:
        ctypes.CDLL("libEGL_nvidia.so.0")
        logger.info("✓ EGL library loaded successfully")
    except:  # noqa: E722
        logger.warning("✗ Failed to load EGL library")


def main():
    test_imports()
    eval()  # shared eval from rho/training/eval.py

    # mujoco_only_test()


if __name__ == "__main__":
    main()

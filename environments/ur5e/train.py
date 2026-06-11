#!/usr/bin/env python3
"""UR5e training entry point.

Thin wrapper that dispatches to ``rho.training.train`` (single-GPU) or
``rho.training.train_accelerate`` (multi-GPU) depending on launch context.
Robot-agnostic in practice -- the active config (e.g.
``environments/ur5e/configs/train_toolbox_phi5.yaml``) drives all
robot- and task-specific behavior.
"""

import os
from datetime import timedelta

import torch.distributed as dist

from rho.training.train import train  # noqa: E402
from rho.training.train_accelerate import train as train_accelerate  # noqa: E402


def is_accelerate_launch():
    """Detect whether we're running under ``accelerate launch``."""
    accelerate_env_vars = [
        "ACCELERATE_USE_CPU",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_NUM_PROCESSES",
    ]

    if any(var in os.environ for var in accelerate_env_vars):
        return True

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


if __name__ == "__main__":
    use_accelerate = is_accelerate_launch()
    print(f"Accelerate session status is : {use_accelerate}")

    if use_accelerate:
        print("Using accelerate training...")

        if "RANK" in os.environ and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=1),
            )
            print("Manually initialized process group with 60-minute timeout")

        train_accelerate()
    else:
        print("Using standard training...")
        train()

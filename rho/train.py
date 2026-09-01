"""Training entry point that dispatches based on the launch environment."""

import os
from datetime import timedelta


def is_accelerate_launch() -> bool:
    """Return whether the process was launched by Accelerate or a distributed runner."""
    accelerate_env_vars = (
        "ACCELERATE_USE_CPU",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_NUM_PROCESSES",
    )
    return (
        any(name in os.environ for name in accelerate_env_vars) or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def main() -> None:
    """Run the single-process or Accelerate training implementation."""
    if not is_accelerate_launch():
        from rho.training.train import train

        train()
        return

    import torch.distributed as dist

    from rho.training.train_accelerate import train

    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    train()


if __name__ == "__main__":
    main()

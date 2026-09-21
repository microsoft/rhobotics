import os
import random
from typing import Any

import numpy as np
import torch


def configure_training_determinism(seed: int, *, deterministic: bool) -> None:
    """Seed training RNGs and optionally require deterministic Torch kernels."""
    if deterministic:
        cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_workspace_config not in {None, ":4096:8", ":16:8"}:
            raise ValueError(
                "deterministic_training requires CUBLAS_WORKSPACE_CONFIG to be ':4096:8' or ':16:8'"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNG streams used by single-GPU training."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

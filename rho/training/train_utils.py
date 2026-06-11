import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from rho.common.serialization import serialize_to_dict
from rho.policies.base import PreTrainedPolicy

logger = logging.getLogger(__name__)

# Re-export serialize_to_dict as serialize_train_config for backward compatibility
serialize_train_config = serialize_to_dict


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the most recent checkpoint in a directory.

    Looks for ``checkpoint_latest.pt`` first.  If that doesn't exist, falls
    back to the highest-step ``checkpoint_step_XXXXXXX.pt`` file.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or ``None`` if the directory doesn't
        exist or contains no checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    # Prefer the explicit "latest" symlink / copy
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest

    # Fall back to highest numbered checkpoint_step_*.pt
    step_files = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if step_files:
        return step_files[-1]

    return None


def make_policy_interface(
    dataset_config,
    policy,
    device,
    eval_mode: str = "standard",
    inference_delay: int | None = None,
    beta: float | None = None,
    eval_dataset_root_dir: str | None = None,
):
    """Create a PolicyInterface for evaluation during training.

    When training with a MultiDatasetConfig, PolicyInterface needs a single
    LeRobotDatasetConfig (for normalization stats, transforms, etc.).
    This function extracts the correct sub-dataset using eval_dataset_root_dir,
    or falls back to the first dataset_cfg.

    Args:
        dataset_config: The training DataConfig (LeRobotDatasetConfig or MultiDatasetConfig)
        policy: The policy model (unwrapped if using DDP)
        device: Device to run on
        eval_mode: Evaluation mode ('standard' or 'rtc')
        inference_delay: Number of action steps for inference delay (RTC)
        beta: Weighting of RTC update vs flow matching update
        eval_dataset_root_dir: root_dir to match when selecting from a MultiDatasetConfig.
            If None and dataset_config is a MultiDatasetConfig, defaults to the first dataset_cfg.

    Returns:
        PolicyInterface instance configured for evaluation
    """
    from rho.datasets.multi_dataset import MultiDatasetConfig
    from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig

    eval_data_config = dataset_config

    if isinstance(dataset_config, MultiDatasetConfig):
        if eval_dataset_root_dir is not None:
            # Search for matching sub-dataset by root_dir
            matched = None
            for cfg in dataset_config.dataset_cfgs:
                if hasattr(cfg, "root_dir") and cfg.root_dir == eval_dataset_root_dir:
                    matched = cfg
                    break
            if matched is None:
                avail = [getattr(c, "root_dir", None) for c in dataset_config.dataset_cfgs]
                raise ValueError(
                    f"No dataset in MultiDatasetConfig with root_dir: "
                    f"{eval_dataset_root_dir}. Available: {avail}"
                )
            eval_data_config = matched
            logger.info(f"Selected sub-dataset for eval PolicyInterface: root_dir={eval_dataset_root_dir}")
        else:
            # Default to first dataset_cfg
            eval_data_config = dataset_config.dataset_cfgs[0]
            logger.info(
                f"No eval_dataset_root_dir provided, defaulting to first dataset: "
                f"root_dir={getattr(eval_data_config, 'root_dir', 'N/A')}"
            )

    policy_interface_cfg = PolicyInterfaceConfig(
        data_config=eval_data_config,
        policy=policy,
        device=device,
        eval_mode=eval_mode,
        inference_delay=inference_delay,
        beta=beta,
    )
    return PolicyInterface(policy_interface_cfg)


def make_optimizer_and_scheduler(
    cfg, policy: PreTrainedPolicy
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    if hasattr(policy.config, "get_optimizer_preset"):
        logger.info("Using optimizer preset from policy config")
        optimizer_factory = policy.config.get_optimizer_preset()
        optimizer = optimizer_factory.build(policy.parameters())
    else:
        logger.info("Using default Adam optimizer")
        optimizer = Adam(policy.parameters(), lr=cfg.learning_rate)

    if hasattr(policy.config, "get_scheduler_preset"):
        logger.info("Using scheduler preset from policy config")
        lr_scheduler_factory = policy.config.get_scheduler_preset()
        lr_scheduler = lr_scheduler_factory.build(optimizer, cfg.steps)
    else:
        logger.info("Using default StepLR scheduler")
        # Simple step scheduler that reduces LR every 10000 steps
        lr_scheduler = StepLR(optimizer, step_size=10000, gamma=0.9)

    return optimizer, lr_scheduler


def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    output_dir: str,
    keep_checkpoint_interval: int | None = None,
    sampler=None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    train_logger=None,
) -> None:
    """Save a training checkpoint.

    Args:
        policy: The policy model to save
        optimizer: The optimizer to save
        step: Current training step
        metrics: Training metrics to save
        output_dir: Directory to save checkpoints
        keep_checkpoint_interval: If set, only keep checkpoints at this interval
            and delete others
        sampler: Optional sampler with save_state() support for
            deterministic resumption
        lr_scheduler: Optional learning-rate scheduler to save
        train_logger: Optional TrainLogger with save_state() support
    """
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:07d}.pt"

    checkpoint = {
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save lr_scheduler state
    if lr_scheduler is not None:
        if hasattr(lr_scheduler, "state_dict"):
            checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        else:
            logger.warning(
                f"LR scheduler {type(lr_scheduler).__name__} does not "
                "support state_dict(); its state will not be saved."
            )

    # Save policy config and feature_dict if available
    if hasattr(policy, "config"):
        checkpoint["policy_config"] = policy.config
        if hasattr(policy.config, "feature_dict"):
            checkpoint["feature_dict"] = policy.config.feature_dict

    # Save sampler state for deterministic resumption
    if sampler is not None:
        if hasattr(sampler, "save_state"):
            checkpoint["sampler_state_dict"] = sampler.save_state()
        else:
            logger.warning(
                f"Sampler {type(sampler).__name__} does not support save_state(); "
                "sampler state will not be saved in checkpoint."
            )

    # Save train logger state (cumulative counters, epoch progress, etc.)
    if train_logger is not None:
        if hasattr(train_logger, "save_state"):
            checkpoint["train_logger_state_dict"] = train_logger.save_state()
        else:
            logger.warning(
                f"TrainLogger {type(train_logger).__name__} does not support save_state(); "
                "logger state will not be saved in checkpoint."
            )

    torch.save(checkpoint, checkpoint_path)  # nosec B614
    logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")

    # Also save as "latest" checkpoint
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)  # nosec B614

    # Handle checkpoint cleanup if keep_checkpoint_interval is set
    if keep_checkpoint_interval is not None:
        cleanup_old_checkpoints(checkpoint_dir, step, keep_checkpoint_interval)


def cleanup_old_checkpoints(checkpoint_dir: Path, current_step: int, keep_checkpoint_interval: int) -> None:
    """Remove old checkpoints, keeping only those at keep_checkpoint_interval.

    Args:
        checkpoint_dir: Directory containing checkpoints
        current_step: Current training step
        keep_checkpoint_interval: Interval for keeping checkpoints
    """
    # Find all checkpoint files matching the pattern checkpoint_step_XXXXXXX.pt
    checkpoint_pattern = "checkpoint_step_*.pt"
    checkpoint_files = list(checkpoint_dir.glob(checkpoint_pattern))

    for checkpoint_file in checkpoint_files:
        # Extract step number from filename
        try:
            filename = checkpoint_file.name
            # Format is checkpoint_step_XXXXXXX.pt, extract the step number
            step_str = filename.replace("checkpoint_step_", "").replace(".pt", "")
            file_step = int(step_str)

            # Skip if this is the current checkpoint
            if file_step == current_step:
                continue

            # Skip if this checkpoint is at the keep interval
            if file_step % keep_checkpoint_interval == 0:
                continue

            # Remove the old checkpoint
            checkpoint_file.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint_file}")

        except (ValueError, AttributeError) as e:
            # Skip files that don't match the expected pattern
            logger.warning(f"Could not parse checkpoint filename {checkpoint_file}: {e}")
            continue


def load_training_state(
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    sampler=None,
    train_logger=None,
) -> tuple:
    """Load training state from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        optimizer: The optimizer whose state will be restored.
        lr_scheduler: The learning rate scheduler to restore.
        sampler: Optional sampler with load_state() support for deterministic resumption.
        train_logger: Optional TrainLogger with load_state() support.

    Returns:
        Tuple of (step, optimizer, lr_scheduler, sampler, train_logger).
    """
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0, optimizer, lr_scheduler, sampler, train_logger

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")  # nosec B614

    step = checkpoint["step"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore lr_scheduler state if available
    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        if hasattr(lr_scheduler, "load_state_dict"):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            logger.info("LR scheduler state restored from checkpoint.")
        else:
            logger.warning(
                f"LR scheduler {type(lr_scheduler).__name__} does not "
                "support load_state_dict(); its state will not be "
                "restored."
            )
    elif lr_scheduler is not None:
        logger.warning("No lr_scheduler state found in checkpoint; scheduler will start from scratch.")

    # Restore sampler state if available
    if sampler is not None and "sampler_state_dict" in checkpoint:
        if hasattr(sampler, "load_state"):
            sampler.load_state(checkpoint["sampler_state_dict"])
            logger.info("Sampler state restored from checkpoint.")
        else:
            logger.warning(
                f"Sampler {type(sampler).__name__} does not support load_state(); "
                "sampler state will not be restored."
            )
    elif sampler is not None:
        logger.warning("No sampler state found in checkpoint; sampler will start from scratch.")

    # Restore train logger state if available
    if train_logger is not None and "train_logger_state_dict" in checkpoint:
        if hasattr(train_logger, "load_state"):
            train_logger.load_state(checkpoint["train_logger_state_dict"])
        else:
            logger.warning(
                f"TrainLogger {type(train_logger).__name__} does not support load_state(); "
                "logger state will not be restored."
            )
    elif train_logger is not None:
        logger.warning("No train_logger state found in checkpoint; logger will start from scratch.")

    # Clear any stale gradients, set_to_none=True frees memory
    optimizer.zero_grad(set_to_none=True)

    # clean up
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return step, optimizer, lr_scheduler, sampler, train_logger


class BaseMetric:
    """Base class for metrics."""

    def __init__(self, name: str, value=None):
        self.name = name
        self.value = value

    def update(self, value) -> None:
        """Update the metric with new value."""
        raise NotImplementedError("Subclasses should implement this method.")

    def reset(self) -> None:
        """Reset the metric."""
        raise NotImplementedError("Subclasses should implement this method.")


class LatestValue(BaseMetric):
    """This class maintains the latest value for a metric"""

    def __init__(self, name, value=None):
        super().__init__(name, value)

    def update(self, value) -> None:
        """Update the latest value with new metrics."""
        self.value = value

    def reset(self) -> None:
        self.value = None


class CumulativeValue(BaseMetric):
    """This class maintains a cumulative value for a metric"""

    def __init__(self, name, value=0):
        super().__init__(name, value)

    def update(self, value) -> None:
        """Update the cumulative value with new metrics."""
        self.value += value

    def reset(self) -> None:
        self.value = 0


class RollingAverage(BaseMetric):
    """This class accepts a dictionary of metrics and maintains a rolling average for each key"""

    def __init__(self, name, value=None):
        super().__init__(name, value)
        self.count = 0 if value is None else 1

    def update(self, value) -> None:
        """Update the rolling average with new metrics."""
        self.count += 1
        if self.value is None:
            self.value = value
        else:
            self.value = self.value * ((self.count - 1) / self.count) + value / (self.count)

    def reset(self) -> None:
        """Reset the rolling averages."""
        self.value = None
        self.count = 0


class TrainLogger:
    """Logger for training metrics."""

    default_latest_value_metrics = ["steps", "epoch_progress"]
    default_cumulative_value_metrics = ["samples"]

    def __init__(
        self,
        latest_value_metrics: list[str] = None,
        cumulative_value_metrics: list[str] = None,
        dataset_length: int | None = None,
    ):
        self.latest_value_metrics = latest_value_metrics or []
        self.cumulative_value_metrics = cumulative_value_metrics or []
        self.dataset_length = dataset_length

        self.latest_value_metrics += [
            n for n in self.default_latest_value_metrics if n not in self.latest_value_metrics
        ]
        self.cumulative_value_metrics += [
            n for n in self.default_cumulative_value_metrics if n not in self.cumulative_value_metrics
        ]

        self.metrics: dict[str, BaseMetric] = {}
        for metric_name in self.latest_value_metrics:
            self.metrics[metric_name] = LatestValue(metric_name)
        for metric_name in self.cumulative_value_metrics:
            self.metrics[metric_name] = CumulativeValue(metric_name)

    def log(self, metrics: dict[str, float]) -> None:
        """Log metrics at a specific training step."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = RollingAverage(key)
            self.metrics[key].update(value)

    def log_batch(self, step: int, batch_size: int) -> None:
        """Log batch information.

        Args:
            step: The current training step.
            batch_size: The number of samples in the batch (should account for
                distributed training if applicable).
        """
        self.metrics["steps"].update(step)
        self.metrics["samples"].update(batch_size)
        if self.dataset_length is not None and self.dataset_length > 0:
            epoch_progress = self.metrics["samples"].value / self.dataset_length
            self.metrics["epoch_progress"].update(epoch_progress)

    def get_metrics(self) -> dict[str, float]:
        """Get the current metrics."""
        return {k: v.value for k, v in self.metrics.items()}

    def get_metrics_and_reset_rolling(self) -> dict[str, float]:
        """Get the current metrics and reset only rolling averages.

        This preserves cumulative and latest value metrics while resetting
        rolling averages for the next logging interval.
        """
        result = {k: v.value for k, v in self.metrics.items()}
        for metric in self.metrics.values():
            if isinstance(metric, RollingAverage):
                metric.reset()
        return result

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def save_state(self) -> dict:
        """Save the logger state for checkpoint resumption.

        Persists LatestValue and CumulativeValue metrics so that
        cumulative counters (e.g. samples) and progress indicators
        (e.g. epoch_progress, steps) survive a restart.
        RollingAverage metrics are transient and are not saved.

        Returns:
            A dict suitable for inclusion in a training checkpoint.
        """
        state: dict = {"dataset_length": self.dataset_length, "metrics": {}}
        for name, metric in self.metrics.items():
            if isinstance(metric, (LatestValue, CumulativeValue)):
                state["metrics"][name] = {
                    "type": type(metric).__name__,
                    "value": metric.value,
                }
        return state

    def load_state(self, state: dict) -> None:
        """Restore logger state from a checkpoint.

        Args:
            state: Dict previously returned by :meth:`save_state`.
        """
        if "dataset_length" in state and state["dataset_length"] is not None:
            self.dataset_length = state["dataset_length"]

        for name, entry in state.get("metrics", {}).items():
            if name in self.metrics:
                self.metrics[name].value = entry["value"]
            else:
                # Re-create the metric from saved type
                cls_name = entry.get("type", "LatestValue")
                if cls_name == "CumulativeValue":
                    self.metrics[name] = CumulativeValue(name, entry["value"])
                else:
                    self.metrics[name] = LatestValue(name, entry["value"])
        logger.info("TrainLogger state restored from checkpoint.")

    @contextmanager
    def log_time(self, metric_name: str):
        """Context manager for automatically timing and logging execution time.

        Args:
            metric_name: The name of the metric to log the timing under.

        Example:
            with training_metrics_recorder.log_time('dataloading_s'):
                batch = next(training_iter)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_time = time.perf_counter() - start_time
            self.log({metric_name: elapsed_time})

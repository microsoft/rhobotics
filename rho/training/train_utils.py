import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from rho.checkpoints import (
    delete_checkpoint_bundle,
    is_checkpoint_bundle,
    load_bundle_training_state,
    load_manifest,
    remove_bundle_training_state,
    resolve_latest_checkpoint,
    save_checkpoint_bundle,
    validate_checkpoint,
)
from rho.common.determinism import capture_rng_state, restore_rng_state
from rho.common.serialization import serialize_to_dict
from rho.policies.base import PreTrainedPolicy

logger = logging.getLogger(__name__)

# Re-export serialize_to_dict as serialize_train_config for backward compatibility
serialize_train_config = serialize_to_dict


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the highest valid completed checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or ``None`` if the directory doesn't
        exist or contains no checkpoints.
    """
    return resolve_latest_checkpoint(checkpoint_dir)


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

        named_groups = None
        if hasattr(policy, "get_named_param_groups"):
            named_groups = policy.get_named_param_groups()

        if named_groups:
            for g in named_groups:
                n_params = sum(p.numel() for p in g["params"])
                logger.info(f"Optimizer group '{g['name']}': {n_params:,} params, lr={g['lr']:.2e}")
            optimizer = optimizer_factory.build(named_groups)
        else:
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
    data_config=None,
    max_shard_size: int | str = "5GB",
    keep_training_state_interval: int | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
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
        grad_scaler: Optional AMP loss scaler to preserve across resumption.
    """
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:07d}"

    training_state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "rng_state_dict": capture_rng_state(),
    }
    if grad_scaler is not None:
        training_state["grad_scaler_state_dict"] = grad_scaler.state_dict()

    # Save lr_scheduler state
    if lr_scheduler is not None:
        if hasattr(lr_scheduler, "state_dict"):
            training_state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        else:
            logger.warning(
                f"LR scheduler {type(lr_scheduler).__name__} does not "
                "support state_dict(); its state will not be saved."
            )

    # Save sampler state for deterministic resumption
    if sampler is not None:
        if hasattr(sampler, "save_state"):
            training_state["sampler_state_dict"] = sampler.save_state()
        else:
            logger.warning(
                f"Sampler {type(sampler).__name__} does not support save_state(); "
                "sampler state will not be saved in checkpoint."
            )

    # Save train logger state (cumulative counters, epoch progress, etc.)
    if train_logger is not None:
        if hasattr(train_logger, "save_state"):
            training_state["train_logger_state_dict"] = train_logger.save_state()
        else:
            logger.warning(
                f"TrainLogger {type(train_logger).__name__} does not support save_state(); "
                "logger state will not be saved in checkpoint."
            )

    save_checkpoint_bundle(
        policy,
        checkpoint_path,
        step=step,
        training_state=training_state,
        data_config=data_config,
        max_shard_size=max_shard_size,
    )
    validate_checkpoint(checkpoint_path)
    logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")

    prune_training_states(
        checkpoint_dir,
        current_step=step,
        keep_training_state_interval=keep_training_state_interval,
    )

    # Handle checkpoint cleanup if keep_checkpoint_interval is set
    if keep_checkpoint_interval is not None:
        cleanup_old_checkpoints(checkpoint_dir, step, keep_checkpoint_interval)


def prune_training_states(
    checkpoint_dir: Path,
    *,
    current_step: int,
    keep_training_state_interval: int | None,
) -> None:
    """Keep resume state only for the newest checkpoint and configured milestones."""
    if keep_training_state_interval is not None and keep_training_state_interval <= 0:
        raise ValueError("keep_training_state_interval must be positive or None")

    for checkpoint_path in checkpoint_dir.glob("checkpoint_step_*"):
        if not is_checkpoint_bundle(checkpoint_path):
            continue
        checkpoint_step = load_manifest(checkpoint_path)["step"]
        if checkpoint_step == current_step:
            continue
        if keep_training_state_interval is not None and checkpoint_step % keep_training_state_interval == 0:
            continue
        if remove_bundle_training_state(checkpoint_path):
            logger.info(f"Removed training state from retained checkpoint: {checkpoint_path}")


def cleanup_old_checkpoints(checkpoint_dir: Path, current_step: int, keep_checkpoint_interval: int) -> None:
    """Remove old checkpoints, keeping only those at keep_checkpoint_interval.

    Args:
        checkpoint_dir: Directory containing checkpoints
        current_step: Current training step
        keep_checkpoint_interval: Interval for keeping checkpoints
    """
    current_bundle = checkpoint_dir / f"checkpoint_step_{current_step:07d}"
    current_legacy_checkpoint = current_bundle.with_suffix(".pt")
    if current_bundle.exists():
        validate_checkpoint(current_bundle)
    elif current_legacy_checkpoint.exists():
        validate_checkpoint(current_legacy_checkpoint)
    else:
        raise FileNotFoundError(
            f"Current checkpoint does not exist for step {current_step}: {current_bundle}"
        )

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*"))

    for checkpoint_file in checkpoint_files:
        # Extract step number from filename
        try:
            filename = checkpoint_file.name
            step_str = filename.replace("checkpoint_step_", "").replace(".pt", "")
            file_step = int(step_str)

            # Skip if this is the current checkpoint
            if file_step == current_step:
                continue

            # Skip if this checkpoint is at the keep interval
            if file_step % keep_checkpoint_interval == 0:
                continue

            # Remove the old checkpoint
            if checkpoint_file.is_dir():
                if not delete_checkpoint_bundle(checkpoint_file):
                    logger.warning(
                        f"Retaining checkpoint while another process is reading it: {checkpoint_file}"
                    )
                    continue
            else:
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
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple:
    """Load training state from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        optimizer: The optimizer whose state will be restored.
        lr_scheduler: The learning rate scheduler to restore.
        sampler: Optional sampler with load_state() support for deterministic resumption.
        train_logger: Optional TrainLogger with load_state() support.
        grad_scaler: Optional AMP loss scaler to restore.

    Returns:
        Tuple of (step, optimizer, lr_scheduler, sampler, train_logger).
    """
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0, optimizer, lr_scheduler, sampler, train_logger

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    if is_checkpoint_bundle(checkpoint_path):
        checkpoint = load_bundle_training_state(checkpoint_path)
        if checkpoint is None:
            logger.info(
                "Checkpoint contains model artifacts only; optimizer, scheduler, sampler, "
                "logger, and step state will start from scratch."
            )
            optimizer.zero_grad(set_to_none=True)
            return 0, optimizer, lr_scheduler, sampler, train_logger
    else:
        legacy_load_started = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")  # nosec B614
        logger.info(
            "Legacy training checkpoint deserialized in %.2fs from %s",
            time.perf_counter() - legacy_load_started,
            checkpoint_path,
        )

    if "optimizer_state_dict" not in checkpoint:
        logger.info(
            "Checkpoint contains policy weights only; optimizer, scheduler, sampler, "
            "logger, and step state will start from scratch."
        )
        optimizer.zero_grad(set_to_none=True)
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return 0, optimizer, lr_scheduler, sampler, train_logger

    step = checkpoint["step"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if grad_scaler is not None:
        if "grad_scaler_state_dict" in checkpoint:
            grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        else:
            logger.warning("No GradScaler state found in checkpoint; loss scaling will start from scratch.")

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

    if "rng_state_dict" in checkpoint:
        restore_rng_state(checkpoint["rng_state_dict"])
        logger.info("Training RNG state restored from checkpoint.")
    else:
        logger.warning(
            "No RNG state found in checkpoint; random streams will restart from the configured seed."
        )

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
        if self.dataset_length is None and state.get("dataset_length") is not None:
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
        if self.dataset_length is not None and self.dataset_length > 0:
            self.metrics["epoch_progress"].value = self.metrics["samples"].value / self.dataset_length
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

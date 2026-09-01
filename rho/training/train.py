import logging
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

import draccus
import torch
from flask import json
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import cycle
from tqdm import tqdm

from rho.checkpoints import resolve_checkpoint
from rho.common.serialization import serialize_to_dict
from rho.common.transforms import ConsolidateTransform, TransformConfig
from rho.common.wandb_logging import WandBConfig, WandBLogger
from rho.datasets import LeRobotDatasetConfig, MultiDatasetConfig, make_dataloader
from rho.environment import EnvironmentConfig, make_environment
from rho.environment.env import evaluate_policy
from rho.policies import PolicyConfig, make_policy
from rho.training.action_sampling_monitor import ActionSamplingMonitor
from rho.training.train_utils import (
    TrainLogger,
    find_latest_checkpoint,
    load_training_state,
    make_optimizer_and_scheduler,
    make_policy_interface,
    save_checkpoint,
    serialize_train_config,
)
from rho.training.validation_probe import ValidationProbe
from rho.utils import init_logging

# Initialize logging early (before draccus parsing) - will be reconfigured later with accelerator
# Uses RHO_LOG_LEVEL if set, otherwise defaults to INFO.
init_logging()

logger = logging.getLogger(__name__)

DatasetConfig = LeRobotDatasetConfig | MultiDatasetConfig


def resolve_training_checkpoint(cfg: "TrainConfig") -> Path | None:
    """Resolve the model source selected by training configuration."""
    if cfg.run_name is not None:
        existing = find_latest_checkpoint(cfg.checkpoint_folder)
        if existing is not None:
            logger.info("run_name=%r: resuming from %s", cfg.run_name, existing)
            cfg.resume = True
            return existing

    if cfg.resume and cfg.pretrained_checkpoint is None:
        raise ValueError("resume=True requires a checkpoint in run_name or an explicit pretrained_checkpoint")

    source = cfg.pretrained_checkpoint
    if source is None:
        source = getattr(cfg.policy, "pretrained_repo_id", None)
    if source is None:
        return None

    return resolve_checkpoint(
        source,
        revision=cfg.checkpoint_revision,
        cache_dir=cfg.checkpoint_cache_dir,
        include_training_state=cfg.resume,
    )


def initialize_checkpoint_folder(cfg: "TrainConfig", *, timestamp: str | None = None) -> Path:
    """Create the checkpoint folder once training execution begins.

    Timestamped folders cannot be created in ``TrainConfig.__post_init__``
    because every distributed worker decodes the config independently. The
    Accelerate entry point supplies one timestamp broadcast by the main
    process; single-process training lets this function generate it locally.
    """
    if cfg.checkpoint_folder is not None:
        cfg.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        return cfg.checkpoint_folder

    requested_local_checkpoint = (
        cfg.pretrained_checkpoint is not None and Path(cfg.pretrained_checkpoint).expanduser().exists()
    )
    if cfg.resume and cfg.resolved_checkpoint is not None and requested_local_checkpoint:
        cfg.checkpoint_folder = cfg.resolved_checkpoint.parent
    else:
        timestamp = timestamp or datetime.now().strftime("%m%d_%H%M%S")
        cfg.checkpoint_folder = Path(cfg.output_dir) / timestamp / "checkpoints"
    cfg.checkpoint_folder.mkdir(parents=True, exist_ok=True)
    return cfg.checkpoint_folder


def collect_lookahead_monitor_batches(training_iter, lookahead_batches: deque, num_samples: int) -> list:
    """Return enough future batches for monitoring without growing the lookahead buffer unnecessarily.

    Batches already in ``lookahead_batches`` have been fetched but not yet trained
    on, so they are the preferred monitor source. Only the missing batches are
    fetched from ``training_iter`` and added to the buffer for later training.
    """
    sample_count = 0
    monitor_batches = []

    for monitor_batch in lookahead_batches:
        monitor_batches.append(monitor_batch)
        sample_count += monitor_batch["action"].shape[0]
        if sample_count >= num_samples:
            return monitor_batches

    while sample_count < num_samples:
        monitor_batch = next(training_iter)
        monitor_batches.append(monitor_batch)
        lookahead_batches.append(monitor_batch)
        sample_count += monitor_batch["action"].shape[0]

    return monitor_batches


@dataclass
class TrainConfig:
    wandb: WandBConfig = field(default_factory=WandBConfig)

    dataset: DatasetConfig = None
    validation_dataset: DatasetConfig | None = None

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    # Explicit local path or Hugging Face repository overriding the policy default.
    pretrained_checkpoint: str | Path | None = None
    checkpoint_revision: str | None = None
    checkpoint_cache_dir: str | None = None
    resolved_checkpoint: Path | None = field(default=None, init=False, repr=False, compare=False)
    # Whether to resume training from the last checkpoint
    resume: bool = False
    # A fixed name for the output run folder.  When set, the output directory
    # will be ``<output_dir>/<run_name>/checkpoints`` instead of using a
    # timestamp. If that folder already contains a checkpoint, runtime
    # resolution automatically resumes it without replacing the authored
    # ``pretrained_checkpoint`` source.
    run_name: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training parameters
    batch_size: int = None  # Default will be set by the dataset config. This value overwrites it
    num_workers: int = 4
    learning_rate: float = 1e-4
    steps: int = 50000  # Total training steps
    # validation_interval: int = 1000  # Steps between validation

    # Checkpoint and output settings
    output_dir: str = "outputs/training"
    save_checkpoint_every: int = 1000  # Save checkpoint every N steps
    keep_checkpoint_interval: int = 10000
    # Older retained checkpoints keep model weights but drop optimizer and
    # scheduler state unless their step matches this interval. None keeps full
    # training state only in the newest checkpoint.
    keep_training_state_interval: int | None = None
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1
    logging_interval: int = 100  # Log metrics every N steps
    grad_clip_norm: float | None = 10.0  # Gradient clipping norm
    # Lightweight host-memory logging for diagnosing dataloader / eval memory
    # growth. Logged under train/memory/* on the main process.
    memory_monitoring: bool = False
    memory_monitor_interval: int = 100

    # Evaluation settings
    validation_interval: int = 1000  # Steps between validation checks
    eval_interval: int = 1000  # Steps between evaluations
    eval_batch_size: int = 32  # Batch size for evaluation
    eval_num_episodes: int = 4  # Number of workers for evaluation dataloader
    eval_mode: str = "standard"
    record_videos: bool = False  # Whether to record videos during evaluation
    environment: EnvironmentConfig | None = None

    training_transforms: list[TransformConfig] | None = None

    # Key to select which dataset's config to use for PolicyInterface
    # when training with multidataset (selects by root_dir)
    eval_dataset_root_dir: str | None = None

    # Action sampling monitoring settings
    action_monitoring: bool = False  # Enable/disable action sampling monitoring
    action_monitor_interval: int = 1000  # Steps between action sampling monitoring
    action_monitor_samples: int = 16  # Number of samples to use for monitoring
    action_monitor_source: str = "lookahead"  # "lookahead", "training_fresh", or "validation"
    set_static_graph: bool = False

    validation_probe: bool = False  # Enable/disable the validation probe
    validation_probe_interval: int = 1000
    validation_probe_batches: int = 100
    max_validation_batches: int = 100  # Upper bound on batches checked during validation loss computation

    inference_delay: int | None = None
    beta: float | None = None

    # Logging configuration
    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR

    def to_dict(self) -> dict:
        """Serialize authored settings while excluding runtime checkpoint state."""
        return {
            config_field.name: serialize_to_dict(getattr(self, config_field.name))
            for config_field in fields(self)
            if config_field.name != "resolved_checkpoint"
        }

    def __post_init__(self):
        """Validate the configuration"""
        logger.debug("Validating training configuration...")
        logger.debug(self.policy)
        if self.keep_training_state_interval is not None and self.keep_training_state_interval <= 0:
            raise ValueError("keep_training_state_interval must be positive or None")
        if self.action_monitor_source not in {"lookahead", "training_fresh", "validation"}:
            raise ValueError(
                "action_monitor_source must be one of "
                "'lookahead', 'training_fresh', or 'validation'; "
                f"got {self.action_monitor_source!r}"
            )

        # Expand environment variables in eval_dataset_root_dir
        if self.eval_dataset_root_dir is not None:
            import os

            self.eval_dataset_root_dir = os.path.expandvars(self.eval_dataset_root_dir)

        if self.batch_size is not None:
            self.dataset.batch_size = self.batch_size
        if self.num_workers is not None:
            self.dataset.num_workers = self.num_workers
        if self.validation_dataset is not None:
            self.validation_dataset.batch_size = self.batch_size
            if self.num_workers is not None:
                self.validation_dataset.num_workers = self.num_workers

        if self.run_name is not None:
            # Fixed, named run folder enables automatic restart/resume.
            self.checkpoint_folder = Path(self.output_dir) / self.run_name / "checkpoints"
        else:
            # Timestamped folders are initialized by the training entry point.
            # In distributed training the main process broadcasts one timestamp
            # so workers cannot create competing run directories.
            self.checkpoint_folder = None

        if not isinstance(self.policy, PolicyConfig):
            raise ValueError("Policy must be a PolicyConfig instance or a dict")

        if self.policy.feature_dict is None:
            self.policy.feature_dict = self.dataset.transformed_feature_dict

        if (
            self.dataset.chunk_size is None
            and hasattr(self.policy, "chunk_size")
            and self.policy.chunk_size is not None
        ):
            self.dataset.chunk_size = self.policy.chunk_size


def train_policy_step(policy, batch, optimizer, lr_scheduler, step, device, use_amp=False):
    """Performs a single training step on the policy"""
    policy.train()
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, loss_dict = policy.compute_loss(batch)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    # Log metrics
    metrics = {
        "loss": loss.item(),
        "learning_rate": lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]["lr"],
    }
    if loss_dict:
        metrics.update(loss_dict)
    return metrics


def validate_policy(
    policy, validation_dataloader, device, eval_batch_size=32, eval_num_episodes=4, max_batches=None
):
    """Validate the policy on the validation dataset"""
    policy.eval()

    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            if max_batches is not None and total_steps >= max_batches:
                break
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            if hasattr(policy, "module"):
                loss, loss_dict = policy.module.compute_loss(batch)
            else:
                loss, loss_dict = policy.compute_loss(batch)
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    validation_dict = {
        "validation_loss": avg_loss,
    }
    if loss_dict:
        validation_dict.update(loss_dict)
    return validation_dict


def make_everything(cfg: TrainConfig, device: torch.device = None):
    """
    Initializes the training components: dataloader, policy, optimizer, and scheduler

    This is to make it slightly easier to make custom training loops for different environments
    or to use different policies without changing the main training loop.

    Args:
        cfg (TrainConfig): The training configuration.
        device (torch.device, optional): The device to use for training.
            If None, it will be determined from cfg.device.
    """

    if device is None:
        device = get_safe_torch_device(cfg.device, log=True)
    training_dataloader, training_sampler = make_dataloader(cfg.dataset, cfg.policy)
    validation_dataloader = None
    if cfg.validation_dataset is not None:
        validation_dataloader, _ = make_dataloader(cfg.validation_dataset, cfg.policy)

    policy = make_policy(cfg.policy)

    policy.to(device)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    transforms_list = None
    if cfg.training_transforms is not None:
        transforms_list = [transform.build() for transform in cfg.training_transforms]
    train_transforms = ConsolidateTransform(transforms_list if transforms_list is not None else [])

    dataset_length = cfg.dataset.get_length()
    training_metrics_recorder = TrainLogger(dataset_length=dataset_length)

    if cfg.resolved_checkpoint is None:
        cfg.resolved_checkpoint = resolve_training_checkpoint(cfg)
    if cfg.resolved_checkpoint is not None:
        checkpoint_path = cfg.resolved_checkpoint
        logger.info(
            "Loading %s checkpoint %s (requested source: %s)",
            "resume" if cfg.resume else "pretrained",
            checkpoint_path,
            cfg.pretrained_checkpoint or getattr(cfg.policy, "pretrained_repo_id", None),
        )
        if cfg.resume:
            step, optimizer, lr_scheduler, training_sampler, training_metrics_recorder = load_training_state(
                checkpoint_path, optimizer, lr_scheduler, training_sampler, training_metrics_recorder
            )
        policy.load_from_pretrained(checkpoint_path)

    policy.train()
    return (
        training_dataloader,
        validation_dataloader,
        policy,
        optimizer,
        lr_scheduler,
        train_transforms,
        step,
        training_sampler,
        training_metrics_recorder,
    )


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    """Main training function"""
    # Initialize logging
    init_logging(console_level=cfg.log_level)

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.resolved_checkpoint = resolve_training_checkpoint(cfg)
    initialize_checkpoint_folder(cfg)
    logger.info(f"Checkpoints will be saved to: {cfg.checkpoint_folder}")

    if cfg.wandb.id is None:
        cfg.wandb.id = (
            "default_job_" + cfg.checkpoint_folder.parent.stem
        )  # grab the timestamp to match wandb job name with checkpoint storage

    serializable_cfg = serialize_train_config(cfg)
    with open(Path(cfg.checkpoint_folder.parent / "train_config.json"), "w") as f:
        json.dump(serializable_cfg, f, indent=4)

    device = get_safe_torch_device(cfg.device, log=True)

    env = make_environment(cfg.environment)
    (
        training_dataloader,
        validation_dataloader,
        policy,
        optimizer,
        lr_scheduler,
        train_transforms,
        step,
        training_sampler,
        training_metrics_recorder,
    ) = make_everything(cfg, device)

    if cfg.action_monitoring:
        action_sampling_monitor = ActionSamplingMonitor(
            policy=policy,
            monitor_interval=cfg.action_monitor_interval,
            num_samples=cfg.action_monitor_samples,
            device=device,
        )

    validation_probe = None
    if (
        cfg.validation_probe
        and cfg.validation_probe_interval > 0
        and cfg.validation_dataset is not None
        and validation_dataloader is not None
    ):
        try:
            denorm_transform = cfg.validation_dataset.get_action_denormalization()
        except Exception:
            logger.warning(
                "Failed to get action denormalization, validation probe will run without denormalization",
                exc_info=True,
            )
            denorm_transform = None
        validation_probe = ValidationProbe(
            validation_dataloader,
            cfg.validation_probe_batches,
            training_dataloader,
            denorm_transform=denorm_transform,
        )

    training_iter = cycle(training_dataloader)
    lookahead_batches = deque()
    wandb_logger = WandBLogger(cfg.wandb)
    # Create progress bar
    progress_bar = tqdm(range(step, cfg.steps), desc="Training", initial=step, total=cfg.steps, unit="step")
    wandb_logger.log_config(cfg)

    # Create PolicyInterface for evaluation (extracts sub-dataset if MultiDatasetConfig)
    policy_interface = make_policy_interface(
        dataset_config=cfg.dataset,
        policy=policy,
        device=device,
        eval_mode=cfg.eval_mode,
        inference_delay=cfg.inference_delay,
        beta=cfg.beta,
        eval_dataset_root_dir=cfg.eval_dataset_root_dir,
    )

    for _ in progress_bar:
        step += 1
        with training_metrics_recorder.log_time("train/dataloading_s"):
            batch = lookahead_batches.popleft() if lookahead_batches else next(training_iter)

        # Apply training transforms and step them
        train_transforms.train()
        batch = train_transforms(batch)
        train_transforms.step()

        with training_metrics_recorder.log_time("train/update_s"):
            training_metrics = train_policy_step(policy, batch, optimizer, lr_scheduler, step, device)

        # Log transform scales if transforms are being used
        if cfg.training_transforms is not None:
            for i, transform in enumerate(train_transforms.transforms):
                if hasattr(transform, "get_current_scale"):
                    scale = transform.get_current_scale()
                    training_metrics[f"transform_{i}_scale"] = scale

        training_metrics_recorder.log(
            training_metrics,
        )
        batch_size = batch["action"].shape[0]
        training_metrics_recorder.log_batch(step, batch_size)
        # Update progress bar with current metrics
        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            {"loss": f"{training_metrics['loss']:.4f}", "lr": f"{current_lr:.2e}", "step": step}
        )

        if step % cfg.validation_interval == 0 and validation_dataloader is not None:
            with training_metrics_recorder.log_time("train/validation_s"):
                # Evaluate the policy on the validation dataset
                validation_metrics = validate_policy(
                    policy,
                    validation_dataloader,
                    device,
                    cfg.eval_batch_size,
                    cfg.eval_num_episodes,
                    max_batches=cfg.max_validation_batches,
                )
            wandb_logger.log(validation_metrics, step=step, prefix="validation")

        if step % cfg.validation_probe_interval == 0 and validation_probe is not None:
            validation_metrics = validation_probe.validate_policy_by_episode(policy, device)
            wandb_logger.log(validation_metrics, step=step, prefix="validation")

        if step % cfg.eval_interval == 0 and env is not None:
            # By default the env should always be None unless this function is being called
            # from an environment-specific training script
            video_path = cfg.checkpoint_folder.parent / "eval" / f"{step:07d}"

            policy.eval()
            with torch.no_grad():
                eval_metrics = evaluate_policy(
                    env=env,
                    policy_interface=policy_interface,
                    num_episodes=cfg.eval_num_episodes,
                    max_steps=cfg.environment.max_episode_steps,
                    seed=12345,
                    record_video=cfg.record_videos,
                    output_dir=str(video_path),
                    eval_mode=cfg.eval_mode,
                    inference_delay=cfg.inference_delay or 8,
                )
            wandb_logger.log(eval_metrics, step=step, prefix="eval")

        if cfg.action_monitoring and step % cfg.action_monitor_interval == 0:
            if cfg.action_monitor_source == "lookahead":
                monitor_batches = collect_lookahead_monitor_batches(
                    training_iter, lookahead_batches, cfg.action_monitor_samples
                )
                action_sampling_monitor.monitor_batches(monitor_batches, step, wandb_logger)
            else:
                monitor_dataloader = training_dataloader
                if cfg.action_monitor_source == "validation":
                    monitor_dataloader = (
                        validation_dataloader if validation_dataloader is not None else training_dataloader
                    )
                action_sampling_monitor.monitor_training_progress(monitor_dataloader, step, wandb_logger)

        if step % cfg.save_checkpoint_every == 0:
            save_checkpoint(
                policy,
                optimizer,
                step,
                training_metrics,
                cfg.checkpoint_folder,
                cfg.keep_checkpoint_interval,
                sampler=training_sampler,
                lr_scheduler=lr_scheduler,
                train_logger=training_metrics_recorder,
                data_config=cfg.dataset,
                keep_training_state_interval=cfg.keep_training_state_interval,
            )

        if step % cfg.logging_interval == 0:
            wandb_logger.log(
                training_metrics_recorder.get_metrics_and_reset_rolling(), step=step, prefix="train"
            )

    if env is not None:
        env.close()


if __name__ == "__main__":
    train()

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import draccus
import torch
from flask import json
from lerobot.datasets.utils import cycle
from lerobot.utils.utils import get_safe_torch_device
from tqdm import tqdm

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
# Uses ALKU_LOG_LEVEL env var if set, otherwise defaults to INFO
init_logging()

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    wandb: WandBConfig = field(default_factory=WandBConfig)

    dataset: LeRobotDatasetConfig | MultiDatasetConfig = None
    validation_dataset: LeRobotDatasetConfig | MultiDatasetConfig | None = None

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    # Absolute path to a pretrained checkpoint
    pretrained_checkpoint: str | Path | None = None
    # Whether to resume training from the last checkpoint
    resume: bool = False
    # A fixed name for the output run folder.  When set, the output directory
    # will be ``<output_dir>/<run_name>/checkpoints`` instead of using a
    # timestamp.  If that folder already contains a checkpoint the job will
    # automatically resume from it (sets ``resume=True`` and
    # ``pretrained_checkpoint`` accordingly).
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
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1
    logging_interval: int = 100  # Log metrics every N steps
    grad_clip_norm: float | None = 10.0  # Gradient clipping norm

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
    set_static_graph: bool = False

    validation_probe: bool = False  # Enable/disable the validation probe
    validation_probe_interval: int = 1000
    validation_probe_batches: int = 100
    max_validation_batches: int = 100  # Upper bound on batches checked during validation loss computation

    inference_delay: int | None = None
    beta: float | None = None

    # Logging configuration
    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR

    def __post_init__(self):
        """Validate the configuration"""
        logger.debug("Validating training configuration...")
        logger.debug(self.policy)

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

        # --- Determine the checkpoint folder ---
        if self.run_name is not None:
            # Fixed, named run folder – enables automatic restart/resume
            self.checkpoint_folder = Path(self.output_dir) / self.run_name / "checkpoints"
            self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

            # Auto-resume: if there's already a checkpoint, use it
            if not self.resume and self.pretrained_checkpoint is None:
                existing = find_latest_checkpoint(self.checkpoint_folder)
                if existing is not None:
                    logger.info(
                        f"run_name='{self.run_name}': found existing checkpoint "
                        f"{existing}, enabling auto-resume."
                    )
                    self.pretrained_checkpoint = str(existing)
                    self.resume = True
        elif not self.resume:
            # Default: timestamp-based folder
            timestamp = str(datetime.now().strftime("%m%d_%H%M%S"))
            self.checkpoint_folder = Path(self.output_dir) / timestamp / "checkpoints"
            self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_folder = Path(self.pretrained_checkpoint).parent

        if not isinstance(self.policy, PolicyConfig):
            raise ValueError("Policy must be a PolicyConfig instance or a dict")

        if self.pretrained_checkpoint and not Path(self.pretrained_checkpoint).exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_checkpoint}")

        if self.pretrained_checkpoint is None and self.resume:
            raise ValueError("Cannot resume training without a pretrained checkpoint")

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
    # Create the dataloader
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

    if cfg.resume or cfg.pretrained_checkpoint is not None:
        if cfg.pretrained_checkpoint is None:
            checkpoint_path = Path(cfg.checkpoint_folder / "checkpoint_latest.pt")
        else:
            checkpoint_path = Path(cfg.pretrained_checkpoint)

        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Please provide a valid pretrained_checkpoint or set resume to False."
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
    logger.info(f"Checkpoints will be saved to: {cfg.checkpoint_folder}")

    if cfg.wandb.id is None:
        cfg.wandb.id = (
            "default_job_" + cfg.checkpoint_folder.parent.stem
        )  # grab the timestamp to match wandb job name with checkpoint storage

    # save the trainconfig for future reference and manual eval
    # print the TrainConfig for local debugging
    # print("#################### Training config ####################")
    # pprint(asdict(cfg))
    # print("#################### Training config ####################")

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
            denorm_transform = cfg.dataset.get_action_denormalization()
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
            batch = next(training_iter)

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
            )

        if step % cfg.logging_interval == 0:
            wandb_logger.log(
                training_metrics_recorder.get_metrics_and_reset_rolling(), step=step, prefix="train"
            )

    if env is not None:
        env.close()


if __name__ == "__main__":
    train()

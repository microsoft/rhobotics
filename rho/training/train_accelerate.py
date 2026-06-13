import json
import logging
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
from accelerate import Accelerator
from lerobot.utils.utils import cycle
from tqdm import tqdm

from rho.common.wandb_logging import WandBLogger
from rho.environment import make_environment
from rho.environment.env import evaluate_policy
from rho.training.action_sampling_monitor import ActionSamplingMonitor
from rho.training.train import TrainConfig, make_everything, validate_policy
from rho.training.train_utils import make_policy_interface, save_checkpoint, serialize_train_config
from rho.training.validation_probe import ValidationProbe
from rho.utils import init_logging

# Initialize logging early (before draccus parsing) - will be reconfigured later with accelerator
# Uses ALKU_LOG_LEVEL env var if set, otherwise defaults to INFO
init_logging()

logger = logging.getLogger(__name__)


def train_policy_step_accumulate(
    policy, batch, optimizer, lr_scheduler, accelerator, step, device, grad_scaler=None, max_grad_norm=None
):
    """Performs a single training step on the policy"""
    policy.train()
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            # TODO: gmullins- The clone prevents crashes after evaluation. Need to understand why
            batch[key] = batch[key].to(device, non_blocking=True)

    with accelerator.accumulate(policy):
        with accelerator.autocast():
            loss, loss_dict = policy(batch)
            loss = loss.float()  # Ensure loss is float for backward pass
        accelerator.backward(loss)
        # Check gradients BEFORE sync/optimizer step
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        # Step through pytorch scheduler at every batch instead of epoch
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


# def train_policy_step(
#     policy, batch, optimizer, lr_scheduler, accelerator, step, device, grad_scaler=None, max_grad_norm=None
# ):
#     policy.train()
#     """Performs a single training step on the policy"""
#     for key in batch:
#         if isinstance(batch[key], torch.Tensor):
#             batch[key] = batch[key].to(device, non_blocking=True)
#     with accelerator.autocast():
#         loss, loss_dict = policy(batch)
#         loss = loss.float()  # Ensure loss is float for backward pass
#     accelerator.backward(loss)
#     if accelerator.sync_gradients:
#         accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
#         optimizer.step()
#         optimizer.zero_grad()
#     if lr_scheduler is not None:
#         lr_scheduler.step()
#     # Log metrics
#     metrics = {
#         "loss": loss.item(),
#         "step": step,
#         "learning_rate": lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]["lr"],
#     }
#     if loss_dict:
#         metrics.update(loss_dict)

#     return metrics


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    """Main training function"""

    # Patch lerobot's unbounded VideoDecoderCache with an LRU-bounded version
    # to prevent memory growth with large multi-shard video datasets.
    if cfg.dataset.video_decoder_cache_size is not None:
        from rho.datasets.video_decoder_patch import patch_video_decoder_cache

        patch_video_decoder_cache(max_size=cfg.dataset.video_decoder_cache_size)

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {cfg.checkpoint_folder}")

    if cfg.wandb.id is None:
        cfg.wandb.id = "default_job_" + cfg.checkpoint_folder.parent.stem

    # print("#################### Training config ####################")
    # pprint(asdict(cfg))
    # print("#################### Training config ####################")

    # Enable TF32 for safer high-throughput matmuls on Ampere+/Hopper GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,  # This should fix your DDP error
        # broadcast_buffers=True,  # Default
        # bucket_cap_mb=25,        # Default
        # gradient_as_bucket_view=False,  # Default
    )

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # log_with="wandb" if cfg.wandb.project else None,
        project_dir=cfg.output_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # Initialize logging to reduce output from non-main processes
    init_logging(accelerator=accelerator, console_level=cfg.log_level)

    if accelerator.is_main_process:
        serializable_cfg = serialize_train_config(cfg)
        with open(Path(cfg.checkpoint_folder.parent / "train_config.json"), "w") as f:
            json.dump(serializable_cfg, f, indent=4)

    device = accelerator.device
    # Create environment for evaluation (only on main process)
    env = None
    if accelerator.is_main_process and cfg.environment is not None:
        env = make_environment(cfg.environment)

    # According to this we need to increase the learning rate by the number of proceess
    # https://huggingface.co/docs/accelerate/v1.10.1/en/concept_guides/performance
    # I am unsure of how to do this given the different varieties of schedulers we support

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

    if cfg.dataset.streaming:
        # If streaming, we don't want to prepare the dataloaders with accelerate,
        # because accelerate's collate function mishandles string types on iterable datasets.
        # We end up moving these to device in the training loop instead.
        # We can maybe remove this path if https://github.com/huggingface/accelerate/pull/3686 is merged.
        policy, optimizer = accelerator.prepare(policy, optimizer)
    else:
        policy, optimizer, training_dataloader, validation_dataloader = accelerator.prepare(
            policy, optimizer, training_dataloader, validation_dataloader
        )

    if cfg.set_static_graph:
        logger.info("Setting policy to static graph mode")
        policy._set_static_graph()

    validation_probe = None
    if (
        cfg.validation_probe
        and accelerator.is_main_process
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
    if accelerator.is_main_process:
        wandb_logger = WandBLogger(cfg.wandb)

        # Create progress bar only on main process
        progress_bar = tqdm(
            range(step, cfg.steps),
            desc="Training",
            initial=step,
            total=cfg.steps,
            unit="step",
            disable=not accelerator.is_local_main_process,
        )
        wandb_logger.log_config(cfg)

        if cfg.action_monitoring:
            action_sampling_monitor = ActionSamplingMonitor(
                policy=policy,
                monitor_interval=cfg.action_monitor_interval,
                num_samples=cfg.action_monitor_samples,
                device=device,
            )
    else:
        progress_bar = range(step, cfg.steps)

    # Get the unwrapped policy for evaluation
    unwrapped_policy = policy.module if hasattr(policy, "module") else policy

    # Create PolicyInterface for evaluation (extracts sub-dataset if MultiDatasetConfig)
    policy_interface = make_policy_interface(
        dataset_config=cfg.dataset,
        policy=unwrapped_policy,
        device=device,
        eval_mode=cfg.eval_mode,
        inference_delay=cfg.inference_delay,
        beta=cfg.beta,
        eval_dataset_root_dir=cfg.eval_dataset_root_dir,
    )

    for _ in progress_bar:
        step += 1

        validation_metrics = None

        with training_metrics_recorder.log_time("dataloading_s"):
            batch = next(training_iter)

        # Apply training transforms and step them
        train_transforms.train()
        batch = train_transforms(batch)
        train_transforms.step()

        with training_metrics_recorder.log_time("update_s"):
            training_metrics = train_policy_step_accumulate(
                policy,
                batch,
                optimizer,
                lr_scheduler,
                accelerator,
                step,
                device,
                grad_scaler=None,
                max_grad_norm=cfg.grad_clip_norm,
            )

        # Log transform scales if transforms are being used
        if cfg.training_transforms is not None:
            for i, transform in enumerate(train_transforms.transforms):
                if hasattr(transform, "get_current_scale"):
                    scale = transform.get_current_scale()
                    training_metrics[f"transform_{i}_scale"] = scale

        if step % cfg.validation_interval == 0 and validation_dataloader is not None:
            # Evaluate policy on validation dataset
            policy.eval()
            with torch.no_grad(), accelerator.autocast(), training_metrics_recorder.log_time("dataloading_s"):
                validation_metrics = validate_policy(
                    policy,
                    validation_dataloader,
                    device,
                    cfg.eval_batch_size,
                    cfg.eval_num_episodes,
                    max_batches=cfg.max_validation_batches,
                )

        if accelerator.is_main_process:
            # Update progress bar with current metrics
            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                {"loss": f"{training_metrics['loss']:.4f}", "lr": f"{current_lr:.2e}", "step": step}
            )
            training_metrics_recorder.log(training_metrics)
            batch_size = batch["action"].shape[0] * accelerator.num_processes
            training_metrics_recorder.log_batch(step, batch_size)

            if step % cfg.logging_interval == 0:
                wandb_logger.log(
                    training_metrics_recorder.get_metrics_and_reset_rolling(), step=step, prefix="train"
                )

            if validation_metrics is not None:
                # Log validation metrics
                wandb_logger.log(validation_metrics, step=step, prefix="validation")

            if step % cfg.validation_probe_interval == 0 and validation_probe is not None:
                # validation_dataset = validation_dataloader.dataset
                validation_metrics = validation_probe.validate_policy_by_episode(
                    policy.module if hasattr(policy, "module") else policy, device
                )
                wandb_logger.log(validation_metrics, step=step, prefix="validation")

            # Evaluation at regular intervals
            if step % cfg.eval_interval == 0 and env is not None:
                # Evaluate policy on main process only
                video_path = cfg.checkpoint_folder.parent / "eval" / f"{step:07d}"

                policy_interface.policy.eval()
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
                unwrapped_policy.reset()
                if type(eval_metrics) is list:
                    eval_metrics = eval_metrics[
                        0
                    ]  # TODO, decide how to handle the multiple task metrics [tlh]
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
        # Sync all processes after logging/checkpointing/evaluation
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Training loop completed")
        if env is not None:
            env.close()

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as cleanup_error:
        if accelerator.is_main_process:
            logger.warning(f"Failed to destroy process group: {cleanup_error}")

    if accelerator.is_main_process:
        logger.info("All processes completed cleanup")

    # Exit without final synchronization to avoid NCCL timeouts
    import sys

    sys.exit(0)


if __name__ == "__main__":
    train()

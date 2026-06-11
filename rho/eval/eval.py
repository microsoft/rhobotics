import json
import logging
from datetime import datetime
from pathlib import Path

import draccus
import wandb

from rho.common.wandb_logging import WandBLogger
from rho.environment import make_environment
from rho.environment.env import evaluate_policy
from rho.eval.eval_config import SimEvalConfig
from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig
from rho.policies import make_policy
from rho.utils import init_logging

logger = logging.getLogger(__name__)


def init_wandb_from_training_run(cfg: SimEvalConfig) -> WandBLogger | None:
    """
    Initialize wandb by resuming the training run if wandb config is available.

    Args:
        cfg: SimEvalConfig instance with wandb_config loaded from train_config.json

    Returns:
        WandBLogger instance if successful, None otherwise
    """
    if cfg.wandb_config is None or not cfg.update_wandb_run:
        logger.info("No wandb config found in train_config.json or update_wandb_run is False")
        return None

    try:
        wandb_logger = WandBLogger(cfg.wandb_config)
        logger.info("Successfully resumed wandb run")
        return wandb_logger
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None


def log_eval_results(output_dir, results, cfg, wandb_logger: WandBLogger | None = None, step: int = None):
    """
    Log evaluation results to both file and wandb.

    Args:
        output_dir: Directory to save text results
        results: Dictionary of evaluation metrics
        cfg: EvalConfig instance
        wandb_logger: WandBLogger instance for logging to wandb
        step: Training step number (for wandb logging)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "evaluation_results.txt"

    # Log to file
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results for {cfg.pretrained_checkpoint}\n")
        f.write(f"Environment: {cfg.environment.name}\n")
        if hasattr(cfg.environment, "task_suite_name"):
            f.write(f"Task suite: {cfg.environment.task_suite_name}\n")
        if hasattr(cfg.environment, "task_name"):
            f.write(f"Task: {cfg.environment.task_name}\n")
        f.write(f"Episodes: {results['num_episodes']}\n")
        f.write(f"Max steps: {cfg.environment.max_episode_steps}\n")
        f.write(f"Mean success rate: {results['mean_success_rt']:.3f}\n")
        f.write(f"Mean reward: {results['mean_reward']:.3f}\n")
        f.write(f"Mean steps: {results['mean_steps']:.1f}\n")
        f.write(f"Episode rewards: {results['episode_rewards']}\n")
        f.write(f"Episode steps: {results['episode_steps']}\n")
        f.write(f"Episode successes: {results['episode_successes']}\n")
        f.write(f"Subtask progress: {results['episode_subtask_progress']}\n")
        f.write(f"Mean subtask progress: {results['mean_subtask_progress']:.3f}\n")
        f.write(f"Mean failure subtask progress: {results['mean_failure_subtask_progress']:.3f}\n")

    # Save JSON for programmatic consumption (e.g. auto_finetune wandb logging)
    json_file = output_dir / "evaluation_results.json"
    # Enrich results with environment metadata before writing
    results_with_meta = dict(results)
    results_with_meta["environment"] = cfg.environment.name
    if hasattr(cfg.environment, "task_suite_name"):
        results_with_meta["task_suite_name"] = cfg.environment.task_suite_name
    if hasattr(cfg.environment, "task_name"):
        results_with_meta["task_name"] = cfg.environment.task_name
    with open(json_file, "w") as f:
        json.dump(results_with_meta, f, indent=2, default=str)

    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Results JSON saved to: {json_file}")
    logger.info(f"Mean success rate: {results['mean_success_rt']:.3f}")
    logger.info(f"Mean reward: {results['mean_reward']:.3f}")
    logger.info(f"Mean steps: {results['mean_steps']:.1f}")

    # Log to wandb if logger available
    if wandb_logger is not None and wandb_logger.enabled:
        wandb_logger.log(results, step=step, prefix=cfg.name)
    else:
        logger.debug("No wandb logger available, skipping wandb logging")


@draccus.wrap()
def eval(cfg: SimEvalConfig) -> None:
    """
    Given a SimEvalConfig, this function will load a pretrained policy and
    evaluate it on provided EnvironmentConfig

    ALERT! Assuming there is a train_config.json at cfg.checkpoint_folder.parent,
    it will overwrite the LeRobotDatasetConfig and PolicyConfig.
    The idea is to always match the configs used in training, especially the
    normalization mapping and stats that are populated through the LeRobotDatasetConfig

    Logs metrics to wandb if there's an active run (e.g., from training).
    Metrics are logged with prefix '<eval_name>/' to distinguish different eval configs.
    """
    # Initialize logging
    init_logging(console_level=cfg.log_level)

    logger.info("Starting policy evaluation...")
    logger.info(f"Eval name: {cfg.name}")
    logger.info(f"Checkpoint: {cfg.pretrained_checkpoint}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Episodes: {cfg.eval_num_episodes}")
    logger.info(f"Max steps per episode: {cfg.environment.max_episode_steps}")
    logger.info(f"Record video: {cfg.record_videos}")

    # Initialize wandb from training run config if available
    wandb_logger = init_wandb_from_training_run(cfg)

    # 1. Create environment from EvalConfig
    env = make_environment(cfg.environment)
    logger.info(f"Environment initialized: {type(env).__name__}")
    logger.info(f"  num_envs: {env.num_envs}")
    logger.info(f"  is_vectorized: {env.is_vectorized}")

    # 2. Load the pretrained policy from provided checkpoint
    logger.info("Creating policy from config...")
    policy = make_policy(cfg.policy)
    policy.load_from_pretrained(cfg.pretrained_checkpoint)
    policy.eval()

    logger.info(f"Policy loaded: {type(policy).__name__}")
    logger.info(f"   Device: {policy.device}")

    # 3. Create PolicyInterface to wrap the policy with normalization/transforms
    logger.info("Creating PolicyInterface...")
    policy_interface_cfg = PolicyInterfaceConfig(
        data_config=cfg.dataset,
        policy=policy,
        device=cfg.device,
        eval_mode=cfg.eval_mode,
        inference_delay=cfg.inference_delay,
        beta=cfg.beta,
    )
    policy_interface = PolicyInterface(policy_interface_cfg)
    logger.info(f"PolicyInterface created with eval_mode: {cfg.eval_mode}")

    # 4. Determine output directory
    if cfg.output_dir is None or cfg.output_dir == "None":
        step = Path(cfg.pretrained_checkpoint).stem.split("_")[-1]
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        video_dir = Path(cfg.checkpoint_folder).parent / "manual_eval" / f"{step}" / timestamp
        video_dir.mkdir(parents=True, exist_ok=True)
    else:
        video_dir = Path(cfg.output_dir)
    logger.info(f"Saving output videos to: {video_dir}")

    # 5. Run evaluation using the standalone evaluate_policy function
    eval_metrics = evaluate_policy(
        env=env,
        policy_interface=policy_interface,
        num_episodes=cfg.eval_num_episodes,
        max_steps=cfg.environment.max_episode_steps,
        seed=cfg.seed,
        record_video=cfg.record_videos,
        output_dir=str(video_dir),
        eval_mode=cfg.eval_mode,
        inference_delay=cfg.inference_delay,
    )

    # Extract step number from checkpoint filename for wandb logging
    step = None
    try:
        checkpoint_stem = Path(cfg.pretrained_checkpoint).stem
        if "step_" in checkpoint_stem:
            step = int(checkpoint_stem.split("_")[-1])
    except (ValueError, IndexError):
        logger.warning("Could not extract step number from checkpoint name")

    log_eval_results(video_dir, eval_metrics, cfg, wandb_logger=wandb_logger, step=step)

    # Finish wandb run if we initialized it
    if wandb_logger is not None and wandb.run is not None:
        logger.info("Syncing wandb data before closing...")
        # Give wandb time to sync before finishing
        import time

        time.sleep(2)
        wandb_logger.finish()
        logger.info("Closed wandb run")

    # Close environment
    env.close()
    logger.info("Environment closed.")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    eval()  # nosec B307

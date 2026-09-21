import gc
import json
import logging
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import draccus
import torch

from rho.checkpoints import checkpoint_read_lease
from rho.common.wandb_logging import WandBLogger
from rho.environment import make_environment
from rho.environment.env import evaluate_policy
from rho.eval.eval_config import SimEvalConfig
from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig
from rho.policies import make_policy
from rho.utils import init_logging

logger = logging.getLogger(__name__)

# Scalar keys to include in multieval summary
EVAL_SCALAR_KEYS = {
    "mean_success_rt",
    "mean_reward",
    "mean_steps",
    "num_episodes",
    "mean_subtask_progress",
    "mean_failure_subtask_progress",
}
EVAL_META_KEYS = ("environment", "task_suite_name", "task_name")


def _make_policy_interface_config(cfg: SimEvalConfig, policy) -> PolicyInterfaceConfig:
    return PolicyInterfaceConfig(
        data_config=cfg.dataset,
        policy=policy,
        device=cfg.device,
        eval_mode=cfg.eval_mode,
        inference_delay=cfg.inference_delay,
        execution_horizon=cfg.execution_horizon,
        beta=cfg.beta,
        guidance_schedule=cfg.guidance_schedule,
    )


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


def _cleanup_gpu():
    """Best-effort release of GPU memory between evaluation runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_multieval_summary(
    summary_results: dict,
    eval_output_dir: Path,
    checkpoint_path: Path,
    step: int | None,
) -> Path:
    """Write a multieval summary JSON and print a human-readable table.

    Returns the path to the written summary file.
    """
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = eval_output_dir / f"multieval_summary_{date_str}.json"

    completed = {k: v for k, v in summary_results.items() if v.get("status", "success") == "success"}
    failed = [k for k in summary_results if k not in completed]
    success_rates = [v["mean_success_rt"] for v in completed.values() if "mean_success_rt" in v]
    completed_mean = float(sum(success_rates) / len(success_rates)) if success_rates else None
    status = "success" if not failed else ("partial" if completed else "failed")
    summary = {
        "pretrain_step": step,
        "finetuned_checkpoint": str(checkpoint_path),
        "date": date_str,
        "status": status,
        "aggregate": {
            "num_tasks": len(summary_results),
            "num_completed": len(completed),
            "num_failed": len(failed),
            "complete": not failed,
            "mean_success_rt": completed_mean if not failed else None,
            "mean_success_rt_completed": completed_mean,
            "per_task_success_rates": {k: v.get("mean_success_rt") for k, v in summary_results.items()},
        },
        "per_task": summary_results,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Multieval summary saved to: {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"Multieval Summary (step {step}, {status})")
    print(f"{'=' * 60}")
    for task_name, task_results in summary_results.items():
        if task_name in failed:
            print(f"  {task_name:40s} FAILED: {task_results['error']}")
        else:
            sr = task_results.get("mean_success_rt")
            print(f"  {task_name:40s} {sr:.1%}" if sr is not None else f"  {task_name:40s} N/A")
    if success_rates:
        label = "AVERAGE (completed tasks only)" if failed else "AVERAGE"
        print(f"  {label:40s} {completed_mean:.1%}")
    print(f"{'=' * 60}\n")

    return summary_path


def _run_single_eval(cfg: SimEvalConfig) -> dict:
    """Run a single evaluation. Called by both eval() and multi_eval().

    When called from multi_eval(), the policy is already loaded in cfg and
    this function only creates the environment, evaluates, and cleans up.

    Returns:
        Evaluation metrics dict. Failures propagate after resources are closed.
    """
    with ExitStack() as cleanup:
        wandb_logger = init_wandb_from_training_run(cfg)
        if wandb_logger is not None:
            cleanup.callback(wandb_logger.finish)
        env = make_environment(cfg.environment)
        cleanup.callback(env.close)
        return _evaluate_in_environment(cfg, env, wandb_logger)


def _evaluate_in_environment(cfg, env, wandb_logger) -> dict:
    logger.info(f"Environment initialized: {type(env).__name__}")
    logger.info(f"  num_envs: {env.num_envs}")
    logger.info(f"  is_vectorized: {env.is_vectorized}")

    # 2. Load or reuse the policy
    # If cfg already has a loaded policy (from multi_eval), reuse it
    if hasattr(cfg, "_loaded_policy") and cfg._loaded_policy is not None:
        policy = cfg._loaded_policy
        logger.info(f"Reusing pre-loaded policy: {type(policy).__name__}")
    else:
        logger.info("Creating policy from config...")
        policy = make_policy(cfg.policy)
        policy.load_from_pretrained(cfg.pretrained_checkpoint)
        policy.eval()

        # Disable gradient checkpointing during eval — it's unnecessary
        # and produces spurious warnings about reentrant checkpointing
        if hasattr(policy, "model"):
            for module in policy.model.modules():
                if hasattr(module, "enable_gradient_checkpointing"):
                    module.enable_gradient_checkpointing = False

        logger.info(f"Policy loaded: {type(policy).__name__}")
        logger.info(f"   Device: {policy.device}")

    # 3. Create PolicyInterface to wrap the policy with normalization/transforms
    logger.info("Creating PolicyInterface...")
    policy_interface_cfg = _make_policy_interface_config(cfg, policy)
    policy_interface = PolicyInterface(policy_interface_cfg)
    logger.info(f"PolicyInterface created with eval_mode: {cfg.eval_mode}")

    # 4. Determine output directory
    if cfg.output_dir is None or cfg.output_dir == "None":
        step_str = Path(cfg.pretrained_checkpoint).stem.split("_")[-1]
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        video_dir = Path(cfg.checkpoint_folder).parent / "manual_eval" / f"{step_str}" / timestamp
        video_dir.mkdir(parents=True, exist_ok=True)
    else:
        video_dir = Path(cfg.output_dir)
    logger.info(f"Saving output to: {video_dir}")

    # 5. Run evaluation
    eval_metrics = evaluate_policy(
        env=env,
        policy_interface=policy_interface,
        num_episodes=cfg.eval_num_episodes,
        max_steps=cfg.environment.max_episode_steps,
        seed=cfg.seed,
        policy_seed=cfg.policy_seed,
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

    return eval_metrics


def _initialize_default_checkpoint(cfg: SimEvalConfig) -> None:
    """Resolve the policy's hosted default when no checkpoint override is set."""
    if cfg.pretrained_checkpoint is not None:
        return

    default_checkpoint = getattr(cfg.policy, "pretrained_repo_id", None)
    if default_checkpoint is None:
        raise ValueError(
            "No pretrained checkpoint was configured. Set pretrained_checkpoint "
            "or use a policy with pretrained_repo_id."
        )

    cfg.pretrained_checkpoint = default_checkpoint
    cfg.__post_init__()


def _make_multi_eval_config(
    cfg: SimEvalConfig,
    eval_cfg,
    checkpoint: Path,
    output_dir: Path,
    task_name: str,
) -> SimEvalConfig:
    """Build one multi-eval entry while preserving public dataset preprocessing."""
    execution_horizon = getattr(eval_cfg, "execution_horizon", None)
    if execution_horizon is None:
        execution_horizon = cfg.execution_horizon
    return SimEvalConfig(
        pretrained_checkpoint=str(checkpoint),
        eval_num_episodes=getattr(eval_cfg, "eval_num_episodes", cfg.eval_num_episodes),
        record_videos=getattr(eval_cfg, "record_videos", cfg.record_videos),
        output_dir=str(output_dir / task_name),
        device=getattr(eval_cfg, "device", cfg.device),
        name=task_name,
        environment=eval_cfg.environment,
        seed=getattr(eval_cfg, "seed", cfg.seed),
        policy_seed=getattr(eval_cfg, "policy_seed", cfg.policy_seed),
        dataset=getattr(eval_cfg, "dataset", cfg.dataset),
        policy=getattr(eval_cfg, "policy", cfg.policy),
        dataset_root_dir=getattr(eval_cfg, "dataset_root_dir", cfg.dataset_root_dir),
        eval_mode=getattr(eval_cfg, "eval_mode", cfg.eval_mode),
        inference_delay=getattr(eval_cfg, "inference_delay", cfg.inference_delay),
        execution_horizon=execution_horizon,
        beta=getattr(eval_cfg, "beta", cfg.beta),
        guidance_schedule=getattr(eval_cfg, "guidance_schedule", cfg.guidance_schedule),
    )


@draccus.wrap()
def eval(cfg: SimEvalConfig) -> None:
    """Evaluate a pretrained policy on one or more environments.

    When ``cfg.eval_configs`` is empty (standard SimEvalConfig YAML), runs a
    single evaluation.  When ``cfg.eval_configs`` is populated (multieval YAML),
    loads the policy ONCE and iterates through each sub-config, creating and
    destroying environments while reusing the policy.

    ALERT! Assuming there is a train_config.json at cfg.checkpoint_folder.parent,
    it will overwrite the LeRobotDatasetConfig and PolicyConfig.
    The idea is to always match the configs used in training, especially the
    normalization mapping and stats that are populated through the LeRobotDatasetConfig

    Logs metrics to wandb if there's an active run (e.g., from training).
    Metrics are logged with prefix '<eval_name>/' to distinguish different eval configs.
    """
    # Initialize logging
    init_logging(console_level=cfg.log_level)

    _initialize_default_checkpoint(cfg)

    with checkpoint_read_lease(cfg.pretrained_checkpoint):
        if cfg.is_multi_eval:
            _run_multi_eval(cfg)
        else:
            logger.info("Starting policy evaluation...")
            logger.info(f"Eval name: {cfg.name}")
            logger.info(f"Checkpoint: {cfg.pretrained_checkpoint}")
            logger.info(f"Device: {cfg.device}")
            logger.info(f"Episodes: {cfg.eval_num_episodes}")
            logger.info(f"Max steps per episode: {cfg.environment.max_episode_steps}")
            logger.info(f"Record video: {cfg.record_videos}")

            _run_single_eval(cfg)
            logger.info("Evaluation completed successfully!")


def _run_multi_eval(cfg: SimEvalConfig) -> None:
    """Run multiple evaluations with a shared policy from a single checkpoint.

    The policy is loaded ONCE from ``cfg.pretrained_checkpoint`` and reused
    across all eval configs.  For each config the environment is created,
    evaluation runs, and the environment is torn down — but the policy stays
    in memory.
    """
    checkpoint = cfg.pretrained_checkpoint
    if checkpoint is None:
        raise ValueError("pretrained_checkpoint must be set for multi-eval")
    task_names = [getattr(entry, "name", f"eval_{i}") for i, entry in enumerate(cfg.eval_configs)]
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("Multi-eval requires at least one evaluation and unique task names.")

    logger.info("=" * 80)
    logger.info("Starting MULTI-EVAL")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Configs: {len(cfg.eval_configs)}")
    logger.info("=" * 80)

    # Load the policy ONCE — this is the key optimisation.
    logger.info("Loading policy from checkpoint (once)...")
    policy = make_policy(cfg.policy)
    policy.load_from_pretrained(str(cfg.pretrained_checkpoint))
    policy.eval()

    # Disable gradient checkpointing during eval
    if hasattr(policy, "model"):
        for module in policy.model.modules():
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing = False

    logger.info(f"Policy loaded: {type(policy).__name__}")

    # Extract step from checkpoint name
    step = None
    try:
        checkpoint_stem = Path(checkpoint).stem
        if "step_" in checkpoint_stem:
            step = int(checkpoint_stem.split("_")[-1])
    except (ValueError, IndexError):
        pass

    # Determine base output directory
    if cfg.output_dir and cfg.output_dir != "None":
        eval_output_dir = Path(cfg.output_dir)
    else:
        base_output_dir = Path(cfg.checkpoint_folder).parent / "multieval"
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        eval_output_dir = (
            base_output_dir / f"step_{step}" / timestamp if step else base_output_dir / timestamp
        )

    summary_results = {}

    for i, eval_cfg in enumerate(cfg.eval_configs):
        task_name = getattr(eval_cfg, "name", f"eval_{i}")
        logger.info(f"\n--- Evaluation {i + 1}/{len(cfg.eval_configs)}: {task_name} ---")

        try:
            eval_cfg_copy = _make_multi_eval_config(cfg, eval_cfg, checkpoint, eval_output_dir, task_name)
            eval_cfg_copy._loaded_policy = policy
            results = _run_single_eval(eval_cfg_copy)
            summary_results[task_name] = {
                k: v for k, v in results.items() if k in EVAL_SCALAR_KEYS and isinstance(v, (int, float))
            }
            summary_results[task_name]["status"] = "success"
            summary_results[task_name]["environment"] = eval_cfg_copy.environment.name
            for meta_key in EVAL_META_KEYS[1:]:
                if hasattr(eval_cfg_copy.environment, meta_key):
                    summary_results[task_name][meta_key] = getattr(eval_cfg_copy.environment, meta_key)

            logger.info(f"Completed: {task_name}")

        except Exception as e:
            logger.exception("Failed: %s", task_name)
            failure = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            summary_results[task_name] = failure
            results_json = eval_output_dir / task_name / "evaluation_results.json"
            results_json.parent.mkdir(parents=True, exist_ok=True)
            with results_json.open("w") as f:
                json.dump(failure, f, indent=2)
        finally:
            _cleanup_gpu()

    # Write multieval summary
    summary_path = write_multieval_summary(summary_results, eval_output_dir, Path(checkpoint), step)
    failed_tasks = [name for name, result in summary_results.items() if result["status"] == "failed"]
    if failed_tasks:
        raise RuntimeError(
            f"Multi-eval failed for {len(failed_tasks)}/{len(task_names)} tasks: "
            f"{', '.join(failed_tasks)}. Results: {summary_path}"
        )

    logger.info("Multi-eval completed!")


if __name__ == "__main__":
    eval()  # nosec B307

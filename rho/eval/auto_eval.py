#!/usr/bin/env python3
"""
Automatic evaluation utilities and monitoring script.

Provides checkpoint discovery, evaluation orchestration, and a standalone
monitoring loop.  Also contains the multi-eval runner used by the
auto-finetune pipeline (``evaluate_finetuned_checkpoint``).

Standalone usage::

    python rho/eval/auto_eval.py \
        --config_path config/multi_eval_config.yaml \
        --checkpoint_folders /path/to/training/checkpoints \
        --eval_interval 10000 \
        --sleep_interval 300
"""

import argparse
import copy
import gc
import json
import logging
import time
from pathlib import Path

import draccus
import torch

from rho.common.wandb_logging import WandBConfig, WandBLogger
from rho.eval.eval import eval
from rho.eval.eval_config import EvalConfig, MultiEvalConfig, SimEvalConfig
from rho.training.train import TrainConfig

logger = logging.getLogger(__name__)


def parse_checkpoint_step(checkpoint_path: Path) -> int | None:
    """Extract step number from checkpoint filename.

    Expected format: ``checkpoint_step_XXXXXX.pt``.
    Returns ``None`` for ``checkpoint_latest.pt`` or unparseable names.
    """
    stem = checkpoint_path.stem
    if stem == "checkpoint_latest":
        return None
    if stem.startswith("checkpoint_step_"):
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return None
    return None


def find_checkpoints_to_evaluate(
    checkpoint_folder: Path,
    eval_interval: int,
    last_evaluated_step: int = 0,
) -> list[Path]:
    """Find checkpoints that need evaluation, sorted by step."""
    if not checkpoint_folder.exists():
        logger.warning(f"Checkpoint folder does not exist: {checkpoint_folder}")
        return []

    checkpoints_to_eval = []
    for ckpt_path in checkpoint_folder.glob("checkpoint_step_*.pt"):
        step = parse_checkpoint_step(ckpt_path)
        if step is not None and step > last_evaluated_step and step % eval_interval == 0:
            checkpoints_to_eval.append((step, ckpt_path))

    checkpoints_to_eval.sort(key=lambda x: x[0])
    return [ckpt for _, ckpt in checkpoints_to_eval]


def find_latest_finetuned_checkpoint(finetune_output_dir: Path) -> Path | None:
    """Find the latest checkpoint produced by a fine-tuning run.

    Searches for the most recent timestamp subfolder containing a checkpoints
    directory with ``checkpoint_latest.pt``, falling back to the highest-step
    checkpoint file.
    """
    checkpoint_dirs = sorted(finetune_output_dir.glob("*/checkpoints"), reverse=True)

    for ckpt_dir in checkpoint_dirs:
        latest = ckpt_dir / "checkpoint_latest.pt"
        if latest.exists():
            return latest

        step_files = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
        if step_files:
            step_files.sort(key=lambda p: parse_checkpoint_step(p) or -1, reverse=True)
            return step_files[0]

    return None


# ---------------------------------------------------------------------------
# Completion markers
# ---------------------------------------------------------------------------


def is_finetuning_complete(finetune_output_dir: Path) -> bool:
    """Check if fine-tuning fully completed (looks for ``.finetune_complete`` marker)."""
    return (finetune_output_dir / ".finetune_complete").exists()


def is_evaluation_complete(eval_output_dir: Path, eval_configs: list) -> bool:
    """Check if all evaluation tasks completed for a given step."""
    if not eval_output_dir.exists():
        return False
    for i, eval_cfg in enumerate(eval_configs):
        name = getattr(eval_cfg, "name", f"eval_{i}")
        task_dir = eval_output_dir / name
        if not (
            (task_dir / "evaluation_results.txt").exists() or (task_dir / "evaluation_results.json").exists()
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# GPU cleanup
# ---------------------------------------------------------------------------


def cleanup_gpu():
    """Best-effort release of GPU memory between training and evaluation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-eval runner (used by auto_finetune pipeline)
# ---------------------------------------------------------------------------

# Scalar keys to extract from evaluation_results.json
EVAL_SCALAR_KEYS = {
    "mean_success_rt",
    "mean_reward",
    "mean_steps",
    "num_episodes",
    "mean_subtask_progress",
    "mean_failure_subtask_progress",
}

# Environment metadata keys to propagate into summaries
EVAL_META_KEYS = ("environment", "task_suite_name", "task_name")


def init_autoeval_wandb(
    parent_wandb: WandBConfig,
    eval_name: str,
) -> WandBLogger | None:
    """Create a persistent autoeval wandb run for logging across pretrained steps."""
    try:
        autoeval_cfg = copy.deepcopy(parent_wandb)
        ft_suffix = f"_ft_{eval_name}" if eval_name else "_ft"
        autoeval_cfg.project = parent_wandb.project + ft_suffix
        autoeval_cfg.id = f"{parent_wandb.id}_evals"
        autoeval_cfg.resume = "allow"
        autoeval_cfg.enabled = True
        autoeval_cfg.tags = list(parent_wandb.tags or []) + ["autoeval"]
        autoeval_cfg.notes = f"Auto-eval tracking for parent run {parent_wandb.id}"
        wandb_logger = WandBLogger(autoeval_cfg)
        logger.info(f"Autoeval wandb: project={autoeval_cfg.project}, id={autoeval_cfg.id}")
        return wandb_logger
    except Exception as e:
        logger.warning(f"Failed to initialise autoeval wandb: {e}")
        return None


def log_eval_results_to_wandb(
    wandb_logger: WandBLogger,
    eval_name: str,
    results: dict,
    eval_task_dir: Path,
    pretrain_step: int | None,
) -> None:
    """Log scalar eval results and videos to the autoeval wandb run."""
    scalar_results = {
        k: v for k, v in results.items() if k in EVAL_SCALAR_KEYS and isinstance(v, (int, float))
    }

    import wandb as _wandb

    for vf in sorted(eval_task_dir.glob("*.mp4")):
        try:
            scalar_results[f"recording/{vf.stem}"] = _wandb.Video(
                str(vf),
                fps=30,
                format="mp4",
            )
            logger.info(f"  Attaching video: {vf.name}")
        except Exception as ve:
            logger.warning(f"  Failed to attach video {vf.name}: {ve}")

    wandb_logger.log(scalar_results, step=pretrain_step, prefix=eval_name)
    logger.info(f"Logged {eval_name} results to autoeval wandb at step {pretrain_step}")


def write_multieval_summary(
    summary_results: dict,
    eval_output_dir: Path,
    finetuned_checkpoint: Path,
    pretrain_step: int | None,
) -> None:
    """Write a multieval summary JSON and print a human-readable table."""
    from datetime import datetime

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = eval_output_dir / f"multieval_summary_{date_str}.json"

    success_rates = [v["mean_success_rt"] for v in summary_results.values() if "mean_success_rt" in v]
    summary = {
        "pretrain_step": pretrain_step,
        "finetuned_checkpoint": str(finetuned_checkpoint),
        "date": date_str,
        "aggregate": {
            "num_tasks": len(summary_results),
            "mean_success_rt": float(sum(success_rates) / len(success_rates)) if success_rates else 0.0,
            "per_task_success_rates": {k: v.get("mean_success_rt", 0.0) for k, v in summary_results.items()},
        },
        "per_task": summary_results,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Multieval summary saved to: {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"Multieval Summary (pretrain step {pretrain_step})")
    print(f"{'=' * 60}")
    for task_name, task_results in summary_results.items():
        sr = task_results.get("mean_success_rt", 0.0)
        print(f"  {task_name:40s} {sr:.1%}")
    if success_rates:
        print(f"  {'AVERAGE':40s} {sum(success_rates) / len(success_rates):.1%}")
    print(f"{'=' * 60}\n")


def evaluate_finetuned_checkpoint(
    eval_configs: list,
    finetuned_checkpoint: Path,
    eval_output_dir: Path,
    parent_wandb: WandBConfig | None = None,
    pretrain_step: int | None = None,
    eval_name: str = "",
) -> bool:
    """Run multiple evaluations on a fine-tuned checkpoint.

    When *parent_wandb* is provided, eval results are logged to a dedicated
    wandb run.  Each call resumes that run and logs metrics at *pretrain_step*,
    so successive checkpoints accumulate as line charts.
    """
    step = parse_checkpoint_step(finetuned_checkpoint)
    step_str = f"step {step}" if step is not None else "latest"

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {finetuned_checkpoint.name} ({step_str})")
    print(f"  {len(eval_configs)} config(s) | output: {eval_output_dir}")
    print(f"{'=' * 80}\n")

    wandb_logger = init_autoeval_wandb(parent_wandb, eval_name) if parent_wandb is not None else None

    all_success = True
    summary_results = {}

    for i, eval_cfg in enumerate(eval_configs):
        task_name = getattr(eval_cfg, "name", f"eval_{i}")
        print(f"\n--- Evaluation {i + 1}/{len(eval_configs)}: {task_name} ---")

        # Skip tasks that already have results
        existing_results_json = eval_output_dir / task_name / "evaluation_results.json"
        if existing_results_json.exists():
            print(f"  Skipping {task_name}: evaluation_results.json already exists")
            with open(existing_results_json) as f:
                results = json.load(f)
            summary_results[task_name] = {
                k: v for k, v in results.items() if k in EVAL_SCALAR_KEYS and isinstance(v, (int, float))
            }
            for meta_key in EVAL_META_KEYS:
                if meta_key in results:
                    summary_results[task_name][meta_key] = results[meta_key]
            continue

        eval_cfg_copy = SimEvalConfig(
            pretrained_checkpoint=str(finetuned_checkpoint),
            eval_num_episodes=getattr(eval_cfg, "eval_num_episodes", 5),
            record_videos=getattr(eval_cfg, "record_videos", True),
            output_dir=str(eval_output_dir / task_name),
            device=getattr(eval_cfg, "device", "cuda"),
            name=task_name,
            environment=eval_cfg.environment,
            dataset=getattr(eval_cfg, "dataset", None),
            policy=getattr(eval_cfg, "policy", None),
            seed=getattr(eval_cfg, "seed", 12345),
            dataset_root_dir=getattr(eval_cfg, "dataset_root_dir", None),
        )

        try:
            eval(eval_cfg_copy)
            print(f"  Completed: {task_name}")

            results_json = Path(eval_output_dir / task_name) / "evaluation_results.json"
            if results_json.exists():
                with open(results_json) as f:
                    results = json.load(f)

                summary_results[task_name] = {
                    k: v for k, v in results.items() if k in EVAL_SCALAR_KEYS and isinstance(v, (int, float))
                }
                for meta_key in EVAL_META_KEYS:
                    if meta_key in results:
                        summary_results[task_name][meta_key] = results[meta_key]

                if wandb_logger is not None:
                    log_eval_results_to_wandb(
                        wandb_logger,
                        task_name,
                        results,
                        Path(eval_output_dir / task_name),
                        pretrain_step,
                    )
            else:
                logger.warning(f"No evaluation_results.json at {results_json}")

        except Exception as e:
            print(f"  Failed: {task_name}: {e}")
            import traceback

            traceback.print_exc()
            all_success = False

    if summary_results:
        write_multieval_summary(
            summary_results,
            eval_output_dir,
            finetuned_checkpoint,
            pretrain_step,
        )

    if wandb_logger is not None:
        try:
            import wandb

            if wandb.run is not None:
                time.sleep(2)
                wandb_logger.finish()
                logger.info("Closed autoeval wandb run")
        except Exception as e:
            logger.warning(f"Error closing autoeval wandb: {e}")

    cleanup_gpu()
    return all_success


# ---------------------------------------------------------------------------
# Evaluation state persistence (used by standalone auto_eval)
# ---------------------------------------------------------------------------


def load_evaluation_state(checkpoint_folder: Path) -> dict:
    """Load the evaluation state file that tracks which checkpoints have been evaluated."""
    state_file = checkpoint_folder.parent / "auto_eval_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load evaluation state from {state_file}: {e}")

    return {"last_evaluated_step": 0}


def save_evaluation_state(checkpoint_folder: Path, last_evaluated_step: int) -> None:
    """Save the evaluation state file that tracks which checkpoints have been evaluated."""
    state_file = checkpoint_folder.parent / "auto_eval_state.json"
    try:
        with open(state_file, "w") as f:
            json.dump({"last_evaluated_step": last_evaluated_step}, f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to save evaluation state to {state_file}: {e}")


def evaluate_checkpoint(eval_configs: list[EvalConfig], checkpoint_path: Path) -> bool:
    """Run evaluation(s) on a specific checkpoint.

    Returns True if all evaluations succeeded.
    """
    step = parse_checkpoint_step(checkpoint_path)
    logger.info("=" * 80)
    logger.info(f"Evaluating checkpoint: {checkpoint_path.name} (step {step})")
    logger.info(f"Running {len(eval_configs)} evaluation configuration(s)")
    logger.info("=" * 80)

    all_success = True

    for i, eval_cfg in enumerate(eval_configs):
        logger.info(f"--- Evaluation {i + 1}/{len(eval_configs)} ---")

        # Create a new config with the specific checkpoint
        # Avoid using draccus.encode/decode as it can't handle numpy arrays
        eval_cfg_copy = SimEvalConfig(
            pretrained_checkpoint=str(checkpoint_path),
            eval_num_episodes=getattr(eval_cfg, "eval_num_episodes", 5),
            record_videos=getattr(eval_cfg, "record_videos", True),
            output_dir=getattr(eval_cfg, "output_dir", None),
            device=getattr(eval_cfg, "device", "cuda"),
            name=eval_cfg.name,
            environment=eval_cfg.environment,
            dataset=eval_cfg.dataset,
            policy=eval_cfg.policy,
            seed=eval_cfg.seed,
        )

        if eval_cfg_copy.output_dir:
            output_dir = Path(eval_cfg_copy.output_dir)
        else:
            output_dir = checkpoint_path.parent.parent / "auto_eval"

        if len(eval_configs) > 1:
            eval_cfg_copy.output_dir = str(output_dir / f"step_{step:07d}" / f"eval_{i:02d}")
        else:
            eval_cfg_copy.output_dir = str(output_dir / f"step_{step:07d}")

        try:
            eval(eval_cfg_copy)
            logger.info(f"Successfully completed evaluation {i + 1}/{len(eval_configs)} for step {step}")
        except Exception as e:
            logger.error(f"Failed evaluation {i + 1}/{len(eval_configs)} for step {step}: {e}")
            import traceback

            traceback.print_exc()
            all_success = False

    return all_success


# ---------------------------------------------------------------------------
# Config loaders (standalone auto_eval)
# ---------------------------------------------------------------------------


def load_eval_config_from_file(config_path: str) -> EvalConfig:
    """Load evaluation config from either an EvalConfig or TrainConfig file."""
    config_path = Path(config_path)

    try:
        with open(config_path) as f:
            eval_cfg = draccus.load(EvalConfig, f)
        logger.info(f"Loaded as EvalConfig from: {config_path}")
        return eval_cfg
    except Exception as e:
        logger.debug(f"Could not load as EvalConfig: {e}")

    try:
        with open(config_path) as f:
            eval_cfg = draccus.load(MultiEvalConfig, f)
        logger.info(f"Loaded as MultiEvalConfig from: {config_path}")
        return eval_cfg
    except Exception as e:
        logger.debug(f"Could not load as MultiEvalConfig: {e}")

    try:
        with open(config_path) as f:
            train_cfg = draccus.load(TrainConfig, f)
        logger.info(f"Loaded as TrainConfig from: {config_path}")

        # Convert TrainConfig to EvalConfig
        eval_cfg = EvalConfig(
            pretrained_checkpoint=None,
            eval_num_episodes=getattr(train_cfg, "eval_num_episodes", 5),
            record_videos=getattr(train_cfg, "record_videos", True),
            output_dir=None,
            device="cuda",
            environment=train_cfg.environment if hasattr(train_cfg, "environment") else None,
            dataset=train_cfg.dataset,
            policy=train_cfg.policy,
            seed=getattr(train_cfg, "seed", 12345),
        )

        logger.info("Successfully converted TrainConfig to EvalConfig")
        return eval_cfg
    except Exception as e:
        raise ValueError(f"Could not load config from {config_path} as EvalConfig or TrainConfig: {e}") from e


def load_eval_configs_from_file(config_path: str) -> list[EvalConfig]:
    """Load evaluation config(s) from MultiEvalConfig, EvalConfig, or TrainConfig file."""
    config_path = Path(config_path)
    try:
        with open(config_path) as f:
            multi_cfg = draccus.load(MultiEvalConfig, f)
        logger.info(f"Loaded as MultiEvalConfig from: {config_path}")
        logger.info(f"Found {len(multi_cfg.eval_configs)} eval config(s) to run")
        return multi_cfg.eval_configs
    except Exception as e:
        logger.debug(f"Could not load as MultiEvalConfig: {e}")

    eval_cfg = load_eval_config_from_file(config_path)
    return [eval_cfg]


# ---------------------------------------------------------------------------
# Standalone monitoring loop
# ---------------------------------------------------------------------------


def monitor_and_evaluate(
    config_path: str,
    checkpoint_folders: list[str],
    eval_interval: int,
    sleep_interval: int = 300,
    run_once: bool = False,
    max_wait_iterations: int = 10,
) -> None:
    """
    Main monitoring loop that checks for new checkpoints and evaluates them.

    Args:
        config_path: Path to the evaluation or training config file
        checkpoint_folders: List of checkpoint folder paths to monitor
        eval_interval: Only evaluate checkpoints at this interval (e.g., 10000)
        sleep_interval: Seconds to wait between checks (default: 300 = 5 minutes)
        run_once: If True, check once and exit. If False, run continuously
        max_wait_iterations: Maximum number of iterations to wait without finding new
                             checkpoints before exiting (default: 10)
    """
    logger.info("=" * 80)
    logger.info("Auto-Evaluation Script Started")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Monitoring {len(checkpoint_folders)} checkpoint folder(s)")
    logger.info(f"Eval interval: {eval_interval} steps")
    logger.info(f"Sleep interval: {sleep_interval} seconds")
    logger.info(f"Run mode: {'Single pass' if run_once else 'Continuous monitoring'}")
    logger.info("=" * 80)

    eval_configs = load_eval_configs_from_file(config_path)
    logger.info(f"Loaded {len(eval_configs)} evaluation configuration(s)")
    logger.info("=" * 80)

    checkpoint_folders = [Path(folder) for folder in checkpoint_folders]

    iteration = 0
    wait_iterations = 0
    while True:
        iteration += 1
        logger.info(f"[Iteration {iteration}] Checking for new checkpoints...")

        any_evaluated = False

        for checkpoint_folder in checkpoint_folders:
            logger.info(f"  Checking folder: {checkpoint_folder}")

            state = load_evaluation_state(checkpoint_folder)
            last_evaluated_step = state.get("last_evaluated_step", 0)
            logger.info(f"    Last evaluated step: {last_evaluated_step}")

            checkpoints = find_checkpoints_to_evaluate(
                checkpoint_folder,
                eval_interval,
                last_evaluated_step,
            )

            if checkpoints:
                logger.info(f"    Found {len(checkpoints)} checkpoint(s) to evaluate")

                # Evaluate each checkpoint
                for ckpt_path in checkpoints:
                    success = evaluate_checkpoint(eval_configs, ckpt_path)
                    if success:
                        step = parse_checkpoint_step(ckpt_path)
                        save_evaluation_state(checkpoint_folder, step)
                        any_evaluated = True
            else:
                logger.info("    No new checkpoints to evaluate")

        if run_once:
            logger.info("=" * 80)
            logger.info("Single pass completed. Exiting.")
            logger.info("=" * 80)
            break

        if any_evaluated:
            wait_iterations = 0
        else:
            wait_iterations += 1
            if wait_iterations >= max_wait_iterations:
                logger.info("=" * 80)
                logger.info(
                    f"Reached maximum wait iterations ({max_wait_iterations}) "
                    f"without finding new checkpoints."
                )
                logger.info("Exiting auto-eval.")
                logger.info("=" * 80)
                break

        logger.info("=" * 80)
        logger.info(f"Waiting {sleep_interval} seconds before next check...")
        if not any_evaluated:
            logger.info(f"Wait iterations: {wait_iterations}/{max_wait_iterations}")
        logger.info("=" * 80)
        time.sleep(sleep_interval)


def main():
    parser = argparse.ArgumentParser(description="Automatically evaluate new training checkpoints")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to MultiEvalConfig, EvalConfig, or TrainConfig YAML file",
    )
    parser.add_argument(
        "--checkpoint_folders",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint folders to monitor",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10000,
        help="Evaluate checkpoints at this step interval (default: 10000)",
    )
    parser.add_argument(
        "--sleep_interval", type=int, default=300, help="Seconds to wait between checks (default: 300)"
    )
    parser.add_argument(
        "--run_once",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Run once and exit (default: False)",
    )
    parser.add_argument(
        "--max_wait_iterations",
        type=int,
        default=10,
        help="Max iterations without new checkpoints before exiting (default: 10)",
    )

    args = parser.parse_args()

    try:
        monitor_and_evaluate(
            config_path=args.config_path,
            checkpoint_folders=args.checkpoint_folders,
            eval_interval=args.eval_interval,
            sleep_interval=args.sleep_interval,
            run_once=args.run_once,
            max_wait_iterations=args.max_wait_iterations,
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

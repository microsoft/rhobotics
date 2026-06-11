#!/usr/bin/env python3
"""
Automatic fine-tuning pipeline that monitors pretrained checkpoint folders,
fine-tunes new checkpoints on a downstream task, and evaluates the results.

Usage::

    python rho/eval/auto_finetune.py --config_path config/auto_finetune.yaml
"""

import copy
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from rho.common.serialization import make_yaml_safe, serialize_to_dict
from rho.common.wandb_logging import WandBConfig, load_wandb_config_from_train_dir
from rho.eval.auto_eval import (
    evaluate_finetuned_checkpoint,
    find_checkpoints_to_evaluate,
    find_latest_finetuned_checkpoint,
    is_evaluation_complete,
    is_finetuning_complete,
    parse_checkpoint_step,
)
from rho.eval.eval_config import MultiEvalConfig, load_policy_config_from_json
from rho.training.train import TrainConfig

logger = logging.getLogger(__name__)

# Policy config fields that are training behaviour (not architecture).
# These are preserved from the finetune YAML when the pretrained policy
# config is loaded.
FINETUNE_POLICY_OVERRIDE_KEYS: set[str] = {
    "freeze_vision_encoder",
    "freeze_vision_transformer",
    "freeze_vision_projector",
    "train_expert_only",
    "chunk_size",
    "n_action_steps",
    "n_obs_steps",
    "scheduler_decay_steps",
    "scheduler_warmup_steps",
    "scheduler_decay_lr",
    "optimizer_lr",
    "optimizer_betas",
    "optimizer_eps",
    "optimizer_weight_decay",
    "optimizer",
    "lr_scheduler",
    "device",
    "dtype",
    "feature_dict",
    "dropout_p",
    "vlm_backbone_folder",
    "vlm_backend",
}


@dataclass
class AutoFineTuneConfig:
    """Configuration for the automatic fine-tune + evaluate pipeline."""

    train_config: TrainConfig = field(default_factory=TrainConfig)
    eval_config: MultiEvalConfig = field(default_factory=MultiEvalConfig)

    # Pretrain checkpoint monitoring
    pretrain_checkpoint_folder: str = ""
    pretrain_eval_interval: int = 5000
    eval_name: str = "finetune_eval"
    pretrain_checkpoint_folders: list[str] = field(default_factory=list)

    # Loop control
    sleep_interval: int = 300
    run_once: bool = False
    max_wait_iterations: int = 10

    # Output directory (default: {pretrain_training_folder}/auto_eval/)
    output_dir: str = ""

    # Mode control
    skip_eval: bool = False
    skip_finetune: bool = False

    # Subprocess launch configuration
    train_script: str = ""
    num_processes: int = 2
    mixed_precision: str = "bf16"

    # Wandb integration
    update_wandb_run: bool = False


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def _pre_download_dataset_metadata(train_cfg: TrainConfig) -> None:
    """Pre-download LeRobot dataset metadata before accelerate launch.

    When accelerate launches multiple processes, each one deserializes the config
    and triggers LeRobotDatasetConfig.__post_init__, which downloads metadata
    from HuggingFace Hub. Running these downloads concurrently causes a race
    condition on the HF cache. This function forces a single-process download
    so the cache is warm before any subprocess starts.
    """
    from rho.datasets.lerobot_dataset import LeRobotDatasetConfig, metadata_from_lerobot_dataset
    from rho.datasets.multi_dataset import MultiDatasetConfig

    def _download_for_config(ds_cfg: LeRobotDatasetConfig) -> None:
        if hasattr(ds_cfg, "repo_id") and ds_cfg.repo_id:
            logger.info(f"Pre-downloading metadata for {ds_cfg.repo_id}...")
            metadata_from_lerobot_dataset(
                repo_id=ds_cfg.repo_id,
                root=ds_cfg.root_dir,
                observation_mapping=getattr(ds_cfg, "observation_mapping", None),
            )

    ds = train_cfg.dataset
    if ds is None:
        return
    if isinstance(ds, LeRobotDatasetConfig):
        _download_for_config(ds)
    elif isinstance(ds, MultiDatasetConfig):
        for wds in ds.datasets:
            if isinstance(wds.dataset, LeRobotDatasetConfig):
                _download_for_config(wds.dataset)


def _prepare_finetune_config(
    train_config: TrainConfig,
    pretrained_checkpoint: Path,
    output_dir: Path,
    parent_wandb: WandBConfig | None,
    eval_name: str,
) -> TrainConfig:
    """Build a concrete TrainConfig for a single fine-tuning run.

    Handles resume detection, wandb setup, pretrained policy loading,
    and policy override application.
    """
    step = parse_checkpoint_step(pretrained_checkpoint)
    train_cfg = copy.deepcopy(train_config)
    train_cfg.pretrained_checkpoint = str(pretrained_checkpoint)
    train_cfg.output_dir = str(output_dir)

    # --- Resume detection ---
    partial_ckpt = find_latest_finetuned_checkpoint(output_dir)
    if partial_ckpt is not None:
        logger.info(f"Resuming from partial checkpoint: {partial_ckpt}")
        train_cfg.pretrained_checkpoint = str(partial_ckpt)
        train_cfg.resume = True
        train_cfg.checkpoint_folder = partial_ckpt.parent
    else:
        train_cfg.resume = False
        existing_ckpt_dirs = sorted(output_dir.glob("*/checkpoints"))
        if existing_ckpt_dirs:
            train_cfg.checkpoint_folder = existing_ckpt_dirs[0]
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            train_cfg.checkpoint_folder = output_dir / timestamp / "checkpoints"
        train_cfg.checkpoint_folder.mkdir(parents=True, exist_ok=True)

    # --- Wandb ---
    if parent_wandb is not None:
        ft_suffix = f"_ft_{eval_name}" if eval_name else "_ft"
        train_cfg.wandb.enabled = True
        train_cfg.wandb.project = parent_wandb.project + ft_suffix
        train_cfg.wandb.username = parent_wandb.username
        train_cfg.wandb.id = f"{parent_wandb.id}_{step}pt"
        train_cfg.wandb.group = parent_wandb.group
        train_cfg.wandb.tags = list(parent_wandb.tags or []) + ["auto_finetune"]
        train_cfg.wandb.notes = f"Auto fine-tune from pretrained step {step} (parent run: {parent_wandb.id})"
        train_cfg.wandb.resume = "allow" if partial_ckpt is not None else False

    # --- Load pretrained policy config and apply finetune overrides ---
    ckpt_folder = Path(pretrained_checkpoint).parent
    train_config_json = ckpt_folder.parent / "train_config.json"
    policy_config = load_policy_config_from_json(train_config_json)
    if policy_config is not None:
        overrides = {
            k: getattr(train_cfg.policy, k)
            for k in FINETUNE_POLICY_OVERRIDE_KEYS
            if hasattr(train_cfg.policy, k)
        }
        train_cfg.policy = policy_config
        for key, value in overrides.items():
            if hasattr(train_cfg.policy, key):
                setattr(train_cfg.policy, key, value)
                logger.info(f"  Fine-tune override: policy.{key} = {value}")
        logger.info(f"Loaded pretrained policy config from {train_config_json}")

    # --- Re-derive feature_dict ---
    if train_cfg.policy.feature_dict is None:
        train_cfg.policy.feature_dict = train_cfg.dataset.transformed_feature_dict
    if (
        train_cfg.dataset.chunk_size is None
        and hasattr(train_cfg.policy, "chunk_size")
        and train_cfg.policy.chunk_size is not None
    ):
        train_cfg.dataset.chunk_size = train_cfg.policy.chunk_size

    return train_cfg


def _serialize_config_to_yaml(train_cfg: TrainConfig, output_dir: Path) -> Path:
    """Serialize a TrainConfig to a temporary YAML file for subprocess consumption."""
    import yaml as _yaml

    temp_config_path = output_dir / ".train_config_auto.yaml"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = serialize_to_dict(train_cfg)
    encoded = make_yaml_safe(encoded, strip_none=True)

    with open(temp_config_path, "w") as f:
        _yaml.safe_dump(encoded, f, sort_keys=False)
    logger.info(f"Wrote training config to {temp_config_path}")
    return temp_config_path


def _build_accelerate_command(
    config_path: Path,
    train_script: str,
    num_processes: int,
    mixed_precision: str,
) -> list[str]:
    """Build the ``accelerate launch`` command line."""
    cmd = [sys.executable, "-m", "accelerate.commands.launch"]
    if num_processes > 1:
        cmd.append("--multi_gpu")
    cmd.extend(
        [
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            "0",
            "--mixed_precision",
            mixed_precision,
        ]
    )
    if train_script:
        train_script_path = Path(train_script)
        if not train_script_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            candidate = project_root / train_script_path
            if candidate.exists():
                train_script_path = candidate
        cmd.append(str(train_script_path))
    else:
        cmd.extend(["-m", "rho.training.train_accelerate"])
    cmd.extend(["--config_path", str(config_path)])
    return cmd


def finetune_checkpoint(
    pretrained_checkpoint: Path,
    train_config: TrainConfig,
    output_dir: Path,
    parent_wandb: WandBConfig | None = None,
    eval_name: str = "",
    train_script: str = "",
    num_processes: int = 2,
    mixed_precision: str = "bf16",
) -> Path:
    """Fine-tune from a pretrained checkpoint via ``accelerate launch`` subprocess.

    Returns the path to the fine-tuned checkpoint.
    """
    step = parse_checkpoint_step(pretrained_checkpoint)
    logger.info(f"Fine-tuning from step {step}: {pretrained_checkpoint}")

    train_cfg = _prepare_finetune_config(
        train_config,
        pretrained_checkpoint,
        Path(output_dir),
        parent_wandb,
        eval_name,
    )
    config_path = _serialize_config_to_yaml(train_cfg, Path(output_dir))

    _pre_download_dataset_metadata(train_cfg)

    cmd = _build_accelerate_command(config_path, train_script, num_processes, mixed_precision)

    logger.info(f"Launching: {shlex.join(cmd)}")
    result = subprocess.run(cmd, env=os.environ.copy(), cwd=os.getcwd())
    if result.returncode != 0:
        raise RuntimeError(
            f"Fine-tuning subprocess exited with code {result.returncode}. Config: {config_path}"
        )

    finetuned_ckpt = find_latest_finetuned_checkpoint(Path(output_dir))
    if not finetuned_ckpt:
        raise RuntimeError(f"No checkpoint found in {output_dir} after fine-tuning.")
    logger.info(f"Fine-tuned checkpoint: {finetuned_ckpt}")

    marker = Path(output_dir) / ".finetune_complete"
    marker.write_text(str(finetuned_ckpt))
    return finetuned_ckpt


# ---------------------------------------------------------------------------
# Checkpoint processing
# ---------------------------------------------------------------------------


def _resolve_checkpoint_folder(folder_path: str) -> tuple[Path, Path]:
    """Return ``(checkpoint_folder, training_folder)`` from a path.

    Auto-descends into ``checkpoints/`` if present.
    """
    folder = Path(folder_path)
    if (folder / "checkpoints").is_dir():
        return folder / "checkpoints", folder
    return folder, folder.parent


def _process_single_checkpoint(
    ckpt_path: Path,
    cfg: AutoFineTuneConfig,
    auto_eval_base: Path,
    eval_configs: list,
    parent_wandb: WandBConfig | None,
) -> str:
    """Process one pretrained checkpoint. Returns a status string.

    Possible return values:
      "skipped"     — already complete
      "ft_pending"  — fine-tuning not yet done (eval-only mode)
      "processed"   — fine-tune and/or eval ran
    """
    step = parse_checkpoint_step(ckpt_path)
    if step is None:
        print(f"  Skipping {ckpt_path.name}: could not parse step")
        return "skipped"

    finetune_dir = auto_eval_base / f"{step:07d}" / cfg.eval_name
    eval_dir = finetune_dir / "eval"

    ft_done = is_finetuning_complete(finetune_dir)
    eval_done = cfg.skip_eval or is_evaluation_complete(eval_dir, eval_configs)

    if ft_done and eval_done:
        print(f"  Step {step}: already complete — skipping")
        return "skipped"

    if cfg.skip_finetune and not ft_done:
        print(f"  Step {step}: fine-tuning not yet complete (eval-only mode)")
        return "ft_pending"

    print(f"\n{'#' * 80}")
    print(f"Processing: {ckpt_path.name} (step {step})")
    print(f"  ft_output={finetune_dir}  eval_output={eval_dir}")
    print(f"{'#' * 80}")

    # Fine-tune
    if ft_done or cfg.skip_finetune:
        finetuned_ckpt = find_latest_finetuned_checkpoint(finetune_dir)
        assert finetuned_ckpt is not None
    else:
        finetuned_ckpt = finetune_checkpoint(
            pretrained_checkpoint=ckpt_path,
            train_config=cfg.train_config,
            output_dir=finetune_dir,
            parent_wandb=parent_wandb,
            eval_name=cfg.eval_name,
            train_script=cfg.train_script,
            num_processes=cfg.num_processes,
            mixed_precision=cfg.mixed_precision,
        )

    # Evaluate
    if cfg.skip_eval:
        print(f"  Skipping evaluation for step {step}")
        return "processed"

    success = evaluate_finetuned_checkpoint(
        eval_configs=eval_configs,
        finetuned_checkpoint=finetuned_ckpt,
        eval_output_dir=eval_dir,
        parent_wandb=parent_wandb,
        pretrain_step=step,
        eval_name=cfg.eval_name,
    )
    print(f"\nEvaluations {'PASSED' if success else 'FAILED (some)'} for step {step}")
    return "processed"


def _process_checkpoint_folder(
    checkpoint_folder: Path,
    cfg: AutoFineTuneConfig,
    auto_eval_base: Path,
    eval_configs: list,
    parent_wandb: WandBConfig | None,
) -> tuple[bool, bool]:
    """Process all eligible checkpoints in one folder.

    Returns ``(any_processed, any_ft_pending)``.
    """
    checkpoints = find_checkpoints_to_evaluate(
        checkpoint_folder,
        cfg.pretrain_eval_interval,
        last_evaluated_step=0,
    )
    if not checkpoints:
        print(f"  No checkpoints in {checkpoint_folder}")
        return False, False

    print(f"  Found {len(checkpoints)} checkpoint(s) in {checkpoint_folder}")
    any_processed = False
    any_ft_pending = False

    for ckpt_path in checkpoints:
        status = _process_single_checkpoint(
            ckpt_path,
            cfg,
            auto_eval_base,
            eval_configs,
            parent_wandb,
        )
        if status == "processed":
            any_processed = True
        elif status == "ft_pending":
            any_ft_pending = True

    return any_processed, any_ft_pending


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_folder_specs(cfg: AutoFineTuneConfig) -> list[tuple[Path, Path, Path]]:
    """Resolve config into a list of (ckpt_folder, train_folder, output_base) tuples."""
    raw_folders: list[str] = []
    if cfg.skip_finetune and cfg.pretrain_checkpoint_folders:
        raw_folders = cfg.pretrain_checkpoint_folders
    elif cfg.pretrain_checkpoint_folder:
        raw_folders = [cfg.pretrain_checkpoint_folder]

    specs = []
    for raw in raw_folders:
        ckpt_folder, train_folder = _resolve_checkpoint_folder(raw)
        out_base = Path(os.path.expandvars(cfg.output_dir)) if cfg.output_dir else train_folder / "auto_eval"
        specs.append((ckpt_folder, train_folder, out_base))
    return specs


def _print_banner(
    folder_specs: list[tuple[Path, Path, Path]],
    cfg: AutoFineTuneConfig,
    eval_configs: list,
    parent_wandb: WandBConfig | None,
) -> None:
    """Print a startup summary."""
    print("=" * 80)
    print("Auto Fine-Tune + Evaluate")
    print("=" * 80)
    for i, (ckpt_f, _, out_b) in enumerate(folder_specs):
        print(f"  [{i}] {ckpt_f}  ->  {out_b}")
    mode_parts = []
    if cfg.skip_eval:
        mode_parts.append("FINETUNE ONLY")
    if cfg.skip_finetune:
        mode_parts.append("EVAL ONLY")
    mode_parts.append("single pass" if cfg.run_once else f"continuous (sleep={cfg.sleep_interval}s)")
    print(
        f"  interval={cfg.pretrain_eval_interval}  configs={len(eval_configs)}  mode={', '.join(mode_parts)}"
    )
    if parent_wandb:
        print(f"  wandb: {parent_wandb.project} / {parent_wandb.id}")
    print("=" * 80)


def _run_monitoring_loop(
    folder_specs: list[tuple[Path, Path, Path]],
    cfg: AutoFineTuneConfig,
    eval_configs: list,
    parent_wandb: WandBConfig | None,
) -> None:
    """Continuous loop: scan folders, finetune, evaluate, sleep, repeat."""
    iteration = 0
    wait_iterations = 0

    while True:
        iteration += 1
        print(f"\n[Iteration {iteration}] Scanning for pretrained checkpoints...")

        any_processed = False
        any_ft_pending = False

        for ckpt_folder, _, out_base in folder_specs:
            processed, ft_pending = _process_checkpoint_folder(
                ckpt_folder,
                cfg,
                out_base,
                eval_configs,
                parent_wandb,
            )
            any_processed = any_processed or processed
            any_ft_pending = any_ft_pending or ft_pending

        if cfg.run_once:
            print("\nSingle pass completed. Exiting.")
            return

        if any_processed or any_ft_pending:
            wait_iterations = 0
        else:
            wait_iterations += 1
            if wait_iterations >= cfg.max_wait_iterations:
                print(f"\nMax wait iterations ({cfg.max_wait_iterations}) reached. Exiting.")
                return

        print(f"\nWaiting {cfg.sleep_interval}s...")
        if any_ft_pending:
            print("  Fine-tuning in progress — waiting for completion...")
        elif not any_processed:
            print(f"  Wait iterations: {wait_iterations}/{cfg.max_wait_iterations}")
        time.sleep(cfg.sleep_interval)


@draccus.wrap()
def main(cfg: AutoFineTuneConfig) -> None:
    """Entry point: resolve folders, load wandb, print banner, run loop."""
    folder_specs = _build_folder_specs(cfg)
    if not folder_specs:
        print("[ERROR] No checkpoint folders configured.")
        return

    eval_configs = cfg.eval_config.eval_configs

    parent_wandb: WandBConfig | None = None
    if cfg.update_wandb_run:
        parent_wandb = load_wandb_config_from_train_dir(folder_specs[0][1])
        if parent_wandb is None:
            print("[WARNING] update_wandb_run=True but no parent wandb config found.")

    _print_banner(folder_specs, cfg, eval_configs, parent_wandb)
    _run_monitoring_loop(folder_specs, cfg, eval_configs, parent_wandb)


if __name__ == "__main__":
    main()

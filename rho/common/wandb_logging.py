import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import wandb
import yaml

from rho.common.serialization import serialize_to_dict

logger = logging.getLogger(__name__)

# ruff: noqa: UP007


@dataclass
class WandBConfig:
    project: str = "phi4robotics"
    # Use the correct entity for the Microsoft Research WandB instance
    # Users can override this with --wandb.username=<entity> if needed
    username: str = "msrx-eai"  # Default to the team entity
    tags: list = None
    group: str = None
    notes: str = None
    id: str = None
    enabled: bool = True  # Whether to enable wandb logging
    resume: str | bool = None  # Resume mode: "allow", "must", "never", or None/False

    def __post_init__(self):
        if self.resume:
            self.resume = "allow"


class WandBLogger:
    """A helper class to log metrics and objects using wandb."""

    def __init__(self, cfg: WandBConfig):
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.log_dir = os.getcwd()  # Default to current working directory
        timestamp = datetime.now().strftime("%m%d_%H%M")
        self.job_name = cfg.id if cfg.id else f"default_job_{timestamp}"
        self._group = cfg.group if cfg.group else "default_group"

        if self.enabled:
            # Set up WandB.
            init_kwargs = {
                "project": self.cfg.project,
                "entity": self.cfg.username,
                "name": self.job_name,
                "notes": self.cfg.notes,
                "tags": self.cfg.tags,
                "dir": self.log_dir,
                "group": self._group,
                "id": self.cfg.id,
            }

            # Add resume parameter if specified
            if self.cfg.resume:
                init_kwargs["resume"] = self.cfg.resume

            wandb.init(**init_kwargs)
        else:
            logger.info("WandB logging is disabled")

    def log(self, metrics: dict[str, Any], step: int | None = None, prefix: str = None):
        """Log metrics to WandB with an optional step number."""
        if prefix is not None:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if self.enabled:
            metrics = self._handle_video_metrics(metrics)
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def log_policy(self, policy):
        """Log the policy to WandB."""
        if self.enabled:
            wandb.log({"policy": policy})

    def _handle_video_metrics(self, metrics):
        video_metrics = {k: v for k, v in metrics.items() if "video" in k}
        metrics = {k: v for k, v in metrics.items() if "video" not in k}
        for key, v in video_metrics.items():
            if "video_path" in key:
                video_path = v
                video_fps = video_metrics[key.replace("video_path", "video_fps")]
                wandb_video = wandb.Video(video_path, fps=video_fps, format="mp4")
                metrics[key.replace("video_path", "video")] = wandb_video
        return metrics

    def finish(self):
        """Finish the wandb run."""
        if self.enabled:
            wandb.finish()

    def log_config(self, config: dict | Any, name: str = "config"):
        """Log a config dataclass or dictionary to WandB as a YAML artifact.

        Args:
            config: Dataclass instance or dictionary to log
            name: Name for the artifact (default: "config")
        """
        if not self.enabled:
            return

        try:
            # Convert dataclass to dict if necessary
            config_dict = serialize_to_dict(config) if hasattr(config, "__dataclass_fields__") else config

            # Create a temporary YAML file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                temp_path = f.name

            try:
                # Create WandB artifact
                artifact = wandb.Artifact(
                    name=name, type="config", description=f"Configuration file for {self.job_name}"
                )
                artifact.add_file(temp_path, name=f"{name}.yaml")

                # Log the artifact
                wandb.log_artifact(artifact)

                # Also log as wandb.config for easy access in UI
                wandb.config.update(config_dict)

                logger.info(f"Logged config to WandB as artifact '{name}' and wandb.config")

            finally:
                # Clean up temporary file
                os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Failed to log config to WandB: {e}")


def load_wandb_config_from_train_dir(training_folder) -> WandBConfig | None:
    """Load a WandBConfig from a training run's train_config.json.

    Searches for ``train_config.json`` in the *training_folder* root or in
    the first timestamp sub-folder.

    Args:
        training_folder: Root folder of a training run (parent of ``checkpoints/``).

    Returns:
        A ``WandBConfig`` decoded from the ``wandb`` key, or ``None`` if
        unavailable.
    """
    import json
    from pathlib import Path

    import draccus

    training_folder = Path(training_folder)
    train_config_path = training_folder / "train_config.json"

    if not train_config_path.exists():
        # Also check inside the first timestamp sub-folder
        timestamp_dirs = sorted(
            [d for d in training_folder.iterdir() if d.is_dir() and d.name != "checkpoints"],
        )
        for td in timestamp_dirs:
            candidate = td / "train_config.json"
            if candidate.exists():
                train_config_path = candidate
                break

    if not train_config_path.exists():
        logger.warning(f"No train_config.json found under {training_folder}")
        return None

    with open(train_config_path) as f:
        train_config_dict = json.load(f)

    if "wandb" not in train_config_dict:
        logger.warning("No 'wandb' key in parent train_config.json")
        return None

    wandb_config = draccus.decode(WandBConfig, train_config_dict["wandb"])
    logger.info(
        f"WandB config loaded from {train_config_path}: project={wandb_config.project}, id={wandb_config.id}"
    )
    return wandb_config

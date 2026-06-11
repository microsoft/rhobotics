#!/usr/bin/env python3
"""
Evaluation script for pretrained checkpoints using EnvironmentWrapper

This script loads a checkpoint created by the accelerate-based training script
and evaluates the policy in the cfg.environment using the EnvironmentWrapper
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch

from rho.common.wandb_logging import WandBConfig
from rho.datasets.data_config import DataConfig
from rho.environment.env import EnvironmentConfig
from rho.eval.policy_interface import PolicyInterfaceConfig
from rho.policies import PolicyConfig
from rho.policies.dsrl.dsrl_config import DSRLConfig
from rho.policies.dsrl.flowdagger_config import FlowDAggerConfig

logger = logging.getLogger(__name__)


def find_dataset_by_root_dir(dataset_dict: dict, target_root_dir: str) -> dict | None:
    """
    Search for a dataset config with the matching root_dir.
    Handles MultiDatasetConfig structures by checking dataset_cfgs (flattened list)
    or falling back to recursive search through datasets.

    Args:
        dataset_dict: The dataset dictionary to search
        target_root_dir: The root_dir to match

    Returns:
        The matching dataset dict, or None if not found
    """
    # Check if this is a leaf dataset with matching root_dir
    if "root_dir" in dataset_dict and dataset_dict.get("root_dir") == target_root_dir:
        return dataset_dict

    # If flatten_nested is True (default), check dataset_cfgs for the flattened list
    if dataset_dict.get("flatten_nested", True) and "dataset_cfgs" in dataset_dict:
        for cfg in dataset_dict["dataset_cfgs"]:
            if cfg.get("root_dir") == target_root_dir:
                return cfg

    # Fallback: recursively search through nested datasets structure
    if "datasets" in dataset_dict:
        for weighted_dataset in dataset_dict["datasets"]:
            if "dataset" in weighted_dataset:
                result = find_dataset_by_root_dir(weighted_dataset["dataset"], target_root_dir)
                if result is not None:
                    return result

    return None


def _normalize_policy_dict(policy_dict: dict) -> dict:
    """Apply backward-compat fixups to a serialized PolicyConfig dict.

    * Renames ``lr_scheduler.name`` → ``lr_scheduler.type``
    * Falls back to ``name`` when ``type`` is missing at the top level

    Args:
        policy_dict: Raw dict from ``train_config.json["policy"]``.

    Returns:
        The same dict, mutated in-place, ready for ``draccus.decode``.

    Raises:
        ValueError: If neither ``type`` nor ``name`` is present.
    """
    if "lr_scheduler" in policy_dict and "name" in policy_dict["lr_scheduler"]:
        policy_dict["lr_scheduler"]["type"] = policy_dict["lr_scheduler"].pop("name")
    if "type" not in policy_dict and "name" in policy_dict:
        policy_dict["type"] = policy_dict["name"]
    if "type" not in policy_dict:
        raise ValueError("PolicyConfig dict is missing both 'type' and 'name' fields")
    return policy_dict


def _normalize_dataset_dict(
    dataset_dict: dict,
    dataset_root_dir: str | None = None,
) -> dict:
    """Apply backward-compat fixups to a serialized DataConfig dict.

    Handles:
    * Multidataset selection via *dataset_root_dir*
    * Old nested ``features`` format migration
    * ``action_type`` uppercasing (enum compat)
    * ``transform_mapping`` action_type uppercasing
    * Inferring ``action_type`` from transform_mapping when missing
    * Inferring ``type='lerobot'`` from ``repo_id``
    * Stripping ``None``-valued keys

    Args:
        dataset_dict: Raw dict from ``train_config.json["dataset"]``.
        dataset_root_dir: When the training used multiple datasets, selects which
            sub-dataset to use.  Required when the dict contains ``datasets`` or
            ``dataset_cfgs``.

    Returns:
        A cleaned dict ready for ``draccus.decode(DataConfig, ...)``.

    Raises:
        AssertionError: If multidataset is detected but *dataset_root_dir* is ``None``.
        ValueError: If no dataset matches *dataset_root_dir*.
    """
    # --- multidataset selection ---
    if "datasets" in dataset_dict or "dataset_cfgs" in dataset_dict:
        assert dataset_root_dir is not None, (
            "Multiple datasets found in training config but dataset_root_dir not provided "
            "in EvalConfig to select which dataset to use."
        )
        matched_dataset = find_dataset_by_root_dir(dataset_dict, dataset_root_dir)
        if matched_dataset is None:
            raise ValueError(f"No dataset found in training config with root_dir: {dataset_root_dir}")
        dataset_dict = matched_dataset

    # --- old nested-features migration ---
    if "features" in dataset_dict and isinstance(dataset_dict["features"], dict):
        features_dict = dataset_dict["features"]
        if "normalization_mapping" in features_dict:
            logger.info("Detected old train_config.json format with nested features, migrating...")
            dataset_dict["normalization_mapping"] = features_dict.get("normalization_mapping")
            dataset_dict["features"] = features_dict.get("features")
            dataset_dict["stats"] = features_dict.get("stats")
            if "clip_values" in features_dict:
                dataset_dict["clip_values"] = features_dict["clip_values"]
            if "chunk_size" in features_dict:
                dataset_dict["chunk_size"] = features_dict["chunk_size"]

    # --- strip None values ---
    dataset_dict = {k: v for k, v in dataset_dict.items() if v is not None}

    # --- action_type uppercasing ---
    if "action_type" in dataset_dict and isinstance(dataset_dict["action_type"], str):
        old_val = dataset_dict["action_type"]
        dataset_dict["action_type"] = old_val.upper()
        if old_val != dataset_dict["action_type"]:
            logger.info(f"Backward compat: dataset.action_type '{old_val}' ->{dataset_dict['action_type']}'")

    if "transform_mapping" in dataset_dict and isinstance(dataset_dict["transform_mapping"], dict):
        for _tm_key, _tm_val in dataset_dict["transform_mapping"].items():
            transforms = _tm_val if isinstance(_tm_val, list) else [_tm_val]
            for t in transforms:
                if isinstance(t, dict) and "action_type" in t:
                    old_val = t["action_type"]
                    t["action_type"] = t["action_type"].upper()
                    if old_val != t["action_type"]:
                        logger.info(
                            f"Backward compat: transform_mapping['{_tm_key}'].action_type "
                            f"'{old_val}' -> '{t['action_type']}'"
                        )

        # Infer action_type from transform_mapping when missing at the dataset level
        if "action_type" not in dataset_dict:
            for _tm_key, _tm_val in dataset_dict["transform_mapping"].items():
                transforms = _tm_val if isinstance(_tm_val, list) else [_tm_val]
                for t in transforms:
                    if isinstance(t, dict) and "action_type" in t:
                        t_type = t.get("type", "")
                        if t_type in ("delta_actions", "convert_to_6d_actions"):
                            dataset_dict["action_type"] = t["action_type"]
                            logger.info(
                                f"Backward compat: inferred dataset.action_type="
                                f"'{t['action_type']}' from transform_mapping['{_tm_key}'] ({t_type})"
                            )
                            break
                if "action_type" in dataset_dict:
                    break

    # --- infer type from repo_id or root_dir ---
    if "type" not in dataset_dict and ("repo_id" in dataset_dict or "root_dir" in dataset_dict):
        dataset_dict["type"] = "lerobot"
        logger.info("Inferred dataset type='lerobot' from presence of repo_id/root_dir field")

    return dataset_dict


def load_policy_config_from_json(train_config_path):
    """Load *only* the PolicyConfig from a ``train_config.json`` file.

    Unlike :func:`load_configs_from_train_config`, this skips dataset
    normalisation entirely, which avoids the ``dataset_root_dir`` assertion
    when the pretrained model was trained on multiple datasets.

    Args:
        train_config_path: Path to ``train_config.json``.

    Returns:
        A decoded ``PolicyConfig``, or ``None`` if the file is missing or
        has no ``policy`` key.
    """
    train_config_path = Path(train_config_path)
    if not train_config_path.exists():
        logger.warning(f"Training config not found at: {train_config_path}")
        return None

    with open(train_config_path) as f:
        train_config_dict = json.load(f)

    if "policy" not in train_config_dict:
        logger.warning("No 'policy' key found in train_config.json")
        return None

    policy_dict = _normalize_policy_dict(train_config_dict["policy"])
    policy_config = draccus.decode(PolicyConfig, policy_dict)
    logger.info(f"PolicyConfig loaded from {train_config_path}: {policy_config.name}")
    return policy_config


def load_configs_from_train_config(
    train_config_path: str | Path,
    *,
    dataset_root_dir: str | None = None,
) -> tuple[PolicyConfig | None, DataConfig | None]:
    """Parse a ``train_config.json`` and return decoded policy/dataset configs.

    This is the single authoritative place that knows how to read a training
    config, apply backward-compatibility migrations, and produce the
    ``PolicyConfig`` and ``DataConfig`` objects needed at eval time.

    Args:
        train_config_path: Path to ``train_config.json``.
        dataset_root_dir: When the training used multiple datasets, selects
            which sub-dataset to use.

    Returns:
        A ``(policy_config, data_config)`` tuple.  Either element may be
        ``None`` if the corresponding key was absent in the JSON.
    """
    train_config_path = Path(train_config_path)
    if not train_config_path.exists():
        logger.warning(f"Training config not found at: {train_config_path}")
        return None, None

    with open(train_config_path) as f:
        train_config_dict = json.load(f)

    # --- PolicyConfig ---
    policy_config = None
    if "policy" in train_config_dict:
        policy_dict = _normalize_policy_dict(train_config_dict["policy"])
        policy_config = draccus.decode(PolicyConfig, policy_dict)
        logger.info(f"PolicyConfig loaded from {train_config_path}: {policy_config.name}")
    else:
        logger.warning("No 'policy' key found in train_config.json")

    # --- DataConfig ---
    data_config = None
    if "dataset" in train_config_dict:
        dataset_dict = _normalize_dataset_dict(
            train_config_dict["dataset"],
            dataset_root_dir=dataset_root_dir,
        )
        try:
            data_config = draccus.decode(DataConfig, dataset_dict)
            logger.info(f"DataConfig loaded from {train_config_path}: {type(data_config).__name__}")
        except Exception as e:
            logger.warning(f"Failed to decode DataConfig from {train_config_path}: {e}")
    else:
        logger.warning("No 'dataset' key found in train_config.json")

    return policy_config, data_config


@dataclass
class EvalConfig:
    # Core evaluation settings
    pretrained_checkpoint: str | None = None
    name: str = "eval"  # Name for this evaluation config (used as metric prefix in wandb)

    # Environment settings
    # should be defined in subclasses
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # TODO get feedback: make dataset and policy configs optional as they can be populated from
    # the train_config.json
    # Dataset and policy configs (shared with training)
    dataset: DataConfig = field(default_factory=DataConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    policy_interface_cfg: PolicyInterfaceConfig = field(default_factory=PolicyInterfaceConfig)
    seed: int = 12345  # Random seed for reproducibility

    eval_mode: str = "standard"
    inference_delay: int = 6  # number of action steps it takes to inference
    beta: int = 10  # weighting of rtc update vs flow matching update

    # Key to select which dataset transforms to use in event of training
    # with multidataset
    dataset_root_dir: str | None = None

    # Logging configuration
    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR

    # ── HIL trainer launch ──────────────────────────────────────────────────
    # When ``train`` is True, serve_policy.eval() also spawns the trainer
    # named by ``trainer_type`` on a background thread before starting the
    # websocket server. The trainer listens for experience transitions
    # streamed from the robot's hil.data_publisher.ExperiencePublisher on
    # ``experience_port``. Defaults are no-trainer behavior so existing
    # serve configs work unchanged.
    train: bool = False
    trainer_type: str = "debug"  # "debug", "dsrl", or "flowdagger"
    experience_port: int = 5555
    # Trainer-specific configs. Real dataclass types so draccus can decode
    # yaml blocks like ``flowdagger: {bc_lr: 1e-4, ...}``. Both modules are
    # pure-dataclass (no torch imports) so importing them at module load
    # time is cheap.
    dsrl: DSRLConfig | None = None
    flowdagger: FlowDAggerConfig | None = None

    def __post_init__(self):
        # Expand environment variables in dataset_root_dir (e.g. $DATA_DIR/...)
        # so YAML configs can use env vars just like LerobotDatasetConfig.root_dir.
        if self.dataset_root_dir is not None:
            self.dataset_root_dir = os.path.expandvars(self.dataset_root_dir)

        # When pretrained_checkpoint is None, this config is being used as a
        # template (e.g. inside a MultiEvalConfig).  Skip all validation and
        # checkpoint-dependent initialisation; the real config will be
        # constructed later with the actual checkpoint path.
        if self.pretrained_checkpoint is None:
            return

        # 0. Check is user provided a folder or a .pt file
        self.pretrained_checkpoint = Path(self.pretrained_checkpoint)
        if self.pretrained_checkpoint.is_dir():
            self.pretrained_checkpoint = self.pretrained_checkpoint / "checkpoint_latest.pt"

        # 1. Check we can find the pretrained_checkpoint
        if not self.pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.pretrained_checkpoint}")
        else:
            self.checkpoint_folder = Path(self.pretrained_checkpoint).parent

        # Load policy and dataset configs from train_config.json
        train_config_path = Path(self.checkpoint_folder).parent / "train_config.json"
        policy_config, data_config = load_configs_from_train_config(
            train_config_path,
            dataset_root_dir=self.dataset_root_dir,
        )
        if policy_config is not None:
            self.policy = policy_config
        if data_config is not None:
            self.dataset = data_config

        if not isinstance(self.policy, PolicyConfig):
            raise ValueError("Policy must be a PolicyConfig instance or a dict")

        if self.policy.feature_dict is None:
            self.policy.feature_dict = self.dataset.transformed_feature_dict

        # pass chunk size through, since EvalConfig does not initialize datasets fully
        if hasattr(self.policy, "chunk_size") and self.dataset.features is not None:
            self.dataset.chunk_size = self.policy.chunk_size

        # 2. Setup PolicyInterfaceConfig
        # pass in data_config
        self.policy_interface_cfg.data_config = self.dataset
        self.policy_interface_cfg.eval_mode = self.eval_mode
        self.policy_interface_cfg.inference_delay = self.inference_delay
        self.policy_interface_cfg.beta = self.beta


@dataclass
class SimEvalConfig(EvalConfig):
    """Evaluation config for simulation environments.

    Inherits from EvalConfig and can be extended with sim-specific settings.
    """

    eval_num_episodes: int = 5
    record_videos: bool = True
    output_dir: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    update_wandb_run: bool = False  # Whether to post updates to wandb run from training
    wandb_config: WandBConfig | None = None  # Loaded from train_config.json if available

    def __post_init__(self):
        super().__post_init__()  # Call the parent post init to load checkpoint and configs

        # If parent returned early (template mode), skip checkpoint-dependent init
        if self.pretrained_checkpoint is None:
            return

        train_config_path = Path(self.checkpoint_folder).parent / "train_config.json"

        if train_config_path.exists():
            with open(train_config_path) as f:
                train_config_dict = json.load(f)

            #### Load WandB config from train_config.json ####
            if "wandb" in train_config_dict and self.update_wandb_run:
                self.wandb_config = draccus.decode(WandBConfig, train_config_dict["wandb"])
                self.wandb_config.resume = "allow"  # Enable resume mode for eval
                logger.info(f"WandB config loaded from {train_config_path}")
                logger.info(f"  WandB run ID: {self.wandb_config.id}")
            else:
                logger.warning("No 'wandb' key found in train_config.json")
            #### Load WandB config from train_config.json ####


@dataclass
class MultiEvalConfig:
    """
    Configuration for running multiple evaluations with different settings.
    Each eval_configs entry is a SimEvalConfig (or EvalConfig) instance.
    """

    eval_configs: list[SimEvalConfig] = field(default_factory=list)
    pretrained_checkpoint: str | None = None  # Optional shared checkpoint for all evals

    def __post_init__(self):
        # Allow empty list when used as a default / template inside another
        # dataclass (e.g. AutoFineTuneConfig).  Validation is deferred to the
        # caller.
        pass

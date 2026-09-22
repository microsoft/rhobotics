#!/usr/bin/env python3
"""
Evaluation script for pretrained checkpoints using EnvironmentWrapper

This script loads a checkpoint created by the accelerate-based training script
and evaluates the policy in the cfg.environment using the EnvironmentWrapper
"""

import json
import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

import draccus
import torch
from draccus import ChoiceRegistry

from rho.checkpoints import (
    is_checkpoint_bundle,
    load_bundle_metadata,
    resolve_checkpoint,
    resolve_latest_checkpoint,
)
from rho.common.wandb_logging import WandBConfig
from rho.datasets.data_config import DataConfig
from rho.environment.env import EnvironmentConfig
from rho.eval.policy_interface import PolicyInterfaceConfig
from rho.models.schedule import migrate_legacy_scheduler_config
from rho.policies import PolicyConfig

logger = logging.getLogger(__name__)


def _resolve_choice_config_class(config_cls: type, config_dict: dict) -> type:
    """Resolve a draccus ChoiceRegistry base class to the concrete dataclass."""
    if not issubclass(config_cls, ChoiceRegistry):
        return config_cls

    choice_name = config_dict.get("type") or config_dict.get("name") or config_cls.default_choice_name()
    if choice_name is None:
        return config_cls

    try:
        return config_cls.get_choice_class(choice_name)
    except KeyError:
        return config_cls


def _config_class_from_annotation(annotation) -> type | None:
    """Return a dataclass/ChoiceRegistry class from a field annotation if one is obvious."""
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            config_cls = _config_class_from_annotation(arg)
            if config_cls is not None:
                return config_cls
        return None

    if isinstance(annotation, type) and (is_dataclass(annotation) or issubclass(annotation, ChoiceRegistry)):
        return annotation

    return None


def _strip_unknown_config_fields(config_cls: type, config_dict: dict, context: str) -> dict:
    """Drop stale fields from serialized configs before draccus decodes them.

    Checkpoint ``train_config.json`` files can outlive code fields.  Draccus is
    intentionally strict, so eval-time checkpoint loading removes fields that
    are not present in the current dataclass and logs exactly what was ignored.
    """
    if not isinstance(config_dict, dict):
        return config_dict

    concrete_cls = _resolve_choice_config_class(config_cls, config_dict)
    if not is_dataclass(concrete_cls):
        return config_dict

    config_fields = {f.name: f for f in fields(concrete_cls)}
    passthrough_fields = {"type"}
    unknown_fields = sorted(k for k in config_dict if k not in config_fields and k not in passthrough_fields)

    if unknown_fields:
        formatted_fields = ", ".join(f"`{name}`" for name in unknown_fields)
        logger.warning(
            f"Ignoring unsupported fields from train_config.json for {context} "
            f"({concrete_cls.__name__}): {formatted_fields}"
        )

    cleaned = {}
    for key, value in config_dict.items():
        if key in passthrough_fields:
            cleaned[key] = value
            continue
        if key not in config_fields:
            continue

        nested_cls = _config_class_from_annotation(config_fields[key].type)
        if nested_cls is not None and isinstance(value, dict):
            cleaned[key] = _strip_unknown_config_fields(nested_cls, value, f"{context}.{key}")
        else:
            cleaned[key] = value

    return cleaned


def _root_dir_matches(candidate: str, target: str) -> bool:
    """Check if two root_dir paths refer to the same dataset.

    Uses a tiered matching strategy:
    1. Exact match
    2. Last 4 path components match (more precise than last 2)
    3. Last 2 path components match (fallback for mount point differences)
    Also strips version suffixes (``_v5``, ``_v6``, etc.) to handle dataset version
    differences between training and evaluation.
    """
    if candidate == target:
        return True

    # Try matching with more path components first (last 4) before falling back to 2
    c_parts = Path(candidate).parts
    t_parts = Path(target).parts

    # Try last 4 components first (more precise)
    n = min(4, len(c_parts), len(t_parts))
    if n == 4 and c_parts[-n:] == t_parts[-n:]:
        return True

    # Match on the last 2 components (parent dir + dataset name)
    n = min(2, len(c_parts), len(t_parts))
    if c_parts[-n:] == t_parts[-n:]:
        return True

    # Also try matching after stripping version suffixes (_v5, _v6, etc.)
    import re

    c_stripped = tuple(re.sub(r"_v\d+$", "", p) for p in c_parts[-n:])
    t_stripped = tuple(re.sub(r"_v\d+$", "", p) for p in t_parts[-n:])
    return c_stripped == t_stripped


def find_dataset_by_root_dir(dataset_dict: dict, target_root_dir: str) -> dict | None:
    """
    Search for a dataset config with the matching root_dir.
    Handles MultiDatasetConfig structures by checking dataset_cfgs (flattened list)
    or falling back to recursive search through datasets.

    Matching uses the last two path components to handle different mount prefixes
    between training and evaluation environments.

    Args:
        dataset_dict: The dataset dictionary to search
        target_root_dir: The root_dir to match

    Returns:
        The matching dataset dict, or None if not found
    """
    # Collect all available datasets for debugging
    available_datasets = []

    def collect_datasets(d: dict):
        """Recursively collect all datasets with root_dirs for logging."""
        if "root_dir" in d:
            available_datasets.append(d.get("root_dir", "unknown"))
        if d.get("flatten_nested", True) and "dataset_cfgs" in d:
            for cfg in d["dataset_cfgs"]:
                if "root_dir" in cfg:
                    available_datasets.append(cfg.get("root_dir", "unknown"))
        if "datasets" in d:
            for weighted_dataset in d["datasets"]:
                if "dataset" in weighted_dataset:
                    collect_datasets(weighted_dataset["dataset"])

    collect_datasets(dataset_dict)
    if available_datasets:
        logger.info(f"Available datasets in training config: {available_datasets}")

    # Check if this is a leaf dataset with matching root_dir
    if "root_dir" in dataset_dict and _root_dir_matches(dataset_dict["root_dir"], target_root_dir):
        logger.info(f"Matched dataset with root_dir: {dataset_dict['root_dir']}")
        logger.info(f"  Has transform_mapping: {'transform_mapping' in dataset_dict}")
        return dataset_dict

    # If flatten_nested is True (default), check dataset_cfgs for the flattened list
    if dataset_dict.get("flatten_nested", True) and "dataset_cfgs" in dataset_dict:
        for cfg in dataset_dict["dataset_cfgs"]:
            if _root_dir_matches(cfg.get("root_dir", ""), target_root_dir):
                logger.info(f"Matched dataset with root_dir: {cfg.get('root_dir', 'unknown')}")
                logger.info(f"  Has transform_mapping: {'transform_mapping' in cfg}")
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

    * Migrates legacy ``lr_scheduler.name`` discriminator/algorithm fields
    * Falls back to ``name`` when ``type`` is missing at the top level

    Args:
        policy_dict: Raw dict from ``train_config.json["policy"]``.

    Returns:
        The same dict, mutated in-place, ready for ``draccus.decode``.

    Raises:
        ValueError: If neither ``type`` nor ``name`` is present.
    """
    if "lr_scheduler" in policy_dict:
        policy_dict["lr_scheduler"] = migrate_legacy_scheduler_config(policy_dict["lr_scheduler"])
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
        if dataset_root_dir is None:
            logger.warning(
                "Multiple datasets found in training config but dataset_root_dir not provided. "
                "Skipping dataset loading — per-task configs should provide dataset_root_dir."
            )
            return None
        matched_dataset = find_dataset_by_root_dir(dataset_dict, dataset_root_dir)
        if matched_dataset is None:
            logger.warning(
                f"No dataset found in training config with root_dir: {dataset_root_dir}. "
                "Skipping dataset loading — eval config will provide its own dataset."
            )
            return None
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
    policy_dict = _strip_unknown_config_fields(PolicyConfig, policy_dict, "policy")
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
        policy_dict = _strip_unknown_config_fields(PolicyConfig, policy_dict, "policy")
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
        if dataset_dict is None:
            # Multi-dataset without selection — skip dataset loading
            pass
        else:
            try:
                dataset_dict = _strip_unknown_config_fields(DataConfig, dataset_dict, "dataset")
                data_config = draccus.decode(DataConfig, dataset_dict)
                logger.info(f"DataConfig loaded from {train_config_path}: {type(data_config).__name__}")
            except Exception as e:
                logger.warning(f"Failed to decode DataConfig from {train_config_path}: {e}")
    else:
        logger.warning("No 'dataset' key found in train_config.json")

    return policy_config, data_config


def load_configs_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_root_dir: str | None = None,
) -> tuple[PolicyConfig | None, DataConfig | None]:
    """Load portable bundle metadata or fall back to a legacy training config."""
    checkpoint_path = Path(checkpoint_path)
    if not is_checkpoint_bundle(checkpoint_path):
        train_config_path = checkpoint_path.parent.parent / "train_config.json"
        return load_configs_from_train_config(
            train_config_path,
            dataset_root_dir=dataset_root_dir,
        )

    metadata = load_bundle_metadata(checkpoint_path)

    policy_config = None
    policy_dict = metadata["policy"]
    if policy_dict is not None:
        if metadata["features"] is not None:
            policy_dict["feature_dict"] = metadata["features"]
        policy_dict = _normalize_policy_dict(policy_dict)
        policy_dict = _strip_unknown_config_fields(PolicyConfig, policy_dict, "policy")
        policy_config = draccus.decode(PolicyConfig, policy_dict)

    data_config = None
    dataset_dict = metadata["data_config"]
    if dataset_dict is not None:
        if metadata["stats"] is not None:
            dataset_dict["stats"] = metadata["stats"]
        dataset_dict = _normalize_dataset_dict(dataset_dict, dataset_root_dir=dataset_root_dir)
        if dataset_dict is not None:
            dataset_dict = _strip_unknown_config_fields(DataConfig, dataset_dict, "dataset")
            data_config = draccus.decode(DataConfig, dataset_dict)

    return policy_config, data_config


@dataclass
class EvalConfig:
    # Core evaluation settings
    pretrained_checkpoint: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_cache_dir: str | None = None
    name: str = "eval"  # Name for this evaluation config (used as metric prefix in wandb)

    # Environment settings
    # should be defined in subclasses
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Dataset and policy configs may be replaced with checkpoint metadata
    # during initialization.
    dataset: DataConfig = field(default_factory=DataConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    policy_interface_cfg: PolicyInterfaceConfig = field(default_factory=PolicyInterfaceConfig)
    seed: int = 12345  # Base seed for environment and scenario resets
    policy_seed: int | None = None  # Independent policy RNG seed; defaults to seed

    eval_mode: str = "standard"
    inference_delay: int = 6  # number of action steps it takes to inference
    execution_horizon: int | None = None  # actions executed between RTC inferences
    beta: int = 10  # weighting of rtc update vs flow matching update
    guidance_schedule: str = "paper"  # guidance coefficient schedule: 'paper' or 'constant'

    # Key to select which dataset transforms to use in event of training
    # with multidataset
    dataset_root_dir: str | None = None

    # Logging configuration
    log_level: str = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR

    # Evaluation parameters (used by sim eval, safe defaults for other usage)
    eval_num_episodes: int = 5
    record_videos: bool = True
    output_dir: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Expand environment variables in dataset_root_dir (e.g. $DATA_DIR/...)
        # so YAML configs can use env vars just like LerobotDatasetConfig.root_dir.
        if self.dataset_root_dir is not None:
            self.dataset_root_dir = os.path.expandvars(self.dataset_root_dir)

        # When pretrained_checkpoint is None, this config is being used as a
        # template (e.g. inside a MultiEvalConfig).  Skip all validation and
        # checkpoint-dependent initialisation; the real config will be
        # constructed later with the actual checkpoint path.
        if self.pretrained_checkpoint is None or self.pretrained_checkpoint == "None":
            self.pretrained_checkpoint = None
            return

        # Resolve a specific bundle, a checkpoint root, or a legacy .pt file.
        self.pretrained_checkpoint = resolve_checkpoint(
            self.pretrained_checkpoint,
            revision=self.checkpoint_revision,
            cache_dir=self.checkpoint_cache_dir,
            include_training_state=False,
        )
        if self.pretrained_checkpoint.is_dir():
            resolved_checkpoint = resolve_latest_checkpoint(self.pretrained_checkpoint)
            if resolved_checkpoint is None:
                raise FileNotFoundError(
                    f"No completed checkpoint found in directory: {self.pretrained_checkpoint}"
                )
            self.pretrained_checkpoint = resolved_checkpoint

        # 1. Check we can find the pretrained_checkpoint
        if not self.pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.pretrained_checkpoint}")
        else:
            self.checkpoint_folder = Path(self.pretrained_checkpoint).parent

        policy_config, data_config = load_configs_from_checkpoint(
            self.pretrained_checkpoint,
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
        self.policy_interface_cfg.execution_horizon = self.execution_horizon
        self.policy_interface_cfg.beta = self.beta
        self.policy_interface_cfg.guidance_schedule = self.guidance_schedule


@dataclass
class SimEvalConfig(EvalConfig):
    """Evaluation config for simulation environments.

    Inherits from EvalConfig and can be extended with sim-specific settings.

    When ``eval_configs`` is populated (via a MultiEvalConfig-style YAML),
    this acts as a multi-eval runner: the policy is loaded once from
    ``pretrained_checkpoint`` and reused across all sub-configs.
    """

    update_wandb_run: bool = False  # Whether to post updates to wandb run from training
    wandb_config: WandBConfig | None = None  # Loaded from train_config.json if available

    # Multi-eval support: when populated, each entry defines a separate
    # evaluation environment.  The top-level pretrained_checkpoint is shared.
    eval_configs: list[EvalConfig] = field(default_factory=list)

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

    @property
    def is_multi_eval(self) -> bool:
        """Return True if this config defines multiple evaluations."""
        return len(self.eval_configs) > 0


@dataclass
class MultiEvalConfig:
    """
    Configuration for running multiple evaluations with different settings.
    Each eval_configs entry is a SimEvalConfig (or EvalConfig) instance.

    Note: This class is retained for backward compatibility with auto_eval.py
    and auto_finetune pipelines.  For direct CLI usage, prefer SimEvalConfig
    with ``eval_configs`` populated.
    """

    eval_configs: list[SimEvalConfig] = field(default_factory=list)
    pretrained_checkpoint: str | None = None  # Optional shared checkpoint for all evals

    def __post_init__(self):
        # Allow empty list when used as a default / template inside another
        # dataclass (e.g. AutoFineTuneConfig).  Validation is deferred to the
        # caller.
        pass

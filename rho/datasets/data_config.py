import logging
import os  # noqa: I001
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from draccus import ChoiceRegistry
from torchvision.transforms import Compose

from rho.common.constants import ACTION, OBSERVATION_ENVIRONMENT_STATE, OBSERVATION_IMAGE, OBSERVATION_STATE
from rho.common.normalize import Unnormalize
from rho.common.serialization import fixup_feature_shapes, serialize_to_dict
from rho.common.transforms import (
    AbsoluteActions,
    ConvertFrom6dActions,
    ConvertTo6dActions,
    DeltaActions,
    Transform,
)
from rho.common.types import ActionType, FeatureType, NormalizationMode, PolicyFeature, TrainingMode

logger = logging.getLogger(__name__)

NEXT_REWARD_KEY = "next.reward"


def extract_transform_config(
    transform,
    transform_class: type,
    type_name: str,
) -> dict | None:
    """Extract configuration from a transform (either instantiated object or dictionary).

    Automatically extracts all public attributes from the transform class or dict keys
    (excluding 'type' for dicts).

    Args:
        transform: Either an instantiated transform object or a dictionary configuration.
        transform_class: The class type to check against (e.g., CombineKeys, ConvertTo6dActions).
        type_name: The string type name used in dictionary configurations (e.g., "combine_keys").

    Returns:
        A dictionary with the extracted configuration if the transform matches, otherwise None.

    Example:
        config = extract_transform_config(
            transform,
            transform_class=CombineKeys,
            type_name="combine_keys",
        )
    """
    if isinstance(transform, transform_class):
        # Extract all public attributes (non-callable, non-dunder)
        return {
            key: getattr(transform, key)
            for key in dir(transform)
            if not key.startswith("_") and not callable(getattr(transform, key))
        }
    elif isinstance(transform, dict) and transform.get("type") == type_name:
        # Extract all keys except 'type'
        return {key: value for key, value in transform.items() if key != "type"}
    return None


class TransformWrapper:
    def __init__(self, key: str, transform: Callable | list):
        self.key = key
        if isinstance(transform, list):
            self.transform = Compose(transform)
        else:
            self.transform = transform

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.key in sample:
            if isinstance(self.transform, list):
                for transform in self.transform:
                    sample[self.key] = transform(sample[self.key])
            else:
                sample[self.key] = self.transform(sample[self.key])
        return sample


def append_task(sample: dict[str, torch.Tensor], instruction: str) -> dict[str, torch.Tensor]:
    """Append a language instruction to the sample if not already present.

    Args:
        sample (dict[str, torch.Tensor]): The data sample.
        instruction (str): The language instruction to append.

    Returns:
        dict[str, torch.Tensor]: The updated data sample.
    """
    if "task" not in sample and instruction is not None:
        sample["task"] = [instruction]
    return sample


def convert_dict_list_to_array(d: dict):
    """Convert dictionary values that are lists to numpy arrays."""
    d_new = {}
    for k, v in d.items():
        if isinstance(v, list):
            d_new[k] = np.array(v)
        elif isinstance(v, dict):
            d_new[k] = convert_dict_list_to_array(v)
        else:
            d_new[k] = v
    return d_new


@dataclass
class DataConfig(ChoiceRegistry):
    """Configuration for data processing from raw data -> model input."""

    @classmethod
    def default_choice_name(cls) -> str:
        """Return 'base' as default when no 'type' key is provided."""
        return "base"

    @property
    def type(self) -> str:
        """Return the choice registry type name for this config."""
        return self.get_choice_name(self.__class__)

    batch_size: int = 64
    num_workers: int = 4
    shuffle_buffer_size: int = 1000  # Buffer size for shuffling in streaming mode

    # Feature configuration fields (previously in FeatureConfig)
    normalization_mapping: dict[str, str | NormalizationMode] | None = None
    features: dict[str, dict | PolicyFeature] | None = None  # Dictionary of features to be used
    stats: str | dict | None = None  # Statistics for normalization
    clip_values: dict[str, tuple[float, float]] | None = None  # Clipping values for normalization
    chunk_size: int | None = None  # Chunk size for action chunk normalization

    # Optional mapping for observation keys
    observation_mapping: dict[str, str] = None
    # Optional whitelist for observations to include in the samples
    # The whitelist uses the renamed keys

    observation_whitelist: list[str] | None = None

    # Optional blacklist for observations to prevent huggingface from loading
    observation_blacklist: list[str] | None = None

    # Mapping of feature keys to transformations
    transform_mapping: dict[str, list[Transform]] = None

    language_instruction: str | None = None  # Default language instruction to use if not in dataset

    training_mode: TrainingMode = TrainingMode.ROBOT_FLOWMATCH

    action_type: ActionType = ActionType.POSITION  # type of action representation used in the dataset

    # This flag is useful when the stats file is too big to save.
    # Particularly in cases where we are using a MultiDatasetConfig
    serialize_stats: bool = True  # Whether to serialize stats when converting to dict

    video_decoder_cache_size: int | None = None  # Max size for video decoder cache, or None for unlimited

    def __post_init__(self):
        import ast
        import json

        import numpy as np

        if self.action_type == ActionType.EE_EULER_POS:
            import warnings

            warnings.warn(
                "EE_EULER_POS action type is deprecated and will be removed in a future version. "
                "Use EE_6D_POS for gimbal-lock-free rotation representation. "
                "See: rho/common/rotation_helpers.py for details.",
                FutureWarning,
                stacklevel=2,
            )

        # make sure only one obs maps to ACTION
        if self.observation_mapping is not None:
            action_mappings = [k for k, v in self.observation_mapping.items() if v == ACTION]
            if len(action_mappings) > 1:
                raise ValueError(
                    f"Multiple observation keys map to {ACTION}: {action_mappings}. Only one is allowed."
                )

        # create default normalization mapping if not provided,
        # using common sense defaults based on feature type
        if self.normalization_mapping is None:
            self.normalization_mapping = {
                OBSERVATION_IMAGE: NormalizationMode.IDENTITY,
                OBSERVATION_ENVIRONMENT_STATE: NormalizationMode.MEAN_STD,
                OBSERVATION_STATE: NormalizationMode.MEAN_STD,
                ACTION: NormalizationMode.MEAN_STD,
            }

        # Process stats field (from FeatureConfig logic)
        if isinstance(self.stats, str):
            # Detect if the string is a json string and if so decode it into a dictionary
            if "{" in self.stats:
                self.stats = ast.literal_eval(self.stats)  # Assume it's a JSON string
                self.stats = convert_dict_list_to_array(self.stats)
            elif Path(self.stats).suffix == ".json":
                with open(Path(self.stats)) as f:
                    self.stats = json.load(f)
            elif Path(self.stats).suffix == ".npz":
                # Handle npz files
                self.stats = dict(np.load(Path(self.stats), allow_pickle=True))
            else:
                raise ValueError(
                    f"Unsupported stats file format: {Path(self.stats).suffix}. "
                    f"Supported formats: .json, .npz"
                )

        # Convert normalization_mapping strings to NormalizationMode enums
        if self.normalization_mapping is not None:
            for k, v in self.normalization_mapping.items():
                if isinstance(v, str):
                    self.normalization_mapping[k] = NormalizationMode(v)

                # If any features use QUANTILE for normalization mapping set the clip values to -1.05 and 1.05
                if v == NormalizationMode.QUANTILE:
                    if self.clip_values is None:
                        self.clip_values = {}
                    self.clip_values[k] = (-1.05, 1.05)

        # Convert feature dicts to PolicyFeature objects
        if self.features is not None:
            for k, v in self.features.items():
                if isinstance(v, dict):
                    # Convert dict to PolicyFeature if necessary
                    if "shape" in v and isinstance(v["shape"], str):
                        v["shape"] = tuple([int(vi) for vi in v["shape"].strip("()").split(",") if vi != ""])
                    self.features[k] = PolicyFeature(**v)

        # Extract input and output features
        if self.features is not None:
            self.input_features = {k: v for k, v in self.features.items() if "observation" in k}
            self.output_features = {k: v for k, v in self.features.items() if "action" in k}
        else:
            self.input_features = {}
            self.output_features = {}

        # Remap stats keys from original to feature keys if observation_mapping is provided
        if self.observation_mapping is not None and self.stats is not None:
            if isinstance(self.stats, str):
                # Load stats from file and remap
                stats_path = self.stats
                if os.path.exists(stats_path):
                    with open(stats_path) as f:
                        stats_dict = json.load(f)
                    # Remap from original keys to feature keys
                    self.stats = {self.observation_mapping.get(k, k): v for k, v in stats_dict.items()}
                # else: keep as string path, will fail later with clear error
            elif isinstance(self.stats, dict):
                # Stats already loaded as dict - remap now
                self.stats = {self.observation_mapping.get(k, k): v for k, v in self.stats.items()}

        self.validate_actionchunk_transforms()
        self.create_reverse_transforms()
        self.process_transform_mapping()
        self.create_transformed_stats_and_features()

    def create_transformed_stats_and_features(self):
        import copy

        from rho.common.transforms import CombineKeys, CombineStatsKeys

        # Using deep copies to prevent overwriting self.stats and self.features
        # Shallow copy (.copy()) doesn't protect nested dictionaries
        self.transformed_stats = copy.deepcopy(self.stats)
        self.transformed_features = copy.deepcopy(self.features)

        if self.transform_mapping is not None:
            # Check the transform mapping for any CombineKeys transforms
            # This means we will need to combine stats for those keys before normalization
            # first, check the transform mapping for any CombineKeys transforms
            for key in self.transform_mapping:
                for transform in self.transform_mapping[key]:
                    combine_keys_config = extract_transform_config(
                        transform,
                        transform_class=CombineKeys,
                        type_name="combine_keys",
                    )
                    if combine_keys_config is not None and not combine_keys_config["post_norm"]:
                        combine_transform = CombineStatsKeys(**combine_keys_config)
                        self.transformed_stats, self.transformed_features = combine_transform(
                            self.transformed_stats, self.transformed_features
                        )

            from rho.common.transforms import ConvertStatsTo6d, ConvertTo6dActions

            # second, check the transform mapping for any ConvertTo6d transforms
            for key in self.transform_mapping:
                for transform in self.transform_mapping[key]:
                    convert_to_6d_config = extract_transform_config(
                        transform,
                        transform_class=ConvertTo6dActions,
                        type_name="convert_to_6d_actions",
                    )
                    if convert_to_6d_config is not None and not convert_to_6d_config["post_norm"]:
                        convert_to_6d_transform = ConvertStatsTo6d(**convert_to_6d_config)
                        self.transformed_stats, self.transformed_features = convert_to_6d_transform(
                            self.transformed_stats, self.transformed_features
                        )

    def validate_actionchunk_transforms(self):
        logger.info("ACTIONCHUNK normalization validation")
        # Validate ACTIONCHUNK normalization modes
        if self.normalization_mapping is not None:
            actionchunk_modes = [
                NormalizationMode.ACTIONCHUNK_MIN_MAX,
                NormalizationMode.ACTIONCHUNK_MEAN_STD,
                NormalizationMode.ACTIONCHUNK_QUANTILE,
                NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD,
                NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX,
                NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE,
            ]

            # Check 1: ACTIONCHUNK modes should only be used for ACTION feature type
            for feature_type_key, norm_mode in self.normalization_mapping.items():
                if (
                    norm_mode in actionchunk_modes
                    and feature_type_key != FeatureType.ACTION
                    and feature_type_key != "ACTION"
                ):
                    raise ValueError(
                        f"ACTIONCHUNK normalization mode '{norm_mode.value}' "
                        f"is used for feature type '{feature_type_key}', "
                        f"but ACTIONCHUNK modes are only intended for ACTION feature type. "
                        f"Please use MEAN_STD, MIN_MAX, or QUANTILE for non-action features."
                    )

            # Check 2: Validate that ACTIONCHUNK normalization modes have delta_actions transform
            action_norm_mode = self.normalization_mapping.get(FeatureType.ACTION)
            if action_norm_mode in actionchunk_modes:
                # Check if delta_actions transform exists with relative_to_state=True
                action_transforms = self.transform_mapping.get(ACTION, []) if self.transform_mapping else []
                has_valid_delta = False
                for t in action_transforms:
                    # Handle both dict configs (from YAML) and instantiated transforms
                    delta_config = extract_transform_config(
                        t,
                        transform_class=DeltaActions,
                        type_name="delta_actions",
                    )
                    if delta_config is not None and (delta_config.get("relative_to_state") is True):
                        has_valid_delta = True
                        break

                if not has_valid_delta:
                    # Raise error instead of auto-adding the transform
                    warning_msg = (
                        f"ACTIONCHUNK normalization mode '{action_norm_mode.value}' requires "
                        f"a 'delta_actions' transform with relative_to_state=True "
                        f"in your {ACTION} transform mapping. "
                        f"Please add the following to your config:\n\n"
                        f"transform_mapping:\n"
                        f"  {ACTION}:\n"
                        f"    - type: delta_actions\n"
                        f"      action_key: {ACTION}\n"
                        f"      state_key: {OBSERVATION_STATE}\n"
                        f"      action_type: USER-SPECIFIED\n"
                        f"      relative_to_state: true\n"
                        f"      post_norm: false\n\n"
                        f"This transform is required because ACTIONCHUNK normalization modes need actions "
                        f"to be relative to the current robot state for proper chunk-based normalization."
                        f"Be sure to set the action_type field appropriately for your dataset."
                    )
                    logger.warning(warning_msg)
                    raise ValueError(warning_msg)

    def create_reverse_transforms(self):
        logger.debug("Time for reverse transforms")
        # Precompute reverse transforms for denormalization
        self._reverse_transforms = []
        self._reverse_transforms_post_norm = []

        if self.transform_mapping is not None:
            action_transforms = self.transform_mapping.get(ACTION, [])

            for transform in reversed(action_transforms):
                # Check for DeltaActions transform
                delta_config = extract_transform_config(
                    transform,
                    transform_class=DeltaActions,
                    type_name="delta_actions",
                )
                if delta_config is not None:
                    absolute_transform = AbsoluteActions(
                        state_key=delta_config["state_key"],
                        action_key=delta_config["action_key"],
                        relative_to_state=delta_config["relative_to_state"],
                        action_type=delta_config["action_type"],
                        post_norm=delta_config["post_norm"],
                    )

                    if delta_config["post_norm"]:
                        self._reverse_transforms_post_norm.append(absolute_transform)
                    else:
                        self._reverse_transforms.append(absolute_transform)

                # Check for ConvertTo6dActions transform
                convert_6d_config = extract_transform_config(
                    transform,
                    transform_class=ConvertTo6dActions,
                    type_name="convert_to_6d_actions",
                )
                if convert_6d_config is not None:
                    convert_from_6d_transform = ConvertFrom6dActions(
                        action_key=convert_6d_config["action_key"],
                        output_type=convert_6d_config["action_type"],
                    )

                    if convert_6d_config.get("post_norm", False):
                        self._reverse_transforms_post_norm.append(convert_from_6d_transform)
                    else:
                        self._reverse_transforms.append(convert_from_6d_transform)

        logger.info(
            f"DataConfig initialized with features: {self.features}, "
            f"Normalization mapping: {self.normalization_mapping}"
        )

    def process_transform_mapping(self):
        """Process the transform mapping to ensure necessary adjustments are made.

        This includes:
        - Validating that any ConvertTo6dActions transforms have the correct
          action_type set based on the DataConfig's action_type.
        - Validating that any DeltaActions transforms with relative_to_state=True
          have the correct action_type set based on the DataConfig's action_type.
        """
        if self.transform_mapping is not None:
            for key, transform in self.transform_mapping.items():
                # Check if it's a single transform with input_type == "Dict"
                # Handle both single transforms and lists of transforms
                transform_list_to_process = transform if isinstance(transform, (list, tuple)) else [transform]

                for t in transform_list_to_process:
                    convert_to_6d_config = (
                        extract_transform_config(t, ConvertTo6dActions, "convert_to_6d_actions") is not None
                    )
                    if convert_to_6d_config:
                        assert t.action_type == self.action_type, (
                            f"ConvertTo6dActions transform for key {key} has action_type "
                            f"{t.action_type}, but DataConfig action_type is {self.action_type}. "
                            f"Please match these values in the config."
                        )

                    # delta_actions_config = (
                    #     extract_transform_config(t, DeltaActions, "delta_actions") is not None
                    # )
                    # if delta_actions_config:
                    #     if convert_to_6d_found:
                    #         assert t.action_type == ActionType.EE_6D_POS, (
                    #             f"DeltaActions transform found for key {key} after "
                    #             f"ConvertTo6dActions. Please ensure the action_type is "
                    #             f"set to EE_6D_POS for proper normalization. "
                    #             f"Current action_type: {t.action_type}"
                    #         )
                    #     else:
                    #         assert t.action_type == self.action_type, (
                    #             f"DeltaActions transform found for key {key} without "
                    #             f"preceding ConvertTo6dActions. Please ensure the "
                    #             f"action_type is set to {self.action_type} for proper "
                    #             f"normalization. Current action_type: {t.action_type}"
                    #         )

    def remap_features(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap the features in the batch according to the feature configuration."""
        # For now, just return the batch as-is.
        if self.observation_mapping is not None:
            # Remap observation keys according to the mapping
            batch = {self.observation_mapping.get(k, k): v for k, v in batch.items()}

        if self.observation_whitelist is not None:
            # Filter the batch to only include whitelisted observations
            batch = {k: v for k, v in batch.items() if k in self.observation_whitelist}

        # Flatten state observations (keep batch dimension, flatten the rest)
        for key in list(batch.keys()):
            if "image" not in key:
                value = batch[key]
                if isinstance(value, torch.Tensor) and value.ndim > 1:
                    # Flatten all dimensions except the batch dimension
                    batch[key] = value.flatten(start_dim=1)

        return batch

    def get_input_transform(self):
        """This is meant to be used for processing the training dataset.
        The Normalize transform this returns will normalize both the observation field as well as
        the action field.
        """

        from rho.common.normalize import Normalize

        return Normalize(
            self.transformed_features,
            self.normalization_mapping,
            self.transformed_stats,
            self.clip_values,
            self.chunk_size,
        )

    def get_action_denormalization(self):
        """This is meant to be used for denormalizing the outputs of the model
        back into action space. It is primarily used by the Environment class
        for evaluation of the policy.

        The transform pipeline applies the reverse transforms in the correct order:
        - For post_norm=True transforms: reverse transform first, then unnormalize
        - For post_norm=False transforms: unnormalize first, then reverse transform
        """

        # Create unnormalize transform with the properly shaped stats
        unnormalize = Unnormalize(
            self.transformed_features, self.normalization_mapping, self.transformed_stats, self.chunk_size
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        transforms = []
        transforms.extend(self._reverse_transforms_post_norm)
        transforms.append(unnormalize)
        transforms.extend(self._reverse_transforms)

        # Return single transform or composed transforms
        if len(transforms) == 1:
            return transforms[0]
        else:
            return Compose(transforms)

    def _convert_numpy_to_python(self, obj):
        """Convert numpy arrays and types to Python native types for JSON serialization.

        Note: This method is kept for backward compatibility. It delegates to serialize_to_dict.
        """
        return serialize_to_dict(obj)

    def to_yaml(self, config_path: Path):
        """Save the configuration to a YAML file.

        The stats field should be saved as a json file in the same folder as the main yaml file.
        The populating the stats field in the yaml file with the path to the json file.

        """
        import json

        import yaml

        # Create the output directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Simplify the features for YAML output
        simplified_features = {}
        if self.features is not None:
            for k, v in self.features.items():
                is_policy_feature = isinstance(v, PolicyFeature)
                if is_policy_feature:
                    # Format shape as a tuple string like "(2,)"
                    shape_str = (
                        f"({','.join(map(str, v.shape))},)"
                        if len(v.shape) == 1
                        else f"({','.join(map(str, v.shape))})"
                    )
                    type_str = v if isinstance(v, str) else v.type
                    type_str = v.type.name if hasattr(v.type, "name") else type_str
                    new_dict = {
                        "type": type_str,
                        "shape": shape_str,
                    }
                    simplified_features[k] = new_dict
                else:
                    # For non-PolicyFeature objects, we need to ensure they're also serializable
                    if isinstance(v, dict):
                        simplified_features[k] = dict(v)  # Create a new dict copy
                    else:
                        simplified_features[k] = v

        # Create feature-type-based mapping (as in the correct format)
        type_based_mapping = {}
        if self.normalization_mapping is not None:
            for k, v in self.normalization_mapping.items():
                if isinstance(v, NormalizationMode):
                    type_based_mapping[k] = v.name
                else:
                    type_based_mapping[k] = v

        # Save stats to json file
        if isinstance(self.stats, dict) and self.stats:
            stats_path = config_path.parent / f"{config_path.stem}_stats.json"
            # Convert numpy arrays to Python types for JSON serialization
            stats_for_json = self._convert_numpy_to_python(self.stats)
            with open(stats_path, "w") as f:
                json.dump(stats_for_json, f, indent=2)
            # Use relative path for stats
            relative_stats_path = stats_path.relative_to(config_path.parent.parent.parent)
            stats_ref = str(relative_stats_path)
        else:
            stats_ref = self.stats if isinstance(self.stats, str) else ""

        # Create the config dictionary in the expected format
        # Get the choice name for this class (for ChoiceRegistry compatibility)
        try:
            type_name = self.get_choice_name(self.__class__)
        except ValueError:
            type_name = None

        config_dict = {
            "features": simplified_features,
            "normalization_mapping": type_based_mapping,
            "stats": stats_ref,
        }
        if type_name:
            config_dict["type"] = type_name

        # Write YAML file
        with open(config_path, "w") as f:
            # Use safe_dump to avoid Python object serialization
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert the DataConfig to a dictionary.

        Serializes each dataclass field individually via serialize_to_dict so that
        nested dataclasses with their own to_dict() are handled correctly.
        Excludes runtime-computed attributes (transformed_features, transformed_stats)
        and optionally strips stats when serialize_stats is False.
        Feature shapes are converted back to string format for YAML round-trip safety.

        Note: The ChoiceRegistry ``type`` key is intentionally **not** included.
        DataConfig subclasses (LeRobotDatasetConfig, MultiDatasetConfig) are
        concrete types in a Union — draccus 0.8-0.11 has a bug where it fails
        to pop the ``type`` key when decoding known concrete subclasses.
        Omitting ``type`` avoids that bug and keeps the output round-trippable.
        Abstract ChoiceRegistry types like PolicyConfig still get ``type``
        via the generic serialize_to_dict ChoiceRegistry path.
        """
        from dataclasses import fields as dc_fields

        result = {}
        for f in dc_fields(self):
            result[f.name] = serialize_to_dict(getattr(self, f.name))

        if not self.serialize_stats:
            result["stats"] = None

        # Fix feature shapes for YAML round-trip (lists → string tuples)
        if isinstance(result.get("features"), dict):
            fixup_feature_shapes(result["features"])

        return result

    def get_transforms(
        self, remap=True, images_only=False
    ) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None:
        """Get a composed transform that includes both feature-specific transforms and normalization.

        Returns:
            A composed transform function that applies all configured transforms to a batch of data.
        """

        # First transform - Relabel all the keys in the dictionary to match the expected feature names
        transform_list = [] if not remap else [self.remap_features]

        # Keep track of transforms that should be applied after normalization
        post_norm_transforms = []

        # Second transform - Add any additional transforms from the transform_mapping
        if self.transform_mapping is not None:
            for key, transform in self.transform_mapping.items():
                # Check if it's a single transform with input_type == "Dict"
                # Handle both single transforms and lists of transforms
                if images_only and not key.startswith(OBSERVATION_IMAGE):
                    continue
                transform_list_to_process = transform if isinstance(transform, (list, tuple)) else [transform]

                for t in transform_list_to_process:
                    list_to_extend = (
                        post_norm_transforms if hasattr(t, "post_norm") and t.post_norm else transform_list
                    )
                    if hasattr(t, "input_type") and t.input_type == "Dict":
                        # Dict transforms operate on the entire batch, add directly
                        list_to_extend.append(t)
                    else:
                        # Other transforms operate on Tensors, wrap them together
                        list_to_extend.append(TransformWrapper(key, t))

        # Third transform - Add normalization transform
        if self.features is not None and self.normalization_mapping is not None:
            input_transform = self.get_input_transform()
            if input_transform is not None:
                # Keep on CPU — moving to CUDA here breaks DataLoader workers
                # (forked processes cannot access the parent's CUDA context).
                # Data is moved to GPU later in the training loop / policy interface.
                transform_list.append(input_transform)

        # Append post-normalization transforms
        transform_list.extend(post_norm_transforms)
        if self.language_instruction is not None:
            transform_list.append(lambda x: append_task(x, instruction=self.language_instruction))

        # Return composed transform
        if len(transform_list) == 1:
            return transform_list[0]
        else:
            # Create a composed function that applies all transforms in sequence
            def composed_transform(
                batch: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                for transform_fn in transform_list:
                    batch = transform_fn(batch)
                return batch

            return composed_transform

    @property
    def feature_dict(self):
        return self.features

    @property
    def transformed_feature_dict(self):
        return self.transformed_features


@DataConfig.register_subclass("base")
@dataclass
class BaseDatasetConfig(DataConfig):
    pass

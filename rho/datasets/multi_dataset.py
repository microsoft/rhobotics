from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features

from rho.common.constants import ACTION, OBSERVATION_STATE
from rho.common.serialization import serialize_to_dict
from rho.common.transforms import DeltaActions, Transform, build_key_padding_transform
from rho.common.types import PolicyFeature, TrainingMode
from rho.datasets.lerobot_dataset import EpisodeAwareSampler, LeRobotDatasetConfig

if TYPE_CHECKING:
    from rho.policies.base import PolicyConfig

logger = logging.getLogger(__name__)


class ShuffleType(Enum):
    EPISODE = 0  # shuffle episodes and retain frame order within episodes (not currently supported)
    FRAME = 1  # shuffle all frames without regard to episodes
    NONE = 2  # no shuffling


class AlkuMultiDataset(torch.utils.data.Dataset):
    """AlkuMultiDataset
    A multi-dataset that concatenates datasets and works with StreamingDatasets.
    Weighting is handled by the sampler, not the dataset.
    """

    def __init__(
        self,
        datasets: list[LeRobotDataset],
        training_modes: list[TrainingMode] | None = None,
        features: dict | None = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.training_modes = training_modes or [TrainingMode.ROBOT_FLOWMATCH] * len(datasets)
        if len(self.training_modes) != len(self.datasets):
            raise ValueError("training_modes and datasets must have the same length")
        self.cumulative_lengths = []
        sum_so_far = 0
        for dataset in self.datasets:
            sum_so_far += len(dataset)
            self.cumulative_lengths.append(sum_so_far)

        self.delta_timestamps = None
        if len(self.datasets) > 0:
            self.delta_timestamps = self.datasets[0].delta_timestamps

        logger.debug(f"MultiDataset delta_timestamps: {self.delta_timestamps}")

        self.features = features
        if self.features is None and len(self.datasets) > 0:
            self.features = handle_possible_lerobot_features(self.datasets[0].features)

        logger.debug(f"MultiDataset features: {self.features}")
        self.padding_transform = lambda x: x  # Identity by default
        if self.features is not None:
            if hasattr(self.features, "feature_dict"):
                self.features = self.features.feature_dict

            # Convert dict to PolicyFeature if necessary
            if isinstance(self.features, dict):
                for k, v in self.features.items():
                    if isinstance(v, dict) and "type" in v and "shape" in v:
                        # Convert dict to PolicyFeature if necessary
                        if "shape" in v and isinstance(v["shape"], str):
                            v["shape"] = tuple(
                                [int(vi) for vi in v["shape"].strip("()").split(",") if vi != ""]
                            )
                        self.features[k] = PolicyFeature(v["type"], v["shape"])

            self.padding_transform = build_key_padding_transform(
                features=self.features, delta_timestamps=self.delta_timestamps
            )

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    @property
    def num_frames(self):
        return self.__len__()

    def __getitem__(self, idx):
        dataset_idx = 0
        sample_idx = idx

        for i, length in enumerate(self.cumulative_lengths):
            if idx < length:
                dataset_idx = i
                if i > 0:
                    sample_idx = idx - self.cumulative_lengths[i - 1]
                break

        # Return sample with dataset index
        max_retries = 100  # Prevent infinite loops
        attempt = 0
        original_idx = idx

        while attempt < max_retries:
            try:
                sample = self.datasets[dataset_idx][sample_idx]
                sample["dataset_index"] = dataset_idx
                sample["training_mode"] = self.training_modes[dataset_idx].value

                sample = self.padding_transform(sample)

                if attempt > 0:
                    logger.debug(
                        f"Successfully recovered from error at index {original_idx} by using index {idx}"
                    )

                return sample

            except Exception as e:
                logger.warning(
                    f"Error accessing dataset {dataset_idx} at index {sample_idx} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

                # Recalculate dataset_idx and sample_idx for new idx

                dataset_idx = (dataset_idx + 1) % len(self.datasets)

                idx = (idx + 1) % len(self.datasets[dataset_idx])

                attempt += 1

        # If all retries failed, raise the last exception
        raise RuntimeError(
            f"Failed to get valid sample after {max_retries} attempts starting from index {original_idx}"
        )


class AlkuMultiIterableDataset(torch.utils.data.IterableDataset):
    """AlkuMultiIterableDataset
    Streams from multiple iterable datasets.
    Assumes each dataset is an IterableDataset (e.g., streaming) and yields items from them.
    Weighting is handled here because samplers are ignored for IterableDatasets.
    """

    def __init__(
        self,
        datasets: list[Iterable],
        dataset_weights: list[float] = None,
        training_modes: list[TrainingMode] | None = None,
        features: dict | None = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.dataset_weights = dataset_weights or [1.0] * len(datasets)
        self.dataset_lengths = [ds.num_frames if hasattr(ds, "num_frames") else 0 for ds in self.datasets]
        self.training_modes = training_modes or [TrainingMode.ROBOT_FLOWMATCH] * len(datasets)

        if len(self.datasets) != len(self.dataset_weights):
            raise ValueError("datasets and dataset_weights must have the same length")

        if len(self.datasets) != len(self.training_modes):
            raise ValueError("datasets and training_modes must have the same length")

        self.features = features
        if self.features is None and len(self.datasets) > 0:
            # If features are not explicitly defined then assume all datasets have the same features
            self.features = handle_possible_lerobot_features(self.datasets[0].features)

        self.delta_timestamps = None
        if len(self.datasets) > 0:
            # Since delta_timestamps is derived from the policy it should be the same for all datasets
            self.delta_timestamps = self.datasets[0].delta_timestamps

        self.padding_transform = lambda x: x  # Identity by default
        if self.features is not None:
            self.padding_transform = build_key_padding_transform(
                features=self.features, delta_timestamps=self.delta_timestamps
            )

    @property
    def num_frames(self):
        return sum(self.dataset_lengths)

    def __iter__(self) -> Iterator:
        iterators = [iter(ds) for ds in self.datasets]
        weights = self.dataset_weights.copy()  # Keep original weights

        while True:
            # Choose a dataset based on weights
            idx = random.choices(range(len(iterators)), weights=weights, k=1)[0]

            try:
                sample = next(iterators[idx])
                sample["dataset_index"] = idx
                sample["training_mode"] = self.training_modes[idx].value
                # Add padding if necessary (to fill in missing keys)
                sample = self.padding_transform(sample)
                yield sample
            except StopIteration:
                # Reset the exhausted dataset
                iterators[idx] = iter(self.datasets[idx])
                # Try again with the reset iterator
                sample = next(iterators[idx])
                sample["dataset_index"] = idx
                sample["training_mode"] = self.training_modes[idx].value
                # Add padding if necessary (to fill in missing keys)
                sample = self.padding_transform(sample)
                yield sample


class MultiDatasetWeightedSampler:
    """
    Hold several EpisodeAwareSampler objects and on every __iter__ step
    pick one according to sample_weights, then yield one index from it.
    """

    def __init__(
        self,
        samplers: list[EpisodeAwareSampler],
        sample_weights: list[float] | None = None,
        shuffle_type: ShuffleType = ShuffleType.FRAME,
        seed: int | None = None,
    ):
        if shuffle_type == ShuffleType.EPISODE:
            raise NotImplementedError("EPISODE shuffle type is not currently supported")

        self.samplers = samplers
        self.shuffle_type = shuffle_type

        if sample_weights is None:
            sample_weights = [1.0] * len(samplers)
        if len(sample_weights) != len(samplers):
            raise ValueError("samplers and sample_weights must have the same length")

        w = np.asarray(sample_weights, dtype=np.float64)
        self.sample_weights_normalized = w / w.sum()

        self.allocated_sample_sizes = self.get_allocated_sample_sizes()
        logger.info(f"Allocated sample sizes per dataset: {self.allocated_sample_sizes}")
        self.yielded = [0] * len(self.samplers)

        self.cumulative_episode_indices = [
            sum(len(sampler) for sampler in self.samplers[:i]) for i in range(len(self.samplers) + 1)
        ]

        self._rng = np.random.default_rng(seed)
        self._epoch = 0

    def get_allocated_sample_sizes(self) -> list[int]:
        """
        Allocate sample sizes for each sampler based on the normalized weights.
        """
        sample_lengths = [len(sampler) for sampler in self.samplers]

        # get the upper bound on the overall data pool size
        data_pool_size = int(round(min(sample_lengths / self.sample_weights_normalized)))
        allocated_sizes = [int(round(weight * data_pool_size)) for weight in self.sample_weights_normalized]
        return allocated_sizes

    def __len__(self) -> int:
        """
        Return the total number of sampled frames across all samplers.
        """
        return sum(self.allocated_sample_sizes)

    def __iter__(self):
        # indices of samplers that still hold data
        # keep track of active samplers and how many items each sampler
        # has already produced in this pass
        iters = [iter(s) for s in self.samplers]
        while True:
            active = [
                idx
                for idx in range(len(self.samplers))
                if self.yielded[idx] < self.allocated_sample_sizes[idx]
            ]

            while active:
                # restrict weights to the active samplers and re-normalise
                probs = self.sample_weights_normalized[active]
                probs = probs / probs.sum()

                if self.shuffle_type == ShuffleType.FRAME:
                    chosen_dataset_idx = self._rng.choice(active, p=probs)
                else:
                    chosen_dataset_idx = np.argmax(probs)

                try:
                    sample_idx = next(iters[chosen_dataset_idx])
                except StopIteration:
                    # sampler depleted for this pass
                    iters[chosen_dataset_idx] = iter(self.samplers[chosen_dataset_idx])
                    sample_idx = next(iters[chosen_dataset_idx])

                sample_idx += self.cumulative_episode_indices[chosen_dataset_idx]
                logger.debug(
                    f"Yielding sample {sample_idx} from dataset {chosen_dataset_idx}, "
                    f"sample {self.yielded[chosen_dataset_idx] + 1}/"
                    f"{self.allocated_sample_sizes[chosen_dataset_idx]}"
                )
                self.yielded[chosen_dataset_idx] += 1
                # remove sampler once it has reached its allocated quota
                if self.yielded[chosen_dataset_idx] >= self.allocated_sample_sizes[chosen_dataset_idx]:
                    active.remove(chosen_dataset_idx)
                yield sample_idx

            # reset iterators and yielded counts for the next pass
            self.yielded = [0] * len(self.samplers)
            self._epoch += 1
            logger.info("All samplers have hit their allocation.")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def save_state(self) -> dict:
        """Return a serialisable snapshot of the full sampler state.

        Captures the numpy RNG state, per-dataset yield counters, current
        epoch, and the state of every child EpisodeAwareSampler so that
        iteration can be resumed deterministically via load_state().
        """
        sampler_states = []
        for s in self.samplers:
            if hasattr(s, "save_state"):
                sampler_states.append(s.save_state())
            else:
                logger.warning(
                    f"Sampler {type(s).__name__} does not support save_state(); its state will not be saved."
                )
                sampler_states.append(None)
        return {
            "rng_state": self._rng.__getstate__(),
            "yielded": list(self.yielded),
            "epoch": self._epoch,
            "sampler_states": sampler_states,
        }

    def load_state(self, state_dict: dict) -> None:
        """Restore sampler state from a previously saved snapshot."""
        self._rng.__setstate__(state_dict["rng_state"])
        self.yielded = list(state_dict["yielded"])
        self._epoch = state_dict["epoch"]
        for sampler, sampler_state in zip(self.samplers, state_dict["sampler_states"], strict=True):
            if sampler_state is None:
                logger.warning(f"No saved state for sampler {type(sampler).__name__}; skipping load_state().")
                continue
            if hasattr(sampler, "load_state"):
                sampler.load_state(sampler_state)
            else:
                logger.warning(
                    f"Sampler {type(sampler).__name__} does not support load_state(); "
                    "its state will not be restored."
                )


@dataclass
class WeightedDatasetConfig:
    dataset: LeRobotDatasetConfig | MultiDatasetConfig
    weight: float = 1


@dataclass
class MultiDatasetConfig:
    """
    Configuration for a multi-dataset setup.
    """

    datasets: list[WeightedDatasetConfig]  # This is the field we expect to read from the YAML file

    # This is expected to be overwritten in the code by breaking apart the datasets list
    # into individual LeRobotDatasetConfig objects and their corresponding weights.
    dataset_cfgs: list[LeRobotDatasetConfig] = None
    weights: list[float] | None = None
    features: dict[str, dict | PolicyFeature] | None = (
        None  # Dictionary of features to be used in the environment
    )
    transformed_features: dict[str, dict | PolicyFeature] | None = (
        None  # Optional dictionary of transformed features
    )
    split_name: str | None = None  # Default split name
    shuffle_type: ShuffleType = ShuffleType.FRAME  # Default shuffle type
    seed: int = 12345  # Default seed for reproducibility
    num_workers: int = 4  # Default number of workers for DataLoader
    prefetch_factor: int = 2  # Default prefetch factor for DataLoader
    batch_size: int = 64
    streaming: bool = False
    shuffle_buffer_size: int | None = None
    observation_whitelist: list[str] | None = None
    transform_mapping: dict[str, list[Transform]] = None

    chunk_size: int | None = None  # Chunk size for action chunk normalization

    # TODO: Sampling does not work when this is false. Investigate why later.
    flatten_nested: bool = True  # Whether to flatten nested MultiDatasetConfigs

    # This option is useful during pretraining to shrink the size of the saved config file.
    serialize_stats: bool = True  # Whether to serialize stats when converting to dict

    # For pretraining using video caching can be too expensive this option allows
    # us to limit the size of the video decoder cache.
    video_decoder_cache_size: int | None = None

    def __post_init__(self):
        if self.dataset_cfgs is None:
            if self.flatten_nested:
                self.dataset_cfgs = []
                for d in self.datasets:
                    if isinstance(d.dataset, MultiDatasetConfig):
                        # Flatten nested MultiDatasetConfigs
                        self.dataset_cfgs.extend(d.dataset.dataset_cfgs)
                    else:
                        self.dataset_cfgs.append(d.dataset)
            else:
                self.dataset_cfgs = [d.dataset for d in self.datasets]

        if self.weights is None:
            if self.flatten_nested:
                self.weights = []
                for d in self.datasets:
                    if isinstance(d.dataset, MultiDatasetConfig):
                        normalized_weights = np.array([d.weight for d in d.dataset.datasets])
                        normalized_weights = normalized_weights / normalized_weights.sum()
                        self.weights.extend([d.weight * w for w in normalized_weights])
                    else:
                        self.weights.append(d.weight)
            else:
                self.weights = [d.weight for d in self.datasets]

        if len(self.dataset_cfgs) != len(self.weights):
            raise ValueError("dataset_cfgs and weights must have the same length")
        if len(self.dataset_cfgs) == 0:
            raise ValueError("At least one dataset must be provided")

        if self.features is None:
            self.features = self.dataset_cfgs[0].features

        if self.transformed_features is None:
            self.transformed_features = self.dataset_cfgs[0].transformed_features

        for k, v in self.features.items():
            if isinstance(v, dict) and "shape" in v and "type" in v:
                # Convert dict to PolicyFeature if necessary
                if isinstance(v["shape"], str):
                    v["shape"] = tuple([int(vi) for vi in v["shape"].strip("()").split(",") if vi != ""])
                self.features[k] = PolicyFeature(v["type"], v["shape"])

        needs_delta_actions = self.needs_delta_actions()
        perdim_delta_actions = self.needs_perdim_delta_actions()

        # Overwrite the transform mapping
        if self.transform_mapping is not None:
            self.set_attribute("transform_mapping", self.transform_mapping)

        # Auto-add delta_actions transform if needed and not already present
        if needs_delta_actions:
            self.create_delta_transform(perdim_delta_actions)

        # Overwrite the split names
        if self.split_name is not None:
            self.set_attribute("split_name", self.split_name)

        # Overwrite the observation whitelist
        if self.observation_whitelist is not None:
            self.set_attribute("observation_whitelist", self.observation_whitelist)

        # Overwrite streaming mode
        self.set_attribute("streaming", self.streaming)

        # Overwrite num workers
        if self.num_workers is not None:
            self.set_attribute("num_workers", self.num_workers)

        if self.chunk_size is not None:
            self.set_attribute("chunk_size", self.chunk_size)

        if self.shuffle_buffer_size is not None:
            self.set_attribute("shuffle_buffer_size", self.shuffle_buffer_size)

        self.set_attribute("serialize_stats", self.serialize_stats)

    def create_delta_transform(self, perdim_delta_actions: bool):
        for dataset_cfg in self.dataset_cfgs:
            if isinstance(dataset_cfg, LeRobotDatasetConfig):
                if dataset_cfg.transform_mapping is None:
                    dataset_cfg.transform_mapping = {}

                # Check if delta_actions transform already exists
                action_transforms = dataset_cfg.transform_mapping.get(ACTION, [])
                has_delta_actions = any(
                    isinstance(t, DeltaActions)
                    and (getattr(t, "relative_to_state", False) is True or perdim_delta_actions)
                    for t in action_transforms
                )

                if not has_delta_actions:
                    logger.info(
                        "ACTIONCHUNK normalization detected in MultiDatasetConfig. "
                        "Automatically adding delta_actions transform with relative_to_state=True."
                    )

                    # Create the delta_actions transform
                    delta_transform = DeltaActions(
                        action_key=ACTION,
                        state_key=OBSERVATION_STATE,
                        relative_to_state=True,
                        post_norm=False,
                    )

                    # Add to action transforms
                    if ACTION not in dataset_cfg.transform_mapping:
                        dataset_cfg.transform_mapping[ACTION] = [delta_transform]
                    else:
                        # Insert at the beginning to ensure it runs before other transforms
                        dataset_cfg.transform_mapping[ACTION].insert(0, delta_transform)
            elif isinstance(dataset_cfg, MultiDatasetConfig):
                # Recursively handle nested MultiDatasetConfigs
                dataset_cfg.create_delta_transform(perdim_delta_actions)

    @property
    def feature_dict(self):
        if self.features is not None:
            return self.features
        return None

    @property
    def transformed_feature_dict(self):
        if self.transformed_features is not None:
            return self.transformed_features
        return None

    def to_dict(self) -> dict:
        """Serialize MultiDatasetConfig, excluding computed fields.

        Fields like ``dataset_cfgs``, ``weights``, ``features``, and
        ``transformed_features`` are derived from ``datasets`` in
        ``__post_init__`` and should not be persisted — they would bloat
        the output and cause decode failures on round-trip.
        """
        from dataclasses import fields as dc_fields

        _exclude = {"dataset_cfgs", "weights", "features", "transformed_features"}
        result = {}
        for f in dc_fields(self):
            if f.name in _exclude:
                continue
            result[f.name] = serialize_to_dict(getattr(self, f.name))
        return result

    def set_attribute(self, attr_name: str, value) -> None:
        """Set an attribute for the MultiDatasetConfig and propagate to all dataset configs."""
        setattr(self, attr_name, value)
        for dataset_cfg in self.dataset_cfgs:
            dataset_cfg.set_attribute(attr_name, value)

    def get_length(self) -> int | None:
        """Get the total number of frames across all datasets."""
        total = 0
        for dataset_cfg in self.dataset_cfgs:
            length = dataset_cfg.get_length()
            if length is None:
                return None
            total += length
        return total

    def needs_delta_actions(self) -> bool:
        """Check if any dataset uses actionchunk normalization that requires delta actions."""
        # Check if any dataset uses ACTIONCHUNK normalization modes
        # for perdim delta actions, relative to state does not necessarily have to be used
        return any(dataset_cfg.needs_delta_actions() for dataset_cfg in self.dataset_cfgs)

    def needs_perdim_delta_actions(self) -> bool:
        return any(dataset_cfg.needs_perdim_delta_actions() for dataset_cfg in self.dataset_cfgs)

    def make_dataset(self, policy_cfg: PolicyConfig = None) -> AlkuMultiDataset | AlkuMultiIterableDataset:
        # If we have a MultiDatasetConfig, we need to create a list of datasets
        datasets = []
        for cfg in self.dataset_cfgs:
            ds = cfg.make_dataset(policy_cfg)
            datasets.append(ds)
        if self.streaming:
            return AlkuMultiIterableDataset(datasets, self.weights, features=self.features)

        return AlkuMultiDataset(
            datasets,
            features=self.features,
        )

    def make_sampler(
        self,
        dataset: AlkuMultiDataset | AlkuMultiIterableDataset,
        policy_cfg: PolicyConfig = None,
    ) -> MultiDatasetWeightedSampler:
        assert isinstance(dataset, AlkuMultiDataset | AlkuMultiIterableDataset), (
            "Expected dataset to be AlkuMultiDataset or AlkuMultiIterableDataset for MultiDatasetConfig"
        )
        # If we have a MultiDatasetConfig, we need to create a MultiDatasetWeightedSampler
        samplers = []
        for dataset_cfg, individual_dataset in zip(self.dataset_cfgs, dataset.datasets, strict=False):
            # Create a sampler for each individual dataset
            new_sampler = dataset_cfg.make_sampler(individual_dataset, policy_cfg)
            if new_sampler is not None:
                samplers.append(new_sampler)
        if len(samplers) == 0 or samplers is None:
            return None

        return MultiDatasetWeightedSampler(
            samplers=samplers,
            sample_weights=self.weights,
            shuffle_type=self.shuffle_type,
            seed=self.seed,
        )

    def get_contributions(self, parent_weighting=None, parent_prefix=None):
        contributions = []
        name = "MultiDataset"
        length = self.get_length()

        contributions = {"name": name, "length": length, "weighted_length": 0, "child_contributions": []}

        child_lengths = np.array([cfg.get_length() for cfg in self.dataset_cfgs])

        normalized_weights = np.array(self.weights) / np.sum(self.weights)
        # get the upper bound on the overall data pool size
        data_pool_size = int(round(min(child_lengths / normalized_weights)))
        allocated_sizes = [int(round(weight * data_pool_size)) for weight in normalized_weights]

        for i, dataset_cfg in enumerate(self.dataset_cfgs):
            ds_contributions = dataset_cfg.get_contributions()
            ds_contributions["weighted_length"] = allocated_sizes[i]
            contributions["child_contributions"].append(ds_contributions)

        return contributions


def handle_possible_lerobot_features(features):
    if isinstance(features, dict):
        for _, v in features.items():
            if isinstance(v, dict) and "dtype" in v:
                return dataset_to_policy_features(features)
    return features

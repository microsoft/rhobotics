from __future__ import annotations

import copy
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
from rho.datasets.data_config import DataConfig
from rho.datasets.lerobot_dataset import EpisodeAwareSampler

if TYPE_CHECKING:
    from rho.policies.base import PolicyConfig

logger = logging.getLogger(__name__)

DEFAULT_MAX_ANSWER_BYTES = 2048


def _feature_specs_equal(a: dict | None, b: dict | None) -> bool:
    return serialize_to_dict(a) == serialize_to_dict(b)


class ShuffleType(Enum):
    EPISODE = 0  # shuffle episodes and retain frame order within episodes (not currently supported)
    FRAME = 1  # shuffle all frames without regard to episodes
    NONE = 2  # no shuffling


class CombinedDataset(torch.utils.data.Dataset):
    """Map-style view over multiple datasets.
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
        self.target_sequence_lengths = None
        if len(self.datasets) > 0:
            self.delta_timestamps = self.datasets[0].delta_timestamps
            self.target_sequence_lengths = getattr(
                self.datasets[0],
                "target_sequence_lengths",
                None,
            )

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
                features=self.features, target_sequence_lengths=self.target_sequence_lengths
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


class WeightedIterableMixDataset(torch.utils.data.IterableDataset):
    """Weighted mixer over a list of IterableDataset children.

    Three mixing modes:

    * **Per-sample (default, ``homogeneous_batch_size=None``)**: re-rolls the
      child each yield. Right when all children share a training_mode
      (pure-robot LeRobot mix).

    * **Homogeneous-batch (``homogeneous_batch_size`` is an ``int``)**:
      re-rolls only every ``homogeneous_batch_size`` yields. The DataLoader's
      default collate then packs that homogeneous burst into a single batch
      in which every sample shares one ``training_mode``. Required for
      grouped datasets that require homogeneous batches. Only correct under
      ``num_workers=0`` because multiple workers can interleave bursts.

    * **Per-child batch (``homogeneous_batch_size`` is a ``list[int]``)**:
      one entry per child, picked by the weighted roll. The mixer COLLATES
      ``bursts[idx]`` samples internally and yields a pre-batched dict;
      the DataLoader runs with ``batch_size=None`` and an identity
      ``collate_fn``. The only way to support different batch sizes per
      modality (e.g. robot=64, VL=8) inside one DataLoader, since
      ``DataLoader.batch_size`` is a single fixed value. Used by
      cotraining configs that set ``MultiDatasetConfig.vl_batch_size`` --
      VL samples push much longer sequences through the VLM than robot
      samples, so per-rank=64 OOMs at backward when robot at the same
      size fits fine. Required ``num_workers=0`` for the same homogeneity
      reason as the int mode.

    `dataset_index` and `training_mode` are expected to already be set on samples
    by the children (e.g., `MapStyleIterableWrapper`). This
    class only applies the cross-dataset `padding_transform` to fill missing keys
    so the collated batch is rectangular across heterogeneous children.
    """

    def __init__(
        self,
        datasets: list[Iterable],
        dataset_weights: list[float] = None,
        features: dict | None = None,
        homogeneous_batch_size: int | list[int] | None = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.dataset_weights = dataset_weights or [1.0] * len(datasets)
        self.dataset_lengths = [ds.num_frames if hasattr(ds, "num_frames") else 0 for ds in self.datasets]
        self.homogeneous_batch_size = homogeneous_batch_size

        if len(self.datasets) != len(self.dataset_weights):
            raise ValueError("datasets and dataset_weights must have the same length")
        # Resolve per-child burst sizes. Three modes:
        #  None        -> per-sample mixing, DataLoader collates (legacy)
        #  int         -> homogeneous batches of that size, DataLoader collates (legacy)
        #  list[int]   -> per-child burst sizes, mixer yields pre-collated batches
        #                 (new mode for cotraining where VL needs smaller batches
        #                 than robot to fit GPU memory)
        if homogeneous_batch_size is None:
            self._bursts = [1] * len(datasets)
            self.yields_batches = False
        elif isinstance(homogeneous_batch_size, int):
            if homogeneous_batch_size <= 0:
                raise ValueError(f"homogeneous_batch_size must be > 0, got {homogeneous_batch_size}")
            self._bursts = [homogeneous_batch_size] * len(datasets)
            self.yields_batches = False
        elif isinstance(homogeneous_batch_size, list):
            if len(homogeneous_batch_size) != len(datasets):
                raise ValueError(
                    f"homogeneous_batch_size list length {len(homogeneous_batch_size)} "
                    f"must match number of datasets {len(datasets)}"
                )
            for b in homogeneous_batch_size:
                if not isinstance(b, int) or b <= 0:
                    raise ValueError(f"homogeneous_batch_size entries must be positive ints, got {b!r}")
            self._bursts = list(homogeneous_batch_size)
            self.yields_batches = True
        else:
            got_type = type(homogeneous_batch_size).__name__
            raise TypeError(f"homogeneous_batch_size must be int, list[int], or None; got {got_type}")

        self.features = features
        if self.features is None and len(self.datasets) > 0:
            # If features are not explicitly defined then assume all datasets have the same features
            self.features = handle_possible_lerobot_features(self.datasets[0].features)

        self.delta_timestamps = None
        self.target_sequence_lengths = None
        for dataset in self.datasets:
            # Since delta_timestamps is derived from the policy it should be
            # the same for all robot datasets. In robot+VL mixes the first
            # child can be the VL bundle with no delta_timestamps, so use the
            # first non-null value to pad VL placeholder action/state tensors
            # to the same rank as robot batches.
            self.delta_timestamps = getattr(dataset, "delta_timestamps", None)
            if self.delta_timestamps is not None:
                break
        for dataset in self.datasets:
            self.target_sequence_lengths = getattr(
                dataset,
                "target_sequence_lengths",
                None,
            )
            if self.target_sequence_lengths is not None:
                break

        self.padding_transform = lambda x: x  # Identity by default
        if self.features is not None:
            self.padding_transform = build_key_padding_transform(
                features=self.features, target_sequence_lengths=self.target_sequence_lengths
            )

    def _sequence_len_for_key(self, key: str) -> int:
        if self.delta_timestamps is None:
            return 1
        max_length = 0
        for ts_key, timestamps in self.delta_timestamps.items():
            if ts_key.startswith(key):
                max_length = max(max_length, len(timestamps))
        return max(max_length, 1)

    def _normalize_placeholder_sequence_rank(self, batch: dict, key: str) -> None:
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            return
        seq_len = self._sequence_len_for_key(key)
        if batch[key].ndim != 2:
            return
        batch[key] = batch[key].unsqueeze(1)
        if seq_len > 1:
            batch[key] = batch[key].expand(-1, seq_len, -1)
        batch[key] = batch[key].contiguous()
        pad_key = f"{key}_is_pad"
        if (
            pad_key in batch
            and isinstance(batch[pad_key], torch.Tensor)
            and batch[pad_key].ndim == 2
            and batch[pad_key].shape[1] == 1
        ):
            batch[pad_key] = batch[pad_key].expand(-1, seq_len).contiguous()

    def _ensure_common_batch_keys(self, batch: dict) -> dict:
        """Keep cross-rank Accelerate concatenation happy for mixed modalities.

        In distributed mode Accelerate may fetch one batch per rank and
        concatenate them before slicing. Robot batches do not naturally carry
        ``answer_text``, while VL batches do. Add a zero fixed-width answer
        tensor to non-VL batches so every rank exposes the same keys.
        """
        if "answer_text" not in batch:
            ref = None
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    ref = value
                    break
            if ref is not None:
                batch_size = int(ref.shape[0]) if ref.ndim > 0 else 1
                batch["answer_text"] = torch.zeros(
                    (batch_size, DEFAULT_MAX_ANSWER_BYTES),
                    dtype=torch.uint8,
                    device=ref.device,
                )
        self._normalize_placeholder_sequence_rank(batch, ACTION)
        self._normalize_placeholder_sequence_rank(batch, OBSERVATION_STATE)
        return batch

    def _ensure_common_sample_keys(self, sample: dict) -> dict:
        if "answer_text" not in sample:
            sample["answer_text"] = torch.zeros(DEFAULT_MAX_ANSWER_BYTES, dtype=torch.uint8)
        return sample

    @property
    def num_frames(self):
        return sum(self.dataset_lengths)

    def _profile_child_label(self, idx: int) -> str:
        dataset = self.datasets[idx]
        base = getattr(dataset, "dataset", dataset)
        name = getattr(base, "repo_id", None)
        if name is None:
            name = getattr(base, "_manifest_path", None)
        if name is None:
            name = type(base).__name__
        return f"{idx}:{name}"

    def __iter__(self) -> Iterator:
        iterators = [iter(ds) for ds in self.datasets]
        weights = self.dataset_weights

        def _next(i):
            try:
                return next(iterators[i])
            except StopIteration:
                iterators[i] = iter(self.datasets[i])
                return next(iterators[i])

        if not self.yields_batches:
            # Legacy mode: yield one sample per __next__; DataLoader handles collation.
            # Profile-mode: time per-stage cost in 30s windows when env
            # RHO_PROFILE_MIXER=1.
            import os
            import time as _time

            profile = os.environ.get("RHO_PROFILE_MIXER", "0") == "1"
            t_next_total = 0.0
            t_pad_total = 0.0
            child_stats: dict[int, list[float]] = {}
            n_yields = 0
            window_start = _time.perf_counter()
            while True:
                idx = random.choices(range(len(iterators)), weights=weights, k=1)[0]
                burst = self._bursts[idx]
                for _ in range(burst):
                    if profile:
                        t0 = _time.perf_counter()
                        sample = _next(idx)
                        t1 = _time.perf_counter()
                        out = self.padding_transform(sample)
                        t2 = _time.perf_counter()
                        t_next_total += t1 - t0
                        t_pad_total += t2 - t1
                        n_yields += 1
                        stats = child_stats.setdefault(idx, [0.0, 0.0, 0.0])
                        stats[0] += 1
                        stats[1] += t1 - t0
                        stats[2] = max(stats[2], t1 - t0)
                        now = _time.perf_counter()
                        if now - window_start >= 30.0:
                            elapsed = now - window_start
                            child_summary = ", ".join(
                                f"{self._profile_child_label(child_idx)} n={int(stats[0])} "
                                f"next={stats[1]:.2f}s max={stats[2]:.2f}s"
                                for child_idx, stats in sorted(
                                    child_stats.items(), key=lambda item: item[1][1], reverse=True
                                )[:8]
                            )
                            logger.info(
                                "[mixer-profile, legacy] %ss window: %d samples, "
                                "next=%.2fs (%.1fms/sample), padding=%.2fs (%.1fms/sample), "
                                "other=%.2fs, top_children=[%s]",
                                f"{elapsed:.1f}",
                                n_yields,
                                t_next_total,
                                1000 * t_next_total / max(n_yields, 1),
                                t_pad_total,
                                1000 * t_pad_total / max(n_yields, 1),
                                elapsed - t_next_total - t_pad_total,
                                child_summary,
                            )
                            t_next_total = t_pad_total = 0.0
                            child_stats = {}
                            n_yields = 0
                            window_start = now
                        yield self._ensure_common_sample_keys(out)
                    else:
                        yield self._ensure_common_sample_keys(self.padding_transform(_next(idx)))
        else:
            # Per-child burst mode: collect bursts[idx] samples from the chosen
            # child and yield them as a pre-collated batch. DataLoader runs with
            # batch_size=None so each yield IS the batch. This is the only way
            # to support different batch sizes per modality (robot=64, VL=8)
            # without batching at the DataLoader level (which uses a single
            # fixed batch_size).
            import os
            import time as _time

            from torch.utils.data import default_collate

            profile = os.environ.get("RHO_PROFILE_MIXER", "0") == "1"
            t_next_total = 0.0
            t_pad_total = 0.0
            t_collate_total = 0.0
            child_stats: dict[int, list[float]] = {}
            n_batches = 0
            n_samples = 0
            window_start = _time.perf_counter()
            while True:
                idx = random.choices(range(len(iterators)), weights=weights, k=1)[0]
                burst = self._bursts[idx]
                if profile:
                    t0 = _time.perf_counter()
                    raw_samples = [_next(idx) for _ in range(burst)]
                    t1 = _time.perf_counter()
                    samples = [self.padding_transform(s) for s in raw_samples]
                    t2 = _time.perf_counter()
                    out = default_collate(samples)
                    t3 = _time.perf_counter()
                    t_next_total += t1 - t0
                    t_pad_total += t2 - t1
                    t_collate_total += t3 - t2
                    n_batches += 1
                    n_samples += burst
                    stats = child_stats.setdefault(idx, [0.0, 0.0, 0.0, 0.0])
                    stats[0] += 1
                    stats[1] += burst
                    stats[2] += t1 - t0
                    stats[3] = max(stats[3], t1 - t0)
                    now = _time.perf_counter()
                    if now - window_start >= 30.0:
                        elapsed = now - window_start
                        child_summary = ", ".join(
                            f"{self._profile_child_label(child_idx)} batches={int(stats[0])} "
                            f"samples={int(stats[1])} next={stats[2]:.2f}s max={stats[3]:.2f}s"
                            for child_idx, stats in sorted(
                                child_stats.items(), key=lambda item: item[1][2], reverse=True
                            )[:8]
                        )
                        logger.info(
                            "[mixer-profile] %ss window: %d batches (%d samples), "
                            "next=%.2fs (%.1fms/sample), padding=%.2fs (%.1fms/sample), "
                            "collate=%.2fs (%.1fms/batch), other=%.2fs, top_children=[%s]",
                            f"{elapsed:.1f}",
                            n_batches,
                            n_samples,
                            t_next_total,
                            1000 * t_next_total / max(n_samples, 1),
                            t_pad_total,
                            1000 * t_pad_total / max(n_samples, 1),
                            t_collate_total,
                            1000 * t_collate_total / max(n_batches, 1),
                            elapsed - t_next_total - t_pad_total - t_collate_total,
                            child_summary,
                        )
                        t_next_total = t_pad_total = t_collate_total = 0.0
                        child_stats = {}
                        n_batches = n_samples = 0
                        window_start = now
                    yield self._ensure_common_batch_keys(out)
                else:
                    samples = [self.padding_transform(_next(idx)) for _ in range(burst)]
                    yield self._ensure_common_batch_keys(default_collate(samples))


class TrainingModeOverrideIterableWrapper(torch.utils.data.IterableDataset):
    """Overwrite training_mode on samples produced by an iterable child."""

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        training_mode: TrainingMode,
        dataset_index: int | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.training_mode = training_mode
        self.dataset_index = dataset_index
        self.num_frames = getattr(dataset, "num_frames", 0)
        self.yields_batches = getattr(dataset, "yields_batches", False)

    def _like_existing(self, existing, value: int):
        if torch.is_tensor(existing):
            return torch.full_like(existing, value)
        if isinstance(existing, list):
            return [value] * len(existing)
        return value

    def __iter__(self):
        mode_value = int(self.training_mode.value)
        for sample in self.dataset:
            sample = dict(sample)
            sample["training_mode"] = self._like_existing(sample.get("training_mode"), mode_value)
            if self.dataset_index is not None:
                sample["dataset_index"] = self._like_existing(sample.get("dataset_index"), self.dataset_index)
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
        # Consume each child's pending resume offset now, including children
        # whose quota is exhausted, so it cannot leak into the next pass.
        iters = [iter(sampler) for sampler in self.samplers]
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
                    chosen_dataset_idx = active[int(np.argmax(probs))]

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
            iters = [iter(sampler) for sampler in self.samplers]
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
            "rng_state": self._rng.bit_generator.state,
            "yielded": list(self.yielded),
            "epoch": self._epoch,
            "sampler_states": sampler_states,
        }

    def load_state(self, state_dict: dict) -> None:
        """Restore sampler state from a previously saved snapshot."""
        if state_dict["rng_state"] is None:
            logger.warning(
                "Sampler checkpoint has no usable RNG state; dataset selection cannot resume exactly."
            )
        else:
            self._rng.bit_generator.state = state_dict["rng_state"]
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
    # Enumerate concrete types explicitly. Draccus won't dispatch via a
    # ChoiceRegistry (`DataConfig`) inside a union with another dataclass
    # (`MultiDatasetConfig`): it falls back to the registry's default choice
    dataset: DataConfig
    weight: float = 1


@DataConfig.register_subclass("multi")
@dataclass
class MultiDatasetConfig(DataConfig):
    """
    Configuration for a multi-dataset setup.
    """

    datasets: list[WeightedDatasetConfig]  # This is the field we expect to read from the YAML file

    # This is expected to be overwritten in the code by breaking apart the datasets list
    # into individual DataConfig objects and their corresponding weights.
    dataset_cfgs: list[DataConfig] = None
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
    action_time_horizon_s: float | None = None
    min_action_chunk_size: int | None = None

    # Sampling currently requires flattened nested MultiDatasetConfigs.
    flatten_nested: bool = True

    # This option is useful during pretraining to shrink the size of the saved config file.
    serialize_stats: bool = True  # Whether to serialize stats when converting to dict
    # When set, every flattened leaf's training_mode is replaced with this value
    # at dataset-build time, overriding the per-child configured training_mode.
    # Lets the same dataset mix be re-used with a different mode label without
    # editing each leaf config. None preserves each child's configured mode.
    training_mode_override: TrainingMode | None = None

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
                        # Use the inner Multi's already-FLATTENED weights so
                        # multi-level nesting (cotraining -> VL bundle ->
                        # refspatial sub-mix) lines up with dataset_cfgs,
                        # which is also recursively flattened above. Earlier
                        # this iterated d.dataset.datasets (shallow), which
                        # produced len(weights) < len(dataset_cfgs) whenever
                        # any inner Multi was itself nested.
                        inner_weights = np.array(d.dataset.weights, dtype=np.float64)
                        normalized_weights = inner_weights / inner_weights.sum()
                        self.weights.extend([d.weight * w for w in normalized_weights])
                    else:
                        self.weights.append(d.weight)
            else:
                self.weights = [d.weight for d in self.datasets]

        if len(self.dataset_cfgs) != len(self.weights):
            raise ValueError("dataset_cfgs and weights must have the same length")
        if len(self.dataset_cfgs) == 0:
            raise ValueError("At least one dataset must be provided")

        # Normalize training_mode_override: draccus on a `TrainingMode | None`
        # union sometimes leaves the YAML value as a raw string ("VQA") or
        # int (4) without dispatching to the enum decoder. Convert defensively
        # so downstream `is not None` checks and `.value` accesses both work.
        if self.training_mode_override is not None and not isinstance(
            self.training_mode_override, TrainingMode
        ):
            if isinstance(self.training_mode_override, str):
                self.training_mode_override = TrainingMode[self.training_mode_override]
            elif isinstance(self.training_mode_override, int):
                self.training_mode_override = TrainingMode(self.training_mode_override)
            else:
                raise TypeError(
                    f"training_mode_override must be a TrainingMode, str, or int; "
                    f"got {type(self.training_mode_override).__name__}"
                )
        if self.training_mode_override is not None:
            logger.info(
                "MultiDatasetConfig: training_mode_override=%s will be applied to all leaves",
                self.training_mode_override.name,
            )

        parent_features_explicit = self.features is not None
        if self.features is None:
            self.features = self.dataset_cfgs[0].features

        # An inner MultiDatasetConfig (e.g. a VL sub-bundle) may have no
        # features declared and rely on the parent to push them down later
        # (see the for-child loop below in this same __post_init__). Skip
        # the dict-to-PolicyFeature conversion in that case.
        if self.features is not None:
            for k, v in self.features.items():
                if isinstance(v, dict) and "shape" in v and "type" in v:
                    # Convert dict to PolicyFeature if necessary
                    if isinstance(v["shape"], str):
                        v["shape"] = tuple([int(vi) for vi in v["shape"].strip("()").split(",") if vi != ""])
                    self.features[k] = PolicyFeature(v["type"], v["shape"])

        # transformed_features falls back AFTER the dict->PolicyFeature
        # conversion so children receive PolicyFeature objects, not raw dicts.
        if self.transformed_features is None:
            if parent_features_explicit and self.features is not None:
                self.transformed_features = copy.deepcopy(self.features)
            else:
                child_transformed_features = [
                    getattr(dataset_cfg, "transformed_feature_dict", None)
                    or getattr(dataset_cfg, "feature_dict", None)
                    for dataset_cfg in self.dataset_cfgs
                ]
                child_transformed_features = [f for f in child_transformed_features if f is not None]
                if child_transformed_features:
                    first_child_features = child_transformed_features[0]
                    if all(
                        _feature_specs_equal(first_child_features, other_features)
                        for other_features in child_transformed_features[1:]
                    ):
                        self.transformed_features = copy.deepcopy(first_child_features)
                    else:
                        raise ValueError(
                            "MultiDatasetConfig has heterogeneous transformed_features across children. "
                            "Set parent features explicitly so the policy schema is not inferred from the "
                            "first child dataset."
                        )
        if self.transformed_features is None and self.features is not None:
            self.transformed_features = copy.deepcopy(self.features)

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
        if self.action_time_horizon_s is not None:
            self.set_attribute("action_time_horizon_s", self.action_time_horizon_s)
        if self.min_action_chunk_size is not None:
            self.set_attribute("min_action_chunk_size", self.min_action_chunk_size)

        if self.shuffle_buffer_size is not None:
            self.set_attribute("shuffle_buffer_size", self.shuffle_buffer_size)

        self.set_attribute("serialize_stats", self.serialize_stats)

        # Push parent-level features down to children that don't define their
        # own so nested mixes remain usable for get_action_denormalization /
        # create_stats_buffers when selected as the eval sub-config.
        for child in self.dataset_cfgs:
            if self.features is not None and getattr(child, "features", None) is None:
                child.features = copy.deepcopy(self.features)
            if self.transformed_features is not None and getattr(child, "transformed_features", None) is None:
                child.transformed_features = copy.deepcopy(self.transformed_features)

    def create_delta_transform(self, perdim_delta_actions: bool):
        for dataset_cfg in self.dataset_cfgs:
            if isinstance(dataset_cfg, MultiDatasetConfig):
                dataset_cfg.create_delta_transform(perdim_delta_actions)
            elif hasattr(dataset_cfg, "transform_mapping"):
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
        result = {"type": self.type}
        for f in dc_fields(self):
            if f.name in _exclude:
                continue
            value = serialize_to_dict(getattr(self, f.name))
            if value is not None:
                result[f.name] = value
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

    def make_dataset(self, policy_cfg: PolicyConfig = None) -> CombinedDataset:
        """Build a combined map-style dataset from the configured children."""
        children = [cfg.make_dataset(policy_cfg) for cfg in self.dataset_cfgs]
        if any(isinstance(child, torch.utils.data.IterableDataset) for child in children):
            raise TypeError("Public MultiDatasetConfig requires map-style child datasets")
        training_modes = [
            self.training_mode_override
            or getattr(cfg, "training_mode", TrainingMode.ROBOT_FLOWMATCH)
            for cfg in self.dataset_cfgs
        ]
        return CombinedDataset(children, training_modes=training_modes, features=self.features)

    def make_sampler(
        self,
        dataset: CombinedDataset | WeightedIterableMixDataset,
        policy_cfg: PolicyConfig = None,
    ) -> MultiDatasetWeightedSampler | None:
        # IterableDatasets can't use PyTorch samplers — interleaving is done
        # inside WeightedIterableMixDataset itself.
        if isinstance(dataset, torch.utils.data.IterableDataset):
            return None

        assert isinstance(dataset, CombinedDataset), (
            "Expected dataset to be CombinedDataset or WeightedIterableMixDataset for MultiDatasetConfig"
        )
        samplers = []
        for dataset_cfg, individual_dataset in zip(self.dataset_cfgs, dataset.datasets, strict=False):
            new_sampler = dataset_cfg.make_sampler(individual_dataset, policy_cfg)
            if new_sampler is not None:
                samplers.append(new_sampler)
        if len(samplers) == 0:
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

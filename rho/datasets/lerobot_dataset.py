import logging
import os  # noqa: I001
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import torch
from datasets import load_dataset
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features

from rho.common.constants import ACTION, OBSERVATION_PREFIX, OBSERVATION_TACTILE
from rho.common.task_encoding import encode_task_bytes
from rho.common.transforms import build_key_padding_transform, get_target_sequence_lengths
from rho.common.types import TrainingMode
from rho.datasets.data_config import DataConfig, RobotDataConfig

if TYPE_CHECKING:
    from rho.policies.base import PolicyConfig

logger = logging.getLogger(__name__)


NEXT_REWARD_KEY = "next.reward"


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Supports save_state() / load_state() for deterministic resumption.
        When shuffle=True, a dedicated torch.Generator seeded with (seed + epoch)
        is used so that the same permutation can be reproduced on resume.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            shuffle: Whether to shuffle the indices.
            seed: Base seed used for deterministic shuffling (combined with epoch).
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index, end_index))

        self.indices = indices
        self.shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        # Number of samples already yielded in the current epoch.
        # Non-zero only after load_state() to fast-forward on the next __iter__ call.
        self._start_offset = 0

    def _get_order(self) -> list[int] | torch.Tensor:
        """Return the iteration order for the current epoch."""
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._epoch)
            return torch.randperm(len(self.indices), generator=generator)
        return list(self.indices)

    def __iter__(self) -> Iterator[int]:
        order = self._get_order()

        # Fast-forward past samples that were already yielded before a checkpoint.
        start = self._start_offset
        self._start_offset = 0  # consumed — reset so subsequent iters start from 0

        if isinstance(order, torch.Tensor):
            for i in range(start, len(order)):
                yield self.indices[int(order[i])]
        else:
            for i in range(start, len(order)):
                yield order[i]

        self._epoch += 1

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def save_state(self) -> dict:
        """Return a serialisable snapshot of the sampler state.

        The returned dict captures enough information to resume iteration
        from the exact same position via load_state().
        """
        return {
            "indices": self.indices,
            "shuffle": self.shuffle,
            "seed": self._seed,
            "epoch": self._epoch,
        }

    def load_state(self, state_dict: dict) -> None:
        """Restore sampler state from a previously saved snapshot.

        After calling load_state(), the next call to __iter__() will
        reproduce the exact same ordering and skip samples that had
        already been yielded.
        """
        self.indices = state_dict["indices"]
        self.shuffle = state_dict["shuffle"]
        self._seed = state_dict["seed"]
        self._epoch = state_dict["epoch"]


def metadata_from_lerobot_dataset(
    repo_id: str, root: str, observation_mapping: dict[str, str] = None
) -> LeRobotDatasetMetadata:
    """Extract feature statistics from a LeRobotDataset instance."""
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    if observation_mapping is not None:
        # Remap observation keys according to the mapping
        ds_meta.info.features = {
            observation_mapping.get(k, k): v for k, v in ds_meta.info.features.items()
        }
        ds_meta.stats = {observation_mapping.get(k, k): v for k, v in ds_meta.stats.items()}
    return ds_meta


def rho_features_from_lerobot_dataset(
    repo_id: str,
    root: str,
    normalization_mapping: dict[str, str] = None,
    observation_mapping: dict[str, str] = None,
) -> dict:
    """Create feature dict and stats from a LeRobotDataset instance.

    Returns a dict with 'features', 'stats', and 'normalization_mapping' keys
    that can be used to populate a LeRobotDatasetConfig.
    """

    assert normalization_mapping is not None, "Normalization mapping must be provided"

    ds_meta = metadata_from_lerobot_dataset(repo_id, root, observation_mapping=observation_mapping)
    features = dataset_to_policy_features(ds_meta.info.features)
    return {
        "features": features,
        "stats": ds_meta.stats,
        "normalization_mapping": normalization_mapping,
    }


def resolve_delta_timestamps(
    cfg: "PolicyConfig",  # noqa: F821
    ds_meta: "LeRobotDatasetMetadata",  # noqa: F821
    action_delta_indices: list[int] | None = None,
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PolicyConfig.

    Args:
        cfg (PolicyConfig): The PolicyConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == NEXT_REWARD_KEY and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key.startswith(ACTION):
            indices = action_delta_indices if action_delta_indices is not None else cfg.action_delta_indices
            if indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in indices]
        if key.startswith(OBSERVATION_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        if key == OBSERVATION_TACTILE and cfg.tactile_observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.tactile_observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def resolve_action_delta_indices_for_dataset(
    cfg: "PolicyConfig",  # noqa: F821
    ds_meta: "LeRobotDatasetMetadata",  # noqa: F821
    action_time_horizon_s: float | None,
    min_action_chunk_size: int,
) -> list[int] | None:
    if cfg.action_delta_indices is None:
        return None
    if action_time_horizon_s is None:
        return cfg.action_delta_indices
    if action_time_horizon_s <= 0:
        raise ValueError(f"action_time_horizon_s must be positive, got {action_time_horizon_s}")
    if min_action_chunk_size <= 0:
        raise ValueError(f"min_action_chunk_size must be positive, got {min_action_chunk_size}")

    max_chunk = int(getattr(cfg, "chunk_size", len(cfg.action_delta_indices)))
    real_chunk = int(round(float(ds_meta.fps) * action_time_horizon_s))
    real_chunk = max(min_action_chunk_size, min(real_chunk, max_chunk))
    return list(range(real_chunk))


class TransformedLeRobotDataset(LeRobotDataset):
    """
    A transformed version of the LeRobotDataset that applies a transformation to each sample.

    Args:
        repo_id: The repository ID of the dataset
        data_transforms: Optional callable to apply transformations to the data.
            The data_transforms callable includes both the normalization and any additional transforms.
            In most cases it is populated using DataConfig.get_transforms().
        split_name: The name of the dataset split to use (e.g., "train",
    """

    def __init__(
        self,
        repo_id: str,
        data_transforms: Callable | None = None,
        padding_transform: Callable | None = None,
        split_name="train",
        observation_blacklist=None,
        training_mode: TrainingMode = TrainingMode.ROBOT_FLOWMATCH,
        **kwargs,
    ):
        self.split_name = split_name
        self.data_transforms = data_transforms
        self.padding_transform = padding_transform or (lambda x: x)
        self.observation_blacklist = observation_blacklist
        self.training_mode = training_mode
        super().__init__(repo_id=repo_id, **kwargs)

    def _check_cached_episodes_sufficient(self) -> bool:
        """Avoid an exhaustive local video existence scan at dataset init.

        Upstream LeRobot checks every expected video file for every episode
        after loading parquet data. Large local robot datasets can have tens
        of thousands of episodes, making startup perform hundreds of thousands
        of NFS stat calls per rank before training begins. We still verify the
        parquet episode coverage here; missing videos will surface at sample
        decode time with the concrete path.
        """
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }
        requested_episodes = (
            set(range(self.meta.total_episodes)) if self.episodes is None else set(self.episodes)
        )
        return requested_episodes.issubset(available_episodes)

    def load_hf_dataset(self) -> datasets.Dataset:
        if self.observation_blacklist is not None:
            for key in self.observation_blacklist:
                logger.debug(f"Removing blacklisted observation key: {key}")
                if key in self.features:
                    del self.features[key]
                if self.delta_indices is not None and key in self.delta_indices:
                    del self.delta_indices[key]
                if self.delta_timestamps is not None and key in self.delta_timestamps:
                    del self.delta_timestamps[key]

        return super().load_hf_dataset()

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        logger.debug(f"Getting sample from {self.root} at index {index}")
        sample = self.data_transforms(sample) if self.data_transforms is not None else sample
        sample = self.padding_transform(sample)
        # Stamp training_mode so a standalone LeRobotDatasetConfig (validation
        # dataset, action-monitor dataloader) routes through the right policy
        # head. Without this, samples lack the key and policy.forward defaults
        # to ROBOT_FLOWMATCH -- which crashes KI / VL-only policies whose
        # training_modes don't include it.
        sample["training_mode"] = self.training_mode.value
        # Encode `task` to a uint8 byte tensor so distributed collation paths
        # do not have to carry Python strings. Mirrors the same encoding done
        # in MapStyleIterableWrapper.
        if "task" in sample and not isinstance(sample["task"], torch.Tensor):
            raw = sample["task"]
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            sample["task"] = encode_task_bytes(raw)
        return sample


def _decode_with_retry(dataset, idx: int, dataset_index: int, max_retries: int = 100):
    """Fetch ``dataset[idx]`` with retry around corrupted MP4 frames. Returns
    the raw sample dict (pre-stamp, pre-pad) or None if all retries fail.
    """
    dataset_len = max(len(dataset), 1)
    cur_idx = idx
    for attempt in range(max_retries):
        try:
            return dataset[cur_idx]
        except Exception as e:
            logger.warning(
                "MapStyleIterableWrapper: error fetching idx=%d (dataset_index=%d, attempt %d/%d): %s",
                cur_idx,
                dataset_index,
                attempt + 1,
                max_retries,
                e,
            )
            cur_idx = (cur_idx + 1) % dataset_len
    logger.error(
        "MapStyleIterableWrapper: gave up after %d retries starting from idx=%d (dataset_index=%d); skipping",
        max_retries,
        idx,
        dataset_index,
    )
    return None


class MapStyleIterableWrapper(torch.utils.data.IterableDataset):
    """Wrap a map-style dataset + sampler as an IterableDataset.

    Used when `MultiDatasetConfig` dispatches to `WeightedIterableMixDataset`
    for co-training mixes. Map-style children get wrapped here so the mixer can
    interleave them while preserving dataset_index and training_mode metadata.

    Iterates the sampler indefinitely (loops across epochs), stamps `dataset_index`
    and `training_mode` onto each sample, and applies an optional padding transform.

    Pure LeRobot training still uses normal PyTorch DataLoader workers. This
    wrapper is only for iterable co-training mixes where children need
    homogeneous training-mode batches.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sampler,
        training_mode: TrainingMode = TrainingMode.ROBOT_FLOWMATCH,
        dataset_index: int = 0,
        padding_transform: Callable | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.training_mode = training_mode
        self.dataset_index = dataset_index
        self.padding_transform = padding_transform or (lambda x: x)
        self.features = getattr(dataset, "features", None)
        self.delta_timestamps = getattr(dataset, "delta_timestamps", None)
        self.target_sequence_lengths = getattr(dataset, "target_sequence_lengths", None)
        self.num_frames = len(dataset)

    def _stamp_and_pad(self, sample):
        """Stamp dataset_index / training_mode, encode ``task`` to bytes, and
        apply the padding transform.
        """
        sample["dataset_index"] = self.dataset_index
        sample["training_mode"] = self.training_mode.value
        # Normalize ``task`` to a uint8 byte tensor. LeRobotDataset typically
        # emits ``task`` as str or list[str] (via append_task); tolerate both
        # plus an already-encoded tensor.
        if "task" in sample and not isinstance(sample["task"], torch.Tensor):
            raw = sample["task"]
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            sample["task"] = encode_task_bytes(raw)
        return self.padding_transform(sample)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Retry around per-sample fetches: LeRobot's video decoder occasionally
        # hits corrupted MP4 frames ("Invalid data found when processing
        # input"). A single bad frame must not crash the run.
        while True:
            indices = iter(self.sampler) if self.sampler is not None else iter(range(len(self.dataset)))
            for idx in indices:
                sample = _decode_with_retry(self.dataset, idx, self.dataset_index)
                if sample is None:
                    continue
                yield self._stamp_and_pad(sample)


class TransformedStreamingLeRobotDataset(StreamingLeRobotDataset):
    """Streaming LeRobot dataset with rho transforms and tensor task encoding."""

    def __init__(
        self,
        repo_id: str,
        data_transforms: Callable | None = None,
        padding_transform: Callable | None = None,
        observation_blacklist: list[str] | None = None,
        split_name="train",
        training_mode: TrainingMode = TrainingMode.ROBOT_FLOWMATCH,
        **kwargs,
    ):
        self.split_name = split_name
        self.data_transforms = data_transforms
        self.padding_transform = padding_transform or (lambda x: x)
        self.observation_blacklist = observation_blacklist
        self.training_mode = training_mode

        super().__init__(repo_id=repo_id, **kwargs)

        if self.observation_blacklist is not None:
            for key in self.observation_blacklist:
                if key in self.meta.features:
                    del self.meta.features[key]
                if self.delta_timestamps is not None and key in self.delta_timestamps:
                    del self.delta_timestamps[key]
                    del self.delta_indices[key]

            features = get_hf_features_from_features(self.meta.features)
            self.hf_dataset = load_dataset(
                self.repo_id if not self.streaming_from_local else str(self.root),
                split="train",
                streaming=self.streaming,
                data_files="data/*/*.parquet",
                revision=self.revision,
                features=features,
            )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for sample in super().__iter__():
            sample = self.data_transforms(sample) if self.data_transforms else sample
            sample = self.padding_transform(sample)
            sample["training_mode"] = self.training_mode.value
            if "task" in sample and not isinstance(sample["task"], torch.Tensor):
                raw = sample["task"]
                if isinstance(raw, list):
                    raw = raw[0] if raw else ""
                sample["task"] = encode_task_bytes(raw)
            yield sample


@DataConfig.register_subclass("lerobot")
@dataclass
class LeRobotDatasetConfig(RobotDataConfig):
    """DatasetConfig for LeRobotDataset datasets.

    Args:
        repo_id: The repository ID of the dataset.
        root: The root directory where the dataset is stored.
        normalization_mapping: A mapping from feature keys to NormalizationMode values.
        observation_mapping: An optional mapping to rename observation keys.
    """

    repo_id: str = "lerobot/pusht"  # The dataset repository ID
    root_dir: str | Path | None = None  # Local dataset root directory
    streaming: bool = False

    # Tolerance for video frame timings
    tolerance_s: float = 0.1  # 1e-4

    # Data preprocessing options
    use_imagenet_stats: bool = False

    split_name: str = "train"  # Default split name to use

    prefetch_factor: int = 2  # Number of batches to prefetch

    # If set, only the first `max_episodes` episodes are sampled from. Episode
    # indices 0..max_episodes-1 are used; later episodes are ignored. Used for
    # data-scaling curves where we want to train on a subset of demonstrations
    # without duplicating the dataset on PVC.
    max_episodes: int | None = None

    def __post_init__(self):
        if self.root_dir is not None:
            self.root_dir = os.path.expandvars(self.root_dir)
            # Check for the existence of root directory, if not there recursively check its parents existence
            logger.info(f"Checking existence of root directory: {self.root_dir}")
            if not Path(self.root_dir).exists():
                parent = Path(self.root_dir).parent
                while parent != parent.parent:
                    if parent.exists():
                        logger.warning(
                            f"Root directory {self.root_dir} does not exist, last parent is {parent}"
                        )
                        break
                    parent = parent.parent
                else:
                    raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")

        self.ds_meta = None
        # Handle features initialization - create default if not provided
        if self.features is None:
            # Create default features from dataset if not provided
            if self.ds_meta is None:
                ds_meta = metadata_from_lerobot_dataset(
                    repo_id=self.repo_id,
                    root=self.root_dir,
                    observation_mapping=self.observation_mapping,
                )

            if self.features is None:
                self.features = dataset_to_policy_features(ds_meta.info.features)

        if self.stats is None:
            if self.ds_meta is None:
                self.ds_meta = metadata_from_lerobot_dataset(
                    repo_id=self.repo_id,
                    root=self.root_dir,
                    observation_mapping=self.observation_mapping,
                )
            self.stats = self.ds_meta.stats

        if self.stats is None:
            raise ValueError("Stats must be provided or inferred from the dataset.")

        super().__post_init__()

    def set_attribute(self, attr_name: str, value) -> None:
        """Set an attribute for the DatasetConfig."""
        setattr(self, attr_name, value)

    def get_length(self) -> int | None:
        """Get the total number of frames in the dataset."""
        if self.ds_meta is not None:
            return self.ds_meta.total_frames
        return None

    def make_dataset(self, policy_cfg: "PolicyConfig" = None) -> torch.utils.data.Dataset:  # noqa: F821
        """Create a LeRobotDataset instance based on the configuration.

        Args:
            policy_cfg: Optional PolicyConfig for additional configurations.

        Returns:
            LeRobotDataset: Configured LeRobotDataset instance.
        """
        ds_meta = LeRobotDatasetMetadata(self.repo_id, root=self.root_dir)
        logger.info(f"Creating dataset with repo_id: {self.repo_id}, root_dir: {self.root_dir}")
        self.ds_meta = ds_meta
        if self.observation_mapping is not None:
            # Remap observation keys according to the mapping
            ds_meta.info.features = {
                self.observation_mapping.get(k, k): v for k, v in ds_meta.info.features.items()
            }
            ds_meta.stats = {self.observation_mapping.get(k, k): v for k, v in ds_meta.stats.items()}

        delta_timestamps = None
        target_sequence_lengths = None

        if policy_cfg is not None:
            logger.debug("Policy config provided, using resolved delta timestamps")
            policy_cfg.robot_hz = ds_meta.fps
            action_delta_indices = resolve_action_delta_indices_for_dataset(
                policy_cfg,
                ds_meta,
                self.action_time_horizon_s,
                self.min_action_chunk_size,
            )
            delta_timestamps = resolve_delta_timestamps(
                policy_cfg,
                ds_meta,
                action_delta_indices=action_delta_indices,
            )
            # The full policy schedule is immediately reduced to a fixed shape
            # contract. Only ``delta_timestamps`` is retained as a timestamp
            # schedule and passed to LeRobot for data loading.
            target_sequence_lengths = get_target_sequence_lengths(
                resolve_delta_timestamps(policy_cfg, ds_meta)
            )
            # We need the delta_timesteps to be in the format of the dataset, rather than the policy
            if self.observation_mapping is not None:
                reverse_mapping = {v: k for k, v in self.observation_mapping.items()}
                if delta_timestamps is not None:
                    delta_timestamps = {reverse_mapping.get(k, k): v for k, v in delta_timestamps.items()}

            # Pass chunk_size from policy to features for action chunk normalization
            if hasattr(policy_cfg, "chunk_size") and self.features is not None:
                self.chunk_size = policy_cfg.chunk_size
        else:
            logger.debug("Policy config is None, using default delta timestamps")

        # KI-ENDSTATE physical/RPY ablation: when enabled on the policy, append
        # an EndStateTarget transform so each sample carries an un-normalized
        # chunk-final action for the KI CE head. Default config leaves this off.
        # Pad to max_action_dim (the uniform action ceiling) so collate works
        # across mixed-dataset robot batches; the KI head slices back to the
        # real representation dim.
        endstate_target = None
        language_action_target = None
        training_modes = getattr(policy_cfg, "training_modes", []) if policy_cfg is not None else []
        has_ki_endstate = any(
            mode == TrainingMode.ROBOT_KNOWLEDGE_INSULATION_ENDSTATE
            or mode == TrainingMode.ROBOT_KNOWLEDGE_INSULATION_ENDSTATE.name
            or mode == int(TrainingMode.ROBOT_KNOWLEDGE_INSULATION_ENDSTATE)
            for mode in training_modes
        )
        if (
            policy_cfg is not None
            and has_ki_endstate
            and getattr(policy_cfg, "endstate_target_format", "language_action") == "language_action"
        ):
            language_action_target = {
                "chunk_reduction": getattr(policy_cfg, "language_action_chunk_reduction", "last"),
                "include_rotation": getattr(policy_cfg, "language_action_include_rotation", True),
                "eef_frame_prob": getattr(policy_cfg, "language_action_eef_frame_prob", 0.5),
            }
        elif policy_cfg is not None and (
            getattr(policy_cfg, "endstate_target_units", "normalized") == "physical"
        ):
            rotation = getattr(policy_cfg, "endstate_target_rotation", "6d")
            endstate_target = (rotation, policy_cfg.max_action_dim)

        dataset_kwargs = {
            "repo_id": self.repo_id,
            "data_transforms": self.get_transforms(
                endstate_target=endstate_target,
                language_action_target=language_action_target,
            ),
            "padding_transform": build_key_padding_transform(
                features=self.features,
                target_sequence_lengths=target_sequence_lengths,
            ),
            "root": self.root_dir,
            "delta_timestamps": delta_timestamps,
            "split_name": self.split_name,
            "tolerance_s": self.tolerance_s,
            "observation_blacklist": self.observation_blacklist,
            "training_mode": self.training_mode,
        }

        if self.streaming:
            dataset_kwargs["buffer_size"] = self.shuffle_buffer_size
            dataset = TransformedStreamingLeRobotDataset(**dataset_kwargs)
        else:
            dataset = TransformedLeRobotDataset(**dataset_kwargs)
        dataset.target_sequence_lengths = target_sequence_lengths

        if self.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

        return dataset

    def make_sampler(
        self,
        dataset: LeRobotDataset,
        policy_cfg: "PolicyConfig" = None,  # noqa: F821
    ) -> torch.utils.data.DataLoader | None:
        """Create a sampler for the dataset if needed.

        Args:
            dataset: The LeRobotDataset instance.
            policy_cfg: Optional PolicyConfig for additional configurations.

        Returns:
            torch.utils.data.DataLoader | None: Configured sampler or None if not needed.
        """
        if not self.streaming and self.max_episodes is not None:
            episode_indices_to_use = list(range(self.max_episodes)) if self.max_episodes is not None else None
            if episode_indices_to_use is not None:
                logger.info(
                    f"Capping training to first {self.max_episodes} episodes (out of "
                    f"{len(dataset.meta.episodes['dataset_from_index'])} available)."
                )
            # create dataloader for offline training
            sampler = EpisodeAwareSampler(
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=episode_indices_to_use,
                shuffle=True,
            )
        else:
            sampler = None

        return sampler

    def get_contributions(self):
        root_part = Path(self.root_dir).name if self.root_dir else "default"
        name = f"{self.repo_id}@{root_part}"
        length = self.get_length()

        return {"name": name, "length": length, "weighted_length": length, "child_contributions": []}

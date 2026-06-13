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
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.datasets.feature_utils import get_hf_features_from_features

from rho.common.constants import ACTION, OBSERVATION_PREFIX, OBSERVATION_TACTILE
from rho.common.types import FeatureType, NormalizationMode
from rho.datasets.data_config import DataConfig

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
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
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
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Base seed used for deterministic shuffling (combined with epoch).
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        # Number of samples already yielded in the current epoch.
        # Non-zero only after load_state() to fast-forward on the next __iter__ call.
        self._start_offset = 0

    def _get_order(self) -> list[int]:
        """Return the iteration order for the current epoch."""
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._epoch)
            perm = torch.randperm(len(self.indices), generator=generator)
            return [self.indices[i] for i in perm]
        return list(self.indices)

    def __iter__(self) -> Iterator[int]:
        order = self._get_order()

        # Fast-forward past samples that were already yielded before a checkpoint.
        start = self._start_offset
        self._start_offset = 0  # consumed — reset so subsequent iters start from 0

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
        ds_meta.info["features"] = {
            observation_mapping.get(k, k): v for k, v in ds_meta.info["features"].items()
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
    features = dataset_to_policy_features(ds_meta.info["features"])
    return {
        "features": features,
        "stats": ds_meta.stats,
        "normalization_mapping": normalization_mapping,
    }


def resolve_delta_timestamps(
    cfg: "PolicyConfig",  # noqa: F821
    ds_meta: "LeRobotDatasetMetadata",  # noqa: F821
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
        if key.startswith(ACTION) and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBSERVATION_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        if key == OBSERVATION_TACTILE and cfg.tactile_observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.tactile_observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


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
        split_name="train",
        observation_blacklist=None,
        **kwargs,
    ):
        self.split_name = split_name
        self.data_transforms = data_transforms
        self.observation_blacklist = observation_blacklist
        super().__init__(repo_id=repo_id, **kwargs)

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
        return sample


class TransformedStreamingLeRobotDataset(StreamingLeRobotDataset):
    """
    A transformed version of the LeRobotDataset that applies a transformation to each sample.

    Args:
        repo_id: The repository ID of the dataset.
        data_transforms: Optional callable to apply transformations to the data.
        split_name: The name of the dataset split to use (e.g., "train").
    """

    def __init__(
        self,
        repo_id: str,
        data_transforms: Callable | None = None,
        observation_blacklist: list[str] | None = None,
        split_name="train",
        **kwargs,
    ):
        self.split_name = split_name
        self.data_transforms = data_transforms
        self.observation_blacklist = observation_blacklist

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
            yield self.data_transforms(sample) if self.data_transforms else sample


@DataConfig.register_subclass("lerobot")
@dataclass
class LeRobotDatasetConfig(DataConfig):
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
                self.features = dataset_to_policy_features(ds_meta.info["features"])

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

    def needs_delta_actions(self) -> bool:
        actionchunk_modes = [
            NormalizationMode.ACTIONCHUNK_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_QUANTILE,
            NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE,
        ]
        if self.normalization_mapping is not None:
            action_norm_mode = self.normalization_mapping.get(FeatureType.ACTION)
            if action_norm_mode in actionchunk_modes:
                return True
        return False

    def needs_perdim_delta_actions(self) -> bool:
        perdim_actionchunk_modes = [
            NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE,
        ]
        if self.normalization_mapping is not None:
            action_norm_mode = self.normalization_mapping.get(FeatureType.ACTION)
            if action_norm_mode in perdim_actionchunk_modes:
                return True
        return False

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
            ds_meta.info["features"] = {
                self.observation_mapping.get(k, k): v for k, v in ds_meta.info["features"].items()
            }
            ds_meta.stats = {self.observation_mapping.get(k, k): v for k, v in ds_meta.stats.items()}

        delta_timestamps = None

        if policy_cfg is not None:
            logger.debug("Policy config provided, using resolved delta timestamps")
            policy_cfg.robot_hz = ds_meta.fps
            delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
            # We need the delta_timesteps to be in the format of the dataset, rather than the policy
            if self.observation_mapping is not None:
                reverse_mapping = {v: k for k, v in self.observation_mapping.items()}
                delta_timestamps = {reverse_mapping.get(k, k): v for k, v in delta_timestamps.items()}

            # Pass chunk_size from policy to features for action chunk normalization
            if hasattr(policy_cfg, "chunk_size") and self.features is not None:
                self.chunk_size = policy_cfg.chunk_size
        else:
            logger.debug("Policy config is None, using default delta timestamps")

        dataset_kwargs = {
            "repo_id": self.repo_id,
            "data_transforms": self.get_transforms(),
            "root": self.root_dir,
            "delta_timestamps": delta_timestamps,
            "split_name": self.split_name,
            "tolerance_s": self.tolerance_s,
            "observation_blacklist": self.observation_blacklist,
        }

        if self.streaming:
            dataset_kwargs["buffer_size"] = self.shuffle_buffer_size

        if self.streaming:
            dataset = TransformedStreamingLeRobotDataset(**dataset_kwargs)
        else:
            dataset = TransformedLeRobotDataset(**dataset_kwargs)

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
        if not self.streaming and policy_cfg is not None and hasattr(policy_cfg, "drop_n_last_frames"):
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
                drop_n_last_frames=policy_cfg.drop_n_last_frames,
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

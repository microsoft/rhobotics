#!/usr/bin/env python

# It was necessary to adjust this to support more types of normalization
# In particular the quantile outlier normalization


import logging

import numpy as np
import torch
from torch import Tensor, nn

from rho.common.types import FeatureType, NormalizationMode, PolicyFeature

logger = logging.getLogger(__name__)


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
    chunk_size: int | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        # Read shape from stats data if available, otherwise use feature shape
        shape = tuple(ft.shape)

        stat_data = stats[key]["mean"]
        if isinstance(stat_data, (np.ndarray, torch.Tensor)):
            dataset_shape = tuple(stat_data.shape)
        elif isinstance(stat_data, list):
            dataset_shape = (len(stat_data),)

        action_chunk_norm_modes = [
            NormalizationMode.ACTIONCHUNK_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_QUANTILE,
            NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD,
            NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX,
            NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE,
        ]

        if shape != dataset_shape and norm_mode not in action_chunk_norm_modes:
            shape = dataset_shape
            logger.debug(f"Overriding shape for {key} from stats: {shape}")

        if norm_mode in action_chunk_norm_modes:
            assert chunk_size <= 50, f"chunk_size {chunk_size} too large for {key} normalization"

        if ft.type is FeatureType.VISUAL:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.QUANTILE:
            if stats and ("q01" not in stats[key] or "q99" not in stats[key]):
                raise ValueError(f"Quantile normalization requires 'q01' and 'q99' in stats for {key}.")
            q01 = torch.ones(shape, dtype=torch.float32) * torch.inf
            q99 = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "q01": nn.Parameter(q01, requires_grad=False),
                    "q99": nn.Parameter(q99, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.ACTIONCHUNK_MEAN_STD:
            if chunk_size is None:
                raise ValueError(
                    f"chunk_size must be provided for ACTIONCHUNK_MEAN_STD normalization of {key}"
                )
            chunk_mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            chunk_std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    f"mean_chunk{chunk_size}": nn.Parameter(chunk_mean, requires_grad=False),
                    f"std_chunk{chunk_size}": nn.Parameter(chunk_std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.ACTIONCHUNK_MIN_MAX:
            if chunk_size is None:
                raise ValueError(
                    f"chunk_size must be provided for ACTIONCHUNK_MIN_MAX normalization of {key}"
                )
            chunk_min = torch.ones(shape, dtype=torch.float32) * torch.inf
            chunk_max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    f"min_chunk{chunk_size}": nn.Parameter(chunk_min, requires_grad=False),
                    f"max_chunk{chunk_size}": nn.Parameter(chunk_max, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.ACTIONCHUNK_QUANTILE:
            if chunk_size is None:
                raise ValueError(
                    f"chunk_size must be provided for ACTIONCHUNK_QUANTILE normalization of {key}"
                )
            chunk_q02 = torch.ones(shape, dtype=torch.float32) * torch.inf
            chunk_q98 = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    f"q02_chunk{chunk_size}": nn.Parameter(chunk_q02, requires_grad=False),
                    f"q98_chunk{chunk_size}": nn.Parameter(chunk_q98, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD:
            chunk_mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            chunk_std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean_chunk": nn.Parameter(chunk_mean, requires_grad=False),
                    "std_chunk": nn.Parameter(chunk_std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX:
            chunk_min = torch.ones(shape, dtype=torch.float32) * torch.inf
            chunk_max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min_chunk": nn.Parameter(chunk_min, requires_grad=False),
                    "max_chunk": nn.Parameter(chunk_max, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE:
            chunk_q02 = torch.ones(shape, dtype=torch.float32) * torch.inf
            chunk_q98 = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "q02_chunk": nn.Parameter(chunk_q02, requires_grad=False),
                    "q98_chunk": nn.Parameter(chunk_q98, requires_grad=False),
                }
            )

        # we can just index however many actions we want from the chunk50 stats,
        # since all of the stats for earlier chunk sizes are the same
        stats_chunked_size = 50

        if stats:
            if isinstance(stats[key]["mean"], np.ndarray):
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = torch.from_numpy(stats[key]["mean"]).to(dtype=torch.float32)
                    buffer["std"].data = torch.from_numpy(stats[key]["std"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = torch.from_numpy(stats[key]["min"]).to(dtype=torch.float32)
                    buffer["max"].data = torch.from_numpy(stats[key]["max"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.QUANTILE:
                    buffer["q01"].data = torch.from_numpy(stats[key]["q01"]).to(dtype=torch.float32)
                    buffer["q99"].data = torch.from_numpy(stats[key]["q99"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.ACTIONCHUNK_MEAN_STD:
                    buffer[f"mean_chunk{chunk_size}"].data = torch.from_numpy(
                        stats[key][f"mean_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                    buffer[f"std_chunk{chunk_size}"].data = torch.from_numpy(
                        stats[key][f"std_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_MIN_MAX:
                    buffer[f"min_chunk{chunk_size}"].data = torch.from_numpy(
                        stats[key][f"min_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                    buffer[f"max_chunk{chunk_size}"].data = torch.from_numpy(
                        stats[key][f"max_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_QUANTILE:
                    buffer[f"q02_chunk{chunk_size}"].data = torch.from_numpy(
                        stats[key][f"q02_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                    buffer[f"q98_chunk{chunk_size}"].data = torch.from_numpy(
                        stats[key][f"q98_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD:
                    buffer["mean_chunk"].data = torch.from_numpy(
                        stats[key][f"mean_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
                    buffer["std_chunk"].data = torch.from_numpy(
                        stats[key][f"std_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX:
                    buffer["min_chunk"].data = torch.from_numpy(
                        stats[key][f"min_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
                    buffer["max_chunk"].data = torch.from_numpy(
                        stats[key][f"max_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE:
                    buffer["q02_chunk"].data = torch.from_numpy(
                        stats[key][f"q02_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
                    buffer["q98_chunk"].data = torch.from_numpy(
                        stats[key][f"q98_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
            elif isinstance(stats[key]["mean"], torch.Tensor):
                # Note: The clone is needed to make sure that the logic in save_pretrained
                # doesn't see duplicated
                # tensors anywhere (for example, when we use the same stats for normalization and
                # unnormalization). See the logic here
                # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = stats[key]["mean"].clone().to(dtype=torch.float32)
                    buffer["std"].data = stats[key]["std"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = stats[key]["min"].clone().to(dtype=torch.float32)
                    buffer["max"].data = stats[key]["max"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.QUANTILE:
                    buffer["q01"].data = stats[key]["q01"].clone().to(dtype=torch.float32)
                    buffer["q99"].data = stats[key]["q99"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.ACTIONCHUNK_MEAN_STD:
                    buffer[f"mean_chunk{chunk_size}"].data = (
                        stats[key][f"mean_chunk{stats_chunked_size}"].clone().to(dtype=torch.float32)
                    )[:chunk_size]
                    buffer[f"std_chunk{chunk_size}"].data = (
                        stats[key][f"std_chunk{stats_chunked_size}"].clone().to(dtype=torch.float32)
                    )[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_MIN_MAX:
                    buffer[f"min_chunk{chunk_size}"].data = (
                        stats[key][f"min_chunk{stats_chunked_size}"].clone().to(dtype=torch.float32)
                    )[:chunk_size]
                    buffer[f"max_chunk{chunk_size}"].data = (
                        stats[key][f"max_chunk{stats_chunked_size}"].clone().to(dtype=torch.float32)
                    )[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_QUANTILE:
                    buffer[f"q02_chunk{chunk_size}"].data = (
                        stats[key][f"q02_chunk{stats_chunked_size}"].clone().to(dtype=torch.float32)
                    )[:chunk_size]
                    buffer[f"q98_chunk{chunk_size}"].data = (
                        stats[key][f"q98_chunk{stats_chunked_size}"].clone().to(dtype=torch.float32)
                    )[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD:
                    buffer["mean_chunk"].data = stats[key]["mean_chunk"].clone().to(dtype=torch.float32)[0]
                    buffer["std_chunk"].data = stats[key]["std_chunk"].clone().to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX:
                    buffer["min_chunk"].data = stats[key]["min_chunk"].clone().to(dtype=torch.float32)[0]
                    buffer["max_chunk"].data = stats[key]["max_chunk"].clone().to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE:
                    buffer["q02_chunk"].data = stats[key]["q02_chunk"].clone().to(dtype=torch.float32)[0]
                    buffer["q98_chunk"].data = stats[key]["q98_chunk"].clone().to(dtype=torch.float32)[0]

            elif isinstance(stats[key]["mean"], list):
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = torch.tensor(stats[key]["mean"]).to(dtype=torch.float32)[0]
                    buffer["std"].data = torch.tensor(stats[key]["std"]).to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = torch.tensor(stats[key]["min"]).to(dtype=torch.float32)[0]
                    buffer["max"].data = torch.tensor(stats[key]["max"]).to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.QUANTILE:
                    buffer["q01"].data = torch.tensor(stats[key]["q01"]).to(dtype=torch.float32)[0]
                    buffer["q99"].data = torch.tensor(stats[key]["q99"]).to(dtype=torch.float32)[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_MEAN_STD:
                    buffer[f"mean_chunk{chunk_size}"].data = torch.tensor(
                        stats[key][f"mean_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                    buffer[f"std_chunk{chunk_size}"].data = torch.tensor(
                        stats[key][f"std_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_MIN_MAX:
                    buffer[f"min_chunk{chunk_size}"].data = torch.tensor(
                        stats[key][f"min_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                    buffer[f"max_chunk{chunk_size}"].data = torch.tensor(
                        stats[key][f"max_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_QUANTILE:
                    buffer[f"q02_chunk{chunk_size}"].data = torch.tensor(
                        stats[key][f"q02_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                    buffer[f"q98_chunk{chunk_size}"].data = torch.tensor(
                        stats[key][f"q98_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[:chunk_size]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD:
                    buffer["mean_chunk"].data = torch.tensor(
                        stats[key][f"mean_chunk{stats_chunked_size}"]
                    ).to(dtype=torch.float32)[0]
                    buffer["std_chunk"].data = torch.tensor(stats[key][f"std_chunk{stats_chunked_size}"]).to(
                        dtype=torch.float32
                    )[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX:
                    buffer["min_chunk"].data = torch.tensor(stats[key][f"min_chunk{stats_chunked_size}"]).to(
                        dtype=torch.float32
                    )[0]
                    buffer["max_chunk"].data = torch.tensor(stats[key][f"max_chunk{stats_chunked_size}"]).to(
                        dtype=torch.float32
                    )[0]
                elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE:
                    buffer["q02_chunk"].data = torch.tensor(stats[key][f"q02_chunk{stats_chunked_size}"]).to(
                        dtype=torch.float32
                    )[0]
                    buffer["q98_chunk"].data = torch.tensor(stats[key][f"q98_chunk{stats_chunked_size}"]).to(
                        dtype=torch.float32
                    )[0]
            else:
                type_ = type(stats[key]["mean"])
                raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
        clip_values: dict[str, tuple[float, float]] | None = None,
        chunk_size: int | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        self.clip_values = clip_values if clip_values is not None else {}
        self.chunk_size = chunk_size
        stats_buffers = create_stats_buffers(features, norm_map, stats, chunk_size)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                # raise ValueError(f"Missing key in batch during normalization: {key}")
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            elif norm_mode is NormalizationMode.QUANTILE:
                q01 = buffer["q01"]
                q99 = buffer["q99"]
                assert not torch.isinf(q01).any(), _no_stats_error_str("q01")
                assert not torch.isinf(q99).any(), _no_stats_error_str("q99")
                # print(key, batch[key].shape, q01.shape, q99.shape)
                batch[key] = (batch[key] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0  # Using the openpi approach
                clip_range = self.clip_values.get(ft.type)
                if clip_range is not None:
                    min_clip, max_clip = clip_range
                    batch[key] = batch[key].clamp(min_clip, max_clip)
            elif norm_mode is NormalizationMode.ACTIONCHUNK_MEAN_STD:
                chunk_mean = buffer[f"mean_chunk{self.chunk_size}"]
                chunk_std = buffer[f"std_chunk{self.chunk_size}"]
                assert not torch.isinf(chunk_mean).any(), _no_stats_error_str(f"mean_chunk{self.chunk_size}")
                assert not torch.isinf(chunk_std).any(), _no_stats_error_str(f"std_chunk{self.chunk_size}")
                batch[key] = (batch[key] - chunk_mean) / (chunk_std + 1e-8)
            elif norm_mode is NormalizationMode.ACTIONCHUNK_MIN_MAX:
                chunk_min = buffer[f"min_chunk{self.chunk_size}"]
                chunk_max = buffer[f"max_chunk{self.chunk_size}"]
                assert not torch.isinf(chunk_min).any(), _no_stats_error_str(f"min_chunk{self.chunk_size}")
                assert not torch.isinf(chunk_max).any(), _no_stats_error_str(f"max_chunk{self.chunk_size}")
                # normalize to [0,1]
                batch[key] = (batch[key] - chunk_min) / (chunk_max - chunk_min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            elif norm_mode is NormalizationMode.ACTIONCHUNK_QUANTILE:
                chunk_q02 = buffer[f"q02_chunk{self.chunk_size}"]
                chunk_q98 = buffer[f"q98_chunk{self.chunk_size}"]
                assert not torch.isinf(chunk_q02).any(), _no_stats_error_str(f"q02_chunk{self.chunk_size}")
                assert not torch.isinf(chunk_q98).any(), _no_stats_error_str(f"q98_chunk{self.chunk_size}")
                # Formula: y = 2 * ((x - q02) / (q98 - q02)) - 1, then clip to [-1.5, 1.5]
                batch[key] = 2 * ((batch[key] - chunk_q02) / (chunk_q98 - chunk_q02 + 1e-8)) - 1
                batch[key] = batch[key].clamp(-1.5, 1.5)
            elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD:
                chunk_mean = buffer["mean_chunk"]
                chunk_std = buffer["std_chunk"]
                assert not torch.isinf(chunk_mean).any(), _no_stats_error_str("mean_chunk")
                assert not torch.isinf(chunk_std).any(), _no_stats_error_str("std_chunk")
                batch[key] = (batch[key] - chunk_mean) / (chunk_std + 1e-8)
            elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX:
                chunk_min = buffer["min_chunk"]
                chunk_max = buffer["max_chunk"]
                assert not torch.isinf(chunk_min).any(), _no_stats_error_str("min_chunk")
                assert not torch.isinf(chunk_max).any(), _no_stats_error_str("max_chunk")
                # normalize to [0,1]
                batch[key] = (batch[key] - chunk_min) / (chunk_max - chunk_min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE:
                chunk_q02 = buffer["q02_chunk"]
                chunk_q98 = buffer["q98_chunk"]
                assert not torch.isinf(chunk_q02).any(), _no_stats_error_str("q02_chunk")
                assert not torch.isinf(chunk_q98).any(), _no_stats_error_str("q98_chunk")
                # Formula: y = 2 * ((x - q02) / (q98 - q02)) - 1, then clip to [-1.5, 1.5]
                batch[key] = 2 * ((batch[key] - chunk_q02) / (chunk_q98 - chunk_q02 + 1e-8)) - 1
                batch[key] = batch[key].clamp(-1.5, 1.5)
            else:
                raise ValueError(norm_mode)
        return batch


class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
        chunk_size: int | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        self.chunk_size = chunk_size
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(features, norm_map, stats, chunk_size)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                continue
                # raise ValueError(f"Missing key in batch during denormalization: {key}")

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            elif norm_mode is NormalizationMode.QUANTILE:
                q01 = buffer["q01"]
                q99 = buffer["q99"]
                assert not torch.isinf(q01).any(), _no_stats_error_str("q01")
                assert not torch.isinf(q99).any(), _no_stats_error_str("q99")
                batch[key] = (batch[key] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            elif norm_mode is NormalizationMode.ACTIONCHUNK_MEAN_STD:
                chunk_mean = buffer[f"mean_chunk{self.chunk_size}"]
                chunk_std = buffer[f"std_chunk{self.chunk_size}"]
                assert not torch.isinf(chunk_mean).any(), _no_stats_error_str(f"mean_chunk{self.chunk_size}")
                assert not torch.isinf(chunk_std).any(), _no_stats_error_str(f"std_chunk{self.chunk_size}")
                batch[key] = batch[key] * chunk_std + chunk_mean
            elif norm_mode is NormalizationMode.ACTIONCHUNK_MIN_MAX:
                chunk_min = buffer[f"min_chunk{self.chunk_size}"]
                chunk_max = buffer[f"max_chunk{self.chunk_size}"]
                assert not torch.isinf(chunk_min).any(), _no_stats_error_str(f"min_chunk{self.chunk_size}")
                assert not torch.isinf(chunk_max).any(), _no_stats_error_str(f"max_chunk{self.chunk_size}")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (chunk_max - chunk_min) + chunk_min
            elif norm_mode is NormalizationMode.ACTIONCHUNK_QUANTILE:
                chunk_q02 = buffer[f"q02_chunk{self.chunk_size}"]
                chunk_q98 = buffer[f"q98_chunk{self.chunk_size}"]
                assert not torch.isinf(chunk_q02).any(), _no_stats_error_str(f"q02_chunk{self.chunk_size}")
                assert not torch.isinf(chunk_q98).any(), _no_stats_error_str(f"q98_chunk{self.chunk_size}")
                # Inverse: x = ((y + 1) / 2) * (q98 - q02) + q02
                batch[key] = ((batch[key] + 1) / 2) * (chunk_q98 - chunk_q02) + chunk_q02
            elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MEAN_STD:
                chunk_mean = buffer["mean_chunk"]
                chunk_std = buffer["std_chunk"]
                assert not torch.isinf(chunk_mean).any(), _no_stats_error_str("mean_chunk")
                assert not torch.isinf(chunk_std).any(), _no_stats_error_str("std_chunk")
                batch[key] = batch[key] * chunk_std + chunk_mean
            elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_MIN_MAX:
                chunk_min = buffer["min_chunk"]
                chunk_max = buffer["max_chunk"]
                assert not torch.isinf(chunk_min).any(), _no_stats_error_str("min_chunk")
                assert not torch.isinf(chunk_max).any(), _no_stats_error_str("max_chunk")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (chunk_max - chunk_min) + chunk_min
            elif norm_mode is NormalizationMode.ACTIONCHUNK_PERDIM_QUANTILE:
                chunk_q02 = buffer["q02_chunk"]
                chunk_q98 = buffer["q98_chunk"]
                assert not torch.isinf(chunk_q02).any(), _no_stats_error_str("q02_chunk")
                assert not torch.isinf(chunk_q98).any(), _no_stats_error_str("q98_chunk")
                # Inverse: x = ((y + 1) / 2) * (q98 - q02) + q02
                batch[key] = ((batch[key] + 1) / 2) * (chunk_q98 - chunk_q02) + chunk_q02
            else:
                raise ValueError(norm_mode)
        return batch

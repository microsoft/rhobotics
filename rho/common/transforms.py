import abc
import dataclasses
import functools
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch import Tensor

from rho.common.constants import (
    ACTION,
    LANGUAGE_ACTION_FRAME,
    LANGUAGE_ACTION_TARGET,
    OBSERVATION_IMAGE,
    OBSERVATION_STATE,
)
from rho.common.language_actions import (
    ee_6d_actions_to_eef_rpy,
    summarize_ee_6d_language_actions,
    summarize_eef_rpy_language_actions,
)
from rho.common.rotation_helpers import (
    compute_absolute_ee_6d_pos,
    compute_absolute_ee_quat_pos,
    compute_absolute_ee_quat_wxyz_pos,
    compute_absolute_ee_rpy_pos,
    compute_absolute_pos,
    compute_delta_ee_6d_pos,
    compute_delta_ee_quat_pos,
    compute_delta_ee_quat_wxyz_pos,
    compute_delta_ee_rpy_pos,
    compute_delta_pos,
    convert_ee_6d_to_ee_quat,
    convert_ee_6d_to_ee_quat_wxyz,
    convert_ee_6d_to_ee_rpy,
    convert_ee_quat_to_ee_6d,
    convert_ee_quat_wxyz_to_ee_6d,
    convert_ee_rpy_to_ee_6d,
)
from rho.common.task_encoding import encode_task_bytes
from rho.common.types import ActionType, FeatureType, PolicyFeature

logger = logging.getLogger(__name__)


# Per-sample image augmentation transforms (RandomResizedCrop / ColorJitter)
# call torchvision constructors that do internal validation + op registration.
# At eff_bs=512 across multiple image keys this is ~1500 throwaway
# constructions per batch. Cache by params (the transforms re-roll random
# state per call, so reuse is safe across samples).
@functools.lru_cache(maxsize=16)
def _cached_random_resized_crop(height, width, scale, ratio):
    return T.RandomResizedCrop(size=(height, width), scale=scale, ratio=ratio)


@functools.lru_cache(maxsize=16)
def _cached_color_jitter(brightness, contrast, saturation, hue):
    return T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)


try:
    from draccus.choice_types import ChoiceRegistry
except ImportError:
    from draccus import ChoiceRegistry

# ---------------------------------------------------------------------------
# FACTR utility functions for curriculum training transforms
# Based on: https://github.com/RaindragonD/FACTR/blob/main/factr/utils.py
# ---------------------------------------------------------------------------


def gaussian_2d_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution.

    Args:
        kernel_size: integer, the height/width of the kernel (assumed square).
        sigma: standard deviation for the Gaussian.
        device, dtype: optional, to place the kernel on a specific device / dtype.

    Returns:
        kernel: Tensor of shape (kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0  # shift to center
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    kernel_2d = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def gaussian_2d_smoothing(img: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing (blur) to a batch of images.

    Args:
        img: Tensor of shape (..., C, H, W).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel.
               Larger scale corresponds to more smoothing.

    Returns:
        blurred: Tensor of the same shape as img.
    """
    if scale <= 0:
        return img

    sigma = scale
    kernel_size = max(3, 2 * math.ceil(3 * sigma) + 1)
    kernel_2d = gaussian_2d_kernel(kernel_size, sigma, device=img.device, dtype=img.dtype)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    channels = img.shape[-3]
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # shape: (C, 1, kH, kW)
    padding = kernel_size // 2

    original_shape = img.shape
    batch_shape = original_shape[:-3]
    spatial_shape = original_shape[-2:]
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    img_reshaped = img.view(batch_size, channels, *spatial_shape)

    blurred = F.conv2d(img_reshaped, kernel_2d, groups=channels, padding=padding)
    return blurred.view(*original_shape)


def gaussian_1d_smoothing(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian 1D smoothing across the last dimension (feature_dim).

    Args:
        x: Tensor of shape (..., feature_dim).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel.
               Larger scale corresponds to more smoothing.

    Returns:
        smoothed_x: Tensor of the same shape as x, but blurred along the last dimension.
    """
    # Handle edge case: if scale is very small, just return x
    if scale <= 0:
        return x

    # kernel_size = 2 * int(3*sigma) + 1 as a rule-of-thumb.
    sigma = scale
    kernel_size = max(3, 2 * int(3 * sigma) + 1)  # ensure at least 3
    half_size = (kernel_size - 1) // 2

    arange = torch.arange(-half_size, half_size + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-0.5 * (arange / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_1d = kernel_1d.view(1, 1, -1)

    padding = half_size
    original_shape = x.shape
    feature_dim = x.shape[-1]
    batch_size = int(torch.prod(torch.tensor(x.shape[:-1])))
    x_reshaped = x.view(batch_size, 1, feature_dim)

    smoothed = F.conv1d(x_reshaped, kernel_1d, padding=padding)
    smoothed_x = smoothed.view(*original_shape)

    return smoothed_x


def downsample_1d(x, scale=2):
    """
    Downsample 1D tensor and then upsample back to original size.

    Args:
        x: Tensor of shape (..., feature_dim)
        scale: Downsampling factor

    Returns:
        Tensor of same shape as x, downsampled then upsampled
    """
    scale = int(np.round(scale))
    if scale <= 1:
        return x

    original_shape = x.shape
    x_down = F.avg_pool1d(x.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)  # (B, K/2)
    x_up = F.interpolate(x_down.unsqueeze(1), size=original_shape[-1], mode="nearest").squeeze(
        1
    )  # or 'linear'
    return x_up


def downsample_2d(img, scale=2):
    """
    Downsample 2D image and then upsample back to original size.

    Args:
        img: Tensor of shape (..., C, H, W)
        scale: Downsampling factor

    Returns:
        Tensor of same shape as img, downsampled then upsampled
    """
    scale = int(np.round(scale))
    if scale <= 1:
        return img

    original_shape = img.shape
    x_down = F.avg_pool2d(img, kernel_size=scale, stride=scale)  # (B, C, H/2, W/2)
    x_up = F.interpolate(x_down, size=original_shape[-2:], mode="nearest")  # or 'bilinear'
    return x_up


def additive_gaussian_noise(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Add element-wise Gaussian noise to a tensor.

    Args:
        x: Tensor of shape (..., feature_dim).
        scale: Standard deviation of the Gaussian noise to add.
               Larger scale corresponds to more noise.

    Returns:
        noisy_x: Tensor of the same shape as x, with additive Gaussian noise.
    """
    if scale <= 0:
        return x

    noise = torch.randn_like(x) * scale
    return x + noise


def get_scale(scheduler, start, end, cur_step, max_step, ratio=2 / 3):
    """
    Calculate scale value based on scheduler type and current training progress.

    Args:
        scheduler: Type of scheduler ('no', 'const', 'linear', 'cos', 'exp', 'step')
        start: Starting scale value
        end: Ending scale value
        cur_step: Current training step
        max_step: Maximum training steps
        ratio: Fraction of training where start scale is maintained

    Returns:
        Current scale value
    """
    assert start >= end, "Start scale must be larger than end scale"
    assert cur_step >= 0 and max_step > 0, "Steps must be non-negative and max_step must be positive"

    if cur_step >= max_step:
        return end

    if scheduler == "no":
        return 0

    t = cur_step / max_step
    if t <= ratio or scheduler == "const":
        return start

    t_rescaled = (t - ratio) / (1 - ratio)
    if scheduler == "linear":
        scale = start + t_rescaled * (end - start)
    elif scheduler == "cos":
        scale = end + 0.5 * (start - end) * (1 + np.cos(t_rescaled * np.pi))
    elif scheduler == "exp":
        scale = start * np.exp(-5 * t_rescaled)
    elif scheduler == "step":
        steps = 10
        step_index = int(t_rescaled * steps)
        scale = start + step_index * (end - start) / steps
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")

    return scale


##### IMAGE TRANSFORMS START #####


@dataclass(frozen=True)
class Transform(ChoiceRegistry, abc.ABC):
    """Base class for all transforms that operate on individual tensors or dicts."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def __call__(self, data: Tensor | dict[str, Tensor]) -> Tensor | dict[str, Tensor]:
        raise NotImplementedError

    def deterministic(self) -> "Transform | None":
        """Return the inference-time (non-random) form of this transform.

        Deterministic transforms return ``self`` unchanged (the default).
        Random augmentations override this to return either a deterministic
        replacement (e.g. a fixed crop/resize) or ``None`` to drop themselves
        from the pipeline entirely. ``get_transforms(training=False)`` maps
        every transform through this so that eval/serve preprocessing is
        reproducible and free of training augmentation.
        """
        return self


def _first_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item)
            if text is not None:
                return text
    return None


@Transform.register_subclass("task_description_selector")
@dataclass(frozen=True)
class TaskDescriptionSelector(Transform):
    """Sample between a task and subtask description, writing the result to task."""

    task_key: str = "task"
    subtask_key: str = "subtask"
    output_key: str = "task"
    subtask_probability: float = 0.0
    drop_subtask_key: bool = True
    input_type: str = "Dict"
    post_norm: bool = False
    type: str = "task_description_selector"

    def __post_init__(self):
        if not 0.0 <= self.subtask_probability <= 1.0:
            raise ValueError("subtask_probability must be between 0.0 and 1.0")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        task = _first_text(data.get(self.task_key))
        subtask = _first_text(data.get(self.subtask_key))

        if subtask is not None and (task is None or torch.rand(()) < self.subtask_probability):
            data[self.output_key] = subtask
        elif task is not None:
            data[self.output_key] = task

        if self.drop_subtask_key and self.subtask_key != self.output_key:
            data.pop(self.subtask_key, None)

        return data

    def deterministic(self) -> "Transform | None":
        # Eval form: no random sampling between task/subtask. Force the primary
        # task description (falls back to subtask only when task is absent).
        return dataclasses.replace(self, subtask_probability=0.0)


@Transform.register_subclass("center_crop")
@dataclasses.dataclass(frozen=True)
class CenterCrop:
    """Center crop image to target size using torchvision.
    Supports image tensors ending in (C, H, W), including temporal batches.
    """

    height: int
    width: int
    input_type: str = "Tensor"
    type: str = "center_crop"

    def __call__(self, data: Tensor) -> Tensor:
        if data.ndim < 3:
            raise ValueError(f"Expected an image tensor ending in (C, H, W), got shape {tuple(data.shape)}")

        # Torchvision preserves leading batch/temporal dimensions.
        transform = T.CenterCrop(size=(self.height, self.width))
        return transform(data)


@Transform.register_subclass("resize_with_padding")
@dataclass(frozen=True)
class ResizeWithPadding(Transform):
    height: int
    width: int
    mode: str = "bilinear"
    align_corners: bool = False
    input_type: str = "Tensor"
    type: str = "resize_with_padding"

    def __call__(self, data: Tensor) -> Tensor:
        """Resizes image to target height and width with padding to maintain aspect ratio.
        Supports 3D (CHW), 4D (BCHW), and 5D (B, S, C, H, W) tensors.
        """
        if len(data.shape) < 3:
            raise ValueError(
                f"Expected tensor with at least 3 dimensions (CHW or BCHW), got {len(data.shape)}"
            )

        # Get dimensions (works for 3D, 4D, and 5D)
        c, h, w = data.shape[-3:]

        # If already correct size, return early
        if (h, w) == (self.height, self.width):
            return data

        # Handle 5D tensors by reshaping to 4D, processing, then reshaping back
        is_5d = len(data.shape) == 5
        if is_5d:
            batch, seq_len = data.shape[:2]
            # Reshape (B, S, C, H, W) -> (B*S, C, H, W)
            data = data.reshape(batch * seq_len, c, h, w)

        # Add batch dimension if 3D
        is_3d = len(data.shape) == 3
        if is_3d:
            data = data.unsqueeze(0)

        # Calculate scaling to maintain aspect ratio
        scale_h = self.height / h
        scale_w = self.width / w
        scale = min(scale_h, scale_w)

        # Calculate new dimensions after scaling
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize to new dimensions
        if new_h != h or new_w != w:
            resized = F.interpolate(
                data,
                size=(new_h, new_w),
                mode=self.mode,
                align_corners=self.align_corners if self.mode != "nearest" else None,
            )
        else:
            resized = data

        # Calculate padding
        pad_h = self.height - new_h
        pad_w = self.width - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding (pad_left, pad_right, pad_top, pad_bottom)
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

        # Remove batch dimension if input was 3D
        if is_3d:
            padded = padded.squeeze(0)

        # Reshape back to 5D if input was 5D
        if is_5d:
            padded = padded.reshape(batch, seq_len, c, self.height, self.width)

        return padded


@Transform.register_subclass("channel_reorder")
@dataclass(frozen=True)
class ChannelReorder(Transform):
    """Reorders tensor dimensions to move channels to a specific position.
    Supports both 3D (HWC/CHW) and 4D (BHWC/BCHW) tensors.

    Common use cases:
    - Convert from HWC to CHW format (channels last to channels first)
    - Convert from BHWC to BCHW format (for batched data)
    """

    from_format: str  # e.g., "HWC", "BHWC", "WHC"
    to_format: str  # e.g., "CHW", "BCHW", "CWH"
    input_type: str = "Tensor"
    type: str = "channel_reorder"

    def __post_init__(self):
        # Validate that formats have same characters
        if set(self.from_format) != set(self.to_format):
            raise ValueError(
                f"from_format '{self.from_format}' and to_format '{self.to_format}' "
                f"must contain the same characters"
            )

    def __call__(self, data: Tensor) -> Tensor:
        if len(data.shape) != len(self.from_format):
            raise ValueError(
                f"Expected tensor with {len(self.from_format)} dimensions "
                f"({self.from_format}), got {len(data.shape)} dimensions"
            )

        # Create mapping from current positions to target positions
        perm = []
        for target_char in self.to_format:
            source_idx = self.from_format.index(target_char)
            perm.append(source_idx)

        return data.permute(perm)


@dataclass(frozen=True)
class _CenterResizedCrop(Transform):
    """Deterministic, centered counterpart of :class:`RandomResizedCrop`.

    Reproduces the *geometry* that ``RandomResizedCrop`` samples, but with the
    randomness removed: it crops the expected region (the mean ``scale`` area
    fraction at the geometric-mean ``ratio``) from the centre of the image,
    then resizes to ``(height, width)``.

    The crop maths — including torchvision's central-crop fallback for inputs
    whose aspect ratio cannot satisfy ``ratio`` — mirrors
    ``RandomResizedCrop.get_params``, so the eval-time field of view matches
    training. A plain full-frame resize would be wrong twice over: it ignores
    ``scale`` (training zooms in, eval would not) and it stretches non-square
    inputs, whereas the sampled crop already carries the target aspect ratio.

    Caveat: for *wide* ``scale`` ranges, torchvision rejects candidate crops
    that do not fit the source and resamples, which biases its realised mean
    area below the range mean used here. This is negligible for the tight
    ranges used in practice -- ``scale=(0.9, 0.9)`` matches exactly and
    ``(0.9, 1.0)`` differs by ~0.5% of area -- but a wide range such as
    ``(0.08, 1.0)`` can diverge by ~11%. Revisit this if such a range is ever
    configured.

    Not registered as a config subclass; only constructed at runtime by
    ``RandomResizedCrop.deterministic()``, never deserialized from a config.
    """

    height: int
    width: int
    scale: tuple[float, float] = (1.0, 1.0)
    ratio: tuple[float, float] = (1.0, 1.0)
    input_type: str = "Tensor"
    type: str = "center_resized_crop"

    def _crop_size(self, h: int, w: int) -> tuple[int, int]:
        """Expected crop ``(crop_h, crop_w)`` for a ``h x w`` source image."""
        # Area is sampled uniformly -> expected value is the arithmetic mean.
        # Aspect ratio is sampled log-uniformly -> use the geometric mean.
        scale = (float(self.scale[0]) + float(self.scale[1])) / 2.0
        ratio = math.sqrt(float(self.ratio[0]) * float(self.ratio[1]))

        target_area = h * w * scale
        crop_w = int(round(math.sqrt(target_area * ratio)))
        crop_h = int(round(math.sqrt(target_area / ratio)))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            return crop_h, crop_w

        # Central-crop fallback, matching torchvision's get_params: the source
        # aspect ratio cannot satisfy `ratio`, so clamp to the nearest bound.
        min_ratio, max_ratio = float(min(self.ratio)), float(max(self.ratio))
        in_ratio = w / h
        if in_ratio < min_ratio:
            crop_w = w
            crop_h = int(round(crop_w / min_ratio))
        elif in_ratio > max_ratio:
            crop_h = h
            crop_w = int(round(crop_h * max_ratio))
        else:
            crop_h, crop_w = h, w
        return max(1, min(crop_h, h)), max(1, min(crop_w, w))

    def __call__(self, data: Tensor) -> Tensor:
        if data.ndim < 3:
            raise ValueError(f"Expected an image tensor ending in (C, H, W), got shape {tuple(data.shape)}")

        h, w = data.shape[-2:]
        crop_h, crop_w = self._crop_size(h, w)

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = data[..., top : top + crop_h, left : left + crop_w]

        if (crop_h, crop_w) == (self.height, self.width):
            return cropped
        return T.Resize((self.height, self.width), antialias=True)(cropped)


@Transform.register_subclass("random_resized_crop")
@dataclass(frozen=True)
class RandomResizedCrop(Transform):
    """Randomly crop and resize to target size using torchvision.
    Supports image tensors ending in (C, H, W), including temporal batches.
    """

    height: int
    width: int
    scale: tuple[float, float] = (0.9, 1.0)
    ratio: tuple[float, float] = (0.98, 1.02)
    input_type: str = "Tensor"
    type: str = "random_resized_crop"

    def __call__(self, data: Tensor) -> Tensor:
        if data.ndim < 3:
            raise ValueError(f"Expected an image tensor ending in (C, H, W), got shape {tuple(data.shape)}")
        # Cache the torchvision transform instance keyed by params. The
        # constructor does internal validation and op registration which is
        # non-trivial when called per-sample at eff_bs=512 across 3 image keys
        # (~1500 throwaway constructions per batch). Cached instance still
        # generates fresh random params on each call.
        return _cached_random_resized_crop(self.height, self.width, self.scale, self.ratio)(data)

    def deterministic(self) -> "Transform | None":
        # Eval form: crop the *expected* region centred, instead of a random
        # one, preserving the training-time field of view and aspect ratio.
        return _CenterResizedCrop(
            height=self.height,
            width=self.width,
            scale=tuple(self.scale),
            ratio=tuple(self.ratio),
        )


@Transform.register_subclass("color_jitter")
@dataclass(frozen=True)
class ColorJitter(Transform):
    """Randomly adjust brightness, contrast, saturation, and hue using torchvision.
    Supports 3D (CHW), 4D (BCHW), and 5D (B, S, C, H, W) tensors. Requires 3-channel RGB input.
    """

    brightness: float | None = None
    contrast: tuple[float, float] | None = None
    saturation: tuple[float, float] | None = None
    hue: float | None = None
    input_type: str = "Tensor"
    type: str = "color_jitter"

    def __call__(self, data: Tensor) -> Tensor:
        if len(data.shape) not in [3, 4, 5]:
            raise ValueError(
                f"Expected 3D (CHW), 4D (BCHW), or 5D (B,S,C,H,W) tensor, got {len(data.shape)}D"
            )

        # Check channel count (last 3 dims: ...CHW)
        channels = data.shape[-3]
        if channels != 3:
            raise ValueError(f"ColorJitter requires 3-channel RGB input, got {channels} channels")

        # Handle 5D tensors by reshaping to 4D, processing, then reshaping back
        is_5d = len(data.shape) == 5
        if is_5d:
            batch, seq_len, c, h, w = data.shape
            # Reshape (B, S, C, H, W) -> (B*S, C, H, W)
            data = data.reshape(batch * seq_len, c, h, w)

        # Cache the torchvision transform instance keyed by params (see
        # RandomResizedCrop.__call__ for the same rationale).
        result = _cached_color_jitter(self.brightness, self.contrast, self.saturation, self.hue)(data)

        # Reshape back to 5D if input was 5D
        if is_5d:
            result = result.reshape(batch, seq_len, c, h, w)

        return result

    def deterministic(self) -> "Transform | None":
        # Eval form: no color augmentation, drop from the pipeline.
        return None


@Transform.register_subclass("random_flip_left_right")
@dataclass(frozen=True)
class RandomFlipLeftRight(Transform):
    """Randomly flip image horizontally. Supports both 3D (CHW) and 4D (BCHW) tensors."""

    p: float = 0.5
    input_type: str = "Tensor"
    type: str = "random_flip_left_right"

    def __call__(self, data: Tensor) -> Tensor:
        if len(data.shape) not in [3, 4]:
            raise ValueError(f"Expected 3D (CHW) or 4D (BCHW) tensor, got {len(data.shape)}D")

        transform = T.RandomHorizontalFlip(p=self.p)
        return transform(data)

    def deterministic(self) -> "Transform | None":
        return None


@Transform.register_subclass("random_flip_up_down")
@dataclass(frozen=True)
class RandomFlipUpDown(Transform):
    """Randomly flip image vertically. Supports both 3D (CHW) and 4D (BCHW) tensors."""

    p: float = 0.5
    input_type: str = "Tensor"
    type: str = "random_flip_up_down"

    def __call__(self, data: Tensor) -> Tensor:
        if len(data.shape) not in [3, 4]:
            raise ValueError(f"Expected 3D (CHW) or 4D (BCHW) tensor, got {len(data.shape)}D")

        transform = T.RandomVerticalFlip(p=self.p)
        return transform(data)

    def deterministic(self) -> "Transform | None":
        return None


@Transform.register_subclass("random_rot90")
@dataclass(frozen=True)
class RandomRot90(Transform):
    """Randomly rotate image by 90-degree multiples. Supports both 3D (CHW) and 4D (BCHW) tensors."""

    p: float = 0.5
    input_type: str = "Tensor"
    type: str = "random_rot90"

    def __call__(self, data: Tensor) -> Tensor:
        if len(data.shape) not in [3, 4]:
            raise ValueError(f"Expected 3D (CHW) or 4D (BCHW) tensor, got {len(data.shape)}D")

        if torch.rand(1).item() < self.p:
            # Randomly choose 1, 2, or 3 (90°, 180°, or 270°)
            k = torch.randint(1, 4, (1,)).item()
            # torch.rot90 rotates in the last two dimensions (H, W)
            return torch.rot90(data, k=k, dims=(-2, -1))
        return data

    def deterministic(self) -> "Transform | None":
        return None


@Transform.register_subclass("combine_keys")
@dataclass(frozen=True)
class CombineKeys(Transform):
    """Combines multiple dictionary keys into a single tensor."""

    input_list: list[str] | None = None
    output_key: str = "combined"
    delete_keys: bool = False
    input_type: str = "Dict"
    post_norm: bool = True
    type: str = "combine_keys"

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.input_list is None:
            raise ValueError("input_list must be provided for CombineKeys transform")
        # Check if all keys exist - if not, skip this transform gracefully
        missing_keys = [k for k in self.input_list if k not in batch]
        if missing_keys:
            # Skip combining if keys are missing (e.g., OXE datasets don't have Agibot-specific keys)
            return batch
        combined = torch.cat([batch[key] for key in self.input_list], dim=-1)
        batch[self.output_key] = combined
        if self.delete_keys:
            for key in self.input_list:
                del batch[key]
        return batch


@Transform.register_subclass("combine_stats_keys")
@dataclass(frozen=True)
class CombineStatsKeys(Transform):
    """Combines multiple stats dictionary keys into a single key by concatenating their stat values.

    Stats structure is expected to be:
    {
        "key1": {"mean": tensor, "std": tensor, ...},
        "key2": {"mean": tensor, "std": tensor, ...},
        ...
    }

    Features structure is expected to be:
    {
        "key1": PolicyFeature(type=..., shape=(...)),
        "key2": PolicyFeature(type=..., shape=(...)),
        ...
    }

    This transform concatenates the stat tensors across input keys for each stat type,
    and creates a combined feature with concatenated shape (along last dimension) and
    verified matching type.
    """

    input_list: list[str] | None = None
    output_key: str = "combined"
    delete_keys: bool = False
    input_type: str = "Dict"
    post_norm: bool = True
    type: str = "combine_stats_keys"

    def __call__(
        self,
        stats: dict[str, dict[str, Tensor]],
        features: dict[str, PolicyFeature] | None = None,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, PolicyFeature] | None]:
        if self.input_list is None:
            raise ValueError("input_list must be provided for CombineStatsKeys transform")

        missing_keys = [k for k in self.input_list if k not in stats]
        if missing_keys:
            # Skip combining if keys are missing (e.g., OXE datasets don't have Agibot-specific keys)
            return stats, features

        # Get all stat keys from the first input key (e.g., "mean", "std", "min", "max")
        first_key = self.input_list[0]
        stat_keys = list(stats[first_key].keys())

        # Combine stats by concatenating tensors for each stat key
        combined_stats = {}
        for stat_key in stat_keys:
            tensors_to_concat = []
            for input_key in self.input_list:
                if stat_key in stats[input_key]:
                    tensors_to_concat.append(torch.tensor(stats[input_key][stat_key]))

            if tensors_to_concat:
                combined_stats[stat_key] = torch.cat(tensors_to_concat, dim=-1)

        stats[self.output_key] = combined_stats

        if self.delete_keys:
            for key in self.input_list:
                del stats[key]

        # Combine features if provided
        if features is not None:
            missing_feature_keys = [k for k in self.input_list if k not in features]
            if not missing_feature_keys:
                # Verify all feature types match
                first_feature = features[self.input_list[0]]
                feature_type = first_feature.type

                for input_key in self.input_list[1:]:
                    if features[input_key].type != feature_type:
                        raise ValueError(
                            f"Feature type mismatch: '{self.input_list[0]}' has type {feature_type}, "
                            f"but '{input_key}' has type {features[input_key].type}. "
                            f"All input features must have the same type."
                        )

                # Concatenate shapes along the last dimension
                # Assumes shapes are 1D tuples like (dim,) - sum the last dimension
                total_last_dim = sum(features[k].shape[-1] for k in self.input_list)

                # Build new shape: keep all dimensions except last, then use concatenated last dim
                base_shape = first_feature.shape[:-1]
                combined_shape = base_shape + (total_last_dim,)

                # Create combined feature
                features[self.output_key] = PolicyFeature(
                    type=feature_type,
                    shape=combined_shape,
                )

                if self.delete_keys:
                    for key in self.input_list:
                        del features[key]

        return stats, features


@Transform.register_subclass("convert_stats_to_6d")
@dataclass(frozen=True)
class ConvertStatsTo6d(Transform):
    """Converts stats/features from quaternion or rpy format to 6D rotation format.

    This transform expands stats dimensions to match 6D rotation representation by
    replacing rotation stats with identity stats.

    For ee_quat_pos (per arm): [pos(3), quat(4), gripper(1)] = 8 dims
        -> [pos(3), rot6d(6), gripper(1)] = 10 dims

    For ee_rpy_pos (per arm): [pos(3), rpy(3), gripper(1)] = 7 dims
        -> [pos(3), rot6d(6), gripper(1)] = 10 dims

    All rotation dimensions are replaced with identity stats (0 for mean/min, 1 for std/max).
    """

    action_key: str = "action"
    action_type: ActionType = ActionType.EE_QUAT_POS_XYZW
    type: str = "convert_stats_to_6d"
    input_type: str = "Dict"
    post_norm: bool = False

    def __call__(
        self,
        stats: dict[str, dict[str, Tensor]],
        features: dict[str, PolicyFeature] | None = None,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, PolicyFeature] | None]:
        logger.debug(
            f"ConvertStatsTo6d called with action_key={self.action_key}, action_type={self.action_type}"
        )
        logger.debug(f"Stats keys available: {list(stats.keys())}")

        if self.action_key not in stats:
            logger.debug(f"action_key '{self.action_key}' not in stats, returning unchanged")
            return stats, features

        action_key_stats = stats[self.action_key]
        logger.debug(f"Stats for '{self.action_key}': stat_keys={list(action_key_stats.keys())}")

        # Log the dimension of the first stat we find
        for stat_name, stat_val in action_key_stats.items():
            if stat_name != "count":
                if isinstance(stat_val, torch.Tensor):
                    logger.debug(f"  {stat_name}: shape={stat_val.shape}")
                elif isinstance(stat_val, (list, tuple)):
                    logger.debug(f"  {stat_name}: len={len(stat_val)}")
                break

        # Determine dims per arm based on input type
        if self.action_type in (ActionType.EE_QUAT_POS_XYZW, ActionType.EE_QUAT_POS_WXYZ):
            input_dims_per_arm = 8  # pos(3) + quat(4) + gripper(1)
            rotation_dims = 4
            output_dims_per_arm = 10  # pos(3) + rot6d(6) + gripper(1)
        elif self.action_type == ActionType.EE_EULER_POS:
            input_dims_per_arm = 7  # pos(3) + rpy(3) + gripper(1)
            rotation_dims = 3
            output_dims_per_arm = 10  # pos(3) + rot6d(6) + gripper(1)
        elif self.action_type == ActionType.EE_6D_POS:
            input_dims_per_arm = 10  # pos(3) + rot6d(6) + gripper(1)
            rotation_dims = 6
            output_dims_per_arm = 10  # pos(3) + rot6d(6) + gripper(1)
        elif self.action_type in (ActionType.QUAT_XYZW, ActionType.QUAT_WXYZ):
            input_dims_per_arm = 4  # quat(4)
            rotation_dims = 4
            output_dims_per_arm = 6  # rot6d(6)
        elif self.action_type == ActionType.EULER:
            input_dims_per_arm = 3  # rpy(3)
            rotation_dims = 3
            output_dims_per_arm = 6  # rot6d(6)
        elif self.action_type == ActionType.SIX_D:
            input_dims_per_arm = 6  # rot6d(6)
            rotation_dims = 6
            output_dims_per_arm = 6  # rot6d(6)
        else:
            raise ValueError(
                f"Unknown action_type: {self.action_type}. "
                f"Must be one of {list(ActionType) - {ActionType.POSITION}}"
            )

        logger.debug(f"Expected input_dims_per_arm={input_dims_per_arm} for action_type={self.action_type}")

        # Process each stat action_key (mean, std, min, max, etc.)
        converted_stats = {}
        for stat_key, stat_tensor in action_key_stats.items():
            if stat_key == "count":
                converted_stats[stat_key] = stat_tensor
                continue  # skip count stat if present
            if not isinstance(stat_tensor, torch.Tensor):
                stat_tensor = torch.tensor(stat_tensor)
            total_input_dims = stat_tensor.shape[-1]
            num_arms = total_input_dims // input_dims_per_arm

            logger.debug(
                f"Processing stat '{stat_key}': total_input_dims={total_input_dims}, "
                f"input_dims_per_arm={input_dims_per_arm}, computed num_arms={num_arms}"
            )

            if total_input_dims % input_dims_per_arm != 0:
                raise ValueError(
                    f"Stats dimension {total_input_dims} is not divisible by {input_dims_per_arm} "
                    f"(expected for {self.action_type})"
                )

            # Determine identity value based on stat type
            if "min" in stat_key or "q01" in stat_key or "q02" in stat_key:
                identity_val = -1.0
            elif "max" in stat_key or "q99" in stat_key or "q98" in stat_key or "std" in stat_key:
                identity_val = 1.0
            elif "mean" in stat_key:
                identity_val = 0.0
            else:
                identity_val = 0.0  # default

            # Build converted tensor
            converted_parts = []
            for arm_idx in range(num_arms):
                arm_start = arm_idx * input_dims_per_arm
                if output_dims_per_arm == 10:
                    # Position stats (3 dims)
                    pos_stats = stat_tensor[..., arm_start : arm_start + 3]
                    # Replace rotation with identity (6 dims)
                    identity_rot_stats = torch.full(
                        (*stat_tensor.shape[:-1], 6),
                        identity_val,
                        dtype=stat_tensor.dtype,
                    )
                    # Gripper stats (1 dim)
                    gripper_end = arm_start + input_dims_per_arm
                    gripper_stats = stat_tensor[..., arm_start + 3 + rotation_dims : gripper_end]

                    converted_parts.extend([pos_stats, identity_rot_stats, gripper_stats])
                elif output_dims_per_arm == 6:
                    # Replace rotation with identity (6 dims)
                    identity_rot_stats = torch.full(
                        (*stat_tensor.shape[:-1], 6),
                        identity_val,
                        dtype=stat_tensor.dtype,
                    )
                    converted_parts.append(identity_rot_stats)

            converted_stats[stat_key] = torch.cat(converted_parts, dim=-1)

        stats[self.action_key] = converted_stats

        # Update feature shape if provided
        if features is not None and self.action_key in features:
            feature = features[self.action_key]
            old_shape = tuple(feature.shape)
            # Calculate new shape
            total_input_dims = old_shape[-1]
            num_arms = total_input_dims // input_dims_per_arm
            new_last_dim = num_arms * output_dims_per_arm
            new_shape = old_shape[:-1] + (new_last_dim,)

            features[self.action_key] = PolicyFeature(
                type=feature.type,
                shape=new_shape,
            )

        return stats, features


@Transform.register_subclass("delta_actions")
@dataclass(frozen=True)
class DeltaActions(Transform):
    """Converts absolute actions to delta actions.

    When relative_to_state=True:
        All actions in the chunk become relative to the current observation state.
        Result: actions - state (broadcasted across all timesteps)

    When relative_to_state=False:
        The first action is relative to state; subsequent actions are relative
        to the preceding action.

    Gripper channels remain absolute by default for supported end-effector formats.
    absolute_idx keeps additional zero-based channels absolute for any action type,
    independently of use_absolute_grippers. Rotation coordinates must be selected
    as complete blocks so the transform remains invertible.

    When action_type is EE_QUAT_POS_XYZW, rotation should be in quaternion format (x, y, z, w).
    When action_type is EE_QUAT_POS_WXYZ, rotation should be in quaternion format (w, x, y, z).
    Returns a tensor with the same size as the original action tensor.
    """

    state_key: str = "observation.state"
    action_key: str = "action"
    action_type: ActionType = ActionType.POSITION
    input_type: str = "Dict"
    post_norm: bool = False
    relative_to_state: bool = True
    use_absolute_grippers: bool = True
    absolute_idx: list[int] | None = None
    type: str = "delta_actions"

    def __post_init__(self):
        self._validate_absolute_idx(self.absolute_idx)

    @staticmethod
    def _validate_absolute_idx(absolute_idx: list[int] | None) -> None:
        if absolute_idx is not None and (
            not isinstance(absolute_idx, list)
            or any(type(index) is not int or index < 0 for index in absolute_idx)
        ):
            raise ValueError("absolute_idx must be a list of non-negative integer indices or None")

    @staticmethod
    def _gripper_slices(action_type: ActionType, action_dim: int) -> list[slice]:
        dims_per_arm = {
            ActionType.EE_EULER_POS: 7,
            ActionType.EE_QUAT_POS_XYZW: 8,
            ActionType.EE_QUAT_POS_WXYZ: 8,
            ActionType.EE_6D_POS: 10,
        }.get(action_type)
        if dims_per_arm is None:
            return []
        return [
            slice(start + dims_per_arm - 1, start + dims_per_arm)
            for start in range(0, action_dim, dims_per_arm)
        ]

    @staticmethod
    def _absolute_indices(
        action_type: ActionType,
        action_dim: int,
        use_absolute_grippers: bool,
        absolute_idx: list[int] | None,
    ) -> list[int]:
        DeltaActions._validate_absolute_idx(absolute_idx)
        indices = set(absolute_idx or [])
        if any(index >= action_dim for index in indices):
            raise ValueError(
                f"absolute_idx {absolute_idx} contains an index outside action dimension {action_dim}"
            )
        if indices and action_type != ActionType.POSITION:
            rotation_dim = {
                ActionType.EE_EULER_POS: 3,
                ActionType.EULER: 3,
                ActionType.EE_QUAT_POS_XYZW: 4,
                ActionType.QUAT_XYZW: 4,
                ActionType.EE_QUAT_POS_WXYZ: 4,
                ActionType.QUAT_WXYZ: 4,
                ActionType.EE_6D_POS: 6,
                ActionType.SIX_D: 6,
            }.get(action_type)
            if rotation_dim is not None:
                has_gripper = bool(DeltaActions._gripper_slices(action_type, action_dim))
                offset = 3 if has_gripper else 0
                stride = rotation_dim + 4 if has_gripper else rotation_dim
                for start in range(offset, action_dim, stride):
                    rotation = set(range(start, start + rotation_dim))
                    if indices & rotation and not rotation <= indices:
                        raise ValueError("absolute_idx must include an entire rotation block, not part of it")
        if use_absolute_grippers:
            indices.update(s.start for s in DeltaActions._gripper_slices(action_type, action_dim))
        return sorted(indices)

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.state_key not in data or self.action_key not in data:
            # Skip if keys are missing (e.g., dataset doesn't have these keys)
            return data

        state = data[self.state_key]
        actions = data[self.action_key]
        absolute_idx = self._absolute_indices(
            self.action_type, actions.shape[-1], self.use_absolute_grippers, self.absolute_idx
        )

        # Normalize actions to 3D: (batch, chunk, action_dim).
        orig_action_dim = len(actions.shape)
        if orig_action_dim == 2:
            actions = actions.unsqueeze(0)
            # Unbatched input: state is (state_dim,), (history, state_dim),
            # or (1, history, state_dim). Promote to 3D, take most recent step.
            while state.ndim < 3:
                state = state.unsqueeze(0)
            state = state[:, -1, :]
        else:
            # Batched input: state is (batch, state_dim) or
            # (batch, history, state_dim). Reduce history axis if present.
            if state.ndim == 3:
                state = state[:, -1, :]

        # Validate action dimensions based on type
        if self.action_type == ActionType.POSITION:
            delta_actions = compute_delta_pos(actions, state, self.relative_to_state)
        elif self.action_type == ActionType.EE_EULER_POS:
            assert actions.shape[-1] % 7 == 0, (
                f"{ActionType.EE_EULER_POS} action dimension must be multiple of 7"
            )
            delta_actions = compute_delta_ee_rpy_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.EE_QUAT_POS_XYZW:
            assert actions.shape[-1] % 8 == 0, (
                f"{ActionType.EE_QUAT_POS_XYZW} action dimension must be multiple of 8"
            )
            delta_actions = compute_delta_ee_quat_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.EE_QUAT_POS_WXYZ:
            assert actions.shape[-1] % 8 == 0, (
                f"{ActionType.EE_QUAT_POS_WXYZ} action dimension must be multiple of 8"
            )
            delta_actions = compute_delta_ee_quat_wxyz_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.EE_6D_POS:
            assert actions.shape[-1] % 10 == 0, (
                f"{ActionType.EE_6D_POS} action dimension must be multiple of 10"
            )
            delta_actions = compute_delta_ee_6d_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.QUAT_XYZW:
            assert actions.shape[-1] % 4 == 0, (
                f"{ActionType.QUAT_XYZW} action dimension must be multiple of 4"
            )
            delta_actions = compute_delta_ee_quat_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        elif self.action_type == ActionType.QUAT_WXYZ:
            assert actions.shape[-1] % 4 == 0, (
                f"{ActionType.QUAT_WXYZ} action dimension must be multiple of 4"
            )
            delta_actions = compute_delta_ee_quat_wxyz_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        elif self.action_type == ActionType.EULER:
            assert actions.shape[-1] % 3 == 0, f"{ActionType.EULER} action dimension must be multiple of 3"
            delta_actions = compute_delta_ee_rpy_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        elif self.action_type == ActionType.SIX_D:
            assert actions.shape[-1] % 6 == 0, f"{ActionType.SIX_D} action dimension must be multiple of 6"
            delta_actions = compute_delta_ee_6d_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        if absolute_idx:
            delta_actions[..., absolute_idx] = actions[..., absolute_idx]

        if orig_action_dim == 2:
            delta_actions = delta_actions.squeeze(0)

        data[self.action_key] = delta_actions
        return data


@Transform.register_subclass("cumulative_delta_actions")
@dataclass(frozen=True)
class CumulativeDeltaActions(Transform):
    """Accumulate per-step delta actions into chunk-relative deltas.

    Assumes the action sequence is already delta-EE commands per step.
    Output at timestep t is the cumulative sum from 0..t.
    """

    action_key: str = "action"
    input_type: str = "Dict"
    post_norm: bool = False
    type: str = "cumulative_delta_actions"

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.action_key not in data:
            return data

        actions = data[self.action_key]
        orig_action_dim = len(actions.shape)

        if orig_action_dim == 2:
            actions = actions.unsqueeze(0)

        cumulative = torch.cumsum(actions, dim=-2)

        if orig_action_dim == 2:
            cumulative = cumulative.squeeze(0)

        data[self.action_key] = cumulative
        return data


@Transform.register_subclass("absolute_actions")
@dataclass(frozen=True)
class AbsoluteActions(Transform):
    """Converts delta actions to absolute actions.
    When action_type is EE_QUAT_POS_XYZW, rotation should be in quaternion format (x, y, z, w).
    When action_type is EE_QUAT_POS_WXYZ, rotation should be in quaternion format (w, x, y, z).

    Defaults match DeltaActions: add state to each action independently and
    preserve absolute grippers. Set relative_to_state=False for cumulative deltas.
    Channels in absolute_idx pass through unchanged, matching DeltaActions.
    """

    state_key: str = "observation.state"
    action_key: str = "action"
    action_type: ActionType = ActionType.POSITION
    input_type: str = "Dict"
    post_norm: bool = False
    relative_to_state: bool = True
    use_absolute_grippers: bool = True
    absolute_idx: list[int] | None = None
    type: str = "absolute_actions"

    def __post_init__(self):
        DeltaActions._validate_absolute_idx(self.absolute_idx)

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.state_key not in data or self.action_key not in data:
            # Skip if keys are missing (e.g., dataset doesn't have these keys)
            return data

        state = data[self.state_key]
        actions = data[self.action_key]
        absolute_idx = DeltaActions._absolute_indices(
            self.action_type, actions.shape[-1], self.use_absolute_grippers, self.absolute_idx
        )

        # Check the shape of actions
        # if (sequence_length, action_dim) change to (1, sequence_length, action_dim)
        # to handle batch dimension

        orig_action_dim = len(actions.shape)

        if orig_action_dim == 2:
            actions = actions.unsqueeze(0)
            # Unbatched input: state is (state_dim,), (history, state_dim),
            # or (1, history, state_dim). Promote to 3D, take most recent step.
            while state.ndim < 3:
                state = state.unsqueeze(0)
            state = state[:, -1, :]
        else:
            # Batched input: state is (batch, state_dim) or
            # (batch, history, state_dim). Reduce history axis if present.
            if state.ndim == 3:
                state = state[:, -1, :]

        # Validate action dimensions based on type
        if self.action_type == ActionType.POSITION:
            absolute_actions = compute_absolute_pos(actions, state, self.relative_to_state)
        elif self.action_type == ActionType.EE_EULER_POS:
            assert actions.shape[-1] % 7 == 0, (
                f"{ActionType.EE_EULER_POS} action dimension must be multiple of 7"
            )
            absolute_actions = compute_absolute_ee_rpy_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.EE_QUAT_POS_XYZW:
            assert actions.shape[-1] % 8 == 0, (
                f"{ActionType.EE_QUAT_POS_XYZW} action dimension must be multiple of 8"
            )
            absolute_actions = compute_absolute_ee_quat_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.EE_QUAT_POS_WXYZ:
            assert actions.shape[-1] % 8 == 0, (
                f"{ActionType.EE_QUAT_POS_WXYZ} action dimension must be multiple of 8"
            )
            absolute_actions = compute_absolute_ee_quat_wxyz_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.EE_6D_POS:
            assert actions.shape[-1] % 10 == 0, (
                f"{ActionType.EE_6D_POS} action dimension must be multiple of 10"
            )
            absolute_actions = compute_absolute_ee_6d_pos(
                actions, state, self.relative_to_state, with_ee_gripper=True
            )
        elif self.action_type == ActionType.QUAT_XYZW:
            assert actions.shape[-1] % 4 == 0, (
                f"{ActionType.QUAT_XYZW} action dimension must be multiple of 4"
            )
            absolute_actions = compute_absolute_ee_quat_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        elif self.action_type == ActionType.QUAT_WXYZ:
            assert actions.shape[-1] % 4 == 0, (
                f"{ActionType.QUAT_WXYZ} action dimension must be multiple of 4"
            )
            absolute_actions = compute_absolute_ee_quat_wxyz_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        elif self.action_type == ActionType.EULER:
            assert actions.shape[-1] % 3 == 0, f"{ActionType.EULER} action dimension must be multiple of 3"
            absolute_actions = compute_absolute_ee_rpy_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        elif self.action_type == ActionType.SIX_D:
            assert actions.shape[-1] % 6 == 0, f"{ActionType.SIX_D} action dimension must be multiple of 6"
            absolute_actions = compute_absolute_ee_6d_pos(
                actions, state, self.relative_to_state, with_ee_gripper=False
            )
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        if absolute_idx:
            absolute_actions[..., absolute_idx] = actions[..., absolute_idx]

        if orig_action_dim == 2:
            absolute_actions = absolute_actions.squeeze(0)

        data[self.action_key] = absolute_actions
        return data


@Transform.register_subclass("convert_to_6d_actions")
@dataclass(frozen=True)
class ConvertTo6dActions(Transform):
    """Converts quaternion/rpy-based end-effector actions to 6D rotation representation."""

    """This assumes that the input actions are provided in the form
        [left_position, left_orientation, left_gripper, right_position, right_orientation, right_gripper]"""
    """This returns output actions in the form
        [left_position, left_rot6d, left_gripper, right_position, right_rot6d, right_gripper]"""

    """When action_type is EE_QUAT_POS_XYZW, rotation should be in quaternion format (x, y, z, w).
    When action_type is EE_QUAT_POS_WXYZ, rotation should be in quaternion format (w, x, y, z)."""

    action_key: str = "action"
    action_type: ActionType = ActionType.EE_QUAT_POS_XYZW
    type: str = "convert_to_6d_actions"
    input_type: str = "Dict"
    post_norm: bool = False

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.action_key not in data:
            # Skip if key is missing (e.g., dataset doesn't have this key)
            return data

        actions = data[self.action_key]

        # Check the shape of actions
        # if (sequence_length, action_dim) change to (1, sequence_length, action_dim)
        # to handle batch dimension

        orig_action_dim = len(actions.shape)

        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)

        # Validate action dimensions based on type
        if self.action_type == ActionType.EE_EULER_POS:
            if actions.shape[-1] % 7 != 0:
                _ds_name = data.get("dataset_name")
                if hasattr(_ds_name, "decode"):
                    _ds_name = _ds_name.decode("utf-8", errors="replace")
                raise AssertionError(
                    f"{ActionType.EE_EULER_POS} action dimension must be multiple of 7; "
                    f"got actions.shape={tuple(actions.shape)} for action_key={self.action_key!r}; "
                    f"dataset_name={_ds_name!r}, "
                    f"sample_keys={sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}"
                )
            actions_6d = convert_ee_rpy_to_ee_6d(actions, with_ee_gripper=True)
        elif self.action_type == ActionType.EE_QUAT_POS_XYZW:
            assert actions.shape[-1] % 8 == 0, (
                f"{ActionType.EE_QUAT_POS_XYZW} action dimension must be multiple of 8"
            )
            actions_6d = convert_ee_quat_to_ee_6d(actions, with_ee_gripper=True)
        elif self.action_type == ActionType.EE_QUAT_POS_WXYZ:
            assert actions.shape[-1] % 8 == 0, (
                f"{ActionType.EE_QUAT_POS_WXYZ} action dimension must be multiple of 8"
            )
            actions_6d = convert_ee_quat_wxyz_to_ee_6d(actions, with_ee_gripper=True)
        elif self.action_type == ActionType.QUAT_XYZW:
            assert actions.shape[-1] % 4 == 0, (
                f"{ActionType.QUAT_XYZW} action dimension must be multiple of 4"
            )
            actions_6d = convert_ee_quat_to_ee_6d(actions, with_ee_gripper=False)
        elif self.action_type == ActionType.QUAT_WXYZ:
            assert actions.shape[-1] % 4 == 0, (
                f"{ActionType.QUAT_WXYZ} action dimension must be multiple of 4"
            )
            actions_6d = convert_ee_quat_wxyz_to_ee_6d(actions, with_ee_gripper=False)
        elif self.action_type == ActionType.EULER:
            assert actions.shape[-1] % 3 == 0, f"{ActionType.EULER} action dimension must be multiple of 3"
            actions_6d = convert_ee_rpy_to_ee_6d(actions, with_ee_gripper=False)
        elif self.action_type == ActionType.EE_6D_POS:
            assert actions.shape[-1] % 10 == 0, (
                f"{ActionType.EE_6D_POS} action dimension must be multiple of 10"
            )
            return data
        elif self.action_type == ActionType.SIX_D:
            assert actions.shape[-1] % 6 == 0, f"{ActionType.SIX_D} action dimension must be multiple of 6"
            return data
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        if orig_action_dim == 2:
            actions_6d = actions_6d.squeeze(0)
        data[self.action_key] = actions_6d

        return data


@Transform.register_subclass("convert_from_6d_actions")
@dataclass(frozen=True)
class ConvertFrom6dActions(Transform):
    """Converts 6D rotation representation back to quaternion/rpy-based end-effector actions."""

    """This assumes that the input actions are provided in the form
        [left_position, left_rot6d, left_gripper, right_position, right_rot6d, right_gripper]"""
    """This returns output actions in the form
        [left_position, left_orientation, left_gripper, right_position, right_orientation, right_gripper]"""

    """When output_type is EE_QUAT_POS_XYZW, rotation will be in quaternion format (x, y, z, w).
    When output_type is EE_QUAT_POS_WXYZ, rotation will be in quaternion format (w, x, y, z)."""

    action_key: str = "action"
    output_type: ActionType = ActionType.EE_QUAT_POS_XYZW
    type: str = "convert_from_6d_actions"

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.action_key not in data:
            # Skip if key is missing (e.g., dataset doesn't have this key)
            return data

        actions = data[self.action_key]

        # Check the shape of actions
        # if (sequence_length, action_dim) change to (1, sequence_length, action_dim)
        # to handle batch dimension

        orig_action_dim = len(actions.shape)

        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)

        # Validate that input is 6D format
        assert actions.shape[-1] % 10 == 0, "Input action dimension must be multiple of 10 (6D format)"

        if self.output_type == ActionType.EE_QUAT_POS_XYZW:
            # Convert 6D actions to quaternion xyzw representation
            actions_out = convert_ee_6d_to_ee_quat(actions, with_ee_gripper=True)
        elif self.output_type == ActionType.EE_QUAT_POS_WXYZ:
            # Convert 6D actions to quaternion wxyz representation
            actions_out = convert_ee_6d_to_ee_quat_wxyz(actions, with_ee_gripper=True)
        elif self.output_type == ActionType.EE_EULER_POS:
            # Convert 6D actions to RPY representation
            actions_out = convert_ee_6d_to_ee_rpy(actions, with_ee_gripper=True)
        elif self.output_type == ActionType.QUAT_XYZW:
            # Convert 6D actions to quaternion xyzw representation without gripper
            actions_out = convert_ee_6d_to_ee_quat(actions, with_ee_gripper=False)
        elif self.output_type == ActionType.QUAT_WXYZ:
            # Convert 6D actions to quaternion wxyz representation without gripper
            actions_out = convert_ee_6d_to_ee_quat_wxyz(actions, with_ee_gripper=False)
        elif self.output_type == ActionType.EULER:
            # Convert 6D actions to RPY representation without gripper
            actions_out = convert_ee_6d_to_ee_rpy(actions, with_ee_gripper=False)
        elif self.output_type == ActionType.EE_6D_POS:
            assert actions.shape[-1] % 10 == 0, (
                f"{ActionType.EE_6D_POS} action dimension must be multiple of 10"
            )
            return data
        elif self.output_type == ActionType.SIX_D:
            assert actions.shape[-1] % 6 == 0, f"{ActionType.SIX_D} action dimension must be multiple of 6"
            return data
        else:
            raise ValueError(
                f"Unknown output_type: {self.output_type}. "
                f"Must be {ActionType.EE_QUAT_POS_XYZW}, {ActionType.EE_QUAT_POS_WXYZ}, "
                f"{ActionType.EE_EULER_POS}, {ActionType.QUAT_XYZW}, "
                f"{ActionType.QUAT_WXYZ}, {ActionType.EULER}, {ActionType.EE_6D_POS}, or {ActionType.SIX_D}"
            )

        if orig_action_dim == 2:
            actions_out = actions_out.squeeze(0)
        data[self.action_key] = actions_out

        return data


class EndStateTarget:
    """Snapshot the chunk-final action as a KI-ENDSTATE LM target.

    Inserted into the per-dataset transform pipeline right before Normalize,
    so it reads the un-normalized delta-6D action (per-arm
    [pos(3), rot6d(6), grip(1)]). Writes ``sample[output_key]``: the last
    chunk step in physical units (translation cm, RPY degrees), optionally
    converted from 6D rotation to RPY, zero-padded to ``target_dim``.

    Constructed in code (not registered for YAML) and gated by config -- only
    KI-ENDSTATE runs with ``endstate_target_units='physical'`` get it. The
    flow expert is unaffected: it keeps consuming the normalized 6D ``action``.
    No-op for samples without an ``action`` key (VL cotraining batches).
    """

    _ARM_6D = 10  # [pos(3), rot6d(6), grip(1)]
    _ARM_RPY = 7  # [pos(3), rpy(3), grip(1)]

    def __init__(
        self,
        rotation: str,
        target_dim: int,
        action_key: str = "action",
        output_key: str = "endstate_target",
        with_ee_gripper: bool = True,
    ):
        self.rotation = rotation
        self.target_dim = target_dim
        self.action_key = action_key
        self.output_key = output_key
        self.with_ee_gripper = with_ee_gripper

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.action_key not in data:
            return data

        # action[..., -1, :] is the chunk-final step; the `...` keeps this
        # correct whether action is (chunk, dim) or (batch, chunk, dim).
        last = data[self.action_key][..., -1, :].detach().to(torch.float32).clone()
        num_arms = last.shape[-1] // self._ARM_6D

        # Translation m -> cm. Done while still in 6D layout (positions are at
        # a known per-arm offset); the 6D->RPY conversion below passes
        # positions through unchanged.
        for arm in range(num_arms):
            base = arm * self._ARM_6D
            last[..., base : base + 3] *= 100.0

        if self.rotation == "rpy":
            last = convert_ee_6d_to_ee_rpy(last, with_ee_gripper=self.with_ee_gripper)
            for arm in range(num_arms):
                base = arm * self._ARM_RPY
                last[..., base + 3 : base + 6] = torch.rad2deg(last[..., base + 3 : base + 6])

        cur = last.shape[-1]
        if cur < self.target_dim:
            last = F.pad(last, (0, self.target_dim - cur))

        data[self.output_key] = last
        return data


class LanguageActionTarget:
    """Snapshot the chunk-net action as LAP language-action text.

    Runs before normalization, after configured action-space transforms. For
    the common EE_6D_POS setup with ``relative_to_state=True``, the last action
    in the chunk is already the net end-effector displacement from the current
    state to the chunk end. The output is encoded as fixed-width bytes so the
    dataloader batch remains tensor-only.
    """

    def __init__(
        self,
        action_key: str = ACTION,
        output_key: str = LANGUAGE_ACTION_TARGET,
        source_action_type: ActionType = ActionType.EE_6D_POS,
        chunk_reduction: str = "last",
        include_rotation: bool = True,
        eef_frame_prob: float = 0.5,
        state_key: str = OBSERVATION_STATE,
        frame_output_key: str = LANGUAGE_ACTION_FRAME,
    ):
        self.action_key = action_key
        self.output_key = output_key
        self.source_action_type = ActionType(source_action_type)
        self.chunk_reduction = chunk_reduction
        self.include_rotation = include_rotation
        self.eef_frame_prob = eef_frame_prob
        self.state_key = state_key
        self.frame_output_key = frame_output_key

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.action_key not in data:
            return data

        actions = data[self.action_key].detach().to(torch.float32)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self.chunk_reduction == "last":
            net_action = actions[..., -1, :]
        elif self.chunk_reduction == "sum":
            net_action = actions.sum(dim=-2)
        else:
            raise ValueError(
                f"Unsupported chunk_reduction={self.chunk_reduction!r}; expected 'last' or 'sum'"
            )

        if self.source_action_type != ActionType.EE_6D_POS:
            raise ValueError(
                f"LanguageActionTarget currently supports EE_6D_POS, got {self.source_action_type}"
            )

        if self.eef_frame_prob > 0.0 and self.state_key in data:
            use_eef = torch.rand(net_action.shape[0]).lt(self.eef_frame_prob)
        else:
            use_eef = torch.zeros(net_action.shape[0], dtype=torch.bool)
        texts: list[str] = []
        frames: list[str] = []
        if use_eef.any():
            state = data[self.state_key]
            if state.ndim == 3:
                state = state[:, -1]
            eef_actions = ee_6d_actions_to_eef_rpy(net_action[use_eef], state[use_eef])
            eef_texts = iter(
                summarize_eef_rpy_language_actions(
                    eef_actions,
                    include_rotation=self.include_rotation,
                )
            )
        else:
            eef_texts = iter(())
        if (~use_eef).any():
            base_texts = iter(
                summarize_ee_6d_language_actions(net_action[~use_eef], include_rotation=self.include_rotation)
            )
        else:
            base_texts = iter(())

        for selected in use_eef.tolist():
            if selected:
                texts.append(next(eef_texts))
                frames.append("end-effector frame")
            else:
                texts.append(next(base_texts))
                frames.append("robot base frame")
        encoded = torch.stack([encode_task_bytes(text) for text in texts], dim=0)
        data[self.output_key] = encoded.squeeze(0) if squeeze else encoded
        encoded_frames = torch.stack([encode_task_bytes(frame) for frame in frames], dim=0)
        data[self.frame_output_key] = encoded_frames.squeeze(0) if squeeze else encoded_frames
        return data


##### IMAGE TRANSFORMS END #####


def get_target_sequence_lengths(temporal_indices: dict[str, Any] | None) -> dict[FeatureType, int] | None:
    """Convert a per-feature temporal lookup mapping into a model shape contract.

    The values in ``temporal_indices`` may be frame indices or timestamps; padding
    only needs the number of requested steps for each feature type.
    """
    if temporal_indices is None:
        return None

    sequence_lengths = {FeatureType.VISUAL: 1, FeatureType.STATE: 1, FeatureType.ACTION: 1}
    for key_substr, feature_type in zip(
        [OBSERVATION_IMAGE, OBSERVATION_STATE, ACTION],
        [FeatureType.VISUAL, FeatureType.STATE, FeatureType.ACTION],
        strict=False,
    ):
        sequence_lengths[feature_type] = max(
            (len(indices) for key, indices in temporal_indices.items() if key.startswith(key_substr)),
            default=1,
        )
    return sequence_lengths


def build_key_padding_transform(
    features: dict[str, PolicyFeature] | None,
    target_sequence_lengths: dict[FeatureType, int] | None = None,
):
    """Builds a transform that checks to see if all features are keys in the input dict,
        for any transforms that are missing it will create a zero element tensor of the correct shape.
    Args:
        features: FeatureConfig containing all feature parameters.
        target_sequence_lengths: Fixed sequence lengths required by the model,
            keyed by feature type. This is a shape contract, not LeRobot's
            per-dataset temporal loading schedule.
    """
    if features is None:
        return lambda x: x  # Identity if no features provided
    if hasattr(features, "feature_dict"):
        features = features.feature_dict

    sequence_by_type = None
    if target_sequence_lengths is not None:
        sequence_by_type = {FeatureType.VISUAL: 1, FeatureType.STATE: 1, FeatureType.ACTION: 1}
        sequence_by_type.update(target_sequence_lengths)

    def key_padding_transform(
        data: dict[str, Tensor], batch_size: int | None = None, device=None
    ) -> dict[str, Tensor]:
        if data is None:
            return data

        def _sequence_axis(value: Tensor, target_shape: tuple[int, ...], sequence_len: int) -> int | None:
            if sequence_len <= 1:
                return None
            if batch_size is not None:
                return 1 if value.ndim >= len(target_shape) else None
            return 0 if value.ndim == len(target_shape) else None

        def _new_pad_mask(value: Tensor, sequence_len: int) -> Tensor:
            if sequence_len <= 1:
                if batch_size is not None:
                    return torch.zeros((batch_size, 1), dtype=torch.bool, device=value.device)
                return torch.zeros((1,), dtype=torch.bool, device=value.device)
            axis = _sequence_axis(value, key_shape, sequence_len)
            current_len = value.shape[axis] if axis is not None else sequence_len
            if batch_size is not None:
                return torch.zeros((value.shape[0], current_len), dtype=torch.bool, device=value.device)
            return torch.zeros((current_len,), dtype=torch.bool, device=value.device)

        def _pad_sequence(value: Tensor, target_len: int, axis: int) -> tuple[Tensor, int]:
            current_len = value.shape[axis]
            if current_len >= target_len:
                return value, 0
            pad_shape = list(value.shape)
            pad_shape[axis] = target_len - current_len
            pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            return torch.cat([value, pad], dim=axis), target_len - current_len

        def _pad_mask(mask: Tensor, target_len: int, pad_len: int) -> Tensor:
            if mask.shape[-1] >= target_len:
                return mask
            true_len = target_len - mask.shape[-1] if pad_len == 0 else pad_len
            pad_shape = list(mask.shape)
            pad_shape[-1] = true_len
            pad = torch.ones(pad_shape, dtype=torch.bool, device=mask.device)
            return torch.cat([mask, pad], dim=-1)

        def _dim_pad_mask(value: Tensor, target_dim: int) -> Tensor:
            # Transforms such as quaternion -> 6D rotation can widen a real
            # feature before padding runs. Those added dimensions are data, not
            # padding, so the mask must describe the post-transform tensor.
            mask_dim = max(value.shape[-1], target_dim)
            if batch_size is not None:
                mask = torch.zeros((value.shape[0], mask_dim), dtype=torch.bool, device=value.device)
            else:
                mask = torch.zeros((mask_dim,), dtype=torch.bool, device=value.device)
            current_dim = value.shape[-1]
            if current_dim < mask_dim:
                mask[..., current_dim:] = True
            return mask

        def _normalize_dim_pad_mask(mask: Tensor, target_dim: int) -> Tensor:
            if mask.shape[-1] == target_dim:
                return mask
            if mask.shape[-1] > target_dim:
                return mask[..., :target_dim]
            pad_shape = list(mask.shape)
            pad_shape[-1] = target_dim - mask.shape[-1]
            pad = torch.ones(pad_shape, dtype=torch.bool, device=mask.device)
            return torch.cat([mask, pad], dim=-1)

        for key, feature in features.items():
            key_shape = feature.shape
            emit_dim_pad_mask = feature.type != FeatureType.VISUAL
            sequence_len = 1
            if sequence_by_type is not None:
                sequence_len = sequence_by_type[feature.type]
                if sequence_len > 1:
                    key_shape = (sequence_len,) + key_shape
                mask_shape = (sequence_len,)
            else:
                mask_shape = (1,)

            if batch_size is not None:
                key_shape = (batch_size,) + key_shape
                mask_shape = (batch_size, mask_shape[0])
            if key not in data:
                data[key] = torch.zeros(key_shape, dtype=torch.float32)
                data[f"{key}_is_pad"] = torch.ones(mask_shape, dtype=torch.bool)
                if emit_dim_pad_mask:
                    data[f"{key}_dim_is_pad"] = _dim_pad_mask(data[key], key_shape[-1])
            elif f"{key}_is_pad" not in data:
                data[f"{key}_is_pad"] = _new_pad_mask(data[key], sequence_len)
            if emit_dim_pad_mask and f"{key}_dim_is_pad" not in data:
                data[f"{key}_dim_is_pad"] = _dim_pad_mask(data[key], key_shape[-1])
            elif emit_dim_pad_mask:
                data[f"{key}_dim_is_pad"] = _normalize_dim_pad_mask(
                    data[f"{key}_dim_is_pad"], max(data[key].shape[-1], key_shape[-1])
                )

            # Assert that the shape is correct for both original and padded data
            if data[key].shape != key_shape:
                if feature.type == FeatureType.VISUAL:
                    # If the image is the wrong size then apply padding
                    data[key] = ResizeWithPadding(height=key_shape[-2], width=key_shape[-1])(data[key])
                else:
                    if data[key].shape[-1] < key_shape[-1]:
                        # Pad the last dimension with zeros when action/state dims differ.
                        pad_size = key_shape[-1] - data[key].shape[-1]
                        pad = (0, pad_size)
                        data[key] = F.pad(data[key], pad, "constant", 0)

                    axis = _sequence_axis(data[key], key_shape, sequence_len)
                    if axis is not None:
                        data[key], pad_len = _pad_sequence(data[key], sequence_len, axis)
                        data[f"{key}_is_pad"] = _pad_mask(data[f"{key}_is_pad"], sequence_len, pad_len)

            if data[f"{key}_is_pad"].shape != mask_shape and sequence_len > 1:
                data[f"{key}_is_pad"] = _pad_mask(data[f"{key}_is_pad"], sequence_len, 0)
            if device is not None:
                data[key] = data[key].to(device)
                data[f"{key}_is_pad"] = data[f"{key}_is_pad"].to(device)
                if emit_dim_pad_mask:
                    data[f"{key}_dim_is_pad"] = data[f"{key}_dim_is_pad"].to(device)

        return data

    return key_padding_transform


class BatchTransform(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError


class ConsolidateTransform(BatchTransform):
    def __init__(self, transforms: list[BatchTransform]):
        if transforms is None:
            transforms = []
        self.transforms = transforms

    def reset(self):
        for transform in self.transforms:
            transform.reset()

    def train(self):
        for transform in self.transforms:
            transform.train()

    def eval(self):
        for transform in self.transforms:
            transform.eval()

    def step(self):
        for transform in self.transforms:
            transform.step()

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        for transform in self.transforms:
            data = transform(data)
        return data


class NoiseScheduleTransform(BatchTransform):
    """
    Apply scheduled noise/augmentation transforms that decay over training.

    Supports multiple transform operators:
    - 'gaussian_1d': 1D Gaussian smoothing
    - 'gaussian_2d': 2D Gaussian smoothing (for images)
    - 'downsample_1d': 1D downsampling
    - 'downsample_2d': 2D downsampling (for images)

    The scale is computed using the get_scale scheduler function and
    decreases from max_scale to min_scale over training steps.
    """

    def __init__(
        self,
        keys: list[str],
        min_noise_level: float = 0.0,
        max_noise_level: float = 1.0,
        scheduler: str = "linear",
        max_steps: int = 100000,
        ratio: float = 2 / 3,
        operator: str = "gaussian_2d",
    ):
        """
        Args:
            keys: List of data keys to apply the transform to
            min_noise_level: Minimum scale value (end of training)
            max_noise_level: Maximum scale value (start of training)
            scheduler: Type of scheduler ('no', 'const', 'linear', 'cos', 'exp', 'step')
            max_steps: Maximum training steps
            ratio: Fraction of training where max_scale is maintained
            operator: Which transform operator to apply
                     ('gaussian_1d', 'gaussian_2d', 'downsample_1d', 'downsample_2d')
        """
        self.keys = keys if isinstance(keys, list) else [keys]
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.ratio = ratio
        self.operator = operator
        self.current_step = 0
        self._is_training = True

        # Map operator names to functions
        self.operator_functions = {
            "gaussian_1d": gaussian_1d_smoothing,
            "gaussian_2d": gaussian_2d_smoothing,
            "downsample_1d": downsample_1d,
            "downsample_2d": downsample_2d,
            "additive_noise": additive_gaussian_noise,
        }

        if operator not in self.operator_functions:
            raise ValueError(
                f"Unknown operator: {operator}. Must be one of {list(self.operator_functions.keys())}"
            )

    def reset(self):
        """Reset the current step counter."""
        self.current_step = 0

    def train(self):
        """Set to training mode."""
        self._is_training = True

    def eval(self):
        """Set to evaluation mode (no transforms applied)."""
        self._is_training = False

    def step(self):
        """Increment the step counter."""
        self.current_step += 1

    def get_current_scale(self) -> float:
        """Compute the current scale based on scheduler and step."""
        return get_scale(
            scheduler=self.scheduler,
            start=self.max_noise_level,
            end=self.min_noise_level,
            cur_step=self.current_step,
            max_step=self.max_steps,
            ratio=self.ratio,
        )

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Apply the scheduled transform to specified keys in the data dict.

        Args:
            data: Dictionary containing tensors

        Returns:
            Dictionary with transforms applied to specified keys
        """
        # Don't apply transforms in eval mode
        if not self._is_training:
            return data

        # Get current scale
        scale = self.get_current_scale()

        # If scale is effectively zero, skip transforms
        if scale <= 0:
            return data

        # Get the transform function
        transform_fn = self.operator_functions[self.operator]

        # Apply transform to each specified key
        for key in self.keys:
            if key in data:
                data[key] = transform_fn(data[key], scale=scale)

        return data


@dataclass
class TransformConfig(ChoiceRegistry, abc.ABC):
    keys: list[str] | str = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.keys, str):
            self.keys = [self.keys]

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self) -> BatchTransform:
        raise NotImplementedError


@TransformConfig.register_subclass("noise_schedule")
@dataclass
class NoiseScheduleTransformConfig(TransformConfig):
    min_noise_level: float = 0.0
    max_noise_level: float = 1.0
    scheduler: str = "linear"
    max_steps: int = 100000
    ratio: float = 2 / 3
    operator: str = "gaussian_2d"
    name: str = "noise_schedule"

    def build(self) -> BatchTransform:
        # Ensure keys is a list
        keys = self.keys if isinstance(self.keys, list) else [self.keys]

        return NoiseScheduleTransform(
            keys=keys,
            min_noise_level=self.min_noise_level,
            max_noise_level=self.max_noise_level,
            scheduler=self.scheduler,
            max_steps=self.max_steps,
            ratio=self.ratio,
            operator=self.operator,
        )

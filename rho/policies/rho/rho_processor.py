"""Rho observation validation and tensor preparation."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES
from rho.common.constants import OBSERVATION_LANG as OBS_TASK
from rho.common.constants import OBSERVATION_STATE as OBS_ROBOT
from rho.common.transforms import ResizeWithPadding
from rho.policies.rho.configuration_rho import RhoConfig

OBS_IMAGES_IS_PAD = f"{OBS_IMAGES}_is_pad"
_IMAGE_RANGE_TOLERANCE = 1e-3
_AMBIGUOUS_FLOAT_IMAGE_MAX = 2.0


@dataclass(frozen=True)
class RhoProcessedObservations:
    image: Tensor
    image_mask: Tensor
    state: Tensor
    action: Tensor | None = None


class RhoProcessor:
    """Convert external observation dictionaries to Rho model tensors."""

    def __init__(self, config: RhoConfig):
        self.config = config
        self._resize_transform = None
        if config.resize_imgs_with_padding is not None:
            self._resize_transform = ResizeWithPadding(*config.resize_imgs_with_padding)

    def validate_required_observations(self, batch: dict) -> None:
        missing_keys = [key for key in self.config.image_features if key not in batch]
        if OBS_ROBOT not in batch:
            missing_keys.append(OBS_ROBOT)
        if OBS_TASK not in batch:
            missing_keys.append(OBS_TASK)
        if missing_keys:
            raise KeyError(
                "Missing required Rho observation keys "
                f"{missing_keys}. Available keys: {sorted(batch)}. "
                "Check the dataset observation mapping and configured camera names."
            )

    @staticmethod
    def _camera_pad_mask(batch: dict, key: str, image: Tensor) -> Tensor:
        batch_size = image.shape[0]
        num_slots = image.shape[1] if image.ndim == 5 else 1
        mask = batch.get(f"{key}_is_pad")
        if mask is None:
            return torch.zeros((batch_size, num_slots), dtype=torch.bool, device=image.device)
        mask = mask.to(device=image.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        expected_shape = (batch_size, num_slots)
        if mask.shape != expected_shape:
            raise ValueError(
                f"Padding mask for {key!r} has shape {tuple(mask.shape)}; "
                f"expected {expected_shape} for image shape {tuple(image.shape)}."
            )
        return mask

    def consolidate_images(self, batch: dict) -> dict:
        """Validate and consolidate configured cameras into a BNCHW tensor."""
        image_keys = list(self.config.image_features)
        if not image_keys:
            raise ValueError(
                f"No image features found in configuration. Expected mappings with {OBS_IMAGES} prefix."
            )

        missing_keys = [key for key in image_keys if key not in batch]
        if missing_keys:
            raise KeyError(
                f"Missing required camera observations {missing_keys}. "
                f"Configured cameras: {image_keys}; available keys: {sorted(batch)}."
            )

        reference = batch[image_keys[0]]
        if not isinstance(reference, Tensor) or reference.ndim not in (4, 5):
            shape = tuple(reference.shape) if isinstance(reference, Tensor) else type(reference).__name__
            raise ValueError(f"Expected camera tensors with shape BCHW or BTCHW, got {shape}.")

        images = []
        pad_masks = []
        for key in image_keys:
            image = batch[key]  # (B, C, H, W) or (B, T, C, H, W)
            if not isinstance(image, Tensor) or image.ndim != reference.ndim:
                shape = tuple(image.shape) if isinstance(image, Tensor) else type(image).__name__
                raise ValueError(
                    f"Camera {key!r} has shape {shape}; expected rank {reference.ndim} "
                    f"to match {tuple(reference.shape)}."
                )
            if image.shape[0] != reference.shape[0] or image.shape[-3:] != reference.shape[-3:]:
                raise ValueError(
                    f"Camera {key!r} has incompatible shape {tuple(image.shape)}; "
                    f"expected matching batch, channel, height, and width dimensions with "
                    f"{tuple(reference.shape)}."
                )
            images.append(image)
            pad_masks.append(self._camera_pad_mask(batch, key, image))

        if reference.ndim == 4:
            batch[OBS_IMAGES] = torch.stack(images, dim=1)  # N * (B, C, H, W) -> (B, N, C, H, W)
        else:
            batch[OBS_IMAGES] = torch.cat(images, dim=1)  # N * (B, T, C, H, W) -> (B, N*T, C, H, W)
        batch[OBS_IMAGES_IS_PAD] = torch.cat(pad_masks, dim=1)  # N * (B, T) -> (B, N*T)
        return batch

    def _prepare_image_range(self, images: Tensor) -> tuple[Tensor, bool | Tensor]:
        if images.dtype == torch.bool:
            raise TypeError("Boolean image tensors are not supported.")
        if not images.is_floating_point():
            if images.dtype != torch.uint8:
                min_value, max_value = torch.aminmax(images)
                if bool((min_value < 0) | (max_value > 255)):
                    raise ValueError(
                        f"Integer images must be in [0, 255], got [{int(min_value)}, {int(max_value)}]."
                    )
            return images.to(dtype=torch.float32).div_(255.0), False

        min_values = images.amin(dim=(1, 2, 3))  # (B*N, C, H, W) -> (B*N,)
        max_values = images.amax(dim=(1, 2, 3))  # (B*N, C, H, W) -> (B*N,)
        min_value, max_value, minimum_image_max = torch.stack(
            (min_values.min(), max_values.max(), max_values.min())
        ).tolist()
        normalized_limit = 1.0 + _IMAGE_RANGE_TOLERANCE + 1e-6

        if not math.isfinite(min_value) or not math.isfinite(max_value):
            finite = torch.isfinite(min_values) & torch.isfinite(max_values)
            image_index = int((~finite).nonzero(as_tuple=False)[0, 0])
            raise ValueError(f"Floating-point image {image_index} must contain only finite values.")
        if min_value < -_IMAGE_RANGE_TOLERANCE:
            image_index = int((min_values < -_IMAGE_RANGE_TOLERANCE).nonzero(as_tuple=False)[0, 0])
            raise ValueError(
                f"Floating-point image {image_index} must be non-negative, "
                f"got minimum {float(min_values[image_index]):.6g}."
            )
        if max_value <= normalized_limit:
            return images.clamp(0.0, 1.0) if min_value < 0.0 or max_value > 1.0 else images, False
        if max_value > 255.0 + _IMAGE_RANGE_TOLERANCE:
            image_index = int((max_values > 255.0 + _IMAGE_RANGE_TOLERANCE).nonzero(as_tuple=False)[0, 0])
            raise ValueError(
                f"Floating-point image {image_index} must be in [0, 1] or [0, 255], "
                f"got maximum {float(max_values[image_index]):.6g}."
            )
        if minimum_image_max >= _AMBIGUOUS_FLOAT_IMAGE_MAX:
            return images.clamp(0.0, 255.0) if min_value < 0.0 or max_value > 255.0 else images, True

        high_range = max_values > normalized_limit
        ambiguous = high_range & (max_values < _AMBIGUOUS_FLOAT_IMAGE_MAX)
        clamp_normalized = ~high_range & ((min_values < 0.0) | (max_values > 1.0))
        clamp_high_range = high_range & ((min_values < 0.0) | (max_values > 255.0))
        has_ambiguous, needs_clamp = torch.stack(
            (ambiguous.any(), (clamp_normalized | clamp_high_range).any())
        ).tolist()
        if has_ambiguous:
            image_index = int(ambiguous.nonzero(as_tuple=False)[0, 0])
            raise ValueError(
                f"Ambiguous floating-point image {image_index} range "
                f"[{float(min_values[image_index]):.6g}, {float(max_values[image_index]):.6g}]. "
                "Expected [0, 1] or [0, 255]."
            )

        expanded_mask = high_range[:, None, None, None]  # (B*N,) -> (B*N, 1, 1, 1)
        if needs_clamp:
            upper_bounds = torch.where(expanded_mask, 255.0, 1.0)
            images = images.clamp_min(0.0).minimum(upper_bounds)
        return images, expanded_mask

    def prepare_image(self, batch: dict) -> tuple[Tensor, Tensor]:
        images = batch[OBS_IMAGES]  # (B, N, C, H, W)
        if images.ndim != 5:
            raise ValueError(
                f"{OBS_IMAGES} must have shape (batch, images, channels, height, width), "
                f"got {tuple(images.shape)}."
            )
        if images.shape[2] != 3:
            raise ValueError(f"Rho expects RGB images with 3 channels, got {images.shape[2]}.")

        batch_size, num_images, channels, height, width = images.shape
        expected_image_slots = len(self.config.image_features) * self.config.n_obs_steps
        if num_images != expected_image_slots:
            raise ValueError(
                f"{OBS_IMAGES} has {num_images} image slots, but Rho expects "
                f"{expected_image_slots} from {len(self.config.image_features)} configured "
                f"camera(s) and n_obs_steps={self.config.n_obs_steps}."
            )
        flat_images = images.reshape(
            batch_size * num_images, channels, height, width
        )  # (B, N, C, H, W) -> (B*N, C, H, W)
        flat_images, divide_after_resize = self._prepare_image_range(flat_images)
        if self._resize_transform is not None:
            flat_images = self._resize_transform(flat_images)  # (B*N, C, H, W) -> (B*N, C, H', W')
        if isinstance(divide_after_resize, Tensor):
            flat_images = torch.where(divide_after_resize, flat_images.div(255.0), flat_images)
        elif divide_after_resize:
            flat_images = flat_images.div(255.0)
        images = flat_images.reshape(
            batch_size, num_images, *flat_images.shape[-3:]
        )  # (B*N, C, H', W') -> (B, N, C, H', W')

        image_is_pad = batch.get(OBS_IMAGES_IS_PAD)
        if image_is_pad is None:
            image_is_pad = torch.zeros((batch_size, num_images), dtype=torch.bool, device=images.device)
        else:
            image_is_pad = image_is_pad.to(device=images.device, dtype=torch.bool)
            if image_is_pad.ndim == 3 and image_is_pad.shape[-1] == 1:
                image_is_pad = image_is_pad.squeeze(-1)  # (B, N, 1) -> (B, N)
            if image_is_pad.shape != (batch_size, num_images):
                raise ValueError(
                    f"{OBS_IMAGES_IS_PAD} has shape {tuple(image_is_pad.shape)}; "
                    f"expected {(batch_size, num_images)}."
                )
        return images, (~image_is_pad).to(dtype=torch.int32)  # valid-image mask: (B, N)

    def prepare_state(self, batch: dict) -> Tensor:
        state = batch[OBS_ROBOT]
        if state.ndim == 2:
            state = state.unsqueeze(1)  # (B, D_state) -> (B, 1, D_state)
        if state.ndim != 3:
            raise ValueError(
                f"{OBS_ROBOT} must have shape (batch, state) or (batch, time, state), "
                f"got {tuple(state.shape)}."
            )
        return self.pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch: dict) -> Tensor:
        return self.pad_vector(batch[ACTION], self.config.max_action_dim)

    @staticmethod
    def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
        if vector.shape[-1] == new_dim:
            return vector
        if vector.shape[-1] > new_dim:
            raise ValueError(
                f"Cannot pad vector dimension {vector.shape[-1]} to smaller dimension {new_dim}."
            )
        return F.pad(vector, (0, new_dim - vector.shape[-1]))  # (..., D) -> (..., new_dim)

    def prepare(self, batch: dict, *, include_action: bool = False) -> RhoProcessedObservations:
        prepared_batch = dict(batch)
        has_consolidated_images = (
            OBS_IMAGES in prepared_batch
            and isinstance(prepared_batch[OBS_IMAGES], Tensor)
            and prepared_batch[OBS_IMAGES].ndim == 5
            and not any(key in prepared_batch for key in self.config.image_features if key != OBS_IMAGES)
        )
        if not has_consolidated_images:
            prepared_batch = self.consolidate_images(prepared_batch)
        if OBS_ROBOT not in prepared_batch:
            raise KeyError(
                f"Missing required Rho observation key {OBS_ROBOT!r}. "
                f"Available keys: {sorted(prepared_batch)}."
            )
        if OBS_TASK not in prepared_batch:
            raise KeyError(
                f"Missing required Rho observation key {OBS_TASK!r}. "
                f"Available keys: {sorted(prepared_batch)}."
            )
        if include_action and ACTION not in prepared_batch:
            raise KeyError(
                f"Missing required Rho training key {ACTION!r}. Available keys: {sorted(prepared_batch)}."
            )

        image, image_mask = self.prepare_image(prepared_batch)
        return RhoProcessedObservations(
            image=image,
            image_mask=image_mask,
            state=self.prepare_state(prepared_batch),
            action=self.prepare_action(prepared_batch) if include_action else None,
        )

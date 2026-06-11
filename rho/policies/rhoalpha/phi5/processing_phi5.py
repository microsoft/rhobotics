"""
GPU-native NaFlex image processor for Phi5 (Bunny-Phi4).

Accepts (C, H, W) float tensors in [0, 1] directly on GPU, avoiding
CPU round-trips through PIL/numpy that the original
Siglip2ImageProcessorNoUpscale requires.

Drop-in replacement: produces the same BatchFeature keys
(pixel_values, pixel_attention_mask, spatial_shapes) with the same
semantics so that both BunnyPhi4Processor.__call__ and
Phi5Backbone.process_batch work without changes.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature

# ---------------------------------------------------------------------------
# Inlined from transformers.models.siglip2.image_processing_siglip2 so this
# module has no hard dependency on a specific transformers version.
# ---------------------------------------------------------------------------


def _get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    eps: float = 1e-5,
) -> tuple[int, int]:
    """Binary-search for the largest scale that keeps patches <= max."""

    def _scaled(scale: float, size: int) -> int:
        s = math.ceil(size * scale / patch_size) * patch_size
        return max(patch_size, int(s))

    lo, hi = eps / 10, 100.0
    while (hi - lo) >= eps:
        mid = (lo + hi) / 2
        th, tw = _scaled(mid, image_height), _scaled(mid, image_width)
        if (th // patch_size) * (tw // patch_size) <= max_num_patches:
            lo = mid
        else:
            hi = mid

    return _scaled(lo, image_height), _scaled(lo, image_width)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------


class Phi5ImageProcessor(nn.Module):
    """
    GPU-native NaFlex image processor for Phi5.

    Equivalent to ``Siglip2ImageProcessorNoUpscale`` but operates entirely
    on PyTorch tensors, keeping data on whichever device it already lives on.

    Supports both per-image ``__call__`` and fully batched
    ``process_batched`` for same-resolution images (robotics training).
    """

    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(
        self,
        patch_size: int = 16,
        max_num_patches: int = 3600,
        min_num_patches: int = 256,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches

        image_mean = image_mean or [0.5, 0.5, 0.5]
        image_std = image_std or [0.5, 0.5, 0.5]
        self.register_buffer(
            "_image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "_image_std",
            torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1),
        )
        self.to(device)

    @classmethod
    def from_siglip2_processor(cls, processor, device: str = "cuda"):
        """Create from an existing Siglip2ImageProcessorNoUpscale or RhoImageProcessorNoUpscale."""
        # Handle both plain attrs (Siglip2) and registered buffers (Rho)
        if hasattr(processor, "image_mean"):
            mean = processor.image_mean
            std = processor.image_std
        else:
            mean = processor._image_mean.flatten().tolist()
            std = processor._image_std.flatten().tolist()
        return cls(
            patch_size=processor.patch_size,
            max_num_patches=processor.max_num_patches,
            min_num_patches=processor.min_num_patches,
            image_mean=mean,
            image_std=std,
            device=device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_target_size(self, height: int, width: int) -> tuple[int, int]:
        p = self.patch_size
        num_patches = max((height // p) * (width // p), 1)

        if num_patches < self.min_num_patches:
            target = self.min_num_patches
        elif num_patches > self.max_num_patches:
            target = self.max_num_patches
        else:
            target = num_patches

        return _get_image_size_for_max_num_patches(
            image_height=height,
            image_width=width,
            patch_size=p,
            max_num_patches=target,
        )

    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """(C, H, W) → (num_patches, patch_dim)"""
        C, H, W = image.shape  # noqa: N806
        p = self.patch_size
        nph, npw = H // p, W // p
        patches = image.view(C, nph, p, npw, p)
        patches = patches.permute(1, 3, 2, 4, 0).contiguous()
        return patches.view(nph * npw, p * p * C)

    def _extract_patches_batched(self, images: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) → (N, num_patches, patch_dim)"""
        N, C, H, W = images.shape  # noqa: N806
        p = self.patch_size
        nph, npw = H // p, W // p
        # (N, C, nph, p, npw, p)
        patches = images.view(N, C, nph, p, npw, p)
        # (N, nph, npw, p, p, C) → (N, nph*npw, p*p*C)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        return patches.view(N, nph * npw, p * p * C)

    def _pad_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = patches.shape[0]
        if n >= self.max_num_patches:
            return (
                patches[: self.max_num_patches],
                torch.ones(self.max_num_patches, dtype=torch.bool, device=patches.device),
            )
        pad = torch.zeros(
            self.max_num_patches - n,
            patches.shape[1],
            dtype=patches.dtype,
            device=patches.device,
        )
        mask = torch.zeros(self.max_num_patches, dtype=torch.bool, device=patches.device)
        mask[:n] = True
        return torch.cat([patches, pad], dim=0), mask

    # ------------------------------------------------------------------
    # Per-image entry point (original interface)
    # ------------------------------------------------------------------

    def __call__(self, images: list[torch.Tensor], return_tensors: str = "pt") -> BatchFeature:
        """
        Process a list of ``(C, H, W)`` float tensors in ``[0, 1]``.

        Returns a :class:`BatchFeature` with keys ``pixel_values``,
        ``pixel_attention_mask``, and ``spatial_shapes``.
        """
        all_patches: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_shapes: list[torch.Tensor] = []

        for image in images:
            device = image.device
            C, H, W = image.shape  # noqa: N806
            target_h, target_w = self._compute_target_size(H, W)

            if target_h != H or target_w != W:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]

            mean = self._image_mean.to(device=device, dtype=image.dtype)
            std = self._image_std.to(device=device, dtype=image.dtype)
            image = (image - mean) / std

            patches = self._extract_patches(image)
            padded, mask = self._pad_patches(patches)

            nph = target_h // self.patch_size
            npw = target_w // self.patch_size

            all_patches.append(padded)
            all_masks.append(mask)
            all_shapes.append(torch.tensor([nph, npw], device=device))

        return BatchFeature(
            data={
                "pixel_values": torch.stack(all_patches),
                "pixel_attention_mask": torch.stack(all_masks),
                "spatial_shapes": torch.stack(all_shapes),
            },
            tensor_type=return_tensors,
        )

    # ------------------------------------------------------------------
    # Batched entry point — all images same resolution (robotics)
    # ------------------------------------------------------------------

    def process_batched(self, images: torch.Tensor, return_tensors: str = "pt") -> BatchFeature:
        """
        Process a batch of same-resolution images in one shot.

        Args:
            images: ``(N, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            :class:`BatchFeature` with ``pixel_values`` ``(N, max_patches, patch_dim)``,
            ``pixel_attention_mask`` ``(N, max_patches)``,
            ``spatial_shapes`` ``(N, 2)``.
        """
        device = images.device
        N, C, H, W = images.shape  # noqa: N806
        target_h, target_w = self._compute_target_size(H, W)

        # Resize all images at once if needed
        if target_h != H or target_w != W:
            images = F.interpolate(
                images,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize all at once
        mean = self._image_mean.to(device=device, dtype=images.dtype)
        std = self._image_std.to(device=device, dtype=images.dtype)
        images = (images - mean) / std

        # Extract patches for the whole batch
        nph = target_h // self.patch_size
        npw = target_w // self.patch_size
        num_patches = nph * npw
        patches = self._extract_patches_batched(images)  # (N, num_patches, patch_dim)

        # Clamp to max_num_patches but do NOT pad to it — all images
        # share the same resolution so no padding is needed and we avoid
        # sending thousands of useless padding tokens through the vision
        # transformer (e.g. 256 real patches vs 3600 padded).
        if num_patches > self.max_num_patches:
            patches = patches[:, : self.max_num_patches]
            num_patches = self.max_num_patches

        # All patches are real (no padding) — mask is all-True.
        mask = torch.ones(N, num_patches, dtype=torch.bool, device=device)

        spatial = (
            torch.tensor(
                [nph, npw],
                device=device,
            )
            .unsqueeze(0)
            .expand(N, 2)
        )

        return BatchFeature(
            data={
                "pixel_values": patches,
                "pixel_attention_mask": mask,
                "spatial_shapes": spatial,
            },
            tensor_type=return_tensors,
        )

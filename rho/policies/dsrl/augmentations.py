"""Data augmentations for DSRL-SAC training.

Ported from dsrl_pi0 JAX implementation to PyTorch.
"""

import torch
import torch.nn.functional as F


def random_crop(images: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Apply random crop with reflect padding to a batch of images.

    Args:
        images: (B, C, H, W) float tensor
        padding: number of pixels to pad on each side

    Returns:
        Cropped images with original spatial dimensions.
    """
    B, C, H, W = images.shape
    padded = F.pad(images, [padding] * 4, mode="reflect")
    # Random crop offsets per image
    crop_h = torch.randint(0, 2 * padding + 1, (B,), device=images.device)
    crop_w = torch.randint(0, 2 * padding + 1, (B,), device=images.device)
    # Gather crops
    cropped = torch.empty_like(images)
    for i in range(B):
        cropped[i] = padded[i, :, crop_h[i] : crop_h[i] + H, crop_w[i] : crop_w[i] + W]
    return cropped


def random_crop_paired(
    images: torch.Tensor, next_images: torch.Tensor, padding: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same random crop to obs and next_obs images.

    Args:
        images: (B, C, H, W) float tensor
        next_images: (B, C, H, W) float tensor
        padding: number of pixels to pad on each side

    Returns:
        Tuple of (cropped_images, cropped_next_images) with same crop per sample.
    """
    B, C, H, W = images.shape
    padded = F.pad(images, [padding] * 4, mode="reflect")
    next_padded = F.pad(next_images, [padding] * 4, mode="reflect")

    crop_h = torch.randint(0, 2 * padding + 1, (B,), device=images.device)
    crop_w = torch.randint(0, 2 * padding + 1, (B,), device=images.device)

    cropped = torch.empty_like(images)
    next_cropped = torch.empty_like(next_images)
    for i in range(B):
        cropped[i] = padded[i, :, crop_h[i] : crop_h[i] + H, crop_w[i] : crop_w[i] + W]
        next_cropped[i] = next_padded[i, :, crop_h[i] : crop_h[i] + H, crop_w[i] : crop_w[i] + W]
    return cropped, next_cropped


def color_jitter(
    images: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.03,
    apply_prob: float = 0.8,
) -> torch.Tensor:
    """Apply color jitter augmentation to a batch of images.

    Expects images in (B, C, H, W) format with values in [0, 1].
    For multi-camera inputs (C > 3), each camera's 3 channels are jittered independently.

    Args:
        images: (B, C, H, W) float tensor in [0, 1]
        brightness: max brightness delta
        contrast: max contrast delta
        saturation: max saturation delta
        hue: max hue delta
        apply_prob: probability of applying jitter per image

    Returns:
        Augmented images in [0, 1].
    """
    B, C, H, W = images.shape
    num_cameras = C // 3
    result = images.clone()

    # Per-image mask for whether to apply
    apply_mask = torch.rand(B, device=images.device) < apply_prob

    for cam_idx in range(num_cameras):
        c_start = cam_idx * 3
        c_end = c_start + 3
        cam_images = images[:, c_start:c_end]  # (B, 3, H, W)

        # Brightness
        delta = torch.empty(B, 1, 1, 1, device=images.device).uniform_(-brightness, brightness)
        aug = cam_images + delta

        # Contrast
        mean = cam_images.mean(dim=(2, 3), keepdim=True)
        factor = torch.empty(B, 1, 1, 1, device=images.device).uniform_(1 - contrast, 1 + contrast)
        aug = factor * (aug - mean) + mean

        # Saturation & Hue (convert to HSV, adjust, convert back)
        # Use a simplified approach: adjust saturation via desaturation blending
        gray = cam_images.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        sat_factor = torch.empty(B, 1, 1, 1, device=images.device).uniform_(1 - saturation, 1 + saturation)
        aug = sat_factor * aug + (1 - sat_factor) * gray

        # Hue shift (approximate: rotate RGB channels slightly)
        hue_delta = torch.empty(B, device=images.device).uniform_(-hue, hue)
        cos_h = torch.cos(hue_delta * 2 * 3.14159).view(B, 1, 1, 1)
        sin_h = torch.sin(hue_delta * 2 * 3.14159).view(B, 1, 1, 1)
        r, g, b = aug[:, 0:1], aug[:, 1:2], aug[:, 2:3]
        new_r = cos_h * r + sin_h * g
        new_g = -sin_h * r + cos_h * g
        aug = torch.cat([new_r, new_g, b], dim=1)

        aug = aug.clamp(0.0, 1.0)

        # Apply mask: keep original where apply_mask is False
        mask = apply_mask.view(B, 1, 1, 1)
        result[:, c_start:c_end] = torch.where(mask, aug, cam_images)

    return result


def augment_batch(
    images: torch.Tensor,
    next_images: torch.Tensor,
    crop_padding: int = 4,
    use_color_jitter: bool = True,
    aug_next: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full augmentation pipeline to a batch pair.

    Args:
        images: (B, C, H, W) float tensor in [0, 1]
        next_images: (B, C, H, W) float tensor in [0, 1]
        crop_padding: random crop padding
        use_color_jitter: whether to apply color jitter
        aug_next: whether to augment next_images too

    Returns:
        Tuple of (augmented_images, augmented_next_images).
    """
    if aug_next:
        images, next_images = random_crop_paired(images, next_images, crop_padding)
    else:
        images = random_crop(images, crop_padding)

    if use_color_jitter:
        images = color_jitter(images)
        if aug_next:
            next_images = color_jitter(next_images)

    return images, next_images

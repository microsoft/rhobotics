import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def _get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize without distortion via padding.

    Accepts [B, C, H, W] or [B, H, W, C]. If float, expects values in [-1, 1].
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D images, got shape={tuple(images.shape)}")

    channels_last = images.shape[-1] <= 4
    if channels_last:
        images = images.permute(0, 3, 1, 2)

    _, _, cur_height, cur_width = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype.is_floating_point:
        resized = resized.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + rem_w

    pad_value = 0 if images.dtype == torch.uint8 else -1.0
    padded = F.pad(
        resized,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=pad_value,
    )

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)
    return padded


def _preprocess_observation_pytorch(
    observation,
    *,
    train: bool,
    image_keys: tuple[str, ...],
    image_resolution: tuple[int, int],
):
    """Minimal torch preprocessing matching OpenPI behavior.

    - Ensures keys exist
    - Resizes with padding
    - Applies OpenPI-style augmentations (only when train=True)
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    out_images: dict[str, torch.Tensor] = {}
    for key in image_keys:
        image = observation.images[key]

        # LeRobot PushT commonly yields stacked observations with shape [B, T, C, H, W]
        # (or [B, T, H, W, C]). PI0 consumes a single 4D image tensor per key, so
        # we take the most recent frame.
        if image.ndim == 5:
            image = image[:, -1]

        is_channels_first = image.ndim == 4 and image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            image = _resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert [-1,1] -> [0,1]
            image = image / 2.0 + 0.5

            # Color-jitter-like ops (minimal, stays close to OpenPI intent)
            brightness = 0.7 + torch.rand(1, device=image.device) * 0.6
            image = image * brightness
            contrast = 0.6 + torch.rand(1, device=image.device) * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast + mean
            saturation = 0.5 + torch.rand(1, device=image.device) * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation
            image = torch.clamp(image, 0, 1)

            # Back to [-1,1]
            image = image * 2.0 - 1.0

        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image

    out_masks: dict[str, torch.Tensor] = {}
    batch_shape = observation.state.shape[:-1]
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    class _ProcessedObs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    return _ProcessedObs(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )


class _SentencePieceTokenizer:
    def __init__(self, model_path: str | Path, *, max_len: int = 48):
        import sentencepiece

        self._max_len = int(max_len)
        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(self, prompt: str, *, state: torch.Tensor | None = None) -> tuple[Tensor, Tensor]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            state_np = state.detach().to("cpu").to(torch.float32).numpy()
            discretized = (torch.from_numpy(state_np).clamp(-1, 1) + 1) * 0.5
            discretized = (discretized * 255).to(torch.int32).numpy().tolist()
            state_str = " ".join(map(str, discretized))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._sp.encode(full_prompt, add_bos=True)
        else:
            tokens = self._sp.encode(cleaned_text, add_bos=True) + self._sp.encode("\n")

        if len(tokens) < self._max_len:
            mask = [True] * len(tokens) + [False] * (self._max_len - len(tokens))
            tokens = tokens + [0] * (self._max_len - len(tokens))
        else:
            if len(tokens) > self._max_len:
                logger.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return (
            torch.tensor(tokens, dtype=torch.int32),
            torch.tensor(mask, dtype=torch.bool),
        )

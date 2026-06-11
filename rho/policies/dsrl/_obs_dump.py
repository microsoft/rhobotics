"""One-shot obs-input dumper for FlowDAgger verification.

Used by both FlowDAggerPolicy._encode_obs (the inference encoder input) and
FlowDAggerTrainer._ingest (the training ingest path) to verify that what
the noise policy actually sees -- keys, shapes, dtypes, value ranges, and
the actual image bytes -- matches expectations before any DAgger training
kicks in.

Each call site reuses dump_obs_once(...) with a unique tag so the dump
fires exactly once per tag.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FIRED: set[str] = set()


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _describe(name: str, val) -> str:
    if val is None:
        return f"  {name}: None"
    if isinstance(val, (list, tuple)):
        first = type(val[0]).__name__ if val else "empty"
        return f"  {name}: {type(val).__name__}(len={len(val)})  first={first}"
    if isinstance(val, str):
        return f"  {name}: str = {val!r}"
    try:
        arr = _as_numpy(val)
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size == 0:
            stats = "all-non-finite"
        else:
            stats = f"min={finite.min():+.4f} max={finite.max():+.4f} mean={finite.mean():+.4f}"
        return f"  {name}: shape={arr.shape} dtype={arr.dtype} {stats}"
    except Exception as e:
        return f"  {name}: type={type(val).__name__}  (no numpy view: {e})"


def _save_image(arr: np.ndarray, out_path: Path) -> bool:
    """Best-effort image dump as PNG. Strips batch / time / channel-first.

    Returns True if a file was written.
    """
    img = arr
    while img.ndim > 3:
        img = img[0]
    if img.ndim == 2:
        pass  # grayscale, leave alone
    elif img.ndim == 3:
        # If first dim looks like channels, transpose to HWC.
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
    else:
        return False

    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.5:
            img = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    try:
        import cv2

        # cv2 expects BGR. If 3-channel, assume input is RGB (post-process_input)
        # and convert. If grayscale, write as-is.
        if img.ndim == 3 and img.shape[-1] == 3:
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(out_path), img)
        return True
    except Exception as e:
        logger.debug(f"cv2 image write failed: {e}")
        try:
            from PIL import Image

            Image.fromarray(img).save(str(out_path))
            return True
        except Exception as e2:
            logger.warning(f"image write failed for {out_path}: {e2}")
            return False


def dump_obs_once(
    tag: str,
    obs: dict[str, Any],
    *,
    image_keys: Iterable[str] | None = None,
    state_keys: Iterable[str] = ("observation.state", "state", "tcp_pose", "joint_positions"),
    out_dir: str = "/tmp/flowdagger_debug",
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Log obs structure + dump images to disk. Fires once per ``tag``."""
    if tag in _FIRED:
        return
    _FIRED.add(tag)

    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"=== [obs-dump:{tag}] one-shot dump (dir={out}) ==="]
    if isinstance(obs, dict):
        lines.append(f"  obs keys ({len(obs)}): {list(obs.keys())}")
        for k, v in obs.items():
            lines.append(_describe(k, v))
    else:
        lines.append(_describe("obs", obs))

    if extra_fields:
        lines.append("  extra:")
        for k, v in extra_fields.items():
            lines.append(_describe(f"  {k}", v))

    # State values verbatim, since shape doesn't tell you if it's zeros / NaN.
    if isinstance(obs, dict):
        for sk in state_keys:
            if sk in obs:
                try:
                    arr = _as_numpy(obs[sk]).reshape(-1)
                    head = ", ".join(f"{x:+.4f}" for x in arr[:16].tolist())
                    lines.append(f"  state[{sk}] values: [{head}]")
                except Exception:
                    pass

    saved: list[str] = []
    if image_keys is not None and isinstance(obs, dict):
        for k in image_keys:
            if k not in obs:
                lines.append(f"  ! image key '{k}' missing from obs")
                continue
            try:
                arr = _as_numpy(obs[k])
            except Exception as e:
                lines.append(f"  ! could not numpy-ify '{k}': {e}")
                continue
            safe = k.replace("/", "_").replace(".", "_")
            png_path = out / f"{safe}.png"
            if _save_image(arr, png_path):
                saved.append(str(png_path))

    if saved:
        lines.append("  saved images:")
        for p in saved:
            lines.append(f"    {p}")

    logger.info("\n".join(lines))
    print("\n".join(lines), flush=True)

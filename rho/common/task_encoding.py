"""Fixed-width byte-tensor encoding for the per-sample ``task`` field.

Why this exists: accelerate's ``dispatch_batches=True`` data-parallel path
scatters / broadcasts batches across ranks, and its slicing code operates on
tensors. Python strings (or lists of strings after default_collate) are not
tensors and crash the slice path. Encoding each sample's task as a fixed-
length uint8 tensor keeps the batch tensor-only without moving tokenization
into the data pipeline.

Format: ``(MAX_TASK_BYTES,) torch.uint8``, UTF-8 bytes left-aligned, null-
padded with zeros. Decode strips trailing nulls then utf-8 decodes with
``errors="replace"`` so a corrupted byte doesn't crash training.
"""

from __future__ import annotations

import torch

from rho.common.constants import MAX_TASK_BYTES


def encode_task_bytes(s: str | bytes | None, max_bytes: int = MAX_TASK_BYTES) -> torch.Tensor:
    """Encode a single task string as a fixed-width uint8 tensor."""
    if s is None:
        s = ""
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    b = b[:max_bytes]
    out = torch.zeros(max_bytes, dtype=torch.uint8)
    if len(b) > 0:
        out[: len(b)] = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    return out


def decode_task_bytes(t: torch.Tensor) -> list[str]:
    """Decode a (B, max_bytes) uint8 batch tensor back to a list of B strings.

    Also accepts a 1-D (max_bytes,) tensor for single-sample decode, returning
    a length-1 list. Strips trailing nulls before utf-8 decoding.
    """
    if t.dim() == 1:
        t = t.unsqueeze(0)
    out: list[str] = []
    for row in t.cpu().to(dtype=torch.uint8):
        raw = bytes(row.tolist()).rstrip(b"\x00")
        out.append(raw.decode("utf-8", errors="replace"))
    return out


def maybe_decode_task(value) -> list[str]:
    """Normalize a batch ``task`` field to ``list[str]``.

    Accepts any of:
        * ``torch.Tensor`` (B, MAX_TASK_BYTES) - tensor-only training path
        * ``list[str]`` - eval / legacy path (e.g. env.step produces list[str])
        * ``str`` - single-sample path, wrapped to length-1 list
    """
    if isinstance(value, torch.Tensor):
        return decode_task_bytes(value)
    if isinstance(value, str):
        return [value]
    return list(value)


__all__ = ["encode_task_bytes", "decode_task_bytes", "maybe_decode_task"]

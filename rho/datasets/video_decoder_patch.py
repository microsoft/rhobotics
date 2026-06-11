"""LRU-bounded VideoDecoderCache to replace lerobot's unbounded global cache.

The default ``VideoDecoderCache`` in ``lerobot.datasets.video_utils`` keeps
every opened ``VideoDecoder`` alive for the lifetime of the process.  With
large multi-shard datasets (e.g. 10 Scale shards × 4 cameras × 230 episodes)
this results in thousands of cached decoders and significant memory growth.

This module provides a drop-in replacement with an LRU eviction policy.
Call :func:`patch_video_decoder_cache` early in training to install it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class LRUVideoDecoderCache:
    """Thread-safe LRU cache for video decoders with bounded size."""

    def __init__(self, max_size: int = 128):
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_decoder(self, video_path: str):
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        import fsspec

        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
                self._hits += 1
                return self._cache[video_path][0]

            self._misses += 1

            # Evict oldest entries if at capacity
            while len(self._cache) >= self._max_size:
                evicted_path, (_, old_handle) = self._cache.popitem(last=False)
                with contextlib.suppress(Exception):
                    old_handle.close()
                self._evictions += 1

            file_handle = fsspec.open(video_path).__enter__()
            decoder = VideoDecoder(file_handle, seek_mode="approximate")
            self._cache[video_path] = (decoder, file_handle)
            return decoder

    def clear(self):
        with self._lock:
            for _, file_handle in self._cache.values():
                with contextlib.suppress(Exception):
                    file_handle.close()
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }


def patch_video_decoder_cache(max_size: int = 128) -> LRUVideoDecoderCache:
    """Replace lerobot's global VideoDecoderCache with an LRU-bounded version.

    Args:
        max_size: Maximum number of video decoders to keep open simultaneously.

    Returns:
        The installed LRUVideoDecoderCache instance.
    """
    from lerobot.datasets import video_utils

    lru_cache = LRUVideoDecoderCache(max_size=max_size)
    video_utils._default_decoder_cache = lru_cache
    logger.info(f"Patched VideoDecoderCache with LRU cache (max_size={max_size})")
    return lru_cache

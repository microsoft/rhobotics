"""Unified serialization utilities for converting dataclasses and complex objects to serializable formats.

This module provides a common interface for serializing dataclasses, numpy arrays, torch tensors,
and other complex objects to dictionaries, JSON, and YAML formats.

The key convention is that dataclasses can implement a `to_dict()` method to provide custom
serialization logic (e.g., excluding large fields like stats).
"""

import enum
import json
import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def serialize_to_dict(obj: Any) -> Any:
    """Recursively serialize an object to a JSON/YAML-compatible dictionary.

    This function handles:
    - Dataclasses: Checks for `to_dict()` method first, otherwise serializes all fields
    - Numpy arrays: Converts to Python lists
    - Torch tensors: Converts to Python lists
    - Paths: Converts to strings
    - Dicts, lists, tuples: Recursively serializes contents
    - Primitives: Returns as-is

    Args:
        obj: The object to serialize. Can be a dataclass, dict, list, numpy array, etc.

    Returns:
        A JSON/YAML-serializable representation of the object.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str
        ...     values: np.ndarray
        >>> cfg = Config(name="test", values=np.array([1, 2, 3]))
        >>> serialize_to_dict(cfg)
        {'name': 'test', 'values': [1, 2, 3]}
    """
    # Check for custom to_dict method first (allows dataclasses to customize serialization)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        # For ChoiceRegistry subclasses, we need to include the 'type' key for draccus decoding
        # Check if the class has get_choice_name (indicates it's a ChoiceRegistry)
        if hasattr(obj, "get_choice_name"):
            result = {"type": obj.get_choice_name(obj.__class__)}
            for f in fields(obj):
                result[f.name] = serialize_to_dict(getattr(obj, f.name))
            return result
        return {f.name: serialize_to_dict(getattr(obj, f.name)) for f in fields(obj)}

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: serialize_to_dict(v) for k, v in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [serialize_to_dict(v) for v in obj]

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle numpy scalar types (including np.bool_)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Handle torch tensors
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Handle torch dtypes - serialize as just the type name (e.g., "bfloat16" not "torch.bfloat16")
    if isinstance(obj, torch.dtype):
        return str(obj).split(".")[-1]

    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Handle enums
    if isinstance(obj, enum.Enum):
        return obj.name

    # Primitives and None pass through directly
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle type objects
    if isinstance(obj, type):
        return obj.__name__

    # For other types, try to convert to string as last resort
    try:
        # Try JSON encoding to see if it's already serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        logger.debug(f"Converting non-serializable type {type(obj).__name__} to string")
        return str(obj)


def make_yaml_safe(obj: Any, strip_none: bool = False) -> Any:
    """Recursively sanitize an already-serialized object for yaml.safe_dump compatibility.

    This is a post-processing step applied after serialize_to_dict() or draccus.encode()
    to ensure the output can be passed to yaml.safe_dump without errors.

    Handles:
    - Enum objects → their .value (safety net for objects not caught by serialize_to_dict)
    - Path objects → strings
    - dict keys → recursively sanitized
    - Optionally strips None-valued dict entries (needed for draccus round-trip
      where None for complex Union-typed fields triggers decode errors)

    Args:
        obj: The object to sanitize (typically output of serialize_to_dict or draccus.encode).
        strip_none: If True, drop dict entries whose value is None.

    Returns:
        A yaml.safe_dump-compatible representation.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v_safe = make_yaml_safe(v, strip_none=strip_none)
            if strip_none and v_safe is None:
                continue
            out[make_yaml_safe(k)] = v_safe
        return out
    if isinstance(obj, (list, tuple)):
        return [make_yaml_safe(v, strip_none=strip_none) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def fixup_feature_shapes(features_dict: dict) -> None:
    """Convert feature shape lists back to strings for YAML round-trip.

    YAML has no tuple type, so serialization converts tuple shapes to lists.
    DataConfig.__post_init__ expects shapes as strings like ``"(3, 224, 224)"``
    and parses them.  This function converts shape lists back to that string format.

    Args:
        features_dict: A dictionary of feature definitions, where values may contain
            a ``"shape"`` key with a list value (e.g. ``[3, 224, 224]``).

    Modifies the dictionary in-place.
    """
    for v in features_dict.values():
        if isinstance(v, dict) and "shape" in v and isinstance(v["shape"], list):
            shape_items = ", ".join(str(x) for x in v["shape"])
            if len(v["shape"]) == 1:
                shape_items += ","
            v["shape"] = "(" + shape_items + ")"


def register_draccus_encoders() -> None:
    """Register draccus encoders for numpy and torch types.

    Safe to call multiple times — idempotent.  Also called automatically
    at module import time so callers don't need to remember.
    """
    import draccus

    @draccus.encode.register(np.ndarray)
    def _encode_ndarray(obj: np.ndarray):
        return obj.tolist()

    @draccus.encode.register(np.floating)
    def _encode_np_floating(obj):
        return float(obj)

    @draccus.encode.register(np.integer)
    def _encode_np_integer(obj):
        return int(obj)


# Auto-register on import so every consumer gets the encoders for free.
register_draccus_encoders()


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def serialize_to_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize an object and write it to a JSON file.

    Args:
        obj: The object to serialize.
        path: Path to the output JSON file.
        indent: JSON indentation level (default: 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = serialize_to_dict(obj)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=indent)


def serialize_to_yaml(obj: Any, path: str | Path, default_flow_style: bool = False) -> None:
    """Serialize an object and write it to a YAML file.

    Args:
        obj: The object to serialize.
        path: Path to the output YAML file.
        default_flow_style: YAML flow style setting (default: False for block style).
    """
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = serialize_to_dict(obj)
    with open(path, "w") as f:
        yaml.safe_dump(serialized, f, default_flow_style=default_flow_style, sort_keys=False)


def serialize_to_yaml_string(obj: Any, default_flow_style: bool = False) -> str:
    """Serialize an object to a YAML string.

    Args:
        obj: The object to serialize.
        default_flow_style: YAML flow style setting (default: False for block style).

    Returns:
        YAML string representation of the object.
    """
    import yaml

    serialized = serialize_to_dict(obj)
    return yaml.safe_dump(serialized, default_flow_style=default_flow_style, sort_keys=False)


def serialize_to_json_string(obj: Any, indent: int = 2) -> str:
    """Serialize an object to a JSON string.

    Args:
        obj: The object to serialize.
        indent: JSON indentation level (default: 2).

    Returns:
        JSON string representation of the object.
    """
    serialized = serialize_to_dict(obj)
    return json.dumps(serialized, indent=indent)

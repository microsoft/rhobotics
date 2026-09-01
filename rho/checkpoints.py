"""Versioned checkpoint bundles for portable model weights and trusted resume state."""

import fcntl
import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import load_state_dict_from_file, save_torch_state_dict, snapshot_download
from safetensors import safe_open
from torch import nn

from rho.common.serialization import serialize_to_dict

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "rho"
BUNDLE_VERSION = 1
MANIFEST_FILE = "manifest.json"
POLICY_CONFIG_FILE = "policy.json"
FEATURES_FILE = "features.json"
DATA_CONFIG_FILE = "data_config.json"
STATS_FILE = "stats.json"
TRAINING_STATE_FILE = "training_state.pt"
WEIGHTS_INDEX_FILE = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = "5GB"
CHECKPOINT_ALLOW_PATTERNS = (
    MANIFEST_FILE,
    POLICY_CONFIG_FILE,
    FEATURES_FILE,
    DATA_CONFIG_FILE,
    STATS_FILE,
    "model*.safetensors",
    WEIGHTS_INDEX_FILE,
)
MACHINE_SPECIFIC_PATH_FIELDS = {
    "output_dir",
    "pretrained_checkpoint",
    "root_dir",
    "vlm_backbone_folder",
}


def is_checkpoint_bundle(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / MANIFEST_FILE).is_file()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def _policy_for_saving(policy: nn.Module) -> nn.Module:
    if isinstance(policy, torch.nn.parallel.DistributedDataParallel):
        return policy.module
    return policy


def resolve_checkpoint(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    include_training_state: bool = False,
) -> Path:
    """Resolve a local checkpoint or download a portable Hugging Face bundle."""
    local_path = Path(source).expanduser()
    if local_path.exists():
        resolved = resolve_latest_checkpoint(local_path) if local_path.is_dir() else local_path
        if resolved is None:
            raise FileNotFoundError(f"No completed checkpoint found at: {local_path}")
        return resolved

    allow_patterns = list(CHECKPOINT_ALLOW_PATTERNS)
    if include_training_state:
        allow_patterns.append(TRAINING_STATE_FILE)
    try:
        downloaded = snapshot_download(
            repo_id=str(source),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve Hugging Face checkpoint {source!r}"
            + (f" at revision {revision!r}" if revision is not None else "")
        ) from exc

    resolved = resolve_latest_checkpoint(Path(downloaded))
    if resolved is None:
        raise FileNotFoundError(f"Downloaded Hugging Face repository is not a valid checkpoint: {source}")
    return resolved


def save_state_dict_bundle(
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: str | Path,
    *,
    step: int,
    policy_config: dict[str, Any] | None = None,
    data_config: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
    max_shard_size: int | str = DEFAULT_MAX_SHARD_SIZE,
) -> Path:
    """Save a portable bundle from an in-memory policy state dictionary."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")

    temporary_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.{os.getpid()}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary checkpoint path already exists: {temporary_dir}")
    temporary_dir.mkdir()
    save_started = time.perf_counter()

    try:
        weights_started = time.perf_counter()
        save_torch_state_dict(
            state_dict,
            temporary_dir,
            safe_serialization=True,
            max_shard_size=max_shard_size,
            metadata={
                "format": "pt",
                "rho_format": BUNDLE_FORMAT,
                "rho_version": str(BUNDLE_VERSION),
            },
        )
        weights_seconds = time.perf_counter() - weights_started

        metadata_started = time.perf_counter()
        policy_config = serialize_to_dict(policy_config) if policy_config is not None else None
        features = None
        if isinstance(policy_config, dict):
            policy_config = _strip_machine_specific_paths(policy_config)
            features = policy_config.pop("feature_dict", None)
            _write_json(temporary_dir / POLICY_CONFIG_FILE, policy_config)
        if features is not None:
            _write_json(temporary_dir / FEATURES_FILE, features)

        has_stats = False
        if data_config is not None:
            serialized_data_config = serialize_to_dict(data_config)
            if not isinstance(serialized_data_config, dict):
                raise TypeError("Serialized data config must be a dictionary")
            serialized_data_config = _strip_machine_specific_paths(serialized_data_config)
            stats = serialized_data_config.pop("stats", None)
            if stats is not None:
                _write_json(temporary_dir / STATS_FILE, stats)
                has_stats = True
            _write_json(temporary_dir / DATA_CONFIG_FILE, serialized_data_config)
        metadata_seconds = time.perf_counter() - metadata_started

        training_state_seconds = None
        if training_state is not None:
            training_state_started = time.perf_counter()
            torch.save(training_state, temporary_dir / TRAINING_STATE_FILE)
            training_state_seconds = time.perf_counter() - training_state_started

        manifest = {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "step": step,
            "weights": {
                "format": "safetensors",
                "max_shard_size": str(max_shard_size),
            },
            "artifacts": {
                "policy": POLICY_CONFIG_FILE if policy_config is not None else None,
                "features": FEATURES_FILE if features is not None else None,
                "data_config": DATA_CONFIG_FILE if data_config is not None else None,
                "stats": STATS_FILE if has_stats else None,
                "training_state": TRAINING_STATE_FILE if training_state is not None else None,
            },
            "save_timings_seconds": {
                "model_weights": weights_seconds,
                "metadata": metadata_seconds,
                "training_state": training_state_seconds,
                "total_before_publish": time.perf_counter() - save_started,
            },
        }
        _write_json(temporary_dir / MANIFEST_FILE, manifest)
        temporary_dir.replace(checkpoint_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    logger.info(
        "Checkpoint bundle saved in %.2fs (weights %.2fs, metadata %.2fs, training state %s)",
        time.perf_counter() - save_started,
        weights_seconds,
        metadata_seconds,
        f"{training_state_seconds:.2f}s" if training_state_seconds is not None else "not saved",
    )
    return checkpoint_dir


def _strip_machine_specific_paths(value):
    if isinstance(value, dict):
        return {
            key: _strip_machine_specific_paths(item)
            for key, item in value.items()
            if key not in MACHINE_SPECIFIC_PATH_FIELDS
        }
    if isinstance(value, list):
        return [_strip_machine_specific_paths(item) for item in value]
    return value


def save_checkpoint_bundle(
    policy: nn.Module,
    checkpoint_dir: str | Path,
    *,
    step: int,
    training_state: dict[str, Any] | None = None,
    data_config=None,
    max_shard_size: int | str = DEFAULT_MAX_SHARD_SIZE,
) -> Path:
    """Save portable model artifacts and optional trusted training state."""
    model = _policy_for_saving(policy)
    policy_config = model.config if hasattr(model, "config") else None
    return save_state_dict_bundle(
        model.state_dict(),
        checkpoint_dir,
        step=step,
        policy_config=policy_config,
        data_config=data_config,
        training_state=training_state,
        max_shard_size=max_shard_size,
    )


def load_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {manifest_path}: {manifest.get('format')!r}")
    if manifest.get("version") != BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version in {manifest_path}: {manifest.get('version')!r}; "
            f"expected {BUNDLE_VERSION}"
        )
    return manifest


def load_bundle_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load JSON metadata without loading model or optimizer tensors."""
    checkpoint_dir = Path(checkpoint_dir)
    with checkpoint_lease(checkpoint_dir):
        manifest = load_manifest(checkpoint_dir)
        result: dict[str, Any] = {"manifest": manifest}
        for key, filename in (
            ("policy", POLICY_CONFIG_FILE),
            ("features", FEATURES_FILE),
            ("data_config", DATA_CONFIG_FILE),
            ("stats", STATS_FILE),
        ):
            path = checkpoint_dir / filename
            result[key] = _read_json(path) if path.is_file() else None
    return result


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint artifact not found: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Checkpoint artifact is empty: {path}")


@contextmanager
def checkpoint_lease(
    checkpoint_dir: str | Path,
    *,
    exclusive: bool = False,
    blocking: bool = True,
):
    """Coordinate multi-file bundle readers with directory deletion."""
    checkpoint_dir = Path(checkpoint_dir)
    lock_dir = checkpoint_dir.parent / ".checkpoint_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{checkpoint_dir.name}.lock"
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def checkpoint_read_lease(checkpoint_path: str | Path | None):
    """Hold a bundle lease for a complete higher-level operation."""
    if checkpoint_path is None:
        yield
        return
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        with checkpoint_lease(checkpoint_path):
            if not is_checkpoint_bundle(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint bundle no longer exists: {checkpoint_path}")
            yield
    else:
        yield


def _weight_shard_paths(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / WEIGHTS_INDEX_FILE
    if index_path.exists():
        _require_nonempty_file(index_path)
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no weight map: {index_path}")
        return [checkpoint_dir / filename for filename in sorted(set(weight_map.values()))]
    return sorted(checkpoint_dir.glob("*.safetensors"))


def validate_checkpoint_bundle(
    checkpoint_dir: str | Path,
    *,
    verify_safetensors_headers: bool = False,
) -> dict[str, Any]:
    """Validate referenced files without reading large tensor payloads.

    The default validation is intentionally limited to manifests, indexes,
    file existence, and non-zero sizes so checkpoint retention does not add
    work proportional to model size. Header validation is available for
    explicit integrity checks and still does not load tensor data.
    """
    checkpoint_dir = Path(checkpoint_dir)
    _require_nonempty_file(checkpoint_dir / MANIFEST_FILE)
    manifest = load_manifest(checkpoint_dir)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Checkpoint manifest has invalid artifacts section: {checkpoint_dir}")
    for filename in artifacts.values():
        if filename is not None:
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Checkpoint manifest contains an invalid artifact name: {filename!r}")
            _require_nonempty_file(checkpoint_dir / filename)

    shard_paths = _weight_shard_paths(checkpoint_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors model weights found in checkpoint: {checkpoint_dir}")

    for shard_path in shard_paths:
        _require_nonempty_file(shard_path)
    if verify_safetensors_headers:
        tensor_count = 0
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                tensor_count += len(shard.keys())
        if tensor_count == 0:
            raise ValueError(f"Safetensors checkpoint contains no tensors: {checkpoint_dir}")

    return manifest


def validate_checkpoint(checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if is_checkpoint_bundle(checkpoint_path):
        validate_checkpoint_bundle(checkpoint_path)
        return
    _require_nonempty_file(checkpoint_path)


def list_completed_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    """List valid numbered bundles and legacy files in ascending step order."""
    checkpoint_root = Path(checkpoint_root)
    if is_checkpoint_bundle(checkpoint_root) or checkpoint_root.is_file():
        validate_checkpoint(checkpoint_root)
        return [checkpoint_root]
    if not checkpoint_root.is_dir():
        return []

    checkpoints: dict[str, Path] = {}
    for candidate in checkpoint_root.glob("checkpoint_step_*"):
        if not is_checkpoint_bundle(candidate) and not (candidate.is_file() and candidate.suffix == ".pt"):
            continue
        try:
            validate_checkpoint(candidate)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Ignoring invalid checkpoint %s: %s", candidate, exc)
            continue
        # Prefer the portable bundle if both formats exist for the same step.
        key = candidate.name.removesuffix(".pt")
        if key not in checkpoints or is_checkpoint_bundle(candidate):
            checkpoints[key] = candidate
    return [checkpoints[key] for key in sorted(checkpoints)]


def load_bundle_weights(model: nn.Module, checkpoint_dir: str | Path, *, strict: bool = True) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    started = time.perf_counter()
    with checkpoint_lease(checkpoint_dir):
        validate_checkpoint_bundle(checkpoint_dir, verify_safetensors_headers=False)
        state_dict = _load_bundle_state_dict(checkpoint_dir)
        result = model.load_state_dict(state_dict, strict=strict)
    logger.info(
        "Safetensors model weights loaded in %.2fs from %s", time.perf_counter() - started, checkpoint_dir
    )
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            "Checkpoint weights do not match the model. "
            f"Missing keys: {result.missing_keys}; unexpected keys: {result.unexpected_keys}"
        )


def load_bundle_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load bundle tensors for specialized legacy-compatible remapping."""
    checkpoint_dir = Path(checkpoint_dir)
    started = time.perf_counter()
    with checkpoint_lease(checkpoint_dir):
        validate_checkpoint_bundle(checkpoint_dir, verify_safetensors_headers=False)
        state_dict = _load_bundle_state_dict(checkpoint_dir)
    logger.info(
        "Safetensors state dict loaded in %.2fs from %s",
        time.perf_counter() - started,
        checkpoint_dir,
    )
    return state_dict


def _load_bundle_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    state_dict = {}
    shard_paths = _weight_shard_paths(checkpoint_dir)
    aliases = _load_shared_tensor_aliases(checkpoint_dir, shard_paths)
    for shard_path in shard_paths:
        state_dict.update(load_state_dict_from_file(shard_path, map_location="cpu"))

    unresolved = dict(aliases)
    while unresolved:
        restored = {alias: state_dict[target] for alias, target in unresolved.items() if target in state_dict}
        if not restored:
            raise ValueError(f"Checkpoint contains unresolved shared tensor aliases: {unresolved}")
        state_dict.update(restored)
        for alias in restored:
            unresolved.pop(alias)
    return state_dict


def _load_shared_tensor_aliases(
    checkpoint_dir: Path,
    shard_paths: list[Path],
) -> dict[str, str]:
    metadata = {}
    index_path = checkpoint_dir / WEIGHTS_INDEX_FILE
    if index_path.is_file():
        metadata.update(_read_json(index_path).get("metadata", {}))
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            metadata.update(shard.metadata() or {})

    reserved = {"total_size", "format", "rho_format", "rho_version"}
    return {
        alias: target
        for alias, target in metadata.items()
        if alias not in reserved and isinstance(target, str)
    }


def load_bundle_training_state(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    checkpoint_dir = Path(checkpoint_dir)
    with checkpoint_lease(checkpoint_dir):
        manifest = load_manifest(checkpoint_dir)
        filename = manifest.get("artifacts", {}).get("training_state")
        if filename is None:
            return None
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint training state not found: {path}")
        started = time.perf_counter()
        training_state = torch.load(path, weights_only=False, map_location="cpu")  # nosec B614
    logger.info(
        "Trusted training state loaded in %.2fs from %s",
        time.perf_counter() - started,
        path,
    )
    return training_state


def delete_checkpoint_bundle(checkpoint_dir: str | Path) -> bool:
    """Delete a bundle only when no model reader holds a lease."""
    checkpoint_dir = Path(checkpoint_dir)
    try:
        with checkpoint_lease(checkpoint_dir, exclusive=True, blocking=False):
            if not checkpoint_dir.exists():
                return True
            validate_checkpoint_bundle(checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
            return True
    except BlockingIOError:
        return False


def remove_bundle_training_state(checkpoint_dir: str | Path) -> bool:
    """Remove trusted resume state while keeping portable model artifacts."""
    checkpoint_dir = Path(checkpoint_dir)
    try:
        with checkpoint_lease(checkpoint_dir, exclusive=True, blocking=False):
            manifest = load_manifest(checkpoint_dir)
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError(f"Checkpoint manifest has invalid artifacts section: {checkpoint_dir}")

            filename = artifacts.get("training_state")
            if filename is None:
                return False

            artifacts["training_state"] = None
            manifest_path = checkpoint_dir / MANIFEST_FILE
            temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
            _write_json(temporary_manifest, manifest)
            temporary_manifest.replace(manifest_path)

            training_state_path = checkpoint_dir / filename
            if training_state_path.exists():
                training_state_path.unlink()
            return True
    except BlockingIOError:
        return False


def resolve_latest_checkpoint(checkpoint_root: str | Path) -> Path | None:
    """Find the highest completed checkpoint, with legacy format compatibility."""
    checkpoint_root = Path(checkpoint_root)
    if is_checkpoint_bundle(checkpoint_root) or checkpoint_root.is_file():
        validate_checkpoint(checkpoint_root)
        return checkpoint_root
    if not checkpoint_root.is_dir():
        return None

    completed = list_completed_checkpoints(checkpoint_root)
    if completed:
        return completed[-1]

    legacy_latest = checkpoint_root / "checkpoint_latest.pt"
    if legacy_latest.is_file() and legacy_latest.stat().st_size > 0:
        return legacy_latest
    return None

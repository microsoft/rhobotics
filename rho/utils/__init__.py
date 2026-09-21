import logging
import os
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import torch
from accelerate import Accelerator

T = TypeVar("T")


def cycle(iterable: Iterable[T]) -> Iterator[T]:
    """Repeat an iterable, recreating its iterator without caching batches."""
    iterator = iter(iterable)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            try:
                item = next(iterator)
            except StopIteration:
                raise ValueError("Cannot cycle an empty or exhausted iterable.") from None
        yield item


def get_safe_torch_device(try_device: str | torch.device, log: bool = False) -> torch.device:
    """Resolve a device, rejecting unavailable accelerators rather than falling back."""
    device = torch.device(try_device)
    if device.type in {"cuda", "mps", "xpu"}:
        backend = torch.backends.mps if device.type == "mps" else getattr(torch, device.type, None)
        if backend is None or not backend.is_available():
            raise RuntimeError(f"Requested device {device} is not available.")
    if log and device.type == "cpu":
        logging.warning("Using CPU, this will be slow.")
    elif log and device.type not in {"cuda", "mps", "xpu"}:
        logging.warning("Using custom %s device.", device)
    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Avoid float64 on accelerators that do not support it."""
    device_type = torch.device(device).type
    if dtype != torch.float64:
        return dtype
    if device_type == "mps":
        return torch.float32
    if device_type == "xpu":
        capability_query = getattr(getattr(torch, "xpu", None), "get_device_capability", None)
        if capability_query is None or not capability_query().get("has_fp64", False):
            logging.warning("Device %s does not report float64 support; using float32 instead.", device)
            return torch.float32
    return dtype


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: The name of the logger (typically __name__ of the calling module)

    Returns:
        A logger instance that inherits settings from the root logger
    """
    return logging.getLogger(name)


def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str | None = None,
    file_level: str = "DEBUG",
    accelerator: Accelerator | None = None,
):
    """Initialize logging configuration for rho.

    In multi-GPU training, only the main process logs to console to avoid duplicate output.
    Non-main processes have console logging suppressed but can still log to file.

    Args:
        log_file: Optional file path to write logs to
        display_pid: Include process ID in log messages (useful for debugging multi-process)
        console_level: Logging level for console output. Defaults to INFO, can be overridden
                      by RHO_LOG_LEVEL environment variable.
        file_level: Logging level for file output
        accelerator: Optional Accelerator instance (for multi-GPU detection)
    """
    # Determine console log level: explicit arg > env var > default (INFO)
    if console_level is None:
        console_level = os.environ.get("RHO_LOG_LEVEL", "INFO")

    class LevelAwareFormatter(logging.Formatter):
        """Formatter that uses verbose format when DEBUG level is enabled."""

        def __init__(self, display_pid: bool = False, verbose: bool = False):
            super().__init__()
            self.display_pid = display_pid
            self.verbose = verbose

        def format(self, record: logging.LogRecord) -> str:
            pid_str = f"[PID: {os.getpid()}] " if self.display_pid else ""
            if self.verbose:
                # Verbose format with timestamp and file location
                dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fnameline = f"{record.pathname}:{record.lineno}"
                return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"
            else:
                # Simple format for INFO, WARNING, ERROR, etc.
                return f"{record.levelname}: {pid_str}{record.getMessage()}"

    # Use verbose format if DEBUG level is enabled
    use_verbose = console_level.upper() == "DEBUG"
    formatter = LevelAwareFormatter(display_pid=display_pid, verbose=use_verbose)

    logger = logging.getLogger()

    # Clear any existing handlers
    logger.handlers.clear()

    # Determine if this is a non-main process in distributed training
    # Check accelerator first, then fall back to environment variables for early init
    if accelerator is not None:
        is_main_process = accelerator.is_main_process
    else:
        # Check common distributed training environment variables
        # RANK is set by torch.distributed, LOCAL_RANK by accelerate/torchrun
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        is_main_process = rank == "0"

    # Console logging (main process only)
    if is_main_process:
        # Set root logger level to allow all messages through to handlers
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        logger.addHandler(console_handler)
    else:
        # Suppress console output for non-main processes
        # Still add NullHandler to avoid "No handler" warnings
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)  # Effectively suppress all logging

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, file_level.upper()))
        logger.addHandler(file_handler)

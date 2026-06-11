from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from rho.training.train_utils import TrainLogger, find_latest_checkpoint, load_training_state, save_checkpoint


def test_train_logger_initialization():
    """Test TrainLogger initialization"""
    logger = TrainLogger()

    # TrainLogger starts with default metrics (steps, samples, and epoch_progress)
    assert "steps" in logger.metrics
    assert "samples" in logger.metrics
    assert "epoch_progress" in logger.metrics
    assert len(logger.metrics) == 3


def test_train_logger_log():
    """Test TrainLogger log method"""
    logger = TrainLogger()

    metrics = {"loss": 0.5, "accuracy": 0.8}
    logger.log(metrics)

    assert "loss" in logger.metrics
    assert "accuracy" in logger.metrics
    # The metrics are stored as RollingAverage objects
    assert logger.metrics["loss"].value == 0.5
    assert logger.metrics["accuracy"].value == 0.8


def test_train_logger_log_multiple():
    """Test TrainLogger logging multiple values"""
    logger = TrainLogger()

    # Log multiple times
    logger.log({"loss": 0.5})
    logger.log({"loss": 0.4})
    logger.log({"loss": 0.3})

    # Should be rolling average (0.5 + 0.4 + 0.3) / 3 = 0.4
    assert abs(logger.metrics["loss"].value - 0.4) < 1e-6


def test_train_logger_log_time():
    """Test TrainLogger log_time context manager"""
    logger = TrainLogger()

    with logger.log_time("test_operation"):
        # Simulate some operation
        pass

    # The operation should be logged as a metric
    assert "test_operation" in logger.metrics
    assert logger.metrics["test_operation"].value > 0


def test_train_logger_log_batch():
    """Test TrainLogger log_batch method"""
    logger = TrainLogger()

    # log_batch takes step and batch_size, not a batch object
    logger.log_batch(step=100, batch_size=4)

    # Should log step and samples (batch size)
    assert logger.metrics["steps"].value == 100
    # samples is cumulative, so batch size gets added to existing 0
    assert logger.metrics["samples"].value >= 4


def test_train_logger_get_metrics():
    """Test TrainLogger get_metrics method"""
    logger = TrainLogger()

    # Log some metrics using the log method
    logger.log({"accuracy": 0.85, "loss": 0.25})

    metrics_dict = logger.get_metrics()

    # Check that metrics are returned
    assert "accuracy" in metrics_dict
    assert "loss" in metrics_dict

    # Check values with tolerance for floating point
    assert abs(metrics_dict["accuracy"] - 0.85) < 1e-10
    assert abs(metrics_dict["loss"] - 0.25) < 1e-10


def test_train_logger_reset():
    """Test TrainLogger reset method"""
    logger = TrainLogger()

    # Log some data
    logger.log({"loss": 0.5})
    with logger.log_time("test_op"):
        pass

    # Verify data exists
    assert "loss" in logger.metrics
    assert "test_op" in logger.metrics

    # Reset and verify data is cleared (except default metrics)
    logger.reset()
    # After reset, should have default metrics (steps, samples) but they should be reset
    assert "steps" in logger.metrics
    assert "samples" in logger.metrics
    assert logger.metrics["steps"].value is None
    assert logger.metrics["samples"].value == 0


@patch("torch.save")
def test_save_checkpoint_basic(mock_torch_save):
    """Test save_checkpoint function"""
    # Create mock objects
    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {"param1": torch.tensor([1.0])}

    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {"lr": 1e-4}

    step = 1000
    metrics = {"loss": 0.5}
    output_dir = Path("test_checkpoint.pt")

    # Call save_checkpoint
    save_checkpoint(mock_policy, mock_optimizer, step, metrics, output_dir)

    # Verify torch.save was called twice (step checkpoint + latest)
    assert mock_torch_save.call_count == 2


def test_load_training_state_no_scheduler():
    """Test load_training_state function without scheduler"""
    mock_optimizer = MagicMock()
    checkpoint_path = Path("nonexistent.pt")

    # Should return defaults when file doesn't exist
    step, loaded_optimizer, loaded_scheduler, loaded_sampler, loaded_logger = load_training_state(
        checkpoint_path, mock_optimizer, None
    )

    assert step == 0
    assert loaded_optimizer == mock_optimizer
    assert loaded_scheduler is None
    assert loaded_sampler is None
    assert loaded_logger is None


@patch("pathlib.Path.mkdir")
def test_save_checkpoint_creates_directory(mock_mkdir):
    """Test save_checkpoint creates directory"""
    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {}

    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {}

    with patch("torch.save"):
        save_checkpoint(mock_policy, mock_optimizer, 100, {}, Path("test_dir"))

    # Directory creation should be called
    mock_mkdir.assert_called()


def test_train_logger_save_state():
    """Test that save_state captures cumulative and latest metrics."""
    logger = TrainLogger(dataset_length=200)

    # Simulate training: log several batches
    logger.log_batch(step=1, batch_size=8)
    logger.log_batch(step=2, batch_size=8)
    logger.log({"loss": 0.5})

    state = logger.save_state()

    assert state["dataset_length"] == 200
    assert "samples" in state["metrics"]
    assert state["metrics"]["samples"]["type"] == "CumulativeValue"
    assert state["metrics"]["samples"]["value"] == 16  # 8 + 8

    assert "steps" in state["metrics"]
    assert state["metrics"]["steps"]["type"] == "LatestValue"
    assert state["metrics"]["steps"]["value"] == 2

    assert "epoch_progress" in state["metrics"]
    assert state["metrics"]["epoch_progress"]["type"] == "LatestValue"
    assert abs(state["metrics"]["epoch_progress"]["value"] - 16 / 200) < 1e-10

    # Rolling averages (like "loss") should NOT be saved
    assert "loss" not in state["metrics"]


def test_train_logger_load_state_roundtrip():
    """Test that load_state correctly restores saved state."""
    logger = TrainLogger(dataset_length=200)
    logger.log_batch(step=5, batch_size=32)
    logger.log_batch(step=6, batch_size=32)
    logger.log({"loss": 0.3})

    state = logger.save_state()

    # Create a fresh logger and restore
    new_logger = TrainLogger(dataset_length=200)
    new_logger.load_state(state)

    assert new_logger.metrics["samples"].value == 64  # 32 + 32
    assert new_logger.metrics["steps"].value == 6
    assert abs(new_logger.metrics["epoch_progress"].value - 64 / 200) < 1e-10
    assert new_logger.dataset_length == 200


def test_train_logger_load_state_continues_correctly():
    """After restoring state, further log_batch calls should accumulate correctly."""
    logger = TrainLogger(dataset_length=100)
    logger.log_batch(step=10, batch_size=20)

    state = logger.save_state()

    new_logger = TrainLogger(dataset_length=100)
    new_logger.load_state(state)

    # Continue logging
    new_logger.log_batch(step=11, batch_size=10)

    assert new_logger.metrics["samples"].value == 30  # 20 + 10
    assert new_logger.metrics["steps"].value == 11
    assert abs(new_logger.metrics["epoch_progress"].value - 30 / 100) < 1e-10


def test_train_logger_load_state_with_missing_metrics():
    """Loading state with extra metrics should create them; missing ones stay default."""
    logger = TrainLogger()
    state = {
        "dataset_length": 500,
        "metrics": {
            "samples": {"type": "CumulativeValue", "value": 42},
            "custom_counter": {"type": "CumulativeValue", "value": 7},
        },
    }
    logger.load_state(state)

    assert logger.metrics["samples"].value == 42
    assert logger.dataset_length == 500
    # custom_counter should be recreated
    assert "custom_counter" in logger.metrics
    assert logger.metrics["custom_counter"].value == 7
    # steps should remain at default (None)
    assert logger.metrics["steps"].value is None


def test_find_latest_checkpoint_nonexistent_dir(tmp_path):
    """Returns None when the directory does not exist."""
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_find_latest_checkpoint_empty_dir(tmp_path):
    """Returns None when the directory exists but has no checkpoints."""
    assert find_latest_checkpoint(tmp_path) is None


def test_find_latest_checkpoint_prefers_latest(tmp_path):
    """Returns checkpoint_latest.pt when it exists."""
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"fake")
    (tmp_path / "checkpoint_step_0001000.pt").write_bytes(b"fake")
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint_latest.pt"


def test_find_latest_checkpoint_falls_back_to_highest_step(tmp_path):
    """Falls back to the highest numbered step checkpoint."""
    (tmp_path / "checkpoint_step_0001000.pt").write_bytes(b"fake")
    (tmp_path / "checkpoint_step_0005000.pt").write_bytes(b"fake")
    (tmp_path / "checkpoint_step_0003000.pt").write_bytes(b"fake")
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint_step_0005000.pt"

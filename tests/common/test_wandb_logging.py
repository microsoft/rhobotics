from unittest.mock import MagicMock, patch

from rho.common.wandb_logging import WandBConfig, WandBLogger


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_initialization(mock_wandb):
    """Test WandBLogger initialization when enabled"""
    config = WandBConfig(project="test_project", username="test_entity", enabled=True)

    mock_wandb.init.return_value = MagicMock()

    logger = WandBLogger(config)

    assert logger.cfg == config
    assert logger.enabled
    mock_wandb.init.assert_called_once()


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_enabled(mock_wandb):
    """Test WandBLogger log method when enabled"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()

    logger = WandBLogger(config)

    # Test logging
    metrics = {"loss": 0.5, "accuracy": 0.8}
    logger.log(metrics, step=100)

    mock_wandb.log.assert_called_once_with(metrics, step=100)


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_with_prefix(mock_wandb):
    """Test WandBLogger log method with prefix"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()

    logger = WandBLogger(config)

    # Test logging with prefix
    metrics = {"loss": 0.5, "accuracy": 0.8}
    logger.log(metrics, step=100, prefix="train")

    expected_metrics = {"train/loss": 0.5, "train/accuracy": 0.8}
    mock_wandb.log.assert_called_once_with(expected_metrics, step=100)


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_finish(mock_wandb):
    """Test WandBLogger finish method"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()

    logger = WandBLogger(config)
    logger.finish()

    mock_wandb.finish.assert_called_once()


# Test for log_config method
@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_config(mock_wandb):
    """Test WandBLogger log_config method with a placeholder TrainConfig"""
    from dataclasses import dataclass

    # Define a placeholder TrainConfig dataclass
    @dataclass
    class TrainConfig:
        learning_rate: float = 0.001
        batch_size: int = 32
        epochs: int = 10
        model_name: str = "test_model"
        optimizer: str = "adam"

    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    logger = WandBLogger(config)

    # Create a test train config
    train_config = TrainConfig(
        learning_rate=0.002, batch_size=64, epochs=20, model_name="phi4mm", optimizer="adamw"
    )

    # Test logging config
    logger.log_config(train_config, name="train_config")

    # Verify that an artifact was created
    mock_wandb.Artifact.assert_called_once()
    mock_artifact.add_file.assert_called_once()
    mock_wandb.log_artifact.assert_called_once_with(mock_artifact)
    mock_wandb.config.update.assert_called_once()


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_config_dict(mock_wandb):
    """Test WandBLogger log_config method with a dictionary"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    logger = WandBLogger(config)

    # Create a test config as dictionary
    train_config = {
        "learning_rate": 0.003,
        "batch_size": 128,
        "epochs": 50,
        "model_name": "phi4mm_large",
        "optimizer": "sgd",
        "nested_config": {"dropout": 0.1, "layers": 12},
    }

    # Test logging config
    logger.log_config(train_config, name="dict_config")

    # Verify that an artifact was created
    mock_wandb.Artifact.assert_called_once()
    mock_artifact.add_file.assert_called_once()
    mock_wandb.log_artifact.assert_called_once_with(mock_artifact)
    mock_wandb.config.update.assert_called_once()

    # Verify the config was passed correctly to wandb.config.update
    call_args = mock_wandb.config.update.call_args[0][0]
    assert call_args["learning_rate"] == 0.003
    assert call_args["batch_size"] == 128
    assert call_args["nested_config"]["dropout"] == 0.1


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_config_disabled(mock_wandb):
    """Test WandBLogger log_config method when disabled"""
    from dataclasses import dataclass

    @dataclass
    class TrainConfig:
        learning_rate: float = 0.001
        batch_size: int = 32

    config = WandBConfig(enabled=False)
    logger = WandBLogger(config)

    train_config = TrainConfig()

    # Should not raise any errors when disabled
    logger.log_config(train_config, name="train_config")

    # Verify wandb methods were not called
    mock_wandb.Artifact.assert_not_called()
    mock_wandb.log_artifact.assert_not_called()


# Test for _handle_video_metrics method
@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_handle_video_metrics(mock_wandb):
    """Test WandBLogger _handle_video_metrics method"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()
    mock_video = MagicMock()
    mock_wandb.Video.return_value = mock_video

    logger = WandBLogger(config)

    # Create test metrics with video path and fps
    metrics = {
        "loss": 0.5,
        "accuracy": 0.8,
        "eval_video_path": "/tmp/test_video.mp4",
        "eval_video_fps": 30,
        "train_video_path": "/tmp/train_video.mp4",
        "train_video_fps": 25,
    }

    # Test the _handle_video_metrics method directly
    processed_metrics = logger._handle_video_metrics(metrics)

    # Verify that video paths were converted to wandb.Video objects
    assert "loss" in processed_metrics
    assert "accuracy" in processed_metrics
    assert "eval_video" in processed_metrics
    assert "train_video" in processed_metrics
    assert "eval_video_path" not in processed_metrics
    assert "eval_video_fps" not in processed_metrics
    assert "train_video_path" not in processed_metrics
    assert "train_video_fps" not in processed_metrics

    # Verify wandb.Video was called correctly
    assert mock_wandb.Video.call_count == 2
    mock_wandb.Video.assert_any_call("/tmp/test_video.mp4", fps=30, format="mp4")
    mock_wandb.Video.assert_any_call("/tmp/train_video.mp4", fps=25, format="mp4")


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_with_video_metrics(mock_wandb):
    """Test WandBLogger log method with video metrics"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()
    mock_video = MagicMock()
    mock_wandb.Video.return_value = mock_video

    logger = WandBLogger(config)

    # Test logging with video metrics
    metrics = {"loss": 0.5, "test_video_path": "/tmp/test.mp4", "test_video_fps": 30}

    logger.log(metrics, step=100)

    # Verify that wandb.Video was created and logged
    mock_wandb.Video.assert_called_once_with("/tmp/test.mp4", fps=30, format="mp4")

    # Verify the final log call had the video object
    expected_call_args = mock_wandb.log.call_args[0][0]
    assert "loss" in expected_call_args
    assert "test_video" in expected_call_args
    assert "test_video_path" not in expected_call_args
    assert "test_video_fps" not in expected_call_args

    mock_wandb.log.assert_called_once()


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_without_step(mock_wandb):
    """Test WandBLogger log method without step parameter"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()

    logger = WandBLogger(config)

    # Test logging without step
    metrics = {"loss": 0.5, "accuracy": 0.8}
    logger.log(metrics)

    # Verify wandb.log was called without step parameter
    mock_wandb.log.assert_called_once_with(metrics)


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_policy(mock_wandb):
    """Test WandBLogger log_policy method"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()

    logger = WandBLogger(config)

    # Test logging policy
    dummy_policy = {"type": "test_policy", "params": {"lr": 0.001}}
    logger.log_policy(dummy_policy)

    mock_wandb.log.assert_called_once_with({"policy": dummy_policy})


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_policy_disabled(mock_wandb):
    """Test WandBLogger log_policy method when disabled"""
    config = WandBConfig(enabled=False)
    logger = WandBLogger(config)

    # Should not raise any errors when disabled
    dummy_policy = {"type": "test_policy"}
    logger.log_policy(dummy_policy)

    # Verify wandb.log was not called
    mock_wandb.log.assert_not_called()


@patch("rho.common.wandb_logging.wandb")
def test_wandb_logger_log_config_with_nested_dataclass(mock_wandb):
    """Test WandBLogger log_config method with nested dataclasses"""
    from dataclasses import dataclass

    @dataclass
    class OptimizerConfig:
        name: str = "adam"
        lr: float = 0.001

    @dataclass
    class TrainConfig:
        epochs: int = 10
        optimizer: OptimizerConfig = None
        optimizers: list = None

        def __post_init__(self):
            if self.optimizer is None:
                self.optimizer = OptimizerConfig()
            if self.optimizers is None:
                self.optimizers = [OptimizerConfig(name="sgd", lr=0.01)]

    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    logger = WandBLogger(config)

    # Create a test train config with nested dataclass
    train_config = TrainConfig(epochs=20)

    # Test logging config
    logger.log_config(train_config, name="nested_config")

    # Verify that an artifact was created
    mock_wandb.Artifact.assert_called_once()
    mock_artifact.add_file.assert_called_once()
    mock_wandb.log_artifact.assert_called_once_with(mock_artifact)
    mock_wandb.config.update.assert_called_once()


@patch("rho.common.wandb_logging.wandb")
@patch("rho.common.wandb_logging.tempfile.NamedTemporaryFile")
def test_wandb_logger_log_config_exception_handling(mock_tempfile, mock_wandb):
    """Test WandBLogger log_config method exception handling"""
    config = WandBConfig(project="test_project", enabled=True)

    mock_wandb.init.return_value = MagicMock()
    # Simulate an exception in artifact creation
    mock_wandb.Artifact.side_effect = Exception("Test exception")

    logger = WandBLogger(config)

    # Test logging config with exception
    test_config = {"test": "value"}

    # Should not raise exception, but print error message
    logger.log_config(test_config, name="exception_config")

    # Verify artifact creation was attempted
    mock_wandb.Artifact.assert_called_once()


def test_serialize_to_dict_non_dataclass():
    """Test serialize_to_dict with non-dataclass object"""
    from rho.common.serialization import serialize_to_dict

    # Test with non-dataclass object
    regular_dict = {"key": "value", "number": 42}
    result = serialize_to_dict(regular_dict)

    # Should return the same dict contents
    assert result == regular_dict

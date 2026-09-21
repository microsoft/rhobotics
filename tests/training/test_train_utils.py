import ast
from collections import deque
from copy import deepcopy
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from rho.datasets.lerobot_dataset import EpisodeAwareSampler
from rho.datasets.multi_dataset import MultiDatasetWeightedSampler
from rho.training.train import TrainConfig, get_training_dataset_length, train, train_policy_step
from rho.training.train_utils import TrainLogger, find_latest_checkpoint, load_training_state, save_checkpoint
from rho.utils import cycle, get_safe_dtype, get_safe_torch_device


def test_non_dataset_runtime_has_no_direct_lerobot_imports():
    root = Path(__file__).resolve().parents[2] / "rho"
    allowed_files = {
        "policies/diffusion/modeling_diffusion.py",
        "utils/recompute_chunk_lerobot_stats.py",
        "utils/xdof/convert_xdof_to_lerobot.py",
    }
    violations = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "datasets" or relative.as_posix() in allowed_files:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            if any(module == "lerobot" or module.startswith("lerobot.") for module in modules):
                violations.append(f"{relative}:{node.lineno}")
    assert violations == []


def test_cycle_recreates_iterators_without_caching_batches():
    class Batches:
        passes = 0

        def __iter__(self):
            self.passes += 1
            yield self.passes
            yield self.passes

    batches = Batches()
    assert list(islice(cycle(batches), 5)) == [1, 1, 2, 2, 3]


def test_cycle_handles_exhausted_resume_iterator():
    class ResumedBatches:
        passes = 0

        def __iter__(self):
            self.passes += 1
            if self.passes > 1:
                yield 1
                yield 2

    assert list(islice(cycle(ResumedBatches()), 4)) == [1, 2, 1, 2]


@pytest.mark.parametrize("batches", [[], iter(())])
def test_cycle_rejects_empty_sources(batches):
    with pytest.raises(ValueError, match="empty or exhausted"):
        next(cycle(batches))


def test_device_resolution_keeps_explicit_cpu():
    assert get_safe_torch_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device", ["cuda:1", "mps", "xpu"])
def test_device_resolution_rejects_unavailable_accelerators(monkeypatch, device):
    backend = torch.backends.mps if device == "mps" else getattr(torch, torch.device(device).type)
    monkeypatch.setattr(backend, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        get_safe_torch_device(device)


@pytest.mark.parametrize("device", ["cuda:1", "mps", "xpu:2"])
def test_device_resolution_preserves_available_device_index(monkeypatch, device):
    backend = torch.backends.mps if device == "mps" else getattr(torch, torch.device(device).type)
    monkeypatch.setattr(backend, "is_available", lambda: True)
    assert get_safe_torch_device(device) == torch.device(device)


@pytest.mark.parametrize(
    "dtype,device,expected",
    [
        (torch.float64, "cpu", torch.float64),
        (torch.float64, "cuda:1", torch.float64),
        (torch.float64, torch.device("mps"), torch.float32),
        (torch.bfloat16, "cpu", torch.bfloat16),
        (torch.float32, "mps", torch.float32),
    ],
)
def test_safe_dtype_preserves_supported_precision(dtype, device, expected):
    assert get_safe_dtype(dtype, device) == expected


@pytest.mark.parametrize("fp64_supported", [False, True])
def test_safe_dtype_checks_xpu_capabilities(monkeypatch, fp64_supported):
    monkeypatch.setattr(
        torch.xpu, "get_device_capability", lambda: {"has_fp64": fp64_supported}, raising=False
    )
    expected = torch.float64 if fp64_supported else torch.float32
    assert get_safe_dtype(torch.float64, "xpu") == expected


def test_safe_dtype_warns_when_xpu_capabilities_are_unavailable(monkeypatch, caplog):
    monkeypatch.delattr(torch.xpu, "get_device_capability", raising=False)
    assert get_safe_dtype(torch.float64, "xpu") == torch.float32
    assert "does not report float64 support" in caplog.text


def test_rho_policy_uses_local_queue_helper():
    from rho.policies.base import populate_queues
    from rho.policies.rho import rho_policy

    assert rho_policy.populate_queues is populate_queues

    queues = {"observation": deque(maxlen=3), "action": deque(maxlen=2)}
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    populate_queues(queues, {"observation": first, "action": first}, exclude_keys=["action"])
    assert list(queues["observation"]) == [first, first, first]
    assert not queues["action"]
    populate_queues(queues, {"observation": second})
    assert list(queues["observation"]) == [first, first, second]


class _LinearPolicy(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 1, bias=False)
        self.weight.data.fill_(0.25)
        self.prediction_dtypes = []

    def compute_loss(self, batch):
        prediction = super().forward(batch["observation"])
        self.prediction_dtypes.append(prediction.dtype)
        return torch.nn.functional.mse_loss(prediction.float(), batch["action"]), {}


@pytest.mark.parametrize("max_grad_norm", [None, 0.1])
def test_plain_training_accumulation_matches_full_batch(max_grad_norm):
    batch = {"observation": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "action": torch.zeros(2, 1)}
    reference = _LinearPolicy()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss, _ = reference.compute_loss(batch)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(reference.parameters(), max_grad_norm)
    reference_optimizer.step()

    policy = _LinearPolicy()
    initial = policy.weight.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    for index in range(2):
        train_policy_step(
            policy,
            {key: value[index : index + 1] for key, value in batch.items()},
            optimizer,
            scheduler,
            1,
            torch.device("cpu"),
            gradient_accumulation_steps=2,
            accumulation_step=index,
            max_grad_norm=max_grad_norm,
        )
        if index == 0:
            torch.testing.assert_close(policy.weight, initial)
            assert scheduler.last_epoch == 0
    torch.testing.assert_close(policy.weight, reference.weight)
    assert scheduler.last_epoch == 1
    assert all(parameter.grad is None for parameter in policy.parameters())


@pytest.mark.parametrize(
    "precision,dtype", [("no", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]
)
def test_plain_training_amp_dtype_and_scaling(precision, dtype):
    policy = _LinearPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0) if precision == "fp16" else None
    initial = policy.weight.detach().clone()
    train_policy_step(
        policy,
        {"observation": torch.ones(2, 2), "action": torch.zeros(2, 1)},
        optimizer,
        None,
        1,
        torch.device("cpu"),
        mixed_precision=precision,
        grad_scaler=scaler,
        max_grad_norm=0.1,
    )
    assert policy.prediction_dtypes == [dtype]
    assert not torch.equal(policy.weight, initial)
    assert (policy.weight - initial).norm().item() <= 0.010001


def test_plain_training_fp16_overflow_does_not_advance_scheduler():
    policy = _LinearPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    initial = policy.weight.detach().clone()
    train_policy_step(
        policy,
        {"observation": torch.full((1, 2), float("inf")), "action": torch.zeros(1, 1)},
        optimizer,
        scheduler,
        1,
        torch.device("cpu"),
        mixed_precision="fp16",
        grad_scaler=scaler,
    )
    torch.testing.assert_close(policy.weight, initial)
    assert scheduler.last_epoch == 0
    assert scaler.get_scale() == 4.0


@pytest.mark.parametrize("multi_dataset", [False, True])
def test_checkpoint_restores_sampler_and_grad_scaler(tmp_path, monkeypatch, multi_dataset):
    policy = _LinearPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16.0)

    def make_sampler():
        children = [EpisodeAwareSampler([0], [8], shuffle=True, seed=seed) for seed in (11, 22)]
        return MultiDatasetWeightedSampler(children, seed=42) if multi_dataset else children[0]

    sampler = make_sampler()
    iterator = iter(sampler)
    list(islice(iterator, 3))
    captured = {}

    def capture_bundle(policy, path, **kwargs):
        captured.update(deepcopy(kwargs["training_state"]))

    monkeypatch.setattr("rho.training.train_utils.save_checkpoint_bundle", capture_bundle)
    monkeypatch.setattr("rho.training.train_utils.validate_checkpoint", lambda _: None)
    save_checkpoint(policy, optimizer, 3, {}, tmp_path, sampler=sampler, grad_scaler=scaler)
    checkpoint = tmp_path / "resume.pt"
    torch.save(captured, checkpoint)
    restored_sampler = make_sampler()
    restored_scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
    step, *_ = load_training_state(checkpoint, optimizer, None, restored_sampler, grad_scaler=restored_scaler)
    assert step == 3
    assert restored_scaler.state_dict() == scaler.state_dict()
    assert list(islice(iter(restored_sampler), 5)) == list(islice(iterator, 5))


@pytest.mark.parametrize("precision", ["no", "bf16", "fp16"])
def test_plain_training_loop_wires_update_settings(monkeypatch, tmp_path, precision):
    from rho.policies import PolicyConfig

    cfg = TrainConfig(
        dataset=SimpleNamespace(chunk_size=None, batch_size=2, num_workers=0),
        policy=PolicyConfig(feature_dict={}),
        device="cpu",
        output_dir=str(tmp_path),
        run_name="test",
        steps=2,
        gradient_accumulation_steps=2,
        mixed_precision=precision,
        grad_clip_norm=0.1,
        action_monitoring=False,
        validation_probe=False,
        save_checkpoint_every=1,
    )
    policy = _LinearPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    recorder = TrainLogger(dataset_length=16)
    transforms = MagicMock(side_effect=lambda batch: batch)
    seen_scalers = []
    checkpoint_calls = []

    def make_components(config, device, *, grad_scaler=None):
        seen_scalers.append(grad_scaler)
        batches = [{"observation": torch.ones(2, 2), "action": torch.zeros(2, 1)} for _ in range(4)]
        return batches, None, policy, optimizer, scheduler, transforms, 0, None, recorder

    monkeypatch.setattr("rho.training.train.resolve_training_checkpoint", lambda _: None)
    monkeypatch.setattr("rho.training.train.serialize_train_config", lambda _: {})
    monkeypatch.setattr("rho.training.train.make_environment", lambda _: None)
    monkeypatch.setattr("rho.training.train.make_everything", make_components)
    monkeypatch.setattr("rho.training.train.make_policy_interface", lambda **kwargs: None)
    monkeypatch.setattr("rho.training.train.WandBLogger", lambda _: MagicMock())
    monkeypatch.setattr(
        "rho.training.train.save_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append((args[2], kwargs)),
    )
    train.__wrapped__(cfg)

    dtype = {"no": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    assert policy.prediction_dtypes == [dtype] * 4
    assert scheduler.last_epoch == 2
    assert recorder.metrics["samples"].value == 8
    assert transforms.step.call_count == 2
    assert [step for step, _ in checkpoint_calls] == [1, 2]
    assert all(kwargs["grad_scaler"] is seen_scalers[0] for _, kwargs in checkpoint_calls)
    assert (seen_scalers[0] is not None) == (precision == "fp16")


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


@patch("rho.training.train_utils.validate_checkpoint")
@patch("rho.training.train_utils.save_checkpoint_bundle")
def test_save_checkpoint_basic(mock_save_bundle, _mock_validate, tmp_path):
    """Test save_checkpoint function"""
    # Create mock objects
    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {"param1": torch.tensor([1.0])}

    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {"lr": 1e-4}

    step = 1000
    metrics = {"loss": 0.5}
    output_dir = tmp_path / "checkpoints"

    # Call save_checkpoint
    save_checkpoint(mock_policy, mock_optimizer, step, metrics, output_dir)

    mock_save_bundle.assert_called_once()
    assert mock_save_bundle.call_args.args[1] == output_dir / "checkpoint_step_0001000"


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


def test_load_training_state_policy_only_checkpoint_starts_fresh(tmp_path):
    """Policy-only checkpoints should not attempt to restore training state."""
    checkpoint_path = tmp_path / "checkpoint_latest.pt"
    torch.save({"policy_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)

    optimizer = MagicMock()
    scheduler = MagicMock()
    sampler = MagicMock()
    train_logger = MagicMock()

    step, loaded_optimizer, loaded_scheduler, loaded_sampler, loaded_logger = load_training_state(
        checkpoint_path, optimizer, scheduler, sampler, train_logger
    )

    assert step == 0
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler
    assert loaded_sampler is sampler
    assert loaded_logger is train_logger
    optimizer.load_state_dict.assert_not_called()
    scheduler.load_state_dict.assert_not_called()
    sampler.load_state.assert_not_called()
    train_logger.load_state.assert_not_called()
    optimizer.zero_grad.assert_called_once_with(set_to_none=True)


@patch("rho.training.train_utils.validate_checkpoint")
@patch("rho.training.train_utils.save_checkpoint_bundle")
def test_save_checkpoint_creates_directory(mock_save_bundle, _mock_validate, tmp_path):
    """Test save_checkpoint creates directory"""
    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {}

    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {}

    output_dir = tmp_path / "test_dir"
    save_checkpoint(mock_policy, mock_optimizer, 100, {}, output_dir)

    assert output_dir.is_dir()
    mock_save_bundle.assert_called_once()


@pytest.mark.parametrize("sampler_length,expected", [(None, 8), (0, 0), (4, 4)])
def test_training_dataset_length_uses_sampled_or_loaded_length(sampler_length, expected):
    config = SimpleNamespace(get_length=lambda: 10)
    dataloader = SimpleNamespace(dataset=range(8))
    sampler = range(sampler_length) if sampler_length is not None else None
    assert get_training_dataset_length(config, dataloader, sampler) == expected


@pytest.mark.parametrize("metadata_length", [None, 10])
def test_training_dataset_length_falls_back_for_unsized_data(metadata_length):
    config = SimpleNamespace(get_length=lambda: metadata_length)
    dataloader = SimpleNamespace(dataset=iter(range(4)))
    assert get_training_dataset_length(config, dataloader, None) == metadata_length


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


def test_train_logger_load_state_preserves_current_dataset_length():
    saved_logger = TrainLogger(dataset_length=200)
    saved_logger.log_batch(step=5, batch_size=20)

    current_logger = TrainLogger(dataset_length=100)
    current_logger.load_state(saved_logger.save_state())

    assert current_logger.dataset_length == 100
    assert current_logger.metrics["epoch_progress"].value == 20 / 100


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
    assert logger.metrics["epoch_progress"].value == 42 / 500
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


def test_find_latest_checkpoint_prefers_numbered_checkpoint(tmp_path):
    """Numbered checkpoints take precedence over the legacy latest copy."""
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"fake")
    (tmp_path / "checkpoint_step_0001000.pt").write_bytes(b"fake")
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint_step_0001000.pt"


def test_find_latest_checkpoint_falls_back_to_highest_step(tmp_path):
    """Falls back to the highest numbered step checkpoint."""
    (tmp_path / "checkpoint_step_0001000.pt").write_bytes(b"fake")
    (tmp_path / "checkpoint_step_0005000.pt").write_bytes(b"fake")
    (tmp_path / "checkpoint_step_0003000.pt").write_bytes(b"fake")
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint_step_0005000.pt"

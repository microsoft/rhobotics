"""Accelerated training counts successful updates, not overflow attempts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from torch import nn

from rho.policies import PolicyConfig
from rho.training import train_accelerate
from rho.training.train import TrainConfig
from rho.training.train_utils import TrainLogger


@pytest.fixture
def make_accelerator(monkeypatch):
    # Accelerate uses process-global shared state. Keep these CPU tests isolated
    # from each other and from other callers in the test process.
    for state_type in (PartialState, AcceleratorState, GradientState):
        monkeypatch.setattr(state_type, "_shared_state", {})

    def make(accumulation_steps=1, precision="fp16"):
        accelerator = Accelerator(
            cpu=True, mixed_precision=precision, gradient_accumulation_steps=accumulation_steps
        )
        if precision == "fp16":
            # Accelerate does not create a CPU FP16 scaler automatically. Supply
            # PyTorch's real CPU scaler to exercise AcceleratedOptimizer's actual
            # overflow detection, including the gradient-unscaling/clipping path.
            accelerator.scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=100)
            accelerator.native_amp = True
        return accelerator

    return make


class _ScalarPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        return (self.weight * batch["action"]).mean(), {}


def _components(accelerator, with_scheduler=True):
    policy = _ScalarPolicy()
    base_optimizer = torch.optim.SGD([{"params": policy.parameters(), "name": "policy"}], lr=0.1)
    # Match train(): prepare the optimizer, but leave the scheduler unwrapped.
    scheduler = (
        torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=1, gamma=0.1) if with_scheduler else None
    )
    policy, optimizer = accelerator.prepare(policy, base_optimizer)
    return policy, optimizer, scheduler


def _step(policy, optimizer, scheduler, accelerator, value, max_grad_norm=None):
    return train_accelerate.train_policy_step_accumulate(
        policy,
        {"action": torch.tensor([[value]], dtype=torch.float32)},
        optimizer,
        scheduler,
        accelerator,
        0,
        torch.device("cpu"),
        max_grad_norm=max_grad_norm,
    )


@pytest.mark.parametrize("accumulation_steps", [1, 3])
@pytest.mark.parametrize("max_grad_norm", [None, 0.1])
@pytest.mark.parametrize("with_scheduler", [False, True])
@pytest.mark.parametrize(
    "overflow_value",
    [float("inf"), float("nan"), torch.finfo(torch.float32).max / 2],
    ids=["inf", "nan", "finite-loss-scaled-overflow"],
)
def test_overflow_is_not_an_update_and_next_window_recovers(
    make_accelerator, accumulation_steps, max_grad_norm, with_scheduler, overflow_value
):
    accelerator = make_accelerator(accumulation_steps)
    policy, optimizer, scheduler = _components(accelerator, with_scheduler)
    initial = policy.weight.detach().clone()

    for index in range(accumulation_steps):
        metrics, updated = _step(
            policy,
            optimizer,
            scheduler,
            accelerator,
            overflow_value if index == 0 else 1.0,
            max_grad_norm,
        )
        assert updated is False
        torch.testing.assert_close(policy.weight, initial)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
        assert metrics["learning_rate"] == pytest.approx(0.1)
        if scheduler is not None:
            assert scheduler.last_epoch == 0

    assert accelerator.optimizer_step_was_skipped is True
    assert accelerator.scaler.get_scale() == 4.0
    assert policy.weight.grad is None

    for index in range(accumulation_steps):
        metrics, updated = _step(policy, optimizer, scheduler, accelerator, 1.0, max_grad_norm)
        assert updated is (index == accumulation_steps - 1)
        if index < accumulation_steps - 1:
            torch.testing.assert_close(policy.weight, initial)
            assert policy.weight.grad is not None

    expected_gradient = 1.0 if max_grad_norm is None else max_grad_norm
    assert policy.weight.item() == pytest.approx(1.0 - 0.1 * expected_gradient)
    assert accelerator.optimizer_step_was_skipped is False
    assert policy.weight.grad is None
    assert accelerator.scaler.get_scale() == 4.0
    expected_lr = 0.01 if with_scheduler else 0.1
    assert metrics["learning_rate"] == pytest.approx(expected_lr)
    assert metrics["learning_rate/policy"] == pytest.approx(expected_lr)
    if scheduler is not None:
        assert scheduler.last_epoch == 1


@pytest.mark.parametrize("precision", ["no", "bf16", "fp16"])
@pytest.mark.parametrize("accumulation_steps", [1, 3])
def test_finite_updates_preserve_accumulation_and_scheduler(make_accelerator, precision, accumulation_steps):
    accelerator = make_accelerator(accumulation_steps, precision)
    policy, optimizer, scheduler = _components(accelerator)
    updates = []
    for _ in range(2):
        for _ in range(accumulation_steps):
            _, updated = _step(policy, optimizer, scheduler, accelerator, 1.0)
            updates.append(updated)
    assert updates == ([False] * (accumulation_steps - 1) + [True]) * 2
    assert policy.weight.item() == pytest.approx(0.89)
    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_training_loop_advances_schedules_only_after_success(
    make_accelerator, monkeypatch, tmp_path, accumulation_steps
):
    accelerator = make_accelerator(accumulation_steps)
    policy = _ScalarPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    recorder = TrainLogger(dataset_length=100)
    transforms = MagicMock(side_effect=lambda batch: batch)
    logger = MagicMock()
    progress = MagicMock()
    checkpoints = []
    cfg = TrainConfig(
        dataset=SimpleNamespace(chunk_size=None, batch_size=1, num_workers=0),
        policy=PolicyConfig(feature_dict={}),
        device="cpu",
        output_dir=str(tmp_path),
        run_name="overflow-test",
        steps=2,
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision="fp16",
        grad_clip_norm=0.1,
        logging_interval=1,
        validation_interval=1,
        save_checkpoint_every=1,
        action_monitoring=False,
        validation_probe=False,
    )
    # Four windows: skip, succeed, skip, succeed. Use a finite loader so a
    # broken recovery path fails at exhaustion instead of looping indefinitely.
    values = [float("inf"), 1.0, float("inf"), 1.0]
    batches = [
        {"action": torch.tensor([[value if micro == 0 else 1.0]])}
        for value in values
        for micro in range(accumulation_steps)
    ]
    validation_batches = [{"action": torch.ones(1, 1)}]

    def make_components(config, device):
        return batches, validation_batches, policy, optimizer, scheduler, transforms, 0, None, recorder

    def save_checkpoint(*args, **kwargs):
        checkpoints.append((args[2], policy.forward_calls, policy.weight.item()))

    validation_calls = []
    monkeypatch.setattr(train_accelerate, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(train_accelerate, "resolve_training_checkpoint", lambda _: None)
    monkeypatch.setattr(train_accelerate, "serialize_train_config", lambda _: {})
    monkeypatch.setattr(train_accelerate, "make_everything", make_components)
    monkeypatch.setattr(train_accelerate, "make_policy_interface", lambda **kwargs: None)
    monkeypatch.setattr(train_accelerate, "WandBLogger", lambda _: logger)
    monkeypatch.setattr(train_accelerate, "tqdm", lambda **kwargs: progress)
    monkeypatch.setattr(train_accelerate, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(train_accelerate, "cycle", iter)
    monkeypatch.setattr(
        train_accelerate,
        "validate_policy",
        lambda *args, **kwargs: validation_calls.append(policy.forward_calls) or {"validation_loss": 0.0},
    )

    with pytest.raises(SystemExit) as exc:
        train_accelerate.train.__wrapped__(cfg)
    assert exc.value.code == 0
    assert policy.forward_calls == 4 * accumulation_steps
    assert [(step, calls) for step, calls, _ in checkpoints] == [
        (1, 2 * accumulation_steps),
        (2, 4 * accumulation_steps),
    ]
    assert [weight for _, _, weight in checkpoints] == pytest.approx([0.99, 0.989])
    assert validation_calls == [2 * accumulation_steps, 4 * accumulation_steps]
    assert transforms.step.call_count == progress.update.call_count == 2
    assert recorder.metrics["steps"].value == 2
    assert scheduler.last_epoch == 2
    assert accelerator.scaler.get_scale() == 2.0

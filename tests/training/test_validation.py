"""Validation aggregation should weight samples rather than batches."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from rho.training.train import validate_policy


class _SquaredActionPolicy(torch.nn.Module):
    def __init__(self, metric_type=float, metrics=True):
        super().__init__()
        self.metric_type = metric_type
        self.metrics = metrics
        self.batch_sizes = []

    def compute_loss(self, batch):
        assert not self.training
        assert not torch.is_grad_enabled()
        actions = batch["action"]
        self.batch_sizes.append(actions.shape[0])
        loss = actions.square().mean()
        metrics = None
        if self.metrics:
            metrics = {
                "l2_loss": self.metric_type(loss.item()),
                "mean_action": self.metric_type(actions.mean().item()),
            }
        return loss, metrics


class _WrappedPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.module = policy


def _loader(values, batch_size):
    # Use multiple timesteps and action dimensions so the sample count cannot
    # accidentally be confused with the number of elements in the action tensor.
    samples = [{"action": torch.full((2, 3), value, dtype=torch.float64)} for value in values]
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_validation_means_are_independent_of_batch_partition(batch_size, reverse, wrapped):
    values = [1.0, 1.0, 1.0, 1.0, 3.0]
    if reverse:
        values.reverse()
    policy = _SquaredActionPolicy()
    evaluated_policy = _WrappedPolicy(policy) if wrapped else policy

    result = validate_policy(evaluated_policy, _loader(values, batch_size), "cpu", eval_batch_size=99)

    assert result["validation_loss"] == pytest.approx(2.6)
    assert result["l2_loss"] == pytest.approx(2.6)
    assert result["mean_action"] == pytest.approx(1.4)
    assert sum(policy.batch_sizes) == len(values)


def test_validation_aggregates_auxiliary_losses_even_with_equal_batches():
    result = validate_policy(_SquaredActionPolicy(), _loader([1.0, 3.0], 1), "cpu")

    assert result["validation_loss"] == pytest.approx(5.0)
    assert result["l2_loss"] == pytest.approx(5.0)


def test_validation_uses_actual_size_of_precollated_batches():
    batches = [
        {"action": torch.ones(4, 2, 3)},
        {"action": torch.full((1, 2, 3), 3.0)},
    ]
    # Match the loader's pre-collated path, whose nominal batch_size is 1.
    loader = DataLoader(batches, batch_size=1, collate_fn=lambda items: items[0])
    result = validate_policy(_SquaredActionPolicy(), loader, "cpu")

    assert result["validation_loss"] == pytest.approx(2.6)
    assert result["l2_loss"] == pytest.approx(2.6)


@pytest.mark.parametrize("max_batches", [1, 2, 3])
def test_validation_limit_weights_only_evaluated_samples(max_batches):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    policy = _SquaredActionPolicy()
    selected = values[: 2 * max_batches]

    result = validate_policy(policy, _loader(values, 2), "cpu", max_batches=max_batches)

    expected = sum(value**2 for value in selected) / len(selected)
    assert result["validation_loss"] == pytest.approx(expected)
    assert result["l2_loss"] == pytest.approx(expected)
    assert len(policy.batch_sizes) == max_batches
    assert sum(policy.batch_sizes) == len(selected)


@pytest.mark.parametrize("metric_type", [float, np.float32, torch.tensor])
def test_validation_aggregates_scalar_metric_types(metric_type):
    result = validate_policy(_SquaredActionPolicy(metric_type), _loader([1.0, 1.0, 3.0], 2), "cpu")

    assert result["l2_loss"] == pytest.approx(11.0 / 3)
    assert isinstance(result["l2_loss"], float)


def test_validation_tracks_denominators_per_metric():
    class OptionalMetricsPolicy(_SquaredActionPolicy):
        def compute_loss(self, batch):
            loss, _ = super().compute_loss(batch)
            metrics = [
                {"l2_loss": loss.item(), "early_only": 7.0, "label": "first"},
                {"l2_loss": loss.item(), "late_only": 2.0, "label": "second"},
                None,
            ]
            return loss, metrics[len(self.batch_sizes) - 1]

    result = validate_policy(OptionalMetricsPolicy(), _loader([1.0, 1.0, 3.0, 3.0, 5.0], 2), "cpu")

    assert result["validation_loss"] == pytest.approx(9.0)
    assert result["l2_loss"] == pytest.approx(5.0)
    assert result["early_only"] == pytest.approx(7.0)
    assert result["late_only"] == pytest.approx(2.0)
    assert result["label"] == "second"


def test_validation_accepts_absent_auxiliary_metrics():
    result = validate_policy(
        _SquaredActionPolicy(metrics=False), _loader([1.0, 1.0, 1.0, 1.0, 3.0], 4), "cpu"
    )

    assert result == {"validation_loss": pytest.approx(2.6)}


@pytest.mark.parametrize("case", ["empty_loader", "zero_limit"])
def test_validation_handles_no_evaluated_batches(case):
    policy = _SquaredActionPolicy()
    loader = [] if case == "empty_loader" else _loader([3.0], 1)
    limit = None if case == "empty_loader" else 0

    assert validate_policy(policy, loader, "cpu", max_batches=limit) == {"validation_loss": 0.0}
    assert not policy.batch_sizes


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
def test_validation_matches_rho_masked_loss_reduction(batch_size):
    from rho.policies.rho import RhoPolicy

    class SquaredActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow_model = SimpleNamespace(_last_hidden_state_stats={})

        def forward(self, image, prompt, state, actions, **kwargs):
            return actions.square()

    policy = object.__new__(RhoPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(max_action_dim=3)
    policy._has_uninitialized_backbone = False
    policy.model = SquaredActionModel()
    # Stub expensive model inputs while retaining RhoPolicy.compute_loss,
    # forward, action padding masks, and its real mean/l2_loss reduction.
    policy._prepare_model_inputs = lambda batch, **kwargs: SimpleNamespace(
        image=None, prompt=None, state=None, action=batch["action"], image_mask=None
    )
    actions = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0], dtype=torch.float64)[:, None, None].expand(5, 2, 3)
    time_pad = torch.zeros(5, 2, dtype=torch.bool)
    time_pad[1, 1] = True
    dim_pad = torch.zeros(5, 3, dtype=torch.bool)
    dim_pad[4, 2] = True
    samples = [
        {"action": actions[i], "action_is_pad": time_pad[i], "action_dim_is_pad": dim_pad[i]}
        for i in range(5)
    ]
    expected = (actions.square() * (~time_pad)[..., None] * (~dim_pad)[:, None, :]).mean().item()

    result = validate_policy(policy, DataLoader(samples, batch_size=batch_size), "cpu")

    assert result["validation_loss"] == pytest.approx(expected)
    assert result["l2_loss"] == pytest.approx(expected)


@pytest.mark.parametrize("max_batches", [None, 1])
def test_validation_skips_empty_batches_without_computing_nan_loss(max_batches):
    policy = _SquaredActionPolicy()
    batches = [{"action": torch.empty(0, 2, 3)}, {"action": torch.full((1, 2, 3), 3.0)}]

    result = validate_policy(policy, batches, "cpu", max_batches=max_batches)

    if max_batches is None:
        assert result["validation_loss"] == pytest.approx(9.0)
        assert result["l2_loss"] == pytest.approx(9.0)
        assert policy.batch_sizes == [1]
    else:
        assert result == {"validation_loss": 0.0}
        assert not policy.batch_sizes

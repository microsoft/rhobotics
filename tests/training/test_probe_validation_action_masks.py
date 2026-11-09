"""Validation action MSE must not depend on artificial time or action dimensions."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rho.training.validation_probe import ValidationProbe, masked_action_mse, valid_action_elements


class _Rows(Dataset):
    meta = SimpleNamespace(episodes=[])

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _Predictions(torch.nn.Module):
    def sample_actions(self, batch):
        return {"actions": batch["prediction"]}


def _probe(samples, batch_size=2, train_samples=None):
    train_loader = (
        DataLoader(_Rows(train_samples), batch_size=batch_size) if train_samples is not None else None
    )
    probe = ValidationProbe(DataLoader(_Rows(samples), batch_size=batch_size), train_dataloader=train_loader)
    return probe.validate_policy_by_episode(_Predictions(), "cpu")


def _row(prediction, *, time_pad=None, dim_pad=None):
    prediction = torch.tensor(prediction, dtype=torch.float32)
    row = {"action": torch.zeros_like(prediction), "prediction": prediction}
    if time_pad is not None:
        row["action_is_pad"] = torch.tensor(time_pad, dtype=torch.bool)
    if dim_pad is not None:
        row["action_dim_is_pad"] = torch.tensor(dim_pad, dtype=torch.bool)
    return row


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_probe_excludes_padded_timesteps_and_dimensions(batch_size):
    samples = [
        _row(
            [[2, 90, 90], [4, 90, 90], [99, 99, 99]],
            time_pad=[False, False, True],
            dim_pad=[False, True, True],
        ),
        _row(
            [[6, 70, 70], [80, 80, 80], [80, 80, 80]],
            time_pad=[False, True, True],
            dim_pad=[False, True, True],
        ),
        _row(
            [[8, 50, 50], [60, 60, 60], [60, 60, 60]],
            time_pad=[False, True, True],
            dim_pad=[False, True, True],
        ),
    ]
    expected = (2**2 + 4**2 + 6**2 + 8**2) / 4
    assert _probe(samples, batch_size)["validation_action_mse"] == pytest.approx(expected)


def test_probe_preserves_unmasked_mse():
    samples = [_row([[1, 2], [3, 4]]), _row([[5, 6], [7, 8]])]
    assert _probe(samples)["validation_action_mse"] == pytest.approx(sum(i * i for i in range(1, 9)) / 8)


@pytest.mark.parametrize("mask", ["time", "dim"])
def test_probe_supports_individual_padding_masks(mask):
    kwargs = {"time_pad": [False, True]} if mask == "time" else {"dim_pad": [False, True]}
    sample = _row([[3, 40], [5, 80]], **kwargs)
    expected = (3**2 + 40**2) / 2 if mask == "time" else (3**2 + 5**2) / 2
    assert _probe([sample])["validation_action_mse"] == pytest.approx(expected)


def test_probe_omits_metric_when_all_action_elements_are_padded():
    sample = _row([[123, 123], [321, 321]], time_pad=[True, True], dim_pad=[False, False])
    assert "validation_action_mse" not in _probe([sample])


def test_probe_combines_different_valid_counts_across_samples():
    samples = [
        _row([[2, 2], [2, 2]], time_pad=[False, True], dim_pad=[False, False]),
        _row([[4, 4], [4, 4]], time_pad=[False, False], dim_pad=[False, False]),
    ]
    assert _probe(samples, 1)["validation_action_mse"] == pytest.approx((2 * 4 + 4 * 16) / 6)


def test_mask_short_dimension_is_treated_as_padding():
    batch = {
        "action": torch.zeros(1, 2, 3),
        "action_dim_is_pad": torch.tensor([[False]]),
    }
    mask = valid_action_elements(batch, "action")
    assert mask.tolist() == [[[True, False, False], [True, False, False]]]


def test_mask_longer_than_action_is_trimmed():
    batch = {
        "action": torch.zeros(1, 1, 2),
        "action_dim_is_pad": torch.tensor([[False, True, False, False]]),
    }
    assert valid_action_elements(batch, "action").tolist() == [[[True, False]]]


@pytest.mark.parametrize(
    "mask,problem",
    [
        ({"action_is_pad": torch.zeros(2, 4)}, "action_is_pad"),
        ({"action_dim_is_pad": torch.zeros(2, 3)}, "action_dim_is_pad"),
    ],
)
def test_invalid_padding_shape_is_rejected(mask, problem):
    batch = {"action": torch.zeros(1, 2, 3), **mask}
    with pytest.raises(ValueError, match=problem):
        valid_action_elements(batch, "action")


def test_metric_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        masked_action_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool))

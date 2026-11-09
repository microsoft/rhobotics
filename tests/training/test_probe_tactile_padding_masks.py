"""Tactile-action comparisons must ignore artificial time steps and channels."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rho.common.constants import ACTION_TACTILE
from rho.training.validation_probe import ValidationProbe


class _Rows(Dataset):
    meta = SimpleNamespace(episodes=[])

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _Policy(torch.nn.Module):
    def sample_actions(self, batch):
        return {"actions": batch["prediction"], "tactile_action": batch.get("tactile_prediction")}


def _row(tactile, *, time_pad=None, dim_pad=None, no_tactile=False):
    tactile = torch.tensor(tactile, dtype=torch.float32)
    row = {"action": torch.zeros_like(tactile), "prediction": torch.zeros_like(tactile)}
    if not no_tactile:
        row[ACTION_TACTILE] = torch.zeros_like(tactile)
        row["tactile_prediction"] = tactile
        if time_pad is not None:
            row[f"{ACTION_TACTILE}_is_pad"] = torch.tensor(time_pad, dtype=torch.bool)
        if dim_pad is not None:
            row[f"{ACTION_TACTILE}_dim_is_pad"] = torch.tensor(dim_pad, dtype=torch.bool)
    return row


def _metric(val, train=None, batch_size=2):
    train_loader = DataLoader(_Rows(train), batch_size=batch_size) if train is not None else None
    probe = ValidationProbe(DataLoader(_Rows(val), batch_size=batch_size), train_dataloader=train_loader)
    return probe.validate_policy_by_episode(_Policy(), "cpu")


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_validation_tactile_mse_excludes_time_and_dimension_padding(batch_size):
    rows = [
        _row([[2, 50], [99, 99]], time_pad=[False, True], dim_pad=[False, True]),
        _row([[4, 50], [6, 50]], time_pad=[False, False], dim_pad=[False, True]),
        _row([[8, 50], [99, 99]], time_pad=[False, True], dim_pad=[False, True]),
    ]
    metrics = _metric(rows, batch_size=batch_size)
    assert metrics["validation_tactile_action_mse"] == pytest.approx((4 + 16 + 36 + 64) / 4)
    assert metrics["validation_action_mse"] == pytest.approx(0)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_training_tactile_mse_uses_its_own_padding_masks(batch_size):
    val = [_row([[1, 1], [1, 1]])]
    train = [
        _row([[2, 90], [90, 90]], time_pad=[False, True], dim_pad=[False, True]),
        _row([[4, 90], [6, 90]], time_pad=[False, False], dim_pad=[False, True]),
    ]
    metrics = _metric(val, train=train, batch_size=batch_size)
    assert metrics["validation_tactile_action_mse"] == pytest.approx(1)
    assert metrics["train_tactile_action_mse"] == pytest.approx((4 + 16 + 36) / 3)
    assert metrics["train_action_mse"] == pytest.approx(0)


def test_missing_tactile_padding_masks_preserves_unmasked_mse():
    val = [_row([[1, 2], [3, 4]])]
    train = [_row([[5, 6], [7, 8]])]
    metrics = _metric(val, train=train)
    assert metrics["validation_tactile_action_mse"] == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert metrics["train_tactile_action_mse"] == pytest.approx((25 + 36 + 49 + 64) / 4)


def test_tactile_time_mask_is_independent_from_robot_action_mask():
    sample = _row([[2, 50], [99, 99]], time_pad=[False, True], dim_pad=[False, True])
    sample["action_is_pad"] = torch.tensor([False, False])
    metrics = _metric([sample])
    assert metrics["validation_tactile_action_mse"] == pytest.approx(4)


@pytest.mark.parametrize("target", ["validation", "training"])
def test_all_padded_tactile_elements_are_omitted(target):
    padded = [_row([[90, 90], [90, 90]], time_pad=[True, True])]
    valid = [_row([[1, 1], [1, 1]])]
    if target == "validation":
        metrics = _metric(padded, train=valid)
        assert "validation_tactile_action_mse" not in metrics
        assert metrics["train_tactile_action_mse"] == pytest.approx(1)
    else:
        metrics = _metric(valid, train=padded)
        assert metrics["validation_tactile_action_mse"] == pytest.approx(1)
        assert "train_tactile_action_mse" not in metrics


def test_absent_tactile_predictions_leave_metrics_unchanged():
    sample = _row([[1, 2]], no_tactile=True)
    metrics = _metric([sample], train=[sample])
    assert metrics["validation_action_mse"] == pytest.approx(0)
    assert metrics["train_action_mse"] == pytest.approx(0)
    assert "validation_tactile_action_mse" not in metrics
    assert "train_tactile_action_mse" not in metrics


def test_tactile_dimension_mask_shorter_than_tensor_excludes_appended_channels():
    sample = _row([[3, 90, 90]], dim_pad=[False])
    assert _metric([sample])["validation_tactile_action_mse"] == pytest.approx(9)

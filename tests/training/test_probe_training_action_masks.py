"""The optional training comparison must exclude padded action elements."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rho.training.validation_probe import ValidationProbe


class _Rows(Dataset):
    meta = SimpleNamespace(episodes=[])

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _PredictionPolicy(torch.nn.Module):
    def sample_actions(self, batch):
        return {"actions": batch["prediction"]}


def _row(prediction, *, time_pad=None, dim_pad=None):
    predicted = torch.tensor(prediction, dtype=torch.float32)
    row = {"action": torch.zeros_like(predicted), "prediction": predicted}
    if time_pad is not None:
        row["action_is_pad"] = torch.tensor(time_pad, dtype=torch.bool)
    if dim_pad is not None:
        row["action_dim_is_pad"] = torch.tensor(dim_pad, dtype=torch.bool)
    return row


def _metrics(validation, training, batch_size=2):
    probe = ValidationProbe(
        DataLoader(_Rows(validation), batch_size=batch_size),
        train_dataloader=DataLoader(_Rows(training), batch_size=batch_size),
    )
    return probe.validate_policy_by_episode(_PredictionPolicy(), "cpu")


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_training_action_mse_weights_only_genuine_elements(batch_size):
    validation = [_row([[1, 1], [1, 1]])]
    training = [
        _row([[2, 99], [99, 99]], time_pad=[False, True], dim_pad=[False, True]),
        _row([[4, 99], [6, 99]], time_pad=[False, False], dim_pad=[False, True]),
        _row([[8, 99], [99, 99]], time_pad=[False, True], dim_pad=[False, True]),
    ]
    result = _metrics(validation, training, batch_size)
    assert result["validation_action_mse"] == pytest.approx(1.0)
    assert result["train_action_mse"] == pytest.approx((4 + 16 + 36 + 64) / 4)


def test_training_comparison_preserves_unmasked_mse():
    validation = [_row([[3, 3]])]
    training = [_row([[1, 2], [3, 4]])]
    result = _metrics(validation, training)
    assert result["validation_action_mse"] == pytest.approx(9.0)
    assert result["train_action_mse"] == pytest.approx(7.5)


@pytest.mark.parametrize("mask", ["time", "dimension"])
def test_training_supports_individual_masks(mask):
    kwargs = {"time_pad": [False, True]} if mask == "time" else {"dim_pad": [False, True]}
    training = [_row([[3, 30], [5, 50]], **kwargs)]
    expected = (9 + 900) / 2 if mask == "time" else (9 + 25) / 2
    assert _metrics([_row([[0, 0]])], training)["train_action_mse"] == pytest.approx(expected)


def test_training_comparison_omits_all_padded_metric():
    validation = [_row([[1, 2]])]
    training = [_row([[99, 99]], time_pad=[True], dim_pad=[False, False])]
    result = _metrics(validation, training)
    assert result["validation_action_mse"] == pytest.approx(2.5)
    assert "train_action_mse" not in result


def test_training_comparison_rejects_invalid_mask_shape():
    training = [_row([[1, 2], [3, 4]], time_pad=[False, True])]
    training[0]["action_is_pad"] = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="action_is_pad"):
        _metrics([_row([[0, 0]])], training)

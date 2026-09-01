import torch

from rho.common.constants import ACTION
from rho.common.transforms import build_key_padding_transform
from rho.common.types import FeatureType, PolicyFeature
from rho.policies.rho.rho_policy import _apply_action_padding_masks


def test_transform_emitted_dim_mask_zeroes_padded_action_loss_cells():
    transform = build_key_padding_transform(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (32,))},
        target_sequence_lengths={FeatureType.ACTION: 5},
    )
    sample = {
        ACTION: torch.ones(3, 7),
        f"{ACTION}_is_pad": torch.tensor([False, False, False]),
    }
    padded = transform(sample)

    action_loss = torch.ones(1, 5, 32, requires_grad=True)
    masked = _apply_action_padding_masks(
        action_loss,
        padded[f"{ACTION}_is_pad"].unsqueeze(0),
        padded[f"{ACTION}_dim_is_pad"].unsqueeze(0),
    )

    assert masked.shape == action_loss.shape
    assert masked[..., 7:].sum() == 0
    assert masked[:, 3:].sum() == 0
    assert masked[:, :3, :7].sum() > 0

    masked.mean().backward()

    assert action_loss.grad[..., 7:].sum() == 0
    assert action_loss.grad[:, 3:].sum() == 0
    assert action_loss.grad[:, :3, :7].sum() > 0


def test_action_padding_masks_zero_dim_and_time_contributions_without_changing_shape():
    action_loss = torch.ones(2, 4, 32, requires_grad=True)
    action_is_pad = torch.tensor(
        [
            [False, False, True, True],
            [False, True, False, True],
        ]
    )
    action_dim_is_pad = torch.zeros(2, 32, dtype=torch.bool)
    action_dim_is_pad[0, 7:] = True
    action_dim_is_pad[1, 20:] = True

    masked = _apply_action_padding_masks(action_loss, action_is_pad, action_dim_is_pad)

    assert masked.shape == action_loss.shape
    assert masked[0, :, 7:].sum() == 0
    assert masked[1, :, 20:].sum() == 0
    assert masked[action_is_pad].sum() == 0
    assert masked[0, :2, :7].sum() > 0
    assert masked[1, [0, 2], :20].sum() > 0

    masked.mean().backward()

    assert action_loss.grad[0, :, 7:].sum() == 0
    assert action_loss.grad[1, :, 20:].sum() == 0
    assert action_loss.grad[action_is_pad].sum() == 0
    assert action_loss.grad[0, :2, :7].sum() > 0
    assert action_loss.grad[1, [0, 2], :20].sum() > 0


def test_action_dim_mask_shorter_than_loss_masks_missing_tail_dims():
    action_loss = torch.ones(2, 3, 32, requires_grad=True)
    action_dim_is_pad = torch.zeros(2, 20, dtype=torch.bool)
    action_dim_is_pad[0, 16:] = True

    masked = _apply_action_padding_masks(action_loss, action_dim_is_pad=action_dim_is_pad)

    assert masked.shape == action_loss.shape
    assert masked[0, :, 16:].sum() == 0
    assert masked[1, :, 20:].sum() == 0
    assert masked[0, :, :16].sum() > 0
    assert masked[1, :, :20].sum() > 0

    masked.mean().backward()

    assert action_loss.grad[0, :, 16:].sum() == 0
    assert action_loss.grad[1, :, 20:].sum() == 0
    assert action_loss.grad[0, :, :16].sum() > 0
    assert action_loss.grad[1, :, :20].sum() > 0

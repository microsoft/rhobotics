"""Single-action inference must preserve the training image layout."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rho.common.constants import ACTION, OBSERVATION_IMAGE, OBSERVATION_LANG, OBSERVATION_STATE
from rho.common.types import FeatureType, PolicyFeature
from rho.policies.rho import RhoConfig, RhoPolicy


class _RecordingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.flow_model = SimpleNamespace(_backend=None, _last_hidden_state_stats={})
        self.calls = []

    def print_freezing_status(self):
        pass

    def sample_actions(self, image, prompt, state, noise=None, image_mask=None):
        self.calls.append(
            {
                "image": image.clone(),
                "image_mask": image_mask.clone(),
                "state": state.clone(),
                "prompt": list(prompt),
                "noise": None if noise is None else noise.clone(),
            }
        )
        return image.new_zeros((image.shape[0], self.config.chunk_size, self.config.max_action_dim))


def make_history_policy(
    *,
    camera_keys=(f"{OBSERVATION_IMAGE}.wrist", f"{OBSERVATION_IMAGE}.head"),
    n_obs_steps=2,
    n_action_steps=1,
    chunk_size=2,
    resize=None,
):
    features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)) for key in camera_keys}
    features[OBSERVATION_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(2,))
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(2,))
    config = RhoConfig(
        device="cpu",
        dtype="float32",
        feature_dict=features,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        chunk_size=chunk_size,
        max_state_dim=2,
        max_action_dim=2,
        resize_imgs_with_padding=resize,
    )
    with patch("rho.policies.rho.rho_policy.RhoModel", _RecordingModel):
        return RhoPolicy(config)


def frame_batch(camera_values, *, pad_masks=None):
    batch = {
        key: torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2).clone()
        for key, values in camera_values.items()
    }
    batch_size = len(next(iter(camera_values.values())))
    batch[OBSERVATION_STATE] = torch.zeros(batch_size, 2)
    batch[OBSERVATION_LANG] = ["move"] * batch_size
    for key, mask in (pad_masks or {}).items():
        batch[f"{key}_is_pad"] = torch.tensor(mask, dtype=torch.bool)
    return batch


def test_select_action_preserves_camera_order_across_history():
    policy = make_history_policy()
    wrist, head = policy.config.image_features
    policy.select_action(
        frame_batch(
            {wrist: [0.1, 0.5], head: [0.3, 0.7]},
            pad_masks={wrist: [False, True], head: [True, False]},
        )
    )
    policy.select_action(
        frame_batch(
            {wrist: [0.2, 0.6], head: [0.4, 0.8]},
            pad_masks={wrist: [False, False], head: [False, True]},
        )
    )

    expected_images = [
        [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]],
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
    ]
    expected_masks = [[[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 1, 0, 1], [0, 1, 1, 0]]]
    assert len(policy.model.calls) == 2
    for call, values, mask in zip(policy.model.calls, expected_images, expected_masks, strict=True):
        expected = torch.tensor(values)[:, :, None, None, None].expand(-1, -1, 3, 2, 2)
        torch.testing.assert_close(call["image"], expected)
        torch.testing.assert_close(call["image_mask"], torch.tensor(mask, dtype=torch.int32))

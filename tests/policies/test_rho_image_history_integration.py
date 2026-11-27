"""Training and public inference entry points must see the same camera history."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rho.common.constants import ACTION, OBSERVATION_LANG, OBSERVATION_STATE
from rho.eval.policy_interface import PolicyInterface
from tests.eval.test_policy_interface import _identity_transform, _make_pi_config
from tests.policies.test_rho_image_history import frame_batch, make_history_policy

WRIST = "observation.image.wrist"
HEAD = "observation.image.head"
CAMERAS = (WRIST, HEAD)


class _OrderSensitiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.flow_model = SimpleNamespace(_last_hidden_state_stats={})
        self.calls = []

    def _predict(self, kind, image, prompt, state, noise, image_mask):
        self.calls.append(
            {
                "kind": kind,
                "image": image.clone(),
                "image_mask": image_mask.clone(),
                "state": state.clone(),
                "prompt": list(prompt),
            }
        )
        # Every slot contributes differently, so swapping camera/time slots
        # changes the returned actions even when the tensor shape is unchanged.
        weights = torch.arange(1, image.shape[1] + 1, dtype=image.dtype, device=image.device)
        score = (image.mean(dim=(2, 3, 4)) * image_mask * weights).sum(dim=1)
        score = score + state[:, -1].sum(dim=1)
        return noise + score[:, None, None]

    def sample_actions(self, image, prompt, state, noise=None, image_mask=None):
        return self._predict("sample", image, prompt, state, noise, image_mask)

    def forward(self, image, prompt, state, action, noise=None, time=None, image_mask=None):
        prediction = self._predict("train", image, prompt, state, noise, image_mask)
        return (prediction - action).square()


def _frames(*, with_masks=True, offset=0.0):
    observations = []
    for step in range(3):
        masks = None
        if with_masks:
            masks = {WRIST: [step == 1, step == 0], HEAD: [step == 2, step == 1]}
        batch = frame_batch(
            {
                WRIST: [0.1 + 0.1 * step + offset, 0.15 + 0.1 * step + offset],
                HEAD: [0.6 + 0.1 * step + offset, 0.65 + 0.1 * step + offset],
            },
            pad_masks=masks,
        )
        batch[OBSERVATION_STATE] = torch.tensor([[step, step + 1], [step + 2, step + 3]]).float()
        observations.append(batch)
    return observations


def _history_batch(frames):
    batch = {
        key: torch.stack([frame[key] for frame in frames], dim=1) for key in (*CAMERAS, OBSERVATION_STATE)
    }
    for key in CAMERAS:
        mask_key = f"{key}_is_pad"
        if mask_key in frames[0]:
            batch[mask_key] = torch.stack([frame[mask_key].reshape(2) for frame in frames], dim=1)
    batch[OBSERVATION_LANG] = list(frames[-1][OBSERVATION_LANG])
    return batch


@pytest.mark.parametrize("n_action_steps", [1, 2])
def test_training_and_chunk_sampling_match_streamed_camera_history(n_action_steps):
    policy = make_history_policy(camera_keys=CAMERAS, n_obs_steps=3, n_action_steps=n_action_steps)
    policy.model = _OrderSensitiveModel()
    frames = _frames()
    noise = torch.tensor([[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]])

    for frame in frames:
        selected = policy.select_action(dict(frame), noise=noise)
    streamed_inputs = policy.model.calls[-1]
    assert len(policy.model.calls) == (3 if n_action_steps == 1 else 2)

    history = _history_batch(frames)
    sampled = policy.sample_actions(history, noise=noise)["actions"]
    sampled_inputs = policy.model.calls[-1]
    history[ACTION] = sampled.clone()
    policy.train()
    loss, metrics = policy(history, noise=noise, time=torch.full((2,), 0.5))
    training_inputs = policy.model.calls[-1]

    expected_pixels = torch.tensor([[0.1, 0.2, 0.3, 0.6, 0.7, 0.8], [0.15, 0.25, 0.35, 0.65, 0.75, 0.85]])
    expected_mask = torch.tensor([[1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1]], dtype=torch.int32)
    for actual in (streamed_inputs, sampled_inputs, training_inputs):
        torch.testing.assert_close(actual["image"][:, :, 0, 0, 0], expected_pixels)
        torch.testing.assert_close(actual["image_mask"], expected_mask)
        torch.testing.assert_close(actual["state"], history[OBSERVATION_STATE])
        assert actual["prompt"] == training_inputs["prompt"]
    # The weighted camera values contribute 6.9 and 9.25; latest state adds 5 and 9.
    expected_actions = noise + torch.tensor([11.9, 18.25])[:, None, None]
    torch.testing.assert_close(sampled, expected_actions)
    torch.testing.assert_close(selected, expected_actions[:, 0])
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert metrics["l2_loss"] == pytest.approx(0.0, abs=1e-12)

    if n_action_steps == 2:
        calls_before = len(policy.model.calls)
        cached = policy.select_action(dict(frames[-1]), noise=noise + 100)
        torch.testing.assert_close(cached, expected_actions[:, 1])
        assert len(policy.model.calls) == calls_before


def test_policy_interface_matches_direct_actions_after_history_is_populated():
    direct = make_history_policy(camera_keys=CAMERAS, n_obs_steps=3)
    served = make_history_policy(camera_keys=CAMERAS, n_obs_steps=3)
    direct.model = _OrderSensitiveModel()
    served.model = _OrderSensitiveModel()
    interface = PolicyInterface(
        _make_pi_config(
            policy=served,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
        )
    )
    noise = torch.full((2, 2, 2), 0.125)

    for offset in (0.0, 0.1):
        direct.reset()
        interface.reset()
        for frame in _frames(with_masks=False, offset=offset):
            selected = direct.select_action(dict(frame), noise=noise)
            chunk = interface.get_action_chunk(dict(frame), noise=noise)

        # Startup repetition and startup mask conventions belong to each API;
        # compare once both histories consist entirely of observed frames.
        torch.testing.assert_close(selected, chunk[:, 0])
        direct_inputs = direct.model.calls[-1]
        served_inputs = served.model.calls[-1]
        for key in ("image", "image_mask", "state"):
            torch.testing.assert_close(direct_inputs[key], served_inputs[key])
        assert direct_inputs["prompt"] == served_inputs["prompt"]

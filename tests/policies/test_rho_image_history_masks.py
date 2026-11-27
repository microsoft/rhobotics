"""Camera history keeps each image paired with its padding mask."""

import pytest
import torch

from rho.common.constants import OBSERVATION_IMAGE
from tests.policies.test_rho_image_history import frame_batch, make_history_policy

CAMERAS = (
    f"{OBSERVATION_IMAGE}.z_wrist",
    f"{OBSERVATION_IMAGE}.a_front",
    f"{OBSERVATION_IMAGE}.m_side",
)


def _pixel(lane, camera, step):
    return (100 * lane + 10 * camera + step + 1) / 1000


def _is_pad(lane, camera, step):
    return (lane + 2 * camera + step) % 3 == 0


def _frame(camera_keys, batch_size, step):
    # Input dictionary order deliberately differs from the configuration order.
    values = {
        key: [_pixel(lane, camera, step) for lane in range(batch_size)]
        for camera, key in reversed(list(enumerate(camera_keys)))
    }
    masks = {
        key: [_is_pad(lane, camera, step) for lane in range(batch_size)]
        for camera, key in enumerate(camera_keys)
    }
    return frame_batch(values, pad_masks=masks)


@pytest.mark.parametrize(
    ("camera_count", "history_steps", "batch_size", "action_steps"),
    [
        (1, 1, 1, 1),
        (1, 3, 2, 2),
        (2, 1, 2, 2),
        (2, 3, 2, 3),
        (3, 2, 2, 1),
        (3, 3, 1, 2),
    ],
)
def test_camera_history_and_masks_follow_configured_order(
    camera_count, history_steps, batch_size, action_steps
):
    camera_keys = CAMERAS[:camera_count]
    policy = make_history_policy(
        camera_keys=camera_keys,
        n_obs_steps=history_steps,
        n_action_steps=action_steps,
        chunk_size=action_steps + 1,
    )

    for step in range(history_steps + 2 * action_steps + 1):
        action = policy.select_action(_frame(camera_keys, batch_size, step))
        assert action.shape == (batch_size, 2)
        assert len(policy.model.calls) == step // action_steps + 1
        if step % action_steps:
            continue

        # Enumerate camera then time, including the existing first-frame repeat.
        times = [max(0, index) for index in range(step - history_steps + 1, step + 1)]
        expected_pixels = torch.tensor(
            [
                [_pixel(lane, camera, index) for camera in range(camera_count) for index in times]
                for lane in range(batch_size)
            ]
        )
        expected_mask = torch.tensor(
            [
                [int(not _is_pad(lane, camera, index)) for camera in range(camera_count) for index in times]
                for lane in range(batch_size)
            ],
            dtype=torch.int32,
        )
        call = policy.model.calls[-1]
        torch.testing.assert_close(
            call["image"], expected_pixels[:, :, None, None, None].expand_as(call["image"])
        )
        assert torch.equal(call["image_mask"], expected_mask)


@pytest.mark.parametrize("mask_rank", [1, 2, 3])
def test_camera_history_accepts_per_frame_mask_shapes(mask_rank):
    camera_keys = CAMERAS[:2]
    policy = make_history_policy(camera_keys=camera_keys, n_obs_steps=2)

    for step in range(2):
        batch = _frame(camera_keys, 2, step)
        for key in camera_keys:
            mask = torch.tensor(
                [_is_pad(lane, camera_keys.index(key), step) for lane in range(2)], dtype=torch.bool
            )
            batch[f"{key}_is_pad"] = mask.reshape((2,) + (1,) * (mask_rank - 1))
        policy.select_action(batch)

    # Each camera's two times stay adjacent, with different masks in each lane.
    assert torch.equal(
        policy.model.calls[-1]["image_mask"],
        torch.tensor([[0, 1, 1, 0], [1, 1, 0, 1]], dtype=torch.int32),
    )


def test_cached_actions_keep_recording_intermediate_camera_frames(monkeypatch):
    camera_keys = CAMERAS[:2]
    policy = make_history_policy(camera_keys=camera_keys, n_obs_steps=3, n_action_steps=3, chunk_size=4)
    sample = policy.model.sample_actions

    def identifiable_actions(*args, **kwargs):
        actions = sample(*args, **kwargs)
        query = len(policy.model.calls)
        actions.copy_(100 * query + torch.arange(4, dtype=actions.dtype).reshape(1, 4, 1))
        return actions

    monkeypatch.setattr(policy.model, "sample_actions", identifiable_actions)
    actual = [policy.select_action(_frame(camera_keys, 1, step))[0, 0].item() for step in range(7)]

    assert actual == [100, 101, 102, 200, 201, 202, 300]
    assert len(policy.model.calls) == 3
    # Queries at steps 3 and 6 use the latest three simulator frames.
    torch.testing.assert_close(
        policy.model.calls[1]["image"][0, :, 0, 0, 0],
        torch.tensor([0.002, 0.003, 0.004, 0.012, 0.013, 0.014]),
    )
    torch.testing.assert_close(
        policy.model.calls[2]["image"][0, :, 0, 0, 0],
        torch.tensor([0.005, 0.006, 0.007, 0.015, 0.016, 0.017]),
    )


def test_reset_discards_camera_history_masks_and_pending_actions():
    camera_keys = CAMERAS[:2]
    policy = make_history_policy(camera_keys=camera_keys, n_obs_steps=3, n_action_steps=3, chunk_size=3)
    for step in range(5):
        policy.select_action(_frame(camera_keys, 2, step))
    assert len(policy.model.calls) == 2

    policy.reset()
    fresh = frame_batch(
        {camera_keys[1]: [0.8, 0.9], camera_keys[0]: [0.6, 0.7]},
        pad_masks={camera_keys[0]: [True, False], camera_keys[1]: [False, True]},
    )
    policy.select_action(fresh)

    assert len(policy.model.calls) == 3
    call = policy.model.calls[-1]
    torch.testing.assert_close(
        call["image"][:, :, 0, 0, 0],
        torch.tensor([[0.6, 0.6, 0.6, 0.8, 0.8, 0.8], [0.7, 0.7, 0.7, 0.9, 0.9, 0.9]]),
    )
    assert torch.equal(
        call["image_mask"],
        torch.tensor([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.int32),
    )

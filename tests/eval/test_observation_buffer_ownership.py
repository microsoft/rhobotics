"""Keep temporal history stable when environments reuse observation buffers."""

import numpy as np
import pytest
import torch

from rho.environment.env import evaluate_policy
from tests.eval.test_rollout_observation_history import _FrameEnvironment, _make_interface, _observation


class _ReusedBufferEnvironment(_FrameEnvironment):
    def __init__(self, explicit_time_axis):
        super().__init__(max_steps=5)
        prefix = (1, 1) if explicit_time_axis else (1,)
        self.buffers = {
            "joint_positions": np.zeros((*prefix, 1), dtype=np.float32),
            "camera_rgb": np.zeros((*prefix, 3, 2, 2), dtype=np.float32),
        }
        self.observation = {key: torch.from_numpy(value) for key, value in self.buffers.items()}
        self.observation["task"] = ["record reused buffers"]

    def _current_observation(self):
        for buffer in self.buffers.values():
            buffer.fill(self.frame)
        return self.observation

    def next_episode(self, seed=None):
        self.frame = 0
        return self._current_observation(), {}

    def step(self, action):
        self.frame += 1
        return self._current_observation(), 0.0, False, self.frame >= self.max_steps, {}


@pytest.mark.parametrize(
    "explicit_time_axis,eval_mode", [(False, "standard"), (True, "standard"), (False, "rtc")]
)
def test_rollout_history_survives_reused_numpy_buffers(explicit_time_axis, eval_mode):
    env = _ReusedBufferEnvironment(explicit_time_axis)
    interface, policy = _make_interface(eval_mode)

    evaluate_policy(
        env, interface, num_episodes=1, max_steps=env.max_steps, eval_mode=eval_mode, inference_delay=1
    )

    assert len(policy.observations) == 2
    for key in ("observation.state", "observation.image.0"):
        assert policy.observations[-1][key].flatten(start_dim=2)[:, :, 0].tolist() == [[3, 4]]
        assert policy.observations[-1][f"{key}_is_pad"].tolist() == [[False, False]]


def test_multiframe_history_owns_each_batch_lane():
    interface, _ = _make_interface()
    frames = torch.tensor([[[1.0], [2.0], [3.0]], [[11.0], [12.0], [13.0]]])
    interface.process_obs_queue({"observation.state": frames})

    frames.add_(100)
    result = interface.process_obs_queue({"observation.state": torch.tensor([[[4.0]], [[14.0]]])})

    assert result["observation.state"][:, :, 0].tolist() == [[3, 4], [13, 14]]
    assert result["observation.state_is_pad"].tolist() == [[False, False], [False, False]]


def test_equal_values_from_reused_observation_are_distinct_frames():
    interface, policy = _make_interface()
    observation = _observation(7)

    interface.observe(observation)
    interface.observe(observation)
    for key in ("observation.state", "observation.image.0"):
        assert len(interface.obs_queue[key]) == 2

    interface.get_action_chunk(observation)

    for key in ("observation.state", "observation.image.0"):
        assert policy.observations[-1][key].flatten(start_dim=2)[:, :, 0].tolist() == [[7, 7]]
        assert policy.observations[-1][f"{key}_is_pad"].tolist() == [[False, False]]

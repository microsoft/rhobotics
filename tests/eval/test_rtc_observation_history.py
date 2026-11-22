"""RTC history, action overlap, and episode boundaries through the real rollout."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rho.environment.env import evaluate_policy
from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig


def _observation(values):
    state = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
    return {
        "observation.state": state,
        "observation.image.0": state[:, :, :, None, None].expand(-1, -1, 3, 2, 2).clone(),
        "task": ["follow the frame"] * len(values),
    }


class _FrameEnvironment:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.is_vectorized = num_envs > 1
        self.episode = -1
        self.step_index = 0
        self.actions = []

    def _observation(self):
        return _observation(1000 * self.episode + 100 * np.arange(self.num_envs) + self.step_index)

    def next_episode(self, seed):
        self.episode += 1
        self.step_index = 0
        return self._observation(), {}

    def step(self, action):
        self.actions.append(action.detach().clone())
        self.step_index += 1
        return (
            self._observation(),
            np.zeros(self.num_envs),
            np.full(self.num_envs, self.step_index == 5),
            np.zeros(self.num_envs, dtype=bool),
            {"is_success": np.zeros(self.num_envs, dtype=bool)},
        )


class _RecordingPolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            chunk_size=6,
            n_action_steps=2,
            delta_indices_dict={
                "observation.state": [-3, -1, 0],
                "observation.image.0": [-3, -1, 0],
            },
        )
        self.model = SimpleNamespace(action_expert=SimpleNamespace(enable_gradient_checkpointing=False))
        self.calls = []
        self.input_calls = 0
        self.output_calls = 0

    def to_relative(self, obs):
        self.input_calls += 1
        if "action" in obs:
            obs["action"] = obs["action"] - obs["observation.state"][:, -1:, :]
        return obs

    def to_absolute(self, obs):
        self.output_calls += 1
        obs["action"] = obs["action"] + obs["observation.state"][:, -1:, :]
        return obs

    def sample_actions_rtc(self, obs, *, prev_actions, **kwargs):
        snapshot = {key: value.clone() for key, value in obs.items() if isinstance(value, torch.Tensor)}
        snapshot["prev_actions"] = None if prev_actions is None else prev_actions.clone()
        chunk = 100 * len(self.calls) + torch.arange(6, dtype=torch.float32)
        self.calls.append(snapshot)
        return {"actions": chunk.reshape(1, 6, 1).expand(obs["observation.state"].shape[0], -1, -1)}


def _make_interface(inference_delay=1):
    policy = _RecordingPolicy()
    config = PolicyInterfaceConfig(
        policy=policy, device="cpu", eval_mode="rtc", inference_delay=inference_delay, beta=2
    )
    config.input_transforms = policy.to_relative
    config.output_transforms = policy.to_absolute
    return PolicyInterface(config), policy


def _assert_history(call, frames, padding, offsets):
    expected = torch.tensor(frames, dtype=torch.float32)[None, :, None]
    expected = expected + torch.tensor(offsets, dtype=torch.float32)[:, None, None]
    torch.testing.assert_close(call["observation.state"], expected)
    torch.testing.assert_close(
        call["observation.image.0"], expected[:, :, :, None, None].expand(-1, -1, 3, 2, 2)
    )
    for key in ("observation.state", "observation.image.0"):
        torch.testing.assert_close(call[f"{key}_is_pad"], torch.tensor(padding).expand(len(offsets), -1))


@pytest.mark.parametrize("inference_delay", [0, 1])
def test_rtc_sparse_history_preserves_action_reframing_and_overlap(inference_delay):
    interface, policy = _make_interface(inference_delay)
    env = _FrameEnvironment()
    evaluate_policy(
        env, interface, num_episodes=2, max_steps=5, eval_mode="rtc", inference_delay=inference_delay
    )

    assert len(policy.calls) == policy.input_calls == policy.output_calls == 3
    _assert_history(policy.calls[0], [0, 0, 0], [True, True, False], [0, 100])
    _assert_history(policy.calls[1], [0, 1, 2], [True, False, False], [0, 100])
    _assert_history(policy.calls[2], [1, 3, 4], [False, False, False], [0, 100])
    assert policy.calls[0]["prev_actions"] is None
    for call, values in zip(policy.calls[1:], ([0, 1, 2, 3], [100, 101, 102, 103]), strict=True):
        expected = torch.tensor(values, dtype=torch.float32).reshape(1, 4, 1).expand(2, -1, -1)
        torch.testing.assert_close(call["prev_actions"], expected)

    expected = [0, 1, 2, 103, 104] if inference_delay else [0, 1, 102, 103, 204]
    expected = torch.tensor(expected, dtype=torch.float32)[:, None, None]
    expected = expected + torch.tensor([0.0, 100.0])[None, :, None]
    torch.testing.assert_close(torch.stack(env.actions), expected)


def test_rtc_history_and_cached_actions_reset_between_vector_batches():
    interface, policy = _make_interface()
    result = evaluate_policy(
        _FrameEnvironment(), interface, num_episodes=4, max_steps=5, eval_mode="rtc", inference_delay=1
    )

    assert result["num_episodes"] == 4
    assert len(policy.calls) == 6
    for start, offset in ((0, 0), (3, 1000)):
        assert policy.calls[start]["prev_actions"] is None
        _assert_history(policy.calls[start], [0, 0, 0], [True, True, False], [offset, offset + 100])
        _assert_history(policy.calls[start + 2], [1, 3, 4], [False, False, False], [offset, offset + 100])
    assert all(not queue for queue in interface.obs_queue.values())
    assert interface.prev_action_chunk is None


def test_rtc_cloned_sessions_keep_independent_intermediate_frames():
    first, policy = _make_interface()
    first.get_action_chunk(_observation([0]))
    second = first.clone_with_fresh_episode_state()
    first.observe(_observation([1]))
    second.observe(_observation([100]))
    first.get_action_chunk({**_observation([2]), "num_actions_executed": 2})
    second.get_action_chunk({**_observation([101]), "num_actions_executed": 0})

    _assert_history(policy.calls[1], [0, 1, 2], [True, False, False], [0])
    _assert_history(policy.calls[2], [100, 100, 101], [True, False, False], [0])
    assert policy.calls[2]["prev_actions"] is None
    first.reset()
    second.observe(_observation([102]))
    second.get_action_chunk({**_observation([103]), "num_actions_executed": 2})
    _assert_history(policy.calls[3], [100, 102, 103], [False, False, False], [0])
    assert all(not queue for queue in first.obs_queue.values())
    assert first.prev_action_chunk is None
    assert second.prev_action_chunk is not None

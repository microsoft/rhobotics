"""Check observation history through the real simulation evaluation loop."""

from types import SimpleNamespace

import pytest
import torch

from rho.environment.env import evaluate_policy
from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig


def _observation(frame):
    return {
        "joint_positions": torch.full((1, 1, 1), float(frame)),
        "camera_rgb": torch.full((1, 3, 2, 2), float(frame)),
        "task": ["record frame history"],
    }


class _FrameEnvironment:
    num_envs = 1
    is_vectorized = False

    def __init__(self, max_steps=9):
        self.max_steps = max_steps
        self.frame = 0

    def next_episode(self, seed=None):
        self.frame = 0
        return _observation(self.frame), {}

    def step(self, action):
        self.frame += 1
        return _observation(self.frame), 0.0, False, self.frame >= self.max_steps, {}


class _CountingTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return obs


class _RecordingPolicy:
    def __init__(self, execution_horizon):
        self.config = SimpleNamespace(
            chunk_size=8,
            n_action_steps=execution_horizon,
            delta_indices_dict={
                "observation.state": [-1, 0],
                "observation.image.0": [-1, 0],
                "action": list(range(8)),
            },
            feature_dict={},
        )
        self.model = SimpleNamespace(action_expert=SimpleNamespace(enable_gradient_checkpointing=True))
        self.observations = []

    def sample_actions(self, obs, noise=None):
        self.observations.append({key: value.clone() for key, value in obs.items() if torch.is_tensor(value)})
        return {"actions": torch.zeros(1, self.config.chunk_size, 1)}

    def sample_actions_rtc(self, obs, **kwargs):
        return self.sample_actions(obs)


def _make_interface(eval_mode="standard", execution_horizon=4):
    policy = _RecordingPolicy(execution_horizon)
    cfg = PolicyInterfaceConfig(
        policy=policy,
        device="cpu",
        eval_mode=eval_mode,
        inference_delay=1,
        beta=1,
        observation_mapping={
            "joint_positions": "observation.state",
            "camera_rgb": "observation.image.0",
        },
    )
    cfg.input_transforms = _CountingTransform()
    cfg.output_transforms = _CountingTransform()
    return PolicyInterface(cfg), policy


@pytest.mark.parametrize("eval_mode", ["standard", "rtc"])
@pytest.mark.parametrize("execution_horizon", [1, 4])
def test_rollout_history_uses_environment_frames(eval_mode, execution_horizon):
    env = _FrameEnvironment()
    interface, policy = _make_interface(eval_mode, execution_horizon)

    evaluate_policy(
        env,
        interface,
        num_episodes=1,
        max_steps=env.max_steps,
        eval_mode=eval_mode,
        inference_delay=1,
    )

    query_frames = list(range(0, env.max_steps, execution_horizon))
    assert len(policy.observations) == len(query_frames)
    assert interface.input_transforms.calls == len(query_frames)
    assert interface.output_transforms.calls == len(query_frames)
    for frame, obs in zip(query_frames, policy.observations, strict=True):
        for key in ("observation.state", "observation.image.0"):
            history = obs[key].flatten(start_dim=2)[:, :, 0]
            assert history.tolist() == [[max(0, frame - 1), frame]]
            assert obs[f"{key}_is_pad"].tolist() == [[frame == 0, False]]
    assert env.frame == env.max_steps


@pytest.mark.parametrize("eval_mode", ["standard", "rtc"])
def test_observe_only_records_history(eval_mode):
    interface, policy = _make_interface(eval_mode)
    interface.get_action_chunk(_observation(0))
    cached_actions = interface.prev_action_chunk
    obs = _observation(1)
    original_tensors = {key: value.clone() for key, value in obs.items() if torch.is_tensor(value)}

    interface.observe(obs)

    assert set(obs) == {"joint_positions", "camera_rgb", "task"}
    for key, value in original_tensors.items():
        torch.testing.assert_close(obs[key], value)
    assert interface.prev_action_chunk is cached_actions
    assert len(policy.observations) == 1
    assert interface.input_transforms.calls == 1
    assert interface.output_transforms.calls == 1

    interface.get_action_chunk(_observation(2))
    for key in ("observation.state", "observation.image.0"):
        assert policy.observations[-1][key].flatten(start_dim=2)[:, :, 0].tolist() == [[1, 2]]
        assert policy.observations[-1][f"{key}_is_pad"].tolist() == [[False, False]]

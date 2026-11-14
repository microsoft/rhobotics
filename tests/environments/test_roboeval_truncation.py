from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rho.environment.env import EvalMetrics, evaluate_policy


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_roboeval_propagates_episode_end_flags(roboeval_wrapper, monkeypatch, terminated, truncated):
    wrapper = roboeval_wrapper
    monkeypatch.setattr(wrapper, "_process_input", lambda obs: obs)
    wrapper.sim.step.return_value = ({}, 0.0, terminated, truncated, {"task_success": 0.0})

    _, reward, returned_terminated, returned_truncated, info = wrapper.step(np.zeros(16))

    np.testing.assert_array_equal(returned_terminated, [terminated])
    assert returned_truncated.dtype == np.bool_
    np.testing.assert_array_equal(returned_truncated, [truncated])
    np.testing.assert_array_equal(wrapper.done, [terminated or truncated])

    episode = EvalMetrics()
    episode.store(0, reward, returned_terminated, returned_truncated, info)
    assert episode.all_episodes_done() == (terminated or truncated)


def test_evaluation_stops_after_unhealthy_simulation(roboeval_wrapper, monkeypatch):
    monkeypatch.setattr(roboeval_wrapper, "_process_input", lambda obs: obs)
    sim = roboeval_wrapper.sim
    sim.reset.return_value = {}, {}
    sim.step.return_value = ({}, 0.0, False, True, {"task_success": 0.0})
    policy = Mock(execution_horizon=1)
    policy.get_action_chunk.return_value = torch.zeros(1, 1, 16)

    metrics = evaluate_policy(roboeval_wrapper, policy, num_episodes=1, max_steps=3)

    sim.step.assert_called_once()
    assert metrics["episode_steps"] == [1]
    assert metrics["episode_successes"] == [False]

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rho.environment.env import EvalMetrics, evaluate_policy


@pytest.mark.parametrize(
    ("terminated", "task_success"),
    [(False, 0.0), (True, 0.0), (True, 1.0)],
    ids=["running", "dropped-pot", "successful-lift"],
)
def test_roboeval_reports_task_success(roboeval_wrapper, monkeypatch, terminated, task_success):
    wrapper = roboeval_wrapper
    monkeypatch.setattr(wrapper, "_process_input", lambda obs: obs)
    wrapper.sim.step.return_value = ({}, 1.0, terminated, False, {"task_success": task_success})

    _, reward, done, truncated, info = wrapper.step(np.zeros(16))

    np.testing.assert_array_equal(done, [terminated])
    assert info["is_success"].dtype == np.bool_
    np.testing.assert_array_equal(info["is_success"], [bool(task_success)])

    episode = EvalMetrics()
    episode.store(0, reward, done, truncated, info)
    assert episode.current_avg_success == task_success


def test_evaluation_counts_a_dropped_pot_as_a_failure(roboeval_wrapper, monkeypatch):
    monkeypatch.setattr(roboeval_wrapper, "_process_input", lambda obs: obs)
    sim = roboeval_wrapper.sim
    sim.reset.return_value = {}, {}
    sim.step.side_effect = [
        ({}, 0.0, True, False, {"task_success": 0.0}),
        ({}, 1.0, True, False, {"task_success": 1.0}),
    ]
    policy = Mock(execution_horizon=1)
    policy.get_action_chunk.return_value = torch.zeros(1, 1, 16)

    metrics = evaluate_policy(roboeval_wrapper, policy, num_episodes=2, max_steps=3)

    assert metrics["episode_successes"] == [False, True]
    assert metrics["mean_success_rt"] == 0.5

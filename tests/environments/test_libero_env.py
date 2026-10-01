from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("libero")

from environments.libero.env import LiberoEnvWrapper, initial_state_indices


@pytest.mark.parametrize("n_envs", [5, 10])
def test_vectorized_initial_states_advance_between_episode_batches(n_envs):
    num_states = 50

    batches = [
        initial_state_indices(completed, n_envs, num_states) for completed in range(0, num_states, n_envs)
    ]

    assert np.concatenate(batches).tolist() == list(range(num_states))


@pytest.mark.parametrize("n_envs", [5, 10])
def test_vectorized_initial_states_respect_configured_start(n_envs):
    num_states = 50
    init_state_id = 20

    batches = [
        initial_state_indices(
            completed,
            n_envs,
            num_states,
            init_state_id=init_state_id,
        )
        for completed in range(0, num_states, n_envs)
    ]

    expected = list(range(init_state_id, num_states)) + list(range(init_state_id))
    assert np.concatenate(batches).tolist() == expected


def test_vectorized_initial_states_wrap_from_configured_start():
    indices = initial_state_indices(
        state_offset=5,
        n_envs=5,
        num_states=50,
        init_state_id=48,
    )

    assert indices.tolist() == [3, 4, 5, 6, 7]


def test_vectorized_reset_applies_init_state_id():
    wrapper = object.__new__(LiberoEnvWrapper)
    wrapper.config = SimpleNamespace(n_envs=3, init_state_id=4)
    wrapper.initial_states = [f"state-{i}" for i in range(10)]
    wrapper._episodes_completed_for_current_task = 2
    wrapper.num_wait_steps = 0
    wrapper.task_description = "test task"
    wrapper.env = MagicMock()
    wrapper.env.set_init_state.return_value = {"observation": "raw"}
    wrapper._process_input = lambda obs: obs

    obs, info = wrapper.reset()

    wrapper.env.set_init_state.assert_called_once_with(
        ["state-6", "state-7", "state-8"]
    )
    assert obs == {"observation": "raw"}
    assert info == {"task_description": "test task"}

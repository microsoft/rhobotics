import numpy as np
import pytest

pytest.importorskip("libero")

from environments.libero.env import initial_state_indices


@pytest.mark.parametrize("n_envs", [5, 10])
def test_vectorized_initial_states_advance_between_episode_batches(n_envs):
    num_states = 50

    batches = [
        initial_state_indices(completed, n_envs, num_states) for completed in range(0, num_states, n_envs)
    ]

    assert np.concatenate(batches).tolist() == list(range(num_states))

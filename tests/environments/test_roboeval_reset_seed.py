import numpy as np


def test_roboeval_replays_seeded_initial_state(roboeval_wrapper):
    def reset_sim(*, seed=None):
        raw_obs = {
            "proprioception": np.random.default_rng(seed).normal(size=18),
            "proprioception_grippers": np.zeros(2),
            "rgb_head": np.zeros((3, 2, 2), dtype=np.uint8),
        }
        return raw_obs, {}

    roboeval_wrapper.sim.reset.side_effect = reset_sim
    first, _ = roboeval_wrapper.next_episode(seed=0)
    other, _ = roboeval_wrapper.next_episode(seed=1)
    replay, _ = roboeval_wrapper.next_episode(seed=0)

    assert [call.kwargs["seed"] for call in roboeval_wrapper.sim.reset.call_args_list] == [0, 1, 0]
    np.testing.assert_array_equal(first["observation.state"], replay["observation.state"])
    assert not np.array_equal(first["observation.state"], other["observation.state"])

    roboeval_wrapper.reset()
    roboeval_wrapper.sim.reset.assert_called_with(seed=None)

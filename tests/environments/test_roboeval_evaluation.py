import numpy as np
import torch

from rho.environment.env import evaluate_policy


class SeededPolicy:
    execution_horizon = 1

    def reset(self):
        pass

    def get_action_chunk(self, obs):
        # Reuse the wrapper's 16-D joint state so observations reach the policy.
        return obs["observation.state"] + torch.rand_like(obs["observation.state"])


def test_roboeval_episode_scores_and_seeded_policy_replay(roboeval_wrapper):
    raw_obs = {
        "proprioception": np.zeros(18, dtype=np.float32),
        "proprioception_grippers": np.zeros(2, dtype=np.float32),
        "rgb_head": np.zeros((3, 2, 2), dtype=np.uint8),
    }
    sim = roboeval_wrapper.sim
    sim.reset.return_value = raw_obs, {}
    ongoing = (raw_obs, 0.0, False, False, {"task_success": 0.0})
    success = (raw_obs, 1.0, True, False, {"task_success": 1.0})

    def run(policy_seed):
        # A successful lift at step two, then an unsuccessful three-step rollout
        # stopped by the evaluator's horizon.
        sim.step.reset_mock()
        sim.step.side_effect = [ongoing, success, ongoing, ongoing, ongoing]
        result = evaluate_policy(
            roboeval_wrapper, SeededPolicy(), num_episodes=2, max_steps=3, seed=90, policy_seed=policy_seed
        )
        assert result["num_episodes"] == 2
        assert result["episode_successes"] == [True, False]
        assert result["episode_steps"] == [2, 3]
        assert result["episode_rewards"] == [1.0, 0.0]
        assert result["mean_success_rt"] == 0.5
        return np.stack([call.args[0] for call in sim.step.call_args_list])

    actions = run(policy_seed=42)
    assert actions.shape == (5, 16)
    np.testing.assert_array_equal(actions, run(policy_seed=42))
    assert not np.array_equal(actions, run(policy_seed=43))

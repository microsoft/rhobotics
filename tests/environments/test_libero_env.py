from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from draccus.parsers.config_parsers import YAMLParser

pytest.importorskip("libero")

from environments.libero.env import (
    LiberoEnvConfig,
    LiberoEnvWrapper,
    initial_state_indices,
    validate_episodes_per_task,
)
from rho.environment.env import evaluate_policy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIBERO_CONFIG_DIR = REPOSITORY_ROOT / "environments" / "libero" / "configs"


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


@pytest.mark.parametrize(
    "episodes_per_task,n_envs",
    [(50, 1), (50, 2), (50, 5), (50, 10), (50, 25), (50, 50), (1, 1), (2, 2), (10, 5)],
)
def test_episodes_per_task_accepts_multiples_of_n_envs(episodes_per_task, n_envs):
    validate_episodes_per_task(episodes_per_task, n_envs)


@pytest.mark.parametrize(
    "episodes_per_task,n_envs,valid_n_envs",
    [
        (50, 8, "1, 2, 5, 10, 25, 50"),
        # Previous smoke-test setting: one batch ran 10 episodes of the first task only.
        (1, 10, "1"),
        (2, 4, "1, 2"),
    ],
)
def test_episodes_per_task_rejects_non_multiples_of_n_envs(episodes_per_task, n_envs, valid_n_envs):
    with pytest.raises(ValueError, match="must be a multiple of n_envs") as exc_info:
        validate_episodes_per_task(episodes_per_task, n_envs)

    assert f"Valid n_envs values for episodes_per_task={episodes_per_task}: {valid_n_envs}." in str(
        exc_info.value
    )


@pytest.mark.parametrize("episodes_per_task,n_envs", [(0, 1), (-5, 1), (5, 0)])
def test_episodes_per_task_rejects_non_positive_values(episodes_per_task, n_envs):
    with pytest.raises(ValueError, match="must be at least 1"):
        validate_episodes_per_task(episodes_per_task, n_envs)


def test_libero_config_rejects_invalid_episode_batching_at_construction():
    with pytest.raises(ValueError, match=r"episodes_per_task \(50\) must be a multiple of n_envs \(8\)"):
        LiberoEnvConfig(episodes_per_task=50, n_envs=8, device="cpu")

    config = LiberoEnvConfig(episodes_per_task=50, n_envs=10, device="cpu")
    assert (config.episodes_per_task, config.n_envs) == (50, 10)


def _libero_environment_blocks(config):
    """Return the LIBERO environment sections of a single or multi-eval config."""
    if "eval_configs" in config:
        blocks = [entry.get("environment") or {} for entry in config["eval_configs"]]
    else:
        blocks = [config.get("environment") or {}]
    return [block for block in blocks if block.get("type") == "libero"]


@pytest.mark.parametrize(
    "config_name",
    sorted(path.name for path in LIBERO_CONFIG_DIR.glob("*.yaml")),
)
def test_shipped_libero_configs_use_valid_episode_batching(config_name):
    with (LIBERO_CONFIG_DIR / config_name).open() as config_file:
        config = YAMLParser.load_config(config_file)

    environments = _libero_environment_blocks(config)
    if "eval" in config_name:
        assert environments, f"{config_name} has no LIBERO environment section"

    for environment in environments:
        # Constructing the config runs the same validation as draccus decoding.
        LiberoEnvConfig(
            episodes_per_task=environment.get("episodes_per_task", LiberoEnvConfig.episodes_per_task),
            n_envs=environment.get("n_envs", LiberoEnvConfig.n_envs),
            device="cpu",
        )


def _make_stubbed_wrapper(n_envs, episodes_per_task, num_tasks=10, num_states=50):
    """Real LiberoEnvWrapper task cycling and reset logic over a fake simulator.

    Each episode reports its observation task as ``"t<task>/s<initial state>"``
    so tests can check exactly which (task, initial state) pairs were evaluated.
    """
    wrapper = object.__new__(LiberoEnvWrapper)
    wrapper.config = LiberoEnvConfig(n_envs=n_envs, episodes_per_task=episodes_per_task, device="cpu")
    wrapper.episodes_per_task = episodes_per_task
    wrapper.num_tasks_in_suite = num_tasks
    wrapper.initial_states = list(range(num_states))
    wrapper.num_wait_steps = 0
    wrapper.lane_states = []
    wrapper.current_task = 0

    def set_init_state(states):
        wrapper.lane_states = list(np.atleast_1d(states))

    wrapper.env = SimpleNamespace(seed=lambda seed: None, reset=lambda: None, set_init_state=set_init_state)

    def setup_libero_env(task_id):
        wrapper.current_task = task_id
        wrapper.task_description = f"task {task_id}"

    wrapper.setup_libero_env = setup_libero_env
    wrapper._cleanup_task_resources = lambda: None
    setup_libero_env(0)

    def observation():
        return {
            "observation.state": torch.zeros(n_envs, 1, 1),
            "task": [f"t{wrapper.current_task}/s{state}" for state in wrapper.lane_states],
        }

    wrapper._process_input = lambda raw_obs: observation()

    def step(action):
        done = np.ones(n_envs, dtype=bool)
        return observation(), np.zeros(n_envs), done, np.zeros(n_envs, dtype=bool), {"is_success": done}

    wrapper.step = step
    wrapper.render = lambda: None
    return wrapper


class _ZeroActionPolicy:
    execution_horizon = 1

    def __init__(self, n_envs):
        self.n_envs = n_envs

    def reset(self):
        pass

    def get_action_chunk(self, obs):
        return torch.zeros(self.n_envs, 1, 7)


@pytest.mark.parametrize(
    "num_episodes,n_envs,episodes_per_task",
    [
        (500, 10, 50),  # multieval_libero_rho_50.yaml
        (10, 1, 1),  # multieval_libero_rho_smoke.yaml
        (500, 5, 50),
    ],
)
def test_valid_episode_batching_evaluates_every_task_and_initial_state_once(
    num_episodes, n_envs, episodes_per_task
):
    wrapper = _make_stubbed_wrapper(n_envs, episodes_per_task)

    results = evaluate_policy(wrapper, _ZeroActionPolicy(n_envs), num_episodes=num_episodes, max_steps=1)

    labels = results["episode_tasks"]
    episodes_by_task = Counter(label.split("/")[0] for label in labels)
    assert results["num_episodes"] == num_episodes
    assert episodes_by_task == {f"t{task}": episodes_per_task for task in range(10)}
    assert len(set(labels)) == len(labels)


def test_next_episode_rejects_runtime_episodes_per_task_override():
    wrapper = _make_stubbed_wrapper(n_envs=10, episodes_per_task=50)
    wrapper.env = MagicMock()
    wrapper.episodes_per_task = 7

    with pytest.raises(ValueError, match=r"episodes_per_task \(7\) must be a multiple of n_envs \(10\)"):
        wrapper.next_episode()

    wrapper.env.set_init_state.assert_not_called()

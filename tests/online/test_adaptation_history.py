"""Online query history must match ordinary policy inference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from environments.metaworld.online_adapt import run_episode
from rho.eval.policy_interface import PolicyInterface
from rho.online.adaptation import AdaptationConfig, InterventionSchedule, OnlineAdapter
from tests.eval.test_policy_interface import _make_pi_config, _MockPolicy, _MockPolicyConfig

STATE = "observation.state"
IMAGE = "observation.image.0"


def _copy_obs(obs):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in obs.items()}


class _Transforms:
    def __call__(self, obs):
        obs = _copy_obs(obs)
        if "action" in obs:
            obs["action"] = (obs["action"] - obs[STATE][:, -1:, :]) / 2
        obs[STATE] = obs[STATE] / 10
        obs[IMAGE] = obs[IMAGE] / 100
        return obs

    def inverse(self, obs):
        obs["action"] = obs["action"] * 2 + obs[STATE][:, -1:, :] * 10
        return obs


class _NoisePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.head = SimpleNamespace(state_dim=2, head=SimpleNamespace(magnitude=3.0))
        self.seen_states = []

    def forward(self, embed, mask, state, shape):
        self.seen_states.append(state.detach().clone())
        return self.weight.expand(shape)


class _Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=2, max_action_dim=2)
        self.noise_policy = _NoisePolicy()
        self.seen_images = []

    def get_image_text_hidden_state(self, image, prompt, image_mask=None):
        self.seen_images.append(image.detach().clone())
        embed = image.flatten(2)
        return embed, torch.ones(embed.shape[:2], dtype=torch.bool)


class _Policy(_MockPolicy):
    def __init__(self, history_steps):
        super().__init__(
            _MockPolicyConfig(
                chunk_size=2,
                n_action_steps=2,
                delta_indices_dict={
                    STATE: list(range(1 - history_steps, 1)),
                    IMAGE: list(range(1 - history_steps, 1)),
                    "action": [0],
                },
            )
        )
        self.model = nn.Module()
        self.model.flow_model = _Flow()
        self.observations = []

    def consolidate_images(self, obs):
        return obs

    def prepare_image(self, obs):
        return obs[IMAGE], None

    def prepare_prompt(self, obs):
        return obs["task"]

    def prepare_state(self, obs):
        return obs[STATE]

    def prepare_action(self, obs):
        return obs["action"]

    def sample_actions(self, obs, noise=None):
        self.observations.append(_copy_obs(obs))
        return {"actions": noise.clone()}


def _observation(value):
    return {
        "state": torch.full((1, 1, 2), float(value)),
        "camera": torch.full((1, 1, 1, 1, 1), float(value)),
        "task": ["test"],
    }


@pytest.fixture
def make_adapter(monkeypatch):
    def make(history_steps=3):
        policy = _Policy(history_steps)
        transforms = _Transforms()
        interface = PolicyInterface(
            _make_pi_config(
                policy=policy,
                device="cpu",
                observation_mapping={"state": STATE, "camera": IMAGE},
                input_transforms=transforms,
                output_transforms=transforms.inverse,
            )
        )
        config = AdaptationConfig(query_freq=1, max_timesteps=2)
        adapter = OnlineAdapter(interface, policy, config, device="cpu")
        inversions = []

        def invert(flow, state, action, precomputed, **kwargs):
            inversions.append((state.clone(), action.clone()))
            return action.clone(), torch.zeros(state.shape[0])

        monkeypatch.setattr("rho.online.adaptation.perstep_fp_noise_map", invert)
        return adapter, policy, inversions

    return make


class _Env:
    action_dim = 2

    def reset(self, seed=None):
        self.start = seed
        self.step_index = 0
        self._last_raw_obs = None
        return _observation(seed), {}

    def step(self, action):
        self.step_index += 1
        return _observation(self.start + self.step_index), 0.0, False, False, {}


@pytest.mark.parametrize("history_steps", [1, 3])
@pytest.mark.parametrize("use_noise_policy", [False, True])
@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("query_freq", [1, 2])
def test_episode_ingests_each_query_once(
    make_adapter, history_steps, use_noise_policy, corrected, query_freq
):
    adapter, policy, inversions = make_adapter(history_steps)
    adapter.cfg.query_freq = query_freq
    adapter.cfg.max_timesteps = 2 * query_freq
    expert = SimpleNamespace(get_action=lambda obs: np.array([5.0, 5.0], dtype=np.float32))
    schedule = InterventionSchedule(force_immediate=True) if corrected else None
    samples, _, _ = run_episode(adapter, _Env(), expert, schedule, 1, adapter.cfg, use_noise_policy)

    current_values = [1, 1 + query_freq]
    expected = (
        [[0.1], [(1 + query_freq) / 10]]
        if history_steps == 1
        else [[0.1, 0.1, 0.1], [0.1, 0.1, (1 + query_freq) / 10]]
    )
    for actual, values in zip(policy.observations, expected, strict=True):
        torch.testing.assert_close(actual[STATE][0, :, 0], torch.tensor(values))
        torch.testing.assert_close(actual[IMAGE][0, :, 0, 0, 0], torch.tensor(values) / 10)
    expected_masks = [[False], [False]] if history_steps == 1 else [[True, True, False], [True, False, False]]
    assert [obs[f"{STATE}_is_pad"][0].tolist() for obs in policy.observations] == expected_masks
    assert len(adapter.pi.obs_queue[STATE]) == min(history_steps, 2)
    if use_noise_policy:
        for actual, values in zip(policy.model.flow_model.noise_policy.seen_states, expected, strict=True):
            torch.testing.assert_close(actual[0, :, 0], torch.tensor(values))
    if corrected:
        assert len(samples) == len(inversions) == 2
        for (state, target), values, current in zip(inversions, expected, current_values, strict=True):
            torch.testing.assert_close(state[0, :, 0], torch.tensor(values))
            torch.testing.assert_close(target, torch.full((1, 2, 2), (5.0 - current) / 2))
    else:
        assert not inversions


def test_episode_reset_clears_previous_query_history(make_adapter):
    adapter, policy, _ = make_adapter()
    for seed in (1, 7):
        run_episode(adapter, _Env(), None, None, seed, adapter.cfg, True)
    first = policy.observations[2]
    torch.testing.assert_close(first[STATE][0, :, 0], torch.full((3,), 0.7))
    assert first[f"{STATE}_is_pad"][0].tolist() == [True, True, False]


def _assert_snapshot_unchanged(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("multiframe", [False, True])
def test_query_helpers_do_not_advance_or_mutate_snapshot(make_adapter, multiframe):
    adapter, policy, _ = make_adapter()
    raw = _observation(1)
    if multiframe:
        raw["state"] = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
        raw["camera"] = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1, 1)
    query = adapter.prepare_observation(raw)
    expected_history = _copy_obs(query.history)
    expected_processed = _copy_obs(query.processed)
    length = len(adapter.pi.obs_queue[STATE])
    # Reusing the environment's buffers must not change a retained query.
    raw["state"].fill_(99)
    raw["camera"].fill_(99)
    raw["task"][0] = "changed"
    for _ in range(2):
        adapter.embed_for_buffer(query)
        noise = adapter.propose_noise(query)
        actions = adapter.act(query, noise)
        adapter.invert(query, actions)
    assert len(adapter.pi.obs_queue[STATE]) == length
    _assert_snapshot_unchanged(query.history, expected_history)
    _assert_snapshot_unchanged(query.processed, expected_processed)
    for observed in policy.observations:
        torch.testing.assert_close(observed[STATE], expected_processed[STATE])
    np.testing.assert_allclose(actions, 2.0 if multiframe else 1.0)


def test_later_query_does_not_change_inversion_conditioning(make_adapter):
    adapter, _, inversions = make_adapter()
    first = adapter.prepare_observation(_observation(1))
    second = adapter.prepare_observation(_observation(2))
    adapter.act(second, np.zeros((2, 2), dtype=np.float32))
    adapter.invert(first, np.full((2, 2), 5.0, dtype=np.float32))
    state, target = inversions[-1]
    torch.testing.assert_close(state, torch.full((1, 3, 2), 0.1))
    torch.testing.assert_close(target, torch.full((1, 2, 2), 2.0))
    assert [frame[0, 0, 0].item() for frame in adapter.pi.obs_queue[STATE]] == [1, 2]


def test_multiframe_queries_preserve_chronology(make_adapter):
    adapter, policy, _ = make_adapter()
    for values in ([1, 2], [3, 4]):
        raw = _observation(0)
        raw["state"] = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 1).expand(-1, -1, 2)
        raw["camera"] = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 1, 1, 1)
        query = adapter.prepare_observation(raw)
        adapter.embed_for_buffer(query)
        noise = adapter.propose_noise(query)
        adapter.act(query, noise)
    assert [frame[0, 0, 0].item() for frame in adapter.pi.obs_queue[STATE]] == [2, 3, 4]
    torch.testing.assert_close(policy.observations[-1][STATE][0, :, 0], torch.tensor([0.2, 0.3, 0.4]))
    assert policy.observations[0][f"{STATE}_is_pad"][0].tolist() == [True, False, False]
    assert policy.observations[1][f"{STATE}_is_pad"][0].tolist() == [False, False, False]


def test_equal_observations_from_distinct_queries_are_real_frames(make_adapter):
    adapter, _, _ = make_adapter()
    raw = _observation(1)
    for _ in range(2):
        query = adapter.prepare_observation(raw)
        adapter.propose_noise(query)
    assert len(adapter.pi.obs_queue[STATE]) == 2
    assert query.processed[f"{STATE}_is_pad"][0].tolist() == [True, False, False]


def test_prepared_action_matches_ordinary_policy_interface(make_adapter):
    adapter, _, _ = make_adapter()
    reference, _, _ = make_adapter()
    for value in (1, 2, 3):
        raw = _observation(value)
        query = adapter.prepare_observation(raw)
        noise = np.full((2, 2), 0.5, dtype=np.float32)
        actual = adapter.act(query, noise)
        expected = reference.pi.get_action_chunk(_copy_obs(raw), noise=noise).numpy()[0]
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("history_steps", [1, 3])
def test_noise_inversion_round_trip_uses_original_query(make_adapter, monkeypatch, history_steps):
    from rho.online.noise_inversion import perstep_fp_noise_map

    adapter, policy, _ = make_adapter(history_steps)
    flow = policy.model.flow_model
    flow.dtype = torch.float32
    flow.config.num_steps = 2

    # Constant-in-time velocity gives an exact discrete inverse while still
    # conditioning on both the query's image embedding and proprioception.
    def velocity(state, actions, time, precomputed):
        embed, _ = precomputed
        return (state[:, -1:, :] + embed.mean(dim=(1, 2), keepdim=True)).expand_as(actions)

    flow.velocity_eval = velocity
    flow.sample_actions_from_precomputed = lambda state, precomputed, noise, num_steps: (
        noise - velocity(state, noise, 0.0, precomputed)
    )

    def sample_actions(obs, noise=None):
        precomputed = flow.get_image_text_hidden_state(obs[IMAGE], obs["task"])
        return {"actions": flow.sample_actions_from_precomputed(obs[STATE], precomputed, noise, 2)}

    policy.sample_actions = sample_actions
    monkeypatch.setattr("rho.online.adaptation.perstep_fp_noise_map", perstep_fp_noise_map)
    first = adapter.prepare_observation(_observation(1))
    adapter.prepare_observation(_observation(2))
    executed = np.full((2, 2), 5.0, dtype=np.float32)
    noise, diagnostics = adapter.invert(first, executed)
    np.testing.assert_allclose(noise, 2.11, rtol=1e-6)
    assert diagnostics["rt"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(adapter.act(first, noise), executed, rtol=1e-6)

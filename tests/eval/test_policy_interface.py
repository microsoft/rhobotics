"""Tests for rho.eval.policy_interface module."""

import logging
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rho.eval.policy_interface import (
    PolicyInterface,
    PolicyInterfaceConfig,
    _move_transforms_to_device,
    validate_rtc_horizons,
)

# =============================================================================
# Helpers — lightweight mock policy that satisfies PolicyInterface's needs
# =============================================================================


@dataclass
class _MockPolicyConfig:
    """Minimal policy config that provides the attributes PolicyInterface reads."""

    chunk_size: int = 4
    n_action_steps: int = 4
    action_dim: int = 2
    device: str = "cpu"

    # delta_indices_dict tells PolicyInterface which observation history steps to keep.
    # A mapping of feature key -> list of ints (relative timestep offsets).
    delta_indices_dict: dict = field(
        default_factory=lambda: {
            "observation.state": [0],
            "observation.image.0": [0],
            "action": [0],
        }
    )

    # Satisfy PolicyConfig attribute access from PolicyInterface
    feature_dict: dict = field(default_factory=dict)

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        return [0]

    @property
    def reward_delta_indices(self):
        return None


class _MockPolicy(nn.Module):
    """Lightweight mock policy that implements the interface expected by PolicyInterface."""

    def __init__(self, config: _MockPolicyConfig | None = None):
        super().__init__()
        self.config = config or _MockPolicyConfig()
        self.device = self.config.device
        # Register a dummy parameter so .to() works correctly
        self._dummy = nn.Parameter(torch.zeros(1))

    def sample_actions(self, obs: dict[str, Tensor], noise=None) -> dict:
        """Return a dict with an 'actions' key containing a random action chunk."""
        batch_size = 1
        for v in obs.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                batch_size = v.shape[0]
                break
        actions = torch.randn(batch_size, self.config.chunk_size, self.config.action_dim, device=self.device)
        return {"actions": actions}

    def reset(self):
        pass


def _identity_transform(obs):
    """Pass-through transform for testing."""
    return obs


def _make_pi_config(*, input_transforms=None, output_transforms=None, **kwargs):
    """Build a PolicyInterfaceConfig, setting transforms as attrs post-construction.

    ``input_transforms``, ``output_transforms`` and ``normalize_inputs`` are
    bare class-level attributes on PolicyInterfaceConfig (no type annotation),
    so they are **not** dataclass fields and cannot be passed as constructor
    kwargs.  This helper works around that.
    """
    cfg = PolicyInterfaceConfig(**kwargs)
    if input_transforms is not None:
        cfg.input_transforms = input_transforms
    if output_transforms is not None:
        cfg.output_transforms = output_transforms
    return cfg


# =============================================================================
# Tests for _move_transforms_to_device
# =============================================================================


class TestMoveTransformsToDevice:
    """Tests for the _move_transforms_to_device utility function."""

    def test_moves_nn_module_to_device(self):
        """An nn.Module should be moved via .to()."""
        module = nn.Linear(3, 3)
        result = _move_transforms_to_device(module, "cpu")
        assert result is module  # same object returned

    def test_moves_compose_children(self):
        """torchvision Compose — each child should be moved."""
        from torchvision.transforms import Compose

        children = [nn.Linear(3, 3), nn.Linear(3, 3)]
        composed = Compose(children)
        result = _move_transforms_to_device(composed, "cpu")
        # Children should still be present and moved
        assert len(result.transforms) == 2

    def test_handles_plain_callable(self):
        """A plain callable without .to() should be returned unchanged."""
        result = _move_transforms_to_device(_identity_transform, "cpu")
        assert result is _identity_transform

    def test_handles_transform_wrapper(self):
        """TransformWrapper wraps an inner transform."""
        from rho.datasets.data_config import TransformWrapper

        inner = nn.Linear(3, 3)
        wrapper = TransformWrapper(inner, "test_key")
        result = _move_transforms_to_device(wrapper, "cpu")
        assert result is wrapper
        # inner should have been moved
        assert result.transform is not None


# =============================================================================
# Tests for PolicyInterfaceConfig
# =============================================================================


class TestPolicyInterfaceConfig:
    """Tests for the PolicyInterfaceConfig dataclass."""

    def test_observation_mapping_is_populated(self):
        """Default construction should produce a non-empty observation_mapping."""
        cfg = PolicyInterfaceConfig(policy=_MockPolicy())
        assert isinstance(cfg.observation_mapping, dict)
        assert len(cfg.observation_mapping) > 0

    def test_post_init_adds_task_key(self):
        """__post_init__ should ensure 'task' is in observation_mapping values."""
        cfg = PolicyInterfaceConfig(
            policy=_MockPolicy(),
            observation_mapping={"observation.state": "observation.state"},
        )
        # task should have been added by __post_init__
        assert "task" in cfg.observation_mapping

    def test_post_init_adds_action_key(self):
        """__post_init__ should ensure 'action' is in observation_mapping values."""
        cfg = PolicyInterfaceConfig(
            policy=_MockPolicy(),
            observation_mapping={"observation.state": "observation.state"},
        )
        assert "action" in cfg.observation_mapping

    def test_post_init_validates_eval_mode(self):
        """Invalid eval_mode should raise AssertionError."""
        with pytest.raises(AssertionError, match="eval_mode must be"):
            PolicyInterfaceConfig(policy=_MockPolicy(), eval_mode="invalid")

    def test_rtc_eval_mode_accepted(self):
        """eval_mode='rtc' should be accepted."""
        cfg = PolicyInterfaceConfig(
            policy=_MockPolicy(),
            eval_mode="rtc",
            inference_delay=3,
            beta=0.5,
        )
        assert cfg.eval_mode == "rtc"

    def test_post_init_rejects_invalid_guidance_schedule(self):
        with pytest.raises(ValueError, match="guidance_schedule"):
            PolicyInterfaceConfig(policy=_MockPolicy(), guidance_schedule="invalid")


# =============================================================================
# Tests for PolicyInterface
# =============================================================================


@pytest.fixture
def mock_policy():
    """Create a mock policy for testing."""
    return _MockPolicy()


@pytest.fixture
def identity_transforms():
    """Provide identity transforms (no-op)."""
    return _identity_transform


@pytest.fixture
def policy_interface(mock_policy, identity_transforms):
    """Create a PolicyInterface with identity transforms (no data_config)."""
    cfg = _make_pi_config(
        policy=mock_policy,
        device="cpu",
        input_transforms=identity_transforms,
        output_transforms=identity_transforms,
    )
    return PolicyInterface(cfg)


class TestPolicyInterfaceInit:
    """Tests for PolicyInterface initialization."""

    def test_init_wires_config_values(self):
        """PolicyInterface should propagate non-default config values."""
        config = _MockPolicyConfig(chunk_size=7, n_action_steps=3)
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
        )
        pi = PolicyInterface(cfg)
        assert pi.policy is policy
        assert pi.device == "cpu"
        assert pi.horizon == 7
        assert pi.execution_horizon == 3

    def test_execution_horizon_override(self):
        """Serving config can override checkpoint n_action_steps for RTC cadence."""
        policy = _MockPolicy(config=_MockPolicyConfig(chunk_size=32, n_action_steps=32))
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
            eval_mode="rtc",
            inference_delay=9,
            execution_horizon=16,
            beta=15,
        )

        pi = PolicyInterface(cfg)

        assert pi.execution_horizon == 16
        assert policy.config.n_action_steps == 32

    def test_invalid_fallback_horizon_names_explicit_override(self):
        policy = _MockPolicy(config=_MockPolicyConfig(chunk_size=4, n_action_steps=4))
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
            eval_mode="rtc",
            inference_delay=1,
            beta=0.5,
        )

        with pytest.raises(ValueError, match="execution_horizon was not set") as exc_info:
            PolicyInterface(cfg)

        assert "policy.config.n_action_steps (4)" in str(exc_info.value)
        assert "Set execution_horizon explicitly" in str(exc_info.value)

    def test_init_rejects_guidance_schedule_mutated_after_config_creation(self):
        policy = _MockPolicy()
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
        )
        cfg.guidance_schedule = "invalid"

        with pytest.raises(ValueError, match="guidance_schedule"):
            PolicyInterface(cfg)

    def test_obs_queue_created_for_multi_step_deltas(self, identity_transforms):
        """Obs queue should be created when delta_indices has >1 entry."""
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.state": [-2, -1, 0],  # 3-step history
                "observation.image.0": [0],
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)
        assert "observation.state" in pi.obs_queue
        assert pi.obs_queue["observation.state"].maxlen == 3  # max(abs([-2,-1,0])) + 1


class TestRemapObservation:
    """Tests for PolicyInterface.remap_observation."""

    def test_remap_keys_renames_input(self, mock_policy, identity_transforms):
        """Keys present in observation_mapping should be renamed to the mapped value."""
        cfg = _make_pi_config(
            policy=mock_policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
            observation_mapping={
                "env_state": "observation.state",
                "env_image": "observation.image.0",
            },
        )
        pi = PolicyInterface(cfg)
        state_tensor = torch.randn(1, 1, 2)
        image_tensor = torch.randn(1, 3, 64, 64)
        obs = {"env_state": state_tensor, "env_image": image_tensor}

        remapped = pi.remap_observation(obs)

        # Original keys should be gone, mapped keys should appear
        assert "env_state" not in remapped
        assert "env_image" not in remapped
        assert "observation.state" in remapped
        assert "observation.image.0" in remapped
        # Tensor identity should be preserved
        assert remapped["observation.state"] is state_tensor
        assert remapped["observation.image.0"] is image_tensor

    def test_unmapped_keys_survive_alongside_mapped(self, mock_policy, identity_transforms):
        """Keys not in observation_mapping should survive; mapped keys should be renamed."""
        cfg = _make_pi_config(
            policy=mock_policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
            observation_mapping={"env_pos": "observation.state"},
        )
        pi = PolicyInterface(cfg)
        obs = {
            "env_pos": torch.randn(1, 1, 2),
            "extra_sensor": torch.randn(1, 5),
        }
        remapped = pi.remap_observation(obs)
        assert "observation.state" in remapped
        assert "env_pos" not in remapped
        assert "extra_sensor" in remapped

    def test_custom_mapping(self, mock_policy, identity_transforms):
        """Custom observation_mapping should remap keys correctly."""
        cfg = _make_pi_config(
            policy=mock_policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
            observation_mapping={
                "agent_pos": "observation.state",
                "pixels": "observation.image.0",
            },
        )
        pi = PolicyInterface(cfg)
        obs = {
            "agent_pos": torch.randn(1, 1, 2),
            "pixels": torch.randn(1, 3, 64, 64),
        }
        remapped = pi.remap_observation(obs)
        assert "observation.state" in remapped
        assert "observation.image.0" in remapped
        assert "agent_pos" not in remapped


class TestProcessObsQueue:
    """Tests for PolicyInterface.process_obs_queue."""

    def test_single_step_passthrough(self, policy_interface):
        """With single-step delta_indices, obs values should pass through unchanged."""
        t = torch.randn(1, 1, 2)
        obs = {"observation.state": t}
        result = policy_interface.process_obs_queue(obs)
        assert result["observation.state"].shape == (1, 1, 2)
        assert torch.equal(result["observation.state"], t)

    def test_multi_step_accumulation(self, identity_transforms):
        """Multi-step delta_indices should accumulate obs history."""
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.state": [-1, 0],
                "observation.image.0": [0],
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)

        # First observation
        obs1 = {"observation.state": torch.ones(1, 1, 3)}
        result1 = pi.process_obs_queue(obs1)
        # With only 1 observation, the -1 index will pad with oldest (same obs)
        assert result1["observation.state"].shape == (1, 2, 3)

        # Second observation
        obs2 = {"observation.state": torch.ones(1, 1, 3) * 2}
        result2 = pi.process_obs_queue(obs2)
        assert result2["observation.state"].shape == (1, 2, 3)
        # The two timesteps should be different: [obs1, obs2]
        assert result2["observation.state"][0, 0, 0].item() == 1.0  # previous
        assert result2["observation.state"][0, 1, 0].item() == 2.0  # current

    def test_emits_is_pad_mask_matching_dataset(self, identity_transforms):
        """process_obs_queue should emit `{key}_is_pad` mirroring dataset getitem.

        At the start of an episode the earliest history steps are padded with the
        oldest available observation and must be flagged True; once enough history
        has accumulated the mask must be all False.
        """
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.state": [-1, 0],
                "observation.image.0": [0],
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)

        # First observation: the -1 step falls before the start -> padded.
        result1 = pi.process_obs_queue({"observation.state": torch.ones(1, 1, 3)})
        pad1 = result1["observation.state_is_pad"]
        assert pad1.shape == (1, 2)
        assert pad1.dtype == torch.bool
        assert pad1[0, 0].item() is True  # padded oldest step
        assert pad1[0, 1].item() is False  # current step

        # Second observation: full history available -> nothing padded.
        result2 = pi.process_obs_queue({"observation.state": torch.ones(1, 1, 3) * 2})
        pad2 = result2["observation.state_is_pad"]
        assert pad2.shape == (1, 2)
        assert pad2[0, 0].item() is False
        assert pad2[0, 1].item() is False

    def test_multi_step_image_without_temporal_axis(self, identity_transforms):
        """UR5-style images arrive as (B, C, H, W) with no temporal axis.

        process_obs_queue must add the temporal axis itself rather than treating
        the channel dimension as time, so a 2-step history yields
        (B, n_obs, C, H, W) matching the dataset getitem.
        """
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.image.0": [-1, 0],  # 2-step image history
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)

        # (batch, C, H, W) — no temporal axis.
        img1 = torch.ones(1, 3, 4, 4)
        result1 = pi.process_obs_queue({"observation.image.0": img1})
        # Channels must be preserved; a temporal axis of size 2 is added.
        assert result1["observation.image.0"].shape == (1, 2, 3, 4, 4)
        pad1 = result1["observation.image.0_is_pad"]
        assert pad1.shape == (1, 2)
        assert pad1[0, 0].item() is True  # padded oldest step
        assert pad1[0, 1].item() is False

        img2 = torch.ones(1, 3, 4, 4) * 2
        result2 = pi.process_obs_queue({"observation.image.0": img2})
        out = result2["observation.image.0"]
        assert out.shape == (1, 2, 3, 4, 4)
        # [previous, current] along the temporal axis.
        assert out[0, 0, 0, 0, 0].item() == 1.0
        assert out[0, 1, 0, 0, 0].item() == 2.0

    def test_multi_step_image_with_temporal_axis(self, identity_transforms):
        """TabletopSim-style images arrive as (B, seq_len, C, H, W).

        The existing temporal axis must be consumed as time (not re-added).
        """
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.image.0": [-1, 0],
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)

        # (batch, seq_len=1, C, H, W) — explicit temporal axis.
        img = torch.ones(1, 1, 3, 4, 4)
        result = pi.process_obs_queue({"observation.image.0": img})
        assert result["observation.image.0"].shape == (1, 2, 3, 4, 4)

    def test_multi_step_state_without_temporal_axis(self, identity_transforms):
        """UR5-style states arrive as (B, dim) with no temporal axis."""
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.state": [-1, 0],
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)

        result1 = pi.process_obs_queue({"observation.state": torch.ones(1, 3)})
        assert result1["observation.state"].shape == (1, 2, 3)
        result2 = pi.process_obs_queue({"observation.state": torch.ones(1, 3) * 2})
        out = result2["observation.state"]
        assert out.shape == (1, 2, 3)
        assert out[0, 0, 0].item() == 1.0
        assert out[0, 1, 0].item() == 2.0


class TestProcessObservation:
    """Tests for PolicyInterface.process_observation."""

    def test_applies_input_transforms(self):
        """process_observation should invoke the input transform on the obs dict."""
        transform_called_with = {}

        def tracking_transform(obs):
            transform_called_with.update(obs)
            return obs

        policy = _MockPolicy()
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=tracking_transform,
            output_transforms=_identity_transform,
        )
        pi = PolicyInterface(cfg)

        obs = {"observation.state": torch.randn(1, 1, 2)}
        pi.process_observation(obs)
        assert "observation.state" in transform_called_with

    def test_removes_action_in_standard_mode(self, policy_interface):
        """In standard eval_mode, 'action' should be removed but other keys preserved."""
        state = torch.randn(1, 1, 2)
        obs = {
            "observation.state": state,
            "action": torch.randn(1, 4, 2),
        }
        result = policy_interface.process_observation(obs)
        assert "action" not in result
        assert "observation.state" in result

    def test_rtc_mode_pads_and_trims_action(self, identity_transforms):
        """In RTC mode, action should be padded to chunk_size then trimmed back."""
        chunk_size = 8
        n_prev = 3
        action_dim = 2
        config = _MockPolicyConfig(chunk_size=chunk_size, n_action_steps=4, action_dim=action_dim)
        policy = _MockPolicy(config=config)
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()
        policy.model.action_expert.enable_gradient_checkpointing = True

        # Use a transform that records the shape it sees *before* trimming
        seen_shapes = []

        def spy_transform(obs):
            if "action" in obs:
                seen_shapes.append(obs["action"].shape)
            return obs

        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=spy_transform,
            output_transforms=_identity_transform,
            eval_mode="rtc",
            inference_delay=3,
            beta=0.5,
        )
        pi = PolicyInterface(cfg)

        prev_actions = torch.randn(1, n_prev, action_dim)
        obs = {
            "observation.state": torch.randn(1, 1, 2),
            "action": prev_actions,
        }
        result = pi.process_observation(obs)

        assert "action" in result
        # The transform should have seen the padded shape (1, chunk_size, action_dim)
        assert len(seen_shapes) == 1
        assert seen_shapes[0] == (1, chunk_size, action_dim)
        # After trimming, output shape should match the original n_prev
        assert result["action"].shape == (1, n_prev, action_dim)


class TestProcessAction:
    """Tests for PolicyInterface.process_action."""

    def test_identity_output_transform(self, policy_interface):
        """With identity output transform, action should pass through."""
        obs = {"observation.state": torch.randn(1, 1, 2)}
        action = torch.randn(1, 4, 2)
        result = policy_interface.process_action(obs, action)
        assert torch.equal(result, action)


class TestGetActionChunk:
    """Tests for PolicyInterface.get_action_chunk — the main inference entry point."""

    def test_returns_action_tensor(self, policy_interface):
        """get_action_chunk should return a tensor of actions."""
        obs = {
            "observation.state": torch.randn(1, 1, 2),
            "observation.image.0": torch.randn(1, 3, 64, 64),
        }
        action = policy_interface.get_action_chunk(obs)
        assert isinstance(action, torch.Tensor)
        # Shape: (batch, chunk_size, action_dim)
        assert action.shape == (1, 4, 2)

    def test_output_on_policy_device(self, policy_interface):
        """Action output should reside on the same device as the policy."""
        obs = {"observation.state": torch.randn(1, 1, 2)}
        action = policy_interface.get_action_chunk(obs)
        assert action.device.type == policy_interface.device

    def test_non_tensor_obs_values_pass_through(self, policy_interface):
        """Non-tensor values (e.g., task strings) should not cause errors."""
        obs = {
            "observation.state": torch.randn(1, 1, 2),
            "task": ["pick up the cup"],
        }
        action = policy_interface.get_action_chunk(obs)
        assert isinstance(action, torch.Tensor)

    def test_rtc_forwards_explicit_noise(self):
        """RTC should forward caller-supplied noise to the policy."""
        config = _MockPolicyConfig(chunk_size=4, n_action_steps=2, action_dim=2)
        policy = _MockPolicy(config=config)
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()
        policy.sample_actions_rtc = MagicMock(return_value={"actions": torch.zeros(1, 4, 2)})
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
            eval_mode="rtc",
            inference_delay=1,
            beta=0.5,
        )
        pi = PolicyInterface(cfg)
        noise = torch.full((1, 4, 2), 7.0)

        pi.get_action_chunk({"observation.state": torch.zeros(1, 1, 2)}, noise=noise)

        seen_noise = policy.sample_actions_rtc.call_args.kwargs["noise"]
        torch.testing.assert_close(seen_noise, noise)

    @pytest.mark.parametrize(
        "num_actions_executed",
        [
            -1,
            5,
            1.5,
            True,
            torch.tensor([1, 2]),
            np.array([1, 2]),
        ],
    )
    def test_rtc_rejects_invalid_num_actions_executed(self, num_actions_executed):
        config = _MockPolicyConfig(chunk_size=4, n_action_steps=2, action_dim=2)
        policy = _MockPolicy(config=config)
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()
        policy.sample_actions_rtc = MagicMock(return_value={"actions": torch.zeros(1, 4, 2)})
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
            eval_mode="rtc",
            inference_delay=1,
            beta=0.5,
        )
        pi = PolicyInterface(cfg)
        pi.get_action_chunk({"observation.state": torch.zeros(1, 1, 2)})

        with pytest.raises(ValueError, match="num_actions_executed"):
            pi.get_action_chunk(
                {
                    "observation.state": torch.zeros(1, 1, 2),
                    "num_actions_executed": num_actions_executed,
                }
            )

    def test_rtc_accepts_full_cached_chunk_as_executed(self):
        config = _MockPolicyConfig(chunk_size=4, n_action_steps=2, action_dim=2)
        policy = _MockPolicy(config=config)
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()
        policy.sample_actions_rtc = MagicMock(return_value={"actions": torch.zeros(1, 4, 2)})
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
            eval_mode="rtc",
            inference_delay=1,
            beta=0.5,
        )
        pi = PolicyInterface(cfg)
        pi.get_action_chunk({"observation.state": torch.zeros(1, 1, 2)})

        pi.get_action_chunk(
            {
                "observation.state": torch.zeros(1, 1, 2),
                "num_actions_executed": 4,
            }
        )

        assert policy.sample_actions_rtc.call_args.kwargs["prev_actions"] is None

    def test_rtc_reframes_cached_absolute_actions(self):
        """RTC should transform cached absolute actions into the current policy frame."""
        config = _MockPolicyConfig(chunk_size=4, n_action_steps=2, action_dim=2)
        policy = _MockPolicy(config=config)
        policy.model = MagicMock()
        policy.model.action_expert = MagicMock()

        first_policy_chunk = torch.zeros(1, 4, 2)
        second_policy_chunk = torch.ones(1, 4, 2)
        policy.sample_actions_rtc = MagicMock(
            side_effect=[
                {"actions": first_policy_chunk},
                {"actions": second_policy_chunk},
            ]
        )

        def to_current_policy_frame(obs):
            if "action" in obs:
                timestep = torch.arange(obs["action"].shape[-2]).view(1, -1, 1)
                obs["action"] = obs["action"] - obs["observation.state"][:, -1:, :] + timestep
            return obs

        def to_absolute_actions(obs):
            timestep = torch.arange(obs["action"].shape[-2]).view(1, -1, 1)
            obs["action"] = obs["action"] + obs["observation.state"][:, -1:, :] - timestep
            return obs

        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=to_current_policy_frame,
            output_transforms=to_absolute_actions,
            eval_mode="rtc",
            inference_delay=1,
            beta=5.0,
        )
        pi = PolicyInterface(cfg)

        first_action = pi.get_action_chunk({"observation.state": torch.full((1, 1, 2), 10.0)})
        expected_first = torch.tensor([10.0, 9.0, 8.0, 7.0]).view(1, 4, 1).expand(-1, -1, 2)
        torch.testing.assert_close(first_action, expected_first)
        torch.testing.assert_close(pi.prev_action_chunk, first_action)

        second_action = pi.get_action_chunk(
            {
                "observation.state": torch.full((1, 1, 2), 12.0),
                "num_actions_executed": 2,
            }
        )

        rtc_prev_actions = policy.sample_actions_rtc.call_args.kwargs["prev_actions"]
        torch.testing.assert_close(rtc_prev_actions, torch.full((1, 2, 2), -4.0))
        expected_second = torch.tensor([13.0, 12.0, 11.0, 10.0]).view(1, 4, 1).expand(-1, -1, 2)
        torch.testing.assert_close(second_action, expected_second)
        torch.testing.assert_close(pi.prev_action_chunk, second_action)

    def test_reset_flag_clears_history_buffers(self, identity_transforms):
        """When obs['_reset_'] is true, internal history buffers should be cleared first."""
        config = _MockPolicyConfig(
            delta_indices_dict={
                "observation.state": [-1, 0],
                "observation.image.0": [0],
                "action": [0],
            }
        )
        policy = _MockPolicy(config=config)
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)

        # Build up queue with one call.
        first_obs = {"observation.state": torch.ones(1, 1, 2)}
        pi.get_action_chunk(first_obs)
        assert len(pi.obs_queue["observation.state"]) > 0

        # On reset=True, queue should be cleared before current obs is appended.
        reset_obs = {
            "observation.state": torch.ones(1, 1, 2) * 7,
            "_reset_": 1,
        }
        pi.get_action_chunk(reset_obs)

        queued = list(pi.obs_queue["observation.state"])
        assert len(queued) == 1
        assert torch.allclose(queued[0], torch.ones(1, 1, 2) * 7)

    def test_unknown_eval_mode_raises(self, mock_policy, identity_transforms):
        """An eval_mode not in ('standard', 'rtc') should raise ValueError."""
        cfg = _make_pi_config(
            policy=mock_policy,
            device="cpu",
            input_transforms=identity_transforms,
            output_transforms=identity_transforms,
        )
        pi = PolicyInterface(cfg)
        # Force an invalid eval_mode after construction
        pi.eval_mode = "unsupported"

        obs = {"observation.state": torch.randn(1, 1, 2)}
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            pi.get_action_chunk(obs)


# =============================================================================
# Integration test: PolicyInterface with DummyEnvironment
# =============================================================================


class TestPolicyInterfaceWithDummyEnvironment:
    """Integration tests using DummyEnvironment from rho.environment.env."""

    def test_evaluate_loop_with_dummy_env(self):
        """Run a simple eval loop: DummyEnv → PolicyInterface → step."""
        from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

        # Set up dummy environment
        env_config = DummyEnvironmentConfig(obs_dim=5, action_dim=2, max_steps=10)
        env = DummyEnvironment(config=env_config)

        # Set up mock policy matching env dimensions
        policy_config = _MockPolicyConfig(chunk_size=4, n_action_steps=4, action_dim=2)
        policy = _MockPolicy(config=policy_config)

        # Build PolicyInterface
        pi_cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
            observation_mapping={
                "observation.state": "observation.state",
                "observation.image.0": "observation.image.0",
                "task": "task",
                "action": "action",
            },
        )
        pi = PolicyInterface(pi_cfg)

        obs, info = env.reset(seed=0)
        total_reward = 0.0

        for _step in range(10):
            action_chunk = pi.get_action_chunk(obs)
            # Take first action from the chunk
            action = action_chunk[:, 0, :]  # (batch, action_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # Should have completed at least one step
        assert isinstance(total_reward, float)

    def test_multi_step_eval_loop(self):
        """Run multiple episodes to verify reset and continued inference."""
        from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

        env_config = DummyEnvironmentConfig(obs_dim=3, action_dim=2, max_steps=5)
        env = DummyEnvironment(config=env_config)

        policy = _MockPolicy(_MockPolicyConfig(chunk_size=2, n_action_steps=2, action_dim=2))

        pi_cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            input_transforms=_identity_transform,
            output_transforms=_identity_transform,
        )
        pi = PolicyInterface(pi_cfg)

        episodes_completed = 0
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            for _step in range(5):
                action_chunk = pi.get_action_chunk(obs)
                action = action_chunk[:, 0, :]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            episodes_completed += 1

        assert episodes_completed == 3


# =============================================================================
# Tests for PolicyInterface with data_config (transforms initialization)
# =============================================================================


class TestPolicyInterfaceWithDataConfig:
    """Tests for PolicyInterface when data_config is provided."""

    def test_data_config_initializes_transforms(self):
        """When data_config is provided, transforms should be initialized from it."""
        mock_data_config = MagicMock()
        mock_data_config.get_transforms.return_value = _identity_transform
        mock_data_config.get_action_denormalization.return_value = _identity_transform
        mock_data_config.transform_mapping = None

        policy = _MockPolicy()
        cfg = PolicyInterfaceConfig(
            policy=policy,
            device="cpu",
            data_config=mock_data_config,
        )
        pi = PolicyInterface(cfg)

        # Transforms should have been set from data_config
        assert pi.input_transforms is not None
        assert pi.output_transforms is not None
        mock_data_config.get_transforms.assert_called_once_with(remap=False, training=False)
        mock_data_config.get_action_denormalization.assert_called_once()

    def test_explicit_transforms_override_data_config(self):
        """When transforms are explicitly provided, data_config should not override them."""
        mock_data_config = MagicMock()
        mock_data_config.transform_mapping = None

        custom_transform = lambda x: x  # noqa: E731
        policy = _MockPolicy()
        cfg = _make_pi_config(
            policy=policy,
            device="cpu",
            data_config=mock_data_config,
            input_transforms=custom_transform,
            output_transforms=custom_transform,
        )
        pi = PolicyInterface(cfg)

        assert pi.input_transforms is custom_transform
        # output_transforms should also stay custom since it was explicitly provided
        assert pi.output_transforms is custom_transform
        mock_data_config.get_transforms.assert_not_called()


# =============================================================================
# Tests for validate_rtc_horizons
# =============================================================================


class TestValidateRtcHorizons:
    """Tests for the validate_rtc_horizons pure function."""

    def test_raises_when_sum_exceeds_chunk_size(self):
        """inference_delay + execution_horizon > chunk_size must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_rtc_horizons(inference_delay=6, execution_horizon=50, chunk_size=50)
        msg = str(exc_info.value)
        assert "6" in msg
        assert "50" in msg

    def test_valid_config_does_not_raise(self):
        """inference_delay + execution_horizon <= chunk_size should not raise."""
        validate_rtc_horizons(inference_delay=6, execution_horizon=8, chunk_size=32)

    def test_exact_sum_warns_degenerate_mask(self, caplog):
        """When sum == chunk_size and inference_delay >= chunk_size - execution_horizon, warn."""
        with caplog.at_level(logging.WARNING, logger="rho.eval.policy_interface"):
            validate_rtc_horizons(inference_delay=6, execution_horizon=26, chunk_size=32)
        assert any("hard-masking" in r.message or "hard masking" in r.message for r in caplog.records)

    def test_exact_sum_does_not_raise(self):
        """inference_delay + execution_horizon == chunk_size is the boundary and must not raise."""
        validate_rtc_horizons(inference_delay=6, execution_horizon=26, chunk_size=32)

    def test_zero_delay_full_horizon_warns_no_latency_benefit(self, caplog):
        """execution_horizon >= chunk_size triggers the 'no latency benefit' warning."""
        with caplog.at_level(logging.WARNING, logger="rho.eval.policy_interface"):
            validate_rtc_horizons(inference_delay=0, execution_horizon=50, chunk_size=50)
        assert any("no latency benefit" in r.message for r in caplog.records)

    def test_zero_delay_full_horizon_does_not_raise(self):
        """inference_delay=0, execution_horizon=chunk_size is non-fatal."""
        validate_rtc_horizons(inference_delay=0, execution_horizon=50, chunk_size=50)

    def test_healthy_config_no_warnings(self, caplog):
        """A properly configured RTC run should produce no warnings."""
        with caplog.at_level(logging.WARNING, logger="rho.eval.policy_interface"):
            validate_rtc_horizons(inference_delay=4, execution_horizon=8, chunk_size=32)
        assert caplog.records == []

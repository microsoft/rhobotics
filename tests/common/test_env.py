from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from rho.environment import ENV_REGISTRY, make_environment, register_environment
from rho.environment.env import EnvironmentConfig, GymEnvironment
from tests.utils import require_package


@require_package("gym_pusht")
def test_gym_environment_reset_and_step_types(pusht_env_config):
    """Test that GymEnvironment step returns correct types and shapes.

    Observations are now returned as torch tensors with shape:
        (batch, seq_len, **feature_size)
    Actions are expected as torch tensors.
    """
    env = GymEnvironment(config=pusht_env_config)

    obs, info = env.reset()

    # Validate return type matches ResetReturn
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert isinstance(info, dict), "Info should be a dictionary"

    # Observations are torch tensors with shape (batch, seq_len, **feature_size)
    # PushT with pixels_agent_pos observation mapping produces:
    #   "observation.state" (from "agent_pos") and "observation.image.0" (from "pixels")
    if "observation.image.0" in obs:
        img = obs["observation.image.0"]
        assert isinstance(img, torch.Tensor), "Image observation should be a torch.Tensor"
        assert img.ndim == 5, (
            f"Image should be 5D (batch, seq_len, C, H, W), got {img.ndim}D with shape {img.shape}"
        )
        assert img.shape[0] == 1, f"Batch dim should be 1 for single env, got {img.shape[0]}"
        assert img.shape[1] == 1, f"Seq dim should be 1, got {img.shape[1]}"

    if "observation.state" in obs:
        state = obs["observation.state"]
        assert isinstance(state, torch.Tensor), "State observation should be a torch.Tensor"
        assert state.ndim == 3, (
            f"State should be 3D (batch, seq_len, dim), got {state.ndim}D with shape {state.shape}"
        )
        assert state.shape[0] == 1, f"Batch dim should be 1 for single env, got {state.shape[0]}"
        assert state.shape[1] == 1, f"Seq dim should be 1, got {state.shape[1]}"

    # Verify task description is present
    assert "task" in obs, "Observations should contain 'task' key"
    assert isinstance(obs["task"], list), "Task should be a list of strings"

    # Create a valid action for pusht as a torch tensor (batch, action_dim)
    action = torch.tensor([[0.1, -0.1]], dtype=torch.float32)

    # Test step return type
    result = env.step(action)
    assert isinstance(result, tuple), "Step should return a tuple"
    assert len(result) == 5, "Step should return (obs, reward, terminated, truncated, info)"

    next_obs, reward, terminated, truncated, info = result

    assert isinstance(next_obs, dict), "Next observation should be a dictionary"
    assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
    assert isinstance(terminated, (bool, np.bool_)), "Terminated should be boolean"
    assert isinstance(truncated, (bool, np.bool_)), "Truncated should be boolean"
    assert isinstance(info, dict), "Info should be a dictionary"

    # Validate observation shapes after step
    if "observation.image.0" in next_obs:
        img = next_obs["observation.image.0"]
        assert isinstance(img, torch.Tensor), "Image observation should be a torch.Tensor"
        assert img.ndim == 5, f"Image should be 5D after step, got {img.ndim}D"
        assert img.shape[0] == 1, "Batch dim should be 1"
        assert img.shape[1] == 1, "Seq dim should be 1"

    if "observation.state" in next_obs:
        state = next_obs["observation.state"]
        assert isinstance(state, torch.Tensor), "State observation should be a torch.Tensor"
        assert state.ndim == 3, f"State should be 3D after step, got {state.ndim}D"
        assert state.shape[0] == 1, "Batch dim should be 1"
        assert state.shape[1] == 1, "Seq dim should be 1"

    env.close()


@require_package("gym_pusht")
def test_gym_environment_vectorized_types(pusht_vectorized_env_config):
    """Test that vectorized GymEnvironment returns correct types and shapes.

    For vectorized envs, observations have batch dimension = n_envs.
    All observations are torch tensors with shape (n_envs, seq_len, **feature_size).
    """
    env = GymEnvironment(config=pusht_vectorized_env_config)

    assert env.is_vectorized
    assert env.num_envs == 4

    # Test reset
    obs, info = env.reset()

    # For vectorized envs, observations should have batch dimension = n_envs
    if "observation.image.0" in obs:
        img = obs["observation.image.0"]
        assert isinstance(img, torch.Tensor), "Image observation should be a torch.Tensor"
        assert img.shape[0] == 4, f"Vectorized image batch dim should be 4, got {img.shape[0]}"
        assert img.shape[1] == 1, f"Seq dim should be 1, got {img.shape[1]}"
        assert img.ndim == 5, f"Image should be 5D, got {img.ndim}D"

    if "observation.state" in obs:
        state = obs["observation.state"]
        assert isinstance(state, torch.Tensor), "State observation should be a torch.Tensor"
        assert state.shape[0] == 4, f"Vectorized state batch dim should be 4, got {state.shape[0]}"
        assert state.shape[1] == 1, f"Seq dim should be 1, got {state.shape[1]}"
        assert state.ndim == 3, f"State should be 3D, got {state.ndim}D"

    # Test step with vectorized action (torch tensor)
    action = torch.tensor([[0.1, -0.1], [0.0, 0.2], [-0.1, 0.1], [0.2, -0.2]], dtype=torch.float32)

    next_obs, reward, terminated, truncated, info = env.step(action)

    # Validate vectorized returns
    assert isinstance(reward, np.ndarray), "Vectorized reward should be numpy array"
    assert reward.shape == (4,), f"Reward shape should be (4,), got {reward.shape}"
    assert isinstance(terminated, np.ndarray), "Vectorized terminated should be numpy array"
    assert terminated.shape == (4,), f"Terminated shape should be (4,), got {terminated.shape}"
    assert isinstance(truncated, np.ndarray), "Vectorized truncated should be numpy array"
    assert truncated.shape == (4,), f"Truncated shape should be (4,), got {truncated.shape}"

    # Validate vectorized observation shapes
    if "observation.image.0" in next_obs:
        img = next_obs["observation.image.0"]
        assert img.shape[0] == 4, "Image batch should match n_envs after step"

    if "observation.state" in next_obs:
        state = next_obs["observation.state"]
        assert state.shape[0] == 4, "State batch should match n_envs after step"

    env.close()


@require_package("gym_pusht")
def test_gym_environment_render(pusht_env_config):
    """Test GymEnvironment render functionality"""
    env = GymEnvironment(config=pusht_env_config)

    obs, info = env.reset()

    frame = env.render()
    if frame is not None:
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # Should be RGB image

    env.close()


@require_package("gym_pusht")
def test_gym_environment_close(pusht_env_config):
    """Test GymEnvironment close functionality"""
    env = GymEnvironment(config=pusht_env_config)

    # Should not raise an error
    result = env.close()
    # close() should return None
    assert result is None


def test_register_environment_and_make():
    """Test the full register -> make_environment lifecycle."""

    @EnvironmentConfig.register_subclass("lifecycle_test_env")
    @dataclass
    class LifecycleTestConfig(EnvironmentConfig):
        name: str = "lifecycle display name"

    @register_environment("lifecycle_test_env")
    class LifecycleTestEnv:
        def __init__(self, config):
            self.config = config

    # Decorator should bind the runtime class to the registered config type.
    assert "lifecycle_test_env" in ENV_REGISTRY
    assert ENV_REGISTRY["lifecycle_test_env"] == LifecycleTestEnv

    # make_environment should find and instantiate it
    config = LifecycleTestConfig()
    env = make_environment(config)
    assert isinstance(env, LifecycleTestEnv)
    assert env.config == config


def test_make_environment_dispatches_registered_type_not_display_name():
    @EnvironmentConfig.register_subclass("typed_lifecycle_test_env")
    @dataclass
    class TypedLifecycleConfig(EnvironmentConfig):
        name: str = "display-only"

    @register_environment("typed_lifecycle_test_env")
    class TypedLifecycleEnv:
        def __init__(self, config):
            self.config = config

    config = TypedLifecycleConfig()
    env = make_environment(config)

    assert config.type == "typed_lifecycle_test_env"
    assert isinstance(env, TypedLifecycleEnv)


def test_make_environment_with_none():
    """Test make_environment with None config"""
    result = make_environment(None)
    assert result is None


def test_make_environment_with_missing_name():
    """Test make_environment with config missing name"""
    config = EnvironmentConfig()
    config.name = None

    with pytest.raises(ValueError, match="has no registered type"):
        make_environment(config)


def test_make_environment_with_unregistered_env():
    """Test make_environment with unregistered environment name"""
    config = EnvironmentConfig()
    config.name = "nonexistent_environment"

    with pytest.raises(ValueError, match="Environment type 'nonexistent_environment' not found in registry"):
        make_environment(config)


# =============================================================================
# DummyEnvironment tests
# =============================================================================


def test_dummy_environment_reset():
    """Test DummyEnvironment reset returns tensors with correct shapes."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig(obs_dim=10, action_dim=4, n_envs=1)
    env = DummyEnvironment(config=config)

    obs, info = env.reset(seed=42)

    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert isinstance(info, dict), "Info should be a dictionary"

    # Observations should be torch tensors with (batch, seq_len, ...) shape
    assert "observation.state" in obs
    state = obs["observation.state"]
    assert isinstance(state, torch.Tensor)
    assert state.ndim == 3, f"State should be 3D (batch, seq, dim), got {state.ndim}D"
    assert state.shape[-1] == 10, f"State dim should be 10, got {state.shape[-1]}"


def test_dummy_environment_step():
    """Test DummyEnvironment step returns correct types."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig(obs_dim=10, action_dim=4, n_envs=1)
    env = DummyEnvironment(config=config)

    env.reset(seed=42)

    action = torch.zeros(4, dtype=torch.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)
    assert "is_success" in info


def test_dummy_environment_render():
    """Test DummyEnvironment render returns an RGB image."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig(image_size=(3, 64, 64))
    env = DummyEnvironment(config=config)

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (64, 64, 3), f"Expected (64, 64, 3), got {frame.shape}"
    assert frame.dtype == np.uint8


# =============================================================================
# EvalMetrics tests
# =============================================================================


def test_eval_metrics_store_and_summary():
    """Test EvalMetrics store and summary computation."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=1)

    # Simulate a few steps
    metrics.store(step=0, reward=1.0, terminated=False, truncated=False, info={})
    metrics.store(step=1, reward=2.0, terminated=False, truncated=False, info={})
    metrics.store(step=2, reward=3.0, terminated=True, truncated=False, info={"is_success": True})

    # Append batch results
    aggregate = EvalMetrics(num_envs=1)
    aggregate.append(metrics)

    summary = aggregate.summary()
    assert summary["num_episodes"] == 1
    assert len(summary["episode_rewards"]) == 1
    assert summary["episode_rewards"][0] == 6.0  # 1 + 2 + 3
    assert summary["episode_successes"][0] is True


def test_eval_metrics_all_episodes_done():
    """Test EvalMetrics all_episodes_done check."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=2)

    assert not metrics.all_episodes_done()

    # Mark one env as done
    metrics.store(
        step=0,
        reward=1.0,
        terminated=np.array([True, False]),
        truncated=np.array([False, False]),
        info={},
    )
    assert not metrics.all_episodes_done()

    # Mark both as done
    metrics.store(
        step=1,
        reward=1.0,
        terminated=np.array([True, True]),
        truncated=np.array([False, False]),
        info={},
    )
    assert metrics.all_episodes_done()


def test_eval_metrics_record_video_creates_output_dir(tmp_path):
    """Test that EvalMetrics creates output directory when record_video=True."""
    from rho.environment.env import EvalMetrics

    output_dir = str(tmp_path / "video_output")
    metrics = EvalMetrics(num_envs=1, record_video=True, output_dir=output_dir)

    assert metrics.record_video is True
    assert Path(output_dir).exists()


def test_eval_metrics_store_with_tasks():
    """Test that EvalMetrics.store records task descriptions on step 0."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=2)

    tasks = ["pick up cube", "push block"]
    metrics.store(
        step=0,
        reward=1.0,
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        info={},
        tasks=tasks,
    )

    assert metrics._current_tasks == ["pick up cube", "push block"]

    # Tasks should NOT be overwritten on subsequent steps
    metrics.store(
        step=1,
        reward=1.0,
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        info={},
        tasks=["other"],
    )

    assert metrics._current_tasks == ["pick up cube", "push block"]


def test_eval_metrics_store_frame():
    """Test store_frame with ndarray and list inputs."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=1)

    # Store a raw ndarray frame
    frame_array = np.zeros((64, 64, 3), dtype=np.uint8)
    metrics.store_frame(frame_array)
    assert len(metrics._frames) == 1

    # Store a list of frames (vectorized env returns list) — should take first
    frame_list = [np.ones((64, 64, 3), dtype=np.uint8)]
    metrics.store_frame(frame_list)
    assert len(metrics._frames) == 2
    assert np.all(metrics._frames[1] == 1)

    # Store None — should be ignored
    metrics.store_frame(None)
    assert len(metrics._frames) == 2


def test_eval_metrics_current_averages():
    """Test current_avg_success and current_avg_reward properties."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=1)

    # Before any stores, averages should be 0
    assert metrics.current_avg_success == 0.0
    assert metrics.current_avg_reward == 0.0

    # After storing some data
    metrics.store(step=0, reward=4.0, terminated=False, truncated=False, info={"is_success": True})

    assert metrics.current_avg_reward == 4.0
    assert metrics.current_avg_success == 1.0


def test_eval_metrics_get_progress_info():
    """Test get_progress_info returns formatted progress strings."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=1)
    metrics.store(step=0, reward=2.0, terminated=False, truncated=False, info={})

    progress = metrics.get_progress_info(total_episodes=10)

    assert "episode" in progress
    assert "avg_success" in progress
    assert "avg_reward" in progress
    assert "step" in progress
    assert progress["episode"] == "0/10"


def test_eval_metrics_save_video(tmp_path):
    """Test save_video writes a video file from stored frames."""
    from rho.environment.env import EvalMetrics

    output_dir = str(tmp_path / "videos")
    metrics = EvalMetrics(num_envs=1, record_video=True, output_dir=output_dir)

    # Store some frames
    for _ in range(5):
        metrics.store_frame(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

    # Mark success for filename
    metrics.store(
        step=0, reward=1.0, terminated=True, truncated=False, info={"is_success": True}, tasks=["test task"]
    )

    video_path = metrics.save_video(episode=0)

    assert video_path is not None
    assert Path(video_path).exists()
    assert "test_task" in video_path
    assert "success" in video_path


def test_eval_metrics_save_video_no_frames():
    """Test save_video returns None when no frames are stored."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=1)
    result = metrics.save_video(episode=0)
    assert result is None


def test_eval_metrics_save_video_skips_duplicate(tmp_path):
    """Test save_video skips saving when a matching video already exists."""
    from rho.environment.env import EvalMetrics

    output_dir = str(tmp_path / "videos")
    metrics = EvalMetrics(num_envs=1, record_video=True, output_dir=output_dir)

    # Store frames and save first video
    for _ in range(3):
        metrics.store_frame(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    metrics.store(
        step=0, reward=1.0, terminated=True, truncated=False, info={"is_success": False}, tasks=["dup task"]
    )
    first_path = metrics.save_video(episode=0)

    # Reset frames and try saving again with same task/status
    metrics._frames = []
    for _ in range(3):
        metrics.store_frame(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    second_path = metrics.save_video(episode=1)

    # Should return the existing video path, not create a new one
    assert second_path == first_path


def test_eval_metrics_mean_properties_empty():
    """Test mean_reward, mean_steps, mean_success_rate with no episodes."""
    from rho.environment.env import EvalMetrics

    metrics = EvalMetrics(num_envs=1)
    assert metrics.mean_reward == 0.0
    assert metrics.mean_steps == 0.0
    assert metrics.mean_success_rate == 0.0


# =============================================================================
# DummyEnvironment additional coverage
# =============================================================================


def test_dummy_environment_close():
    """Test DummyEnvironment close is a no-op."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig()
    env = DummyEnvironment(config=config)
    result = env.close()
    assert result is None


def test_dummy_environment_next_episode():
    """Test DummyEnvironment next_episode delegates to reset."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig(obs_dim=5, action_dim=2)
    env = DummyEnvironment(config=config)

    obs, info = env.next_episode(seed=99)

    assert isinstance(obs, dict)
    assert "observation.state" in obs
    assert obs["observation.state"].shape[-1] == 5


def test_dummy_environment_process_output_batched():
    """Test _process_output removes batch dim when batch=1."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig(action_dim=4)
    env = DummyEnvironment(config=config)

    # (batch=1, action_dim) should become (action_dim,)
    action = torch.zeros((1, 4), dtype=torch.float32)
    result = env._process_output(action)
    assert result.ndim == 1
    assert result.shape == (4,)


def test_dummy_environment_process_input_string():
    """Test _process_input converts string values to lists."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig

    config = DummyEnvironmentConfig()
    env = DummyEnvironment(config=config)

    raw_obs = {"task": "pick up the cube", "observation.state": np.zeros(10, dtype=np.float32)}
    processed = env._process_input(raw_obs)

    assert processed["task"] == ["pick up the cube"]
    assert isinstance(processed["observation.state"], torch.Tensor)


# =============================================================================
# evaluate_policy integration test using DummyEnvironment
# =============================================================================


class MockPolicyInterface:
    """A mock PolicyInterface that returns random action chunks."""

    def __init__(self, action_dim=4, chunk_size=4, execution_horizon=None):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.execution_horizon = chunk_size if execution_horizon is None else execution_horizon

    def reset(self):
        pass

    def get_action_chunk(self, obs):
        # Determine batch size from observation
        batch_size = 1
        for _key, value in obs.items():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        # Return (batch, chunk_size, action_dim)
        return torch.randn(batch_size, self.chunk_size, self.action_dim)


@pytest.mark.parametrize("eval_mode,inference_delay", [("standard", 0), ("rtc", 2)])
@pytest.mark.parametrize("chunk_size,horizon,max_steps", [(16, 8, 20), (16, 8, 3), (4, 4, 12), (8, 1, 12)])
def test_evaluate_policy_execution_horizon(eval_mode, inference_delay, chunk_size, horizon, max_steps):
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig, evaluate_policy

    if eval_mode == "rtc" and horizon + inference_delay > chunk_size:
        inference_delay = 0

    class RecordingEnvironment(DummyEnvironment):
        def __init__(self):
            super().__init__(DummyEnvironmentConfig(action_dim=1, max_steps=max_steps))
            self.actions = []

        def step(self, action):
            self.actions.append(action.item())
            return super().step(action)

    class RecordingPolicyInterface(MockPolicyInterface):
        def __init__(self):
            super().__init__(action_dim=1, chunk_size=chunk_size, execution_horizon=horizon)
            self.inference_steps = []
            self.executed_counts = []

        def get_action_chunk(self, obs):
            self.inference_steps.append(len(env.actions))
            self.executed_counts.append(obs.get("num_actions_executed"))
            chunk_id = len(self.inference_steps) - 1
            return (100 * chunk_id + torch.arange(chunk_size)).reshape(1, chunk_size, 1)

    env = RecordingEnvironment()
    policy = RecordingPolicyInterface()
    evaluate_policy(
        env, policy, num_episodes=1, max_steps=max_steps, eval_mode=eval_mode, inference_delay=inference_delay
    )

    assert policy.inference_steps == list(range(0, max_steps, horizon))
    if eval_mode == "standard":
        assert env.actions == [100 * (step // horizon) + step % horizon for step in range(max_steps)]
        assert policy.executed_counts == [None] * len(policy.inference_steps)
    else:
        assert env.actions[: min(horizon, max_steps)] == list(range(min(horizon, max_steps)))
        assert policy.executed_counts == [0] + [horizon] * (len(policy.inference_steps) - 1)
        if max_steps > horizon + inference_delay:
            assert env.actions[horizon : horizon + inference_delay] == list(
                range(horizon, horizon + inference_delay)
            )
            assert env.actions[horizon + inference_delay] == 100 + inference_delay


def test_evaluate_policy_with_dummy_env():
    """Test the full evaluate_policy loop using DummyEnvironment."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig, evaluate_policy

    config = DummyEnvironmentConfig(obs_dim=10, action_dim=4, max_steps=15, n_envs=1)
    env = DummyEnvironment(config=config)
    policy = MockPolicyInterface(action_dim=4, chunk_size=4)

    results = evaluate_policy(
        env=env,
        policy_interface=policy,
        num_episodes=2,
        max_steps=15,
        seed=42,
        record_video=False,
    )

    assert isinstance(results, dict)

    required_keys = [
        "mean_reward",
        "mean_steps",
        "mean_success_rt",
        "episode_rewards",
        "episode_steps",
        "episode_successes",
        "num_episodes",
    ]
    for key in required_keys:
        assert key in results, f"Missing key: {key}"

    assert results["num_episodes"] == 2
    assert len(results["episode_rewards"]) == 2
    assert len(results["episode_steps"]) == 2
    assert len(results["episode_successes"]) == 2
    assert isinstance(results["mean_reward"], float)
    assert isinstance(results["mean_steps"], float)
    assert 0 <= results["mean_success_rt"] <= 1


def test_evaluate_policy_uses_independent_policy_seed():
    """Policy sampling can vary without changing the environment seed."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig, evaluate_policy

    class RecordingPolicyInterface(MockPolicyInterface):
        def __init__(self):
            super().__init__(action_dim=4, chunk_size=2)
            self.first_chunk = None

        def get_action_chunk(self, obs):
            chunk = super().get_action_chunk(obs)
            if self.first_chunk is None:
                self.first_chunk = chunk.clone()
            return chunk

    def first_chunk(policy_seed):
        env = DummyEnvironment(config=DummyEnvironmentConfig(obs_dim=10, action_dim=4, max_steps=2, n_envs=1))
        policy = RecordingPolicyInterface()
        evaluate_policy(
            env=env,
            policy_interface=policy,
            num_episodes=1,
            max_steps=2,
            seed=42,
            policy_seed=policy_seed,
            record_video=False,
        )
        return policy.first_chunk

    assert torch.equal(first_chunk(100), first_chunk(100))
    assert not torch.equal(first_chunk(100), first_chunk(101))


def test_evaluate_policy_with_video(tmp_path):
    """Test evaluate_policy with video recording enabled."""
    from rho.environment.env import DummyEnvironment, DummyEnvironmentConfig, evaluate_policy

    output_dir = str(tmp_path / "eval_videos")
    config = DummyEnvironmentConfig(obs_dim=10, action_dim=4, max_steps=10, n_envs=1)
    env = DummyEnvironment(config=config)
    policy = MockPolicyInterface(action_dim=4, chunk_size=2)

    results = evaluate_policy(
        env=env,
        policy_interface=policy,
        num_episodes=1,
        max_steps=10,
        seed=42,
        record_video=True,
        output_dir=output_dir,
    )

    assert results["num_episodes"] == 1
    # Video should have been saved
    if "video_path" in results:
        assert Path(results["video_path"]).exists()
        assert results["video_fps"] == 30

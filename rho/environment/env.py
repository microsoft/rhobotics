import logging
import re
from collections import deque
from dataclasses import dataclass, field  # noqa: I001
from pathlib import Path
from typing import Any

import draccus
import gymnasium as gym
import imageio
import numpy as np
import torch
from tqdm import tqdm

from rho.common.registry import get_registered_choice_type
from rho.eval.policy_interface import PolicyInterface

logger = logging.getLogger(__name__)

# Type aliases
StepReturn = tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
ResetReturn = tuple[dict[str, Any], dict[str, Any]]


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FILENAME_COLLAPSE_RE = re.compile(r"_+")


def sanitize_for_filename(text: str, max_length: int = 50) -> str:
    """Convert an arbitrary string into a filesystem-safe label."""
    if not text:
        return "task"
    label = _FILENAME_SAFE_RE.sub("_", text.strip())
    label = _FILENAME_COLLAPSE_RE.sub("_", label).strip("_.-")
    label = label[:max_length].rstrip("_.-")
    return label or "task"


@dataclass
class EvalMetrics:
    """
    Handles tracking of evaluation metrics during policy evaluation.

    This class manages episode-level and aggregate metrics for both
    single and vectorized environments. Extracted from the original
    EnvironmentWrapper.evaluate_policy method for modularity.

    Attributes:
        num_envs: Number of parallel environments
        episode_rewards: List of cumulative rewards per completed episode
        episode_steps: List of step counts per completed episode
        episode_successes: List of success flags per completed episode
    """

    num_envs: int = 1
    record_video: bool = False
    output_dir: str = "outputs/eval"
    fps: int = 30

    # Aggregate metrics across all episodes
    episode_rewards: list[float] = field(default_factory=list)
    episode_steps: list[float] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)
    episode_tasks: list[str] = field(default_factory=list)
    episode_subtask_progress: list[float] = field(default_factory=list)

    # Current episode tracking (per environment)
    _current_reward: np.ndarray = field(default=None, repr=False)
    _current_step_count: np.ndarray = field(default=None, repr=False)
    _current_successes: np.ndarray = field(default=None, repr=False)
    _current_done: np.ndarray = field(default=None, repr=False)
    _current_tasks: list[str] = field(default_factory=list, repr=False)
    _current_subtask_progress: np.ndarray = field(default=None, repr=False)

    # Video recording
    _frames: list = field(default_factory=list, repr=False)

    # Global counters
    global_step: int = 0
    completed_episodes: int = 0

    def __post_init__(self) -> None:
        """Initialize per-environment tracking arrays."""
        self.reset_episode()
        if self.record_video:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def reset_episode(self) -> None:
        """Reset tracking for a new episode batch."""
        self._current_reward = np.zeros(self.num_envs, dtype=np.float64)
        self._current_step_count = np.zeros(self.num_envs, dtype=np.float64)
        self._current_successes = np.zeros(self.num_envs, dtype=bool)
        self._current_done = np.zeros(self.num_envs, dtype=bool)
        self._current_tasks = [""] * self.num_envs
        self._current_subtask_progress = np.zeros(self.num_envs, dtype=np.float64)
        self._frames = []

    def store(
        self,
        step: int,
        reward: float | np.ndarray,
        terminated: bool | np.ndarray,
        truncated: bool | np.ndarray,
        info: dict[str, Any],
        tasks: list[str] | None = None,
    ) -> None:
        """
        Update metrics after an environment step.

        Args:
            step: Current step number in the episode
            reward: Reward received (scalar or array for vectorized envs)
            terminated: Whether episode terminated (scalar or array)
            truncated: Whether episode was truncated (scalar or array)
            info: Info dict from environment (may contain 'is_success')
            tasks: List of task descriptions for each environment
        """
        # Convert scalars to arrays for uniform handling
        reward = np.atleast_1d(reward)
        terminated = np.atleast_1d(terminated)
        truncated = np.atleast_1d(truncated)

        # Update cumulative reward (only for non-done environments)
        not_done_mask = 1 - self._current_done.astype(int)
        self._current_reward += reward * not_done_mask

        # Update done status
        done = terminated | truncated | self._current_done
        self._current_done = done

        # Update step count (only for non-done environments)
        self._current_step_count += not_done_mask

        # Update success tracking (terminated before max steps = success)
        # Also check info dict for explicit success flag
        is_success = info.get("is_success", terminated) if isinstance(info, dict) else terminated
        is_success = np.atleast_1d(is_success)
        self._current_successes = self._current_successes | is_success

        if info.get("subtask_progress") is not None:
            subtask_progress = np.atleast_1d(info["subtask_progress"])
            self._current_subtask_progress = np.maximum(self._current_subtask_progress, subtask_progress)

        # Store task descriptions (only on first step or when not set)
        if tasks is not None and step == 0:
            self._current_tasks = list(tasks)

        # Update global step counter
        self.global_step += self.num_envs

    def store_frame(self, frame: np.ndarray | list) -> None:
        """
        Store a frame for video recording.

        Args:
            frame: Rendered frame(s) from environment
        """
        if frame is not None:
            if isinstance(frame, list) and len(frame) > 0:
                self._frames.append(frame[0])
            else:
                self._frames.append(frame)

    def all_episodes_done(self) -> bool:
        """Check if all environments in the current batch are done."""
        return np.all(self._current_done)

    def append(self, batch_metrics: "EvalMetrics") -> None:
        """
        Append results from a batch evaluation to this metrics tracker.

        Args:
            batch_metrics: EvalMetrics from a single episode batch
        """
        self.episode_rewards.extend(batch_metrics._current_reward.tolist())
        self.episode_steps.extend(batch_metrics._current_step_count.tolist())
        self.episode_successes.extend(batch_metrics._current_successes.tolist())
        self.episode_tasks.extend(batch_metrics._current_tasks)
        self.episode_subtask_progress.extend(batch_metrics._current_subtask_progress.tolist())
        self.completed_episodes += batch_metrics.num_envs
        self.global_step += batch_metrics.global_step

    @property
    def current_avg_success(self) -> float:
        """Compute current average success rate including ongoing episode."""
        all_successes = self.episode_successes + self._current_successes.tolist()
        return np.mean(all_successes) if all_successes else 0.0

    @property
    def current_avg_reward(self) -> float:
        """Compute current average reward including ongoing episode."""
        all_rewards = self.episode_rewards + self._current_reward.tolist()
        return np.mean(all_rewards) if all_rewards else 0.0

    @property
    def mean_reward(self) -> float:
        """Compute mean reward across all completed episodes."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def mean_steps(self) -> float:
        """Compute mean steps across all completed episodes."""
        return float(np.mean(self.episode_steps)) if self.episode_steps else 0.0

    @property
    def mean_success_rate(self) -> float:
        """Compute mean success rate across all completed episodes."""
        return float(np.mean(self.episode_successes)) if self.episode_successes else 0.0

    def get_progress_info(self, total_episodes: int) -> dict[str, str]:
        """
        Get formatted progress information for display.

        Args:
            total_episodes: Total number of episodes to run

        Returns:
            Dictionary with formatted progress metrics
        """
        return {
            "episode": f"{self.completed_episodes}/{total_episodes}",
            "avg_success": f"{self.current_avg_success:.3f}",
            "avg_reward": f"{self.current_avg_reward:.2f}",
            "step": str(self.global_step),
        }

    def save_video(
        self,
        episode: int,
        task_name: str | None = None,
        success: bool | None = None,
    ) -> str | None:
        """
        Save recorded frames as a video file.

        Args:
            episode: Episode number for filename
            task_name: Optional task description for filename
            success: Whether the episode was successful (for filename)

        Returns:
            Path to saved video file, or None if no frames
        """
        if not self._frames:
            return None

        output_directory = Path(self.output_dir)

        if success is None:
            success = bool(self._current_successes[0]) if len(self._current_successes) > 0 else False

        success_status = "success" if success else "failure"

        # Use task from current tracking if not provided
        if task_name is None and self._current_tasks and self._current_tasks[0]:
            task_name = self._current_tasks[0]

        # Sanitize task name for filename (handles spaces, punctuation, unicode)
        task_label = sanitize_for_filename(task_name) if task_name else "task"

        # Format: {task}_{episode}_{result}.mp4
        video_path = output_directory / f"{task_label}_{success_status}.mp4"

        # Skip if a video with the same task and success status already exists
        existing = list(output_directory.glob(f"{task_label}_{success_status}.mp4"))
        if existing:
            logger.debug(
                f"Video for '{task_label}' with status '{success_status}' "
                f"already exists ({existing[0].name}), skipping save."
            )
            return str(existing[0])

        imageio.mimsave(str(video_path), np.stack(self._frames), fps=self.fps)
        logger.info(f"Saved {success_status} video: {video_path}")

        return str(video_path)

    def summary(self) -> dict[str, Any]:
        """
        Get final evaluation results summary.

        Returns:
            Dictionary containing all evaluation metrics
        """
        # Compute failure-only subtask progress
        failure_subtask_progress = [
            p for p, s in zip(self.episode_subtask_progress, self.episode_successes, strict=False) if not s
        ]

        return {
            "mean_reward": self.mean_reward,
            "mean_steps": self.mean_steps,
            "mean_success_rt": self.mean_success_rate,
            "episode_rewards": self.episode_rewards.copy(),
            "episode_steps": self.episode_steps.copy(),
            "episode_successes": self.episode_successes.copy(),
            "episode_tasks": self.episode_tasks.copy(),
            "episode_subtask_progress": self.episode_subtask_progress.copy(),
            "mean_subtask_progress": float(np.mean(self.episode_subtask_progress))
            if self.episode_subtask_progress
            else 0.0,
            "mean_failure_subtask_progress": float(np.mean(failure_subtask_progress))
            if failure_subtask_progress
            else 0.0,
            "num_episodes": self.completed_episodes,
        }


@dataclass
class EnvironmentConfig(draccus.ChoiceRegistry):
    """Configuration for environment interactions"""

    name: str = "UNDEFINED"  # Name of the environment

    @property
    def type(self) -> str:
        """Return the registered environment identity used for factory dispatch."""
        return get_registered_choice_type(self, legacy_name=self.name)


@dataclass
@EnvironmentConfig.register_subclass("GymEnvironment")
class GymEnvironmentConfig(EnvironmentConfig):
    """Configuration for OpenAI Gym environments."""

    name: str = "GymEnvironment"
    env_name: str = "gym_pusht/PushT-v0"  # Default environment name
    obs_type: str = "pixels_agent_pos"  # Observation type for gym env
    max_episode_steps: int = 300  # Maximum steps per episode
    n_envs: int = 1  # Number of parallel environments
    seed: int = 42  # Random seed

    # Observation mapping from env keys to policy keys
    observation_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "agent_pos": "observation.state",
            "pixels": "observation.image",
            "image": "observation.image",
        }
    )

    # Default task description for the environment
    default_task: str = "Complete the task"


class EnvironmentWrapper:
    """
    Wrapper for interactions between the policy and the environment(s).

    Supports both single and vectorized environments for improved training
    and evaluation efficiency. The wrapper handles the complexities of
    vectorized environments transparently.

    The key responsibilities of this wrapper are:
    - Managing environment step, reset, render, and close operations
    - Supporting both single (n_envs=1) and vectorized (n_envs>1) environments
    - Converting between environment format and policy tensor format internally
    - Providing a consistent interface regardless of vectorization

    All observations returned by step() and reset() are tensors with shape:
        (batch_size, seq_len, **feature_size)

    Actions passed to step() should be tensors with shape:
        (batch_size, action_dim) or (action_dim,)

    Attributes:
        env: The wrapped environment instance (single or vectorized)
        config: Configuration containing normalization and mapping settings
        is_vectorized: Boolean indicating if using vectorized environments
        num_envs: Property returning the number of parallel environments
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config

    def _process_input(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw environment observations to policy tensor format.

        Internal method called automatically by reset() and step().
        Converts numpy arrays to torch tensors with proper batch/seq dimensions.

        Args:
            raw_obs: Raw observation dict from environment with numpy arrays

        Returns:
            Dict with torch tensors of shape (batch, seq_len, **feature_size):
                - Image keys: (batch, 1, C, H, W), values in [0, 1]
                - State keys: (batch, 1, feature_dim)
                - Task: List of strings
        """
        raise NotImplementedError

    def _process_output(self, action: torch.Tensor) -> Any:
        """Convert policy action tensor to environment format.

        Internal method called automatically by step().
        Converts torch tensor to format expected by underlying environment.

        Args:
            action: Single action tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            Action in environment-expected format (typically numpy array)
        """
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> StepReturn:
        """Step the environment with the given action.

        Automatically converts action from tensor format and returns
        observations as tensors with shape (batch, seq_len, **feature_size).

        Args:
            action: Action tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            StepReturn tuple containing:
            - obs: Dict with torch tensors of shape (batch, seq_len, **feature_size)
            - reward: Float (single env) or numpy array of shape [n_envs] (vectorized)
            - terminated: Bool or numpy array indicating episode termination
            - truncated: Bool or numpy array indicating episode truncation
            - info: Dict with string keys and Any values
        """
        raise NotImplementedError("Step method not implemented for wrapper.")

    def reset(self, seed: int | None = None) -> ResetReturn:
        """Reset the environment with optional seed.

        Returns observations as tensors with shape (batch, seq_len, **feature_size).

        Returns:
            ResetReturn tuple containing:
            - obs: Dict with torch tensors of shape (batch, seq_len, **feature_size)
            - info: Dict with string keys and Any values
        """
        raise NotImplementedError("Reset method not implemented for wrapper.")

    def render(self) -> Any | list[Any]:
        """Render the environment(s).

        Returns:
            For single environment: Raw render output (numpy array, None, etc.)
            For vectorized environments: List of render outputs from each env
        """
        raise NotImplementedError("Render method not implemented for wrapper.")

    def _process_action(self, actions):
        """Convert a policy action (or action chunk) into a step-wise sequence.

        `evaluate_policy()` uses action chunking: the policy returns a horizon of actions,
        which we enqueue and execute one-by-one. This helper standardizes shapes and
        converts tensors to numpy.

        Important: do NOT unnormalize here. `step()` already applies `unnormalize_outputs`
        if configured.
        """
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().to("cpu").to(torch.float32).numpy()
        elif isinstance(actions, np.ndarray):
            actions_np = actions
        else:
            # Try a best-effort conversion (e.g., list -> np)
            actions_np = np.asarray(actions)

        # Expected shapes:
        # - single env: [H, A] (or [A])
        # - vectorized: [B, H, A] (or [B, A])
        if actions_np.ndim == 1:
            return [actions_np]

        if actions_np.ndim == 2:
            # Treat as a sequence: [H, A] or a single-step batch [B, A].
            # In our eval loop, non-vectorized envs use chunking, so [H, A] is typical.
            return [actions_np[t] for t in range(actions_np.shape[0])]

        if actions_np.ndim == 3:
            # [B, H, A] -> list length H with items shaped [B, A]
            return [actions_np[:, t, :] for t in range(actions_np.shape[1])]

        raise ValueError(f"Unsupported action shape for eval: {actions_np.shape}")

    def close(self) -> None:
        """Close the environment(s)."""
        raise NotImplementedError("Close method not implemented for wrapper.")

    def next_episode(self, seed: int | None = None) -> ResetReturn:
        """Prepare for the next episode and reset the environment.

        This method is called between episodes during evaluation. By default,
        it simply calls reset(). Subclasses can override this to perform
        additional setup between episodes, such as cycling through different
        tasks in a task suite.

        Args:
            seed: Optional random seed for the reset

        Returns:
            ResetReturn tuple containing:
            - obs: Dict with torch tensors of shape (batch, seq_len, **feature_size)
            - info: Dict with string keys and Any values
        """
        return self.reset(seed=seed)

    @property
    def num_envs(self) -> int:
        """Return the number of parallel environments."""
        return getattr(self, "n_envs", 1)

    @property
    def is_vectorized(self) -> bool:
        """Return True if using vectorized environments."""
        return getattr(self, "_is_vectorized", False)


def evaluate_policy(
    env: EnvironmentWrapper,
    policy_interface: PolicyInterface,
    num_episodes: int = 3,
    max_steps: int = 100,
    seed: int = 12345,
    policy_seed: int | None = None,
    record_video: bool = False,
    output_dir: str = "outputs/eval",
    eval_mode: str = "standard",
    inference_delay: int = 8,
    fps: int = 30,
) -> dict[str, Any]:
    """
    Evaluate the policy in the environment.

    All observations and actions are kept as tensors with shape:
        (batch_size, seq_len, **feature_size)

    The environment's step() and reset() methods handle conversion internally.
    The policy_interface.get_action_chunk() handles normalization/transforms.

    Args:
        env: The environment wrapper to evaluate in
        policy_interface: The policy interface for action generation
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Base seed for environment and scenario resets
        policy_seed: Base seed for policy sampling. Defaults to ``seed``.
        record_video: Whether to record videos during evaluation
        output_dir: Directory to save videos
        eval_mode: Evaluation mode ('standard' or 'rtc')
        inference_delay: Number of actions to execute before re-inference (RTC mode)
        fps: Frames per second for video recording

    Returns:
        dict: Evaluation metrics including mean_success_rt, mean_reward, mean_steps
    """
    # Calculate number of episode batches based on vectorization
    num_episodes_batch = int(np.ceil(num_episodes / env.num_envs)) if env.is_vectorized else num_episodes

    # Initialize aggregate metrics tracker
    eval_metrics = EvalMetrics(
        num_envs=env.num_envs,
        record_video=record_video,
        output_dir=output_dir,
        fps=fps,
    )

    # Calculate total expected steps for progress bar
    total_expected_steps = num_episodes * max_steps

    # Initialize progress bar tracking total steps
    progress_bar = tqdm(
        total=total_expected_steps,
        desc="Evaluating Policy",
        unit="step",
        ncols=150,
    )

    video_path = None
    total_steps_completed = 0

    for episode in range(num_episodes_batch):
        # Reset environment - returns obs as tensors (batch, seq_len, **feature_size)
        # Use next_episode() which allows environments to cycle through tasks
        obs, info = env.next_episode(seed=seed + episode)

        policy_interface.reset()
        # Initialize batch metrics for this episode
        batch_metrics = EvalMetrics(
            num_envs=env.num_envs,
            record_video=record_video,
            output_dir=output_dir,
            fps=fps,
        )

        # Reset policy state if needed
        if hasattr(policy_interface, "reset"):
            policy_interface.reset()

        episode_policy_seed = (seed if policy_seed is None else policy_seed) + episode
        torch.manual_seed(episode_policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_policy_seed)

        # Initialize action queue using deque (like Phi4MMPolicy.select_action)
        # Queue holds tensors of shape (batch, action_dim)
        action_queue: deque[torch.Tensor] = deque(maxlen=max_steps)

        # RTC: number of actions popped from the current chunk since the last
        # inference. Instead of sending the still-in-flight actions back to the
        # policy (which, after process_action denormalization + output-action-type
        # conversion, no longer live in the policy's trained action space), we send
        # this count. PolicyInterface caches the last predicted chunk in policy
        # space and indexes into it with num_actions_executed to recover the
        # remaining actions for RTC blending. This index-based recovery is exact
        # as long as inference_delay + execution_horizon <= chunk_size (true for
        # any sensible RTC config).
        #
        # Re-inference fires once this counter reaches execution_horizon, matching
        # the paper's s_min threshold (Algorithm 1, line 13). At that point the
        # queue still holds chunk_size - execution_horizon actions, so
        # len(prev_actions) == chunk_size - execution_horizon, which is exactly
        # what _prepare_rtc_mask uses to derive the soft-mask zero boundary.
        num_executed_since_inference = 0

        for step in range(max_steps):
            # ============================================================
            # Core loop - all tensors with shape (batch, seq_len, **)
            # ============================================================

            # For RTC mode, tell the policy how many actions of the previously
            # predicted chunk have already been executed and let it recover the
            # still-in-flight actions from its cached policy-space chunk.
            #
            # Trigger re-inference by counting executed actions, not by watching
            # the queue length. After execution_horizon actions have been consumed
            # the queue holds chunk_size - execution_horizon entries, so
            # len(prev_actions) inside _prepare_rtc_mask equals
            # chunk_size - execution_horizon. That is exactly the value the mask's
            # zero boundary is derived from (s_eff = H - len(prev_actions)), so
            # the soft-mask decay region aligns with the real action overlap.
            if (
                eval_mode == "rtc"
                and num_executed_since_inference >= policy_interface.execution_horizon
                and len(action_queue) > 0
            ):
                obs["num_actions_executed"] = num_executed_since_inference

                action_chunk = policy_interface.get_action_chunk(obs)

                # Emulate async execution: while the server was inferencing, the
                # first `inference_delay` still-in-flight actions keep executing on
                # the robot, so we retain them from the old chunk and then continue
                # from the freshly (RTC-blended) chunk at index `inference_delay`.
                # action_chunk[:, i] and the old queued action at position i both
                # correspond to the same future frame, so this stays aligned.
                old_in_flight = list(action_queue)[:inference_delay]
                action_queue.clear()
                action_queue.extend(old_in_flight)
                actions_transposed = action_chunk[:, inference_delay:].transpose(0, 1)
                action_queue.extend(actions_transposed)
                num_executed_since_inference = 0

            # Check if we need new actions from policy
            elif len(action_queue) == 0:
                # Get new action chunk from policy
                # Returns tensor of shape (batch, chunk_size, action_dim)
                if eval_mode == "rtc":
                    obs["num_actions_executed"] = num_executed_since_inference
                action_chunk = policy_interface.get_action_chunk(obs)

                # Populate queue: transpose to (chunk_size, batch, action_dim)
                # then extend queue with each timestep
                actions_transposed = action_chunk.transpose(0, 1)
                action_queue.extend(actions_transposed)
                num_executed_since_inference = 0

            # Pop next action from queue: (batch, action_dim)
            action = action_queue.popleft()
            num_executed_since_inference += 1

            # Step environment - handles _process_output internally
            # Returns obs as tensors (batch, seq_len, **feature_size)

            obs, reward, terminated, truncated, info = env.step(action)

            # Store metrics (pass task descriptions for tracking)
            tasks = obs.get("task", None)
            batch_metrics.store(step, reward, terminated, truncated, info, tasks)

            # Update progress bar with steps completed this iteration
            steps_this_iter = env.num_envs  # Each env takes a step
            total_steps_completed += steps_this_iter
            progress_bar.update(steps_this_iter)

            # Combine aggregate metrics with in-progress batch for display
            all_successes = eval_metrics.episode_successes + batch_metrics._current_successes.tolist()
            all_rewards = eval_metrics.episode_rewards + batch_metrics._current_reward.tolist()
            progress_bar.set_postfix(
                {
                    "episode": f"{eval_metrics.completed_episodes + env.num_envs}/{num_episodes}",
                    "avg_success": f"{np.mean(all_successes):.3f}" if all_successes else "0.000",
                    "avg_reward": f"{np.mean(all_rewards):.2f}" if all_rewards else "0.00",
                }
            )

            # Record video frame if requested
            if record_video:
                frame = env.render()
                batch_metrics.store_frame(frame)

            # Check if all episodes in batch are done
            if batch_metrics.all_episodes_done():
                break

        # Save video for this episode batch
        if record_video:
            video_path = batch_metrics.save_video(episode)

        # Append batch results to aggregate metrics
        eval_metrics.append(batch_metrics)

    # Close progress bar
    progress_bar.close()

    # Get final results
    results = eval_metrics.summary()

    if record_video and video_path is not None:
        results["video_path"] = str(video_path)
        results["video_fps"] = fps

    policy_interface.reset()  # Reset policy state after evaluation

    # Delete all locals that may hold GPU tensor references
    del obs, action_queue, batch_metrics, eval_metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Dummy Environment for Testing
# =============================================================================


@dataclass
@EnvironmentConfig.register_subclass("DummyEnvironment")
class DummyEnvironmentConfig(EnvironmentConfig):
    """Configuration for dummy test environment."""

    name: str = "DummyEnvironment"
    obs_dim: int = 10
    action_dim: int = 4
    image_size: tuple[int, int, int] = (3, 64, 64)  # C, H, W
    max_steps: int = 100
    n_envs: int = 1


class DummyEnvironment(EnvironmentWrapper):
    """
    A dummy environment for testing the evaluate_policy loop.

    This environment generates random observations and accepts any actions.
    It's useful for testing the evaluation pipeline without requiring
    a real environment or policy.

    All observations are returned as tensors with shape (batch, seq_len, **feature_size).
    Actions are accepted as tensors with shape (batch, action_dim).

    Attributes:
        config: DummyEnvironmentConfig with environment parameters
        obs_dim: Dimension of state observations
        action_dim: Dimension of actions
        image_size: Shape of image observations (C, H, W)
        current_step: Current step in the episode
        max_steps: Maximum steps per episode
    """

    def __init__(self, config: DummyEnvironmentConfig) -> None:
        super().__init__(config)
        self.config = config
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.image_size = config.image_size
        self.max_steps = config.max_steps
        self.n_envs = config.n_envs
        self._is_vectorized = config.n_envs > 1

        self.current_step = 0
        self._rng = np.random.default_rng(seed=42)

    def _process_input(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw numpy observations to policy tensor format.

        Args:
            raw_obs: Dict with numpy array observations from environment

        Returns:
            Dict with torch tensors of shape (batch, seq_len, **feature_size)
        """
        processed = {}
        for key, value in raw_obs.items():
            if isinstance(value, np.ndarray):
                # State: (obs_dim,) -> (batch=1, seq_len=1, obs_dim)
                if value.ndim == 1 or value.ndim == 3:
                    value = value[np.newaxis, np.newaxis, :]
                processed[key] = torch.from_numpy(value).float()
            elif isinstance(value, str):
                processed[key] = [value]
            else:
                processed[key] = value
        return processed

    def _process_output(self, action: torch.Tensor) -> np.ndarray:
        """Convert policy action tensor to environment numpy format.

        Args:
            action: Tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            Numpy array in format expected by underlying environment
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Remove batch dimension if present and batch=1
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        return action

    def step(self, action: torch.Tensor) -> StepReturn:
        """Take a step in the dummy environment.

        Args:
            action: Action tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
            where obs is a dict of tensors with shape (batch, seq_len, **feature_size)
        """
        # Convert action (validates format, but dummy env doesn't use it)
        _ = self._process_output(action)

        self.current_step += 1

        # Generate random observation (raw numpy)
        raw_obs = self._generate_raw_observation()

        # Convert to tensor format
        obs = self._process_input(raw_obs)

        # Random reward
        reward = self._rng.uniform(-1, 1)

        # Terminate randomly or at max steps
        terminated = self.current_step >= self.max_steps or self._rng.random() < 0.01
        truncated = False

        info = {
            "is_success": terminated and self._rng.random() < 0.5,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None) -> ResetReturn:
        """Reset the dummy environment.

        Args:
            seed: Optional random seed

        Returns:
            Tuple of (obs, info)
            where obs is a dict of tensors with shape (batch, seq_len, **feature_size)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        self.current_step = 0

        # Generate raw observation and convert to tensor format
        raw_obs = self._generate_raw_observation()
        obs = self._process_input(raw_obs)

        info = {"episode_start": True}

        return obs, info

    def render(self) -> np.ndarray:
        """Render a dummy frame.

        Returns:
            Random RGB image array of shape (H, W, 3)
        """
        h, w = self.image_size[1], self.image_size[2]
        return (self._rng.random((h, w, 3)) * 255).astype(np.uint8)

    def close(self) -> None:
        """Close the dummy environment (no-op)."""
        pass

    def _generate_raw_observation(self) -> dict[str, Any]:
        """Generate a random raw observation dict (numpy arrays).

        Returns:
            Dict with numpy arrays:
                - 'observation.state': (obs_dim,)
                - 'observation.image.0': (C, H, W)
                - 'task': str
        """
        return {
            "observation.state": self._rng.uniform(-1, 1, size=(self.obs_dim,)).astype(np.float32),
            "observation.image.0": self._rng.uniform(0, 1, size=self.image_size).astype(np.float32),
            "task": "dummy task description",
        }


# =============================================================================
# Gym Environment
# =============================================================================


class GymEnvironment(EnvironmentWrapper):
    """
    A wrapper for OpenAI Gym environments.

    This class handles the interaction with Gym environments, converting
    observations to policy tensor format and actions from tensor to numpy format.
    Supports both single and vectorized environments.

    All observations are returned as tensors with shape (batch, seq_len, **feature_size).
    Actions are accepted as tensors with shape (batch, action_dim).
    """

    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config)
        self.config = config
        self.observation_mapping = config.observation_mapping or {}
        self.default_task = config.default_task

        # Hack to only import gym_pusht if needed
        if config.env_name == "gym_pusht/PushT-v0":
            import gym_pusht  # noqa: F401

        # Create vectorized environment if n_envs > 1, otherwise single env
        if config.n_envs > 1:
            self.env = gym.vector.SyncVectorEnv(
                [
                    lambda: gym.make(
                        config.env_name,
                        obs_type=config.obs_type,
                        max_episode_steps=config.max_episode_steps,
                    )
                    for _ in range(config.n_envs)
                ]
            )
            self._is_vectorized = True
            self.n_envs = config.n_envs
            self.done = [False] * config.n_envs
            self.total_reward = [0.0] * config.n_envs
        else:
            self.env = gym.make(
                config.env_name,
                obs_type=config.obs_type,
                max_episode_steps=config.max_episode_steps,
            )
            self._is_vectorized = False
            self.n_envs = 1
            self.done = False
            self.total_reward = 0.0

        # Initialize the environment
        self.env.reset()

    def _process_input(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw environment observations to policy tensor format.

        Converts numpy arrays to torch tensors with proper dimensions:
        - Images: (batch, seq_len=1, C, H, W) with values in [0, 1]
        - States: (batch, seq_len=1, feature_dim)

        Args:
            raw_obs: Raw observation dict from gym environment

        Returns:
            Dict with torch tensors of shape (batch, seq_len, **feature_size)
        """
        processed_obs = {}

        # Apply observation mapping (common for both vectorized and single env)
        for env_key, policy_key in self.observation_mapping.items():
            if env_key in raw_obs:
                processed_obs[policy_key] = raw_obs[env_key]

        # Convert numpy arrays to tensors with proper shapes
        for key, value in processed_obs.items():
            if not isinstance(value, np.ndarray):
                continue

            tensor_value = torch.from_numpy(value).float()
            is_image = "image" in key

            if is_image:
                tensor_value = self._process_image_tensor(tensor_value)
            else:
                tensor_value = self._process_state_tensor(tensor_value)

            processed_obs[key] = tensor_value

        # Add task description
        processed_obs["task"] = [self.default_task] * self.n_envs

        return processed_obs

    def _process_image_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process image tensor to policy format.

        Handles both vectorized (batch, H, W, C) and single (H, W, C) inputs.
        Output shape: (batch, seq_len=1, C, H, W) with values in [0, 1].

        Args:
            tensor: Image tensor from environment

        Returns:
            Processed tensor with shape (batch, 1, C, H, W)
        """
        # Add batch dimension for single env: (H, W, C) -> (1, H, W, C)
        if not self.is_vectorized:
            tensor = tensor.unsqueeze(0)

        # Now tensor is (batch, H, W, C) for both cases
        # Convert channels-last to channels-first: (batch, H, W, C) -> (batch, C, H, W)
        if tensor.shape[-1] <= 4:  # Channels last
            tensor = tensor.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        # Add seq_len dimension: (batch, C, H, W) -> (batch, 1, C, H, W)
        return tensor.unsqueeze(1)

    def _process_state_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process state tensor to policy format.

        Handles both vectorized (batch, dim) and single (dim,) inputs.
        Output shape: (batch, seq_len=1, dim).

        Args:
            tensor: State tensor from environment

        Returns:
            Processed tensor with shape (batch, 1, dim)
        """
        # Add batch dimension for single env: (dim,) -> (1, dim)
        if not self.is_vectorized:
            tensor = tensor.unsqueeze(0)

        # Add seq_len dimension: (batch, dim) -> (batch, 1, dim)
        return tensor.unsqueeze(1)

    def _process_output(self, action: torch.Tensor) -> np.ndarray:
        """Convert policy action tensor to gym environment format.

        Args:
            action: Tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            Numpy array in format expected by gym environment
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Remove batch dimension if single env and batch=1
        if not self.is_vectorized and action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        return action

    def step(self, action: torch.Tensor) -> StepReturn:
        """Take a step in the gym environment.

        Args:
            action: Action tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
            where obs is a dict of tensors with shape (batch, seq_len, **feature_size)
        """
        # Convert action tensor to numpy
        action_np = self._process_output(action)

        # Step the environment
        step_result = self.env.step(action_np)

        if len(step_result) == 4:
            # Old gym format: obs, reward, done, info
            raw_obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            # New gymnasium format: obs, reward, terminated, truncated, info
            raw_obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Unexpected step result format: {len(step_result)} elements")

        # Convert observation to tensor format
        obs = self._process_input(raw_obs)

        # Update total reward tracking
        if self.is_vectorized:
            if isinstance(reward, (list, tuple)):
                for i, r in enumerate(reward):
                    self.total_reward[i] += r
            else:
                for i in range(len(self.total_reward)):
                    self.total_reward[i] += reward
        else:
            self.total_reward += reward

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None) -> ResetReturn:
        """Reset the gym environment.

        Args:
            seed: Optional random seed

        Returns:
            Tuple of (obs, info)
            where obs is a dict of tensors with shape (batch, seq_len, **feature_size)
        """
        if self.is_vectorized:
            self.done = [False] * self.n_envs
            self.total_reward = [0.0] * self.n_envs
        else:
            self.done = False
            self.total_reward = 0.0

        raw_obs, info = self.env.reset(seed=seed)
        obs = self._process_input(raw_obs)

        return obs, info

    def render(self) -> Any | list[Any]:
        """Render the environment(s).

        Returns:
            For single environment: Raw render output (numpy array, None, etc.)
            For vectorized environments: List of render outputs from each env
        """
        if self.is_vectorized:
            if isinstance(self.env, gym.vector.SyncVectorEnv):
                return [env.render() for env in self.env.envs]
            return self.env.call("render")
        return self.env.render()

    def close(self) -> None:
        """Close the gym environment(s)."""
        self.env.close()

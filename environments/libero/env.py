#!/usr/bin/env python3
"""
Libero Environment Configuration and Wrapper Classes.

This module provides specialized environment configurations and wrappers for the
LIBERO benchmark suite, extending the base environment classes to handle
Libero-specific observation spaces, actions, and rendering.
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

# Import base classes - adjust path as needed
from rho.environment import EnvironmentConfig, EnvironmentWrapper, register_environment

logger = logging.getLogger(__name__)


def _quat2axisangle(quat):
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def list_of_dicts_to_batch(obs: list[dict]) -> dict:
    """
    Convert a list of dictionaries to a single dictionary with batched values.

    Args:
        obs: List of dictionaries where each dictionary represents an observation.

    Returns:
        A single dictionary with batched values for each key.
    """
    batch = {}
    if len(obs) > 0:
        for key in obs[0]:
            batch[key] = np.stack([o[key] for o in obs], axis=0)
    return batch


def initial_state_indices(state_offset: int, n_envs: int, num_states: int) -> np.ndarray:
    """Return the next contiguous vectorized batch of LIBERO initial states."""
    return (state_offset + np.arange(n_envs)) % num_states


@EnvironmentConfig.register_subclass("libero")
@dataclass
class LiberoEnvConfig(EnvironmentConfig):
    """Configuration for Libero environments."""

    # Libero-specific settings
    task_suite_name: str = "libero_object"
    task_id: int = 0  # Task ID within the suite to use for training/evaluation
    init_state_id: int = 0
    resolution: int = 256
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])

    # Number of episodes to run per task before switching to the next task in the suite
    # Override defaults for Libero
    episodes_per_task: int = 2
    name: str = "libero"  # Not using gym registration
    max_episode_steps: int = 250
    obs_type: str = "observation"
    default_task: str = "Manipulate objects in the kitchen environment"
    seed: int = 12345  # Explicitly add seed field
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_envs: int = 1  # Number of parallel environments

    # Libero-specific observation mapping
    observation_mapping: dict[str, str] | None = None


@register_environment("libero")
class LiberoEnvWrapper(EnvironmentWrapper):
    """
    Environment wrapper for LIBERO benchmark environments.

    This wrapper handles the specific requirements of LIBERO environments,
    including initialization with BDDL files, proper observation processing,
    and rendering capabilities.
    """

    def __init__(self, config: LiberoEnvConfig):
        """
        Initialize the Libero environment wrapper.

        Args:
            config: LiberoEnvConfig containing Libero-specific settings
        """
        # Initialize parent class properly to inherit evaluate_policy method
        super().__init__(config=config)

        # Initialize Libero environment
        # self._setup_libero_env()
        self.config = config
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite_name = self.config.task_suite_name
        self.task_suite = self.benchmark_dict[self.task_suite_name]()
        self.num_tasks_in_suite = self.task_suite.n_tasks
        self.episodes_per_task = self.config.episodes_per_task

        # Initialize state tracking
        self.done = False
        self.total_reward = 0.0
        self.device = self.config.device

        # Action space - Libero uses 7D action space (6 DOF + gripper)
        self.action_dim = 7
        self.num_wait_steps = 10
        self.dummy_action = torch.tensor([0.0] * 6 + [-1.0])  # 6 DOF arm + gripper

        if self.n_envs > 1:
            self.dummy_action = self.dummy_action.to(self.device).repeat(self.n_envs, 1)
        else:
            self.dummy_action = self.dummy_action.to(self.device)

        self.last_frame = None

        # Automatically set up the Libero environment with the configured task_id
        self.setup_libero_env(self.config.task_id)

    def setup_libero_env(self, task_id):
        """Setup the LIBERO environment with the specified task."""
        # Get task suite and retrieve specific task

        self.task = self.task_suite.get_task(task_id)
        self.task_description = self.task.language

        # Setup environment arguments
        task_bddl_file = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file

        self.env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": self.config.resolution,
            "camera_widths": self.config.resolution,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": self.config.camera_names,
        }

        # Create environment (single or vectorized)
        if self.config.n_envs > 1:
            # Create vectorized environment using SubprocVectorEnv
            self.env = SubprocVectorEnv(
                [lambda: OffScreenRenderEnv(**self.env_args) for _ in range(self.config.n_envs)]
            )
        else:
            # Create single environment
            self.env = OffScreenRenderEnv(**self.env_args)

        # Seed the environment
        self.env.seed(self.config.seed)

        # Load initial states if available
        self._load_initial_states()

    def _load_initial_states(self):
        """Load initial states for reproducible testing."""
        init_states_folder = get_libero_path("init_states")
        init_states_path = os.path.join(
            init_states_folder, self.task.problem_folder, self.task.init_states_file
        )

        if os.path.exists(init_states_path):
            self.initial_states = torch.load(  # nosec B614
                init_states_path, weights_only=False
            )
        else:
            self.initial_states = None
            raise ValueError("Initial states file not found: " + init_states_path)

    @property
    def is_vectorized(self) -> bool:
        """Return True if using vectorized environments."""
        return self.config.n_envs > 1

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return self.config.n_envs

    @property
    def n_envs(self) -> int:
        """Alias for num_envs for compatibility."""
        return self.config.n_envs

    def reset(self, seed: int | None = None) -> tuple:
        """Reset the environment.

        Returns observations as tensors with shape (batch, seq_len, **feature_size).
        """
        if seed is not None:
            self.env.seed(seed)

        # Initialize done state and total reward
        self.done = np.zeros(self.n_envs, dtype=bool)
        self.total_reward = np.zeros(self.n_envs, dtype=np.float64)

        # Reset environment
        self.env.reset()

        # Set initial state if available
        if self.initial_states is not None:
            state_offset = getattr(self, "_episodes_completed_for_current_task", 0)
            if self.is_vectorized:
                # Advance through every prebuilt state exactly once per task.
                indices = initial_state_indices(state_offset, self.n_envs, len(self.initial_states))
                init_states_batch = [self.initial_states[i] for i in indices]
                raw_obs = self.env.set_init_state(init_states_batch)
            else:
                state_index = (self.config.init_state_id + state_offset) % len(self.initial_states)
                raw_obs = self.env.set_init_state(self.initial_states[state_index])

        # add delay to allow objects to settle
        for _ in range(self.num_wait_steps):
            raw_obs, _, _, _, info = self._raw_step(self.dummy_action)

        # Process observations to tensor format
        obs = self._process_input(raw_obs)

        # Return observation and info
        info = {"task_description": self.task_description}
        return obs, info

    def _raw_step(self, action) -> tuple:
        """Raw step without observation processing - used internally."""
        if isinstance(action, torch.Tensor):
            action = action.to(dtype=torch.float32).detach().cpu().numpy()

        obs, reward, done, info = self.env.step(action)

        if self.is_vectorized:
            info = list_of_dicts_to_batch(info)
        else:
            done = np.array([done])
            reward = np.array([reward])

        return obs, reward, done, False, info

    def step(self, action: torch.Tensor) -> tuple:
        """Step the environment with the given action.

        Args:
            action: Action tensor of shape (batch, action_dim)

        Returns observations as tensors with shape (batch, seq_len, **feature_size).
        """
        # Convert action from tensor to numpy
        action_np = self._process_output(action)

        # Step environment
        raw_obs, reward, done, truncated, info = self._raw_step(action_np)

        # Update state using numpy (works for both single and vectorized)
        self.total_reward += reward
        self.done = self.done | done

        # If done then it must be a success as there is no failure condition other than timeout
        info["is_success"] = done

        # Process observations to tensor format
        obs = self._process_input(raw_obs)

        return obs, reward, done, truncated, info

    def _rotate_image_180(self, img: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees to match preprocessing."""
        return np.ascontiguousarray(img[::-1, ::-1])

    def libero_observation_to_batch(self, obs):
        """Convert raw Libero observation to tensor format.

        Returns tensors with shape (batch=1, seq_len=1, **feature_size).
        """
        img = self._rotate_image_180(obs["agentview_image"])
        wrist_img = self._rotate_image_180(obs["robot0_eye_in_hand_image"])

        # (H, W, C) -> (C, H, W)
        img_reshaped = img.transpose(2, 0, 1)
        wrist_img_reshaped = wrist_img.transpose(2, 0, 1)

        # Build state from robot observations
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )

        element = {
            # Images: (batch=1, seq_len=1, C, H, W)
            "observation.image.agentview": torch.from_numpy(img_reshaped / 255.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0),
            "observation.image.wrist": torch.from_numpy(wrist_img_reshaped / 255.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0),
            # State: (batch=1, seq_len=1, state_dim)
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0),
            # Task: list of strings
            "task": [str(self.task_description)],
        }

        if "action" in obs:
            element["action"] = obs["action"].float().unsqueeze(0).unsqueeze(0)

        return element

    def _process_output(self, action: torch.Tensor) -> np.ndarray:
        """Convert policy action tensor to environment numpy format.

        Note: Denormalization is handled by PolicyInterface, not here.

        Args:
            action: Tensor of shape (batch, action_dim)

        Returns:
            Numpy array in format expected by Libero environment
        """
        # Ensure action is a numpy array for Libero
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()[..., : self.action_dim]

        # Handle action formatting for single vs vectorized environments
        if self.is_vectorized:
            # For vectorized environments, ensure action is 3D (batch_size, chunk_size, action_dim)
            if isinstance(action, np.ndarray) and action.ndim == 2:
                # If 2D action, keep as is for single step
                pass
            elif action.ndim == 3:
                # transpose so its (chunk_size, batch_size, action_dim)
                action = np.transpose(action, (1, 0, 2))
        else:
            # For single environment, ensure action is 1D or 2D
            if isinstance(action, np.ndarray) and action.ndim == 2 and action.shape[0] == 1:
                action = action.squeeze(0)

        return action

    def _process_input(self, obs: dict | list[dict]) -> dict:
        """Convert raw Libero observations to policy tensor format.

        Note: Normalization is handled by PolicyInterface, not here.

        Args:
            obs: Raw observation dictionary from Libero environment
                 For vectorized envs, this will be a list of dicts

        Returns:
            Dict with torch tensors of shape (batch, seq_len, **feature_size)
        """
        processed_obs = {}
        self._store_video_frames(obs)
        if self.is_vectorized:
            # obs is an array of dictionaries, need to reorganize
            batch_size = len(obs)
            for i in range(batch_size):
                obs[i] = self.libero_observation_to_batch(obs[i])
            all_keys = set(obs[0].keys())
            # Iterate through all keys and stack values
            for key in all_keys:
                values = []
                for i in range(batch_size):
                    if key in obs[i]:
                        values.append(obs[i][key])
                if isinstance(values[0], torch.Tensor):
                    # Stack tensors along batch dimension
                    processed_obs[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], np.ndarray):
                    # Concatenate numpy arrays along batch dimension
                    processed_obs[key] = np.concatenate(values, axis=0)
                elif isinstance(values[0], list):
                    # Flatten lists (e.g., task descriptions)
                    processed_obs[key] = [item for sublist in values for item in sublist]
                else:
                    # Pass through other types (strings, etc.)
                    processed_obs[key] = values
        else:
            processed_obs = self.libero_observation_to_batch(obs)

        # Apply observation mapping if configured
        if self.config.observation_mapping is not None:
            remapped = {}
            for key, value in processed_obs.items():
                if key in self.config.observation_mapping:
                    remapped[self.config.observation_mapping[key]] = value
                else:
                    remapped[key] = value
            processed_obs = remapped

        return processed_obs

    def next_episode(self, seed: int | None = None) -> tuple:
        """Prepare for the next episode, cycling through tasks in the suite.

        This method tracks episodes per task and switches to the next task
        after completing the configured number of episodes. Set `env.episodes_per_task`
        before calling evaluate_policy to control how many episodes run per task.

        Args:
            seed: Optional random seed for the reset

        Returns:
            Tuple of (obs, info) from reset()
        """
        # Initialize tracking on first call
        if not hasattr(self, "_current_task_id"):
            self._current_task_id = 0
            self._episodes_completed_for_current_task = 0
            # First call - setup initial task (don't re-setup if already on task 0)
            if self.config.task_id != 0:
                self._cleanup_task_resources()
                self.setup_libero_env(self._current_task_id)
            logger.info(
                f"[Task {self._current_task_id + 1}/{self.num_tasks_in_suite}] {self.task_description}"
            )
        else:
            # Increment episode counter by number of environments (vectorized runs n_envs episodes)
            self._episodes_completed_for_current_task += self.num_envs

            # Check if we should move to next task
            # Default to 1 episode per task if not set
            episodes_per_task = getattr(self, "episodes_per_task", 1)
            if self._episodes_completed_for_current_task >= episodes_per_task:
                # Clean up and move to next task
                self._cleanup_task_resources()
                self._current_task_id = (self._current_task_id + 1) % self.num_tasks_in_suite
                self._episodes_completed_for_current_task = 0
                self.setup_libero_env(self._current_task_id)
                logger.info(
                    f"[Task {self._current_task_id + 1}/{self.num_tasks_in_suite}] {self.task_description}"
                )

        return self.reset(seed=seed)

    def _cleanup_task_resources(self):
        """Clean up resources after each task to prevent GPU memory accumulation."""
        # Close the current environment to free GPU resources
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
            self.env = None

        # Clear any cached frames
        self.last_frame = None

        # Force garbage collection and GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Additional GPU memory cleanup
            torch.cuda.synchronize()

        logger.debug("Cleaned up resources after task evaluation")

    def _store_video_frames(self, obs: dict | list[dict]):
        if isinstance(obs, (list, np.ndarray)):
            renders = []
            for obs_i in obs:
                if "agentview_image" in obs_i:
                    renders.append(self._rotate_image_180(obs_i["agentview_image"]))
                else:
                    renders.append(None)
            self.last_frame = renders
        else:
            self.last_frame = (
                self._rotate_image_180(obs["agentview_image"]) if "agentview_image" in obs else None
            )

    def render(self) -> Any | list[Any]:
        """Due to issues rendering in vectorized environments,
        we return the last frame stored during _process_observation.
        """
        return self.last_frame

    def close(self):
        """Close the environment."""
        self._cleanup_task_resources()

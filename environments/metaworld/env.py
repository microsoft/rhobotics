#!/usr/bin/env python3
"""
MetaWorld MT50 Environment Configuration and Wrapper Classes.

Mirrors the LIBERO wrapper's contract (see environments/libero/env.py) for the
MetaWorld V3 benchmark, in the native pi05_metaworld / mmurz-mt50 shape:
a single `corner3` camera, a 4-D state (hand xyz + gripper) and a 4-D action
(xyz delta + gripper). No state/action widening happens here -- the policy pads
to max_state_dim / max_action_dim itself.

Task indices follow `mt50_tasks.json`, which is derived from the *dataset's*
meta/tasks.jsonl (mmurz/metaworld_mt50_v3), not from RLinf's TASK_NAME_TO_ID --
those two orderings disagree on 27 of 50 tasks, so using the wrong one silently
pairs each episode with another task's language prompt.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rho.environment import EnvironmentConfig, EnvironmentWrapper, register_environment

logger = logging.getLogger(__name__)

_TASKS_PATH = Path(__file__).parent / "mt50_tasks.json"


def load_mt50_tasks() -> dict[int, dict[str, str]]:
    """task_index -> {env_name, prompt, policy}, keyed by int."""
    with open(_TASKS_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


MT50_TASKS = load_mt50_tasks()


@EnvironmentConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnvConfig(EnvironmentConfig):
    """Configuration for MetaWorld MT50 environments."""

    name: str = "metaworld"
    # Index into mt50_tasks.json (matches the training dataset's task_index).
    task_id: int = 0
    # Restrict `next_episode` cycling to this subset of task ids. Empty = all 50.
    task_ids: list[int] = field(default_factory=list)
    resolution: int = 256
    camera_name: str = "corner3"
    # The FlowDAgger paper's MetaWorld runs use 200; RLinf's eval config uses 100.
    max_episode_steps: int = 200
    episodes_per_task: int = 10
    obs_type: str = "observation"
    default_task: str = "Complete the MetaWorld task"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # MetaWorld envs are single-process here; parallelise across workers instead.
    n_envs: int = 1
    observation_mapping: dict[str, str] | None = None


@register_environment("metaworld")
class MetaworldEnvWrapper(EnvironmentWrapper):
    """Environment wrapper for the MetaWorld V3 / MT50 benchmark."""

    def __init__(self, config: MetaworldEnvConfig):
        super().__init__(config=config)
        self.config = config
        self.device = config.device
        self.action_dim = 4

        if config.n_envs != 1:
            raise ValueError("MetaworldEnvWrapper supports n_envs=1 only")

        self.task_ids = list(config.task_ids) if config.task_ids else sorted(MT50_TASKS)
        unknown = [t for t in self.task_ids if t not in MT50_TASKS]
        if unknown:
            raise ValueError(f"unknown MT50 task ids: {unknown}")

        self.num_tasks_in_suite = len(self.task_ids)
        self.episodes_per_task = config.episodes_per_task

        self.env = None
        self.last_frame = None
        self._step_count = 0
        self._seed = int(config.seed)
        self.done = np.zeros(1, dtype=bool)
        self.total_reward = np.zeros(1, dtype=np.float64)

        self.setup_metaworld_env(config.task_id)

    # -- setup ---------------------------------------------------------

    def setup_metaworld_env(self, task_id: int):
        """Build the raw MetaWorld env for `task_id` (an MT50 dataset index)."""
        import metaworld

        spec = MT50_TASKS[int(task_id)]
        self.task_id = int(task_id)
        self.env_name = spec["env_name"]
        self.task_description = spec["prompt"]
        self.policy_name = spec["policy"]

        cls = metaworld.ALL_V3_ENVIRONMENTS[self.env_name]
        env = cls(
            render_mode="rgb_array",
            width=self.config.resolution,
            height=self.config.resolution,
            camera_name=self.config.camera_name,
        )
        # Goal-observable so the scripted expert can read the goal out of
        # obs[36:39]. The policy never sees the raw 39-D obs, so this is
        # invisible to the model.
        env._partially_observable = False
        env._freeze_rand_vec = False
        # V3's _get_state_rand_vec falls back to the *global* np.random unless
        # this is set, which makes init poses unreproducible even with a seed.
        env.seeded_rand_vec = True
        env._set_task_called = True
        try:
            del env.sawyer_observation_space  # drop the cached (partial) space
        except AttributeError:
            pass

        self.env = env
        self._last_raw_obs = None
        self._last_info: dict = {}

    def make_expert(self):
        """Instantiate this task's scripted Sawyer policy (for DAgger/demos)."""
        import metaworld.policies as policies

        return getattr(policies, self.policy_name)()

    # -- properties ----------------------------------------------------

    @property
    def is_vectorized(self) -> bool:
        return False

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def n_envs(self) -> int:
        return 1

    # -- gym-style API -------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple:
        """Reset the environment.

        Returns observations as tensors with shape (batch, seq_len, **feature_size).
        """
        from gymnasium.utils import seeding

        if seed is None:
            self._seed = (self._seed + 1) % (2**31)
            seed = self._seed
        else:
            self._seed = int(seed)

        # MetaWorld V3's Env.reset() ignores its seed argument, so install a
        # fresh Generator directly; _get_state_rand_vec consumes it.
        self.env._np_random, _ = seeding.np_random(int(seed))

        self.done = np.zeros(1, dtype=bool)
        self.total_reward = np.zeros(1, dtype=np.float64)
        self._step_count = 0

        raw_obs, info = self.env.reset()
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self._last_info = dict(info) if info else {}

        obs = self._process_input(self._last_raw_obs)
        return obs, {"task_description": self.task_description}

    def _raw_step(self, action) -> tuple:
        if isinstance(action, torch.Tensor):
            action = action.to(dtype=torch.float32).detach().cpu().numpy()
        a = np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]
        raw_obs, _r, _term, _trunc, info = self.env.step(np.clip(a, -1.0, 1.0))

        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self._last_info = dict(info) if info else {}
        self._step_count += 1

        # MetaWorld gives a dense shaping reward we don't want; the benchmark
        # metric is success_once, so treat success as the reward and terminate
        # the episode on it (the env itself never sets terminated).
        success = float(self._last_info.get("success", 0.0))
        done = np.array([success > 0.0])
        truncated = self._step_count >= self.config.max_episode_steps
        return self._last_raw_obs, np.array([success]), done, truncated, self._last_info

    def step(self, action: torch.Tensor) -> tuple:
        action_np = self._process_output(action)
        raw_obs, reward, done, truncated, info = self._raw_step(action_np)

        self.total_reward = np.maximum(self.total_reward, reward)
        self.done = self.done | done
        info = dict(info)
        info["is_success"] = self.done.copy()

        obs = self._process_input(raw_obs)
        return obs, reward, done, truncated, info

    # -- conversion ----------------------------------------------------

    def _process_output(self, action: torch.Tensor) -> np.ndarray:
        """Policy action tensor -> 4-D numpy action.

        Denormalization is handled by PolicyInterface, not here.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)[..., : self.action_dim]
        if action.ndim == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        return action

    def _process_input(self, raw_obs: np.ndarray) -> dict:
        """Raw 39-D MetaWorld obs -> policy tensor dict.

        Normalization is handled by PolicyInterface; images are IDENTITY-normalized
        so they go out as [0, 1] floats, matching how the MT50 dataset was built.
        """
        frame = self._render_frame()
        self.last_frame = frame

        img = frame.transpose(2, 0, 1)  # HWC -> CHW
        state = np.asarray(raw_obs, dtype=np.float32)[:4]

        processed = {
            "observation.image": torch.from_numpy(img / 255.0).float().unsqueeze(0).unsqueeze(0),
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0),
            "task": [str(self.task_description)],
        }

        if self.config.observation_mapping:
            processed = {
                self.config.observation_mapping.get(k, k): v for k, v in processed.items()
            }
        return processed

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        res = self.config.resolution
        if frame is None:
            return np.zeros((res, res, 3), dtype=np.uint8)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    # -- episode / task cycling ----------------------------------------

    def next_episode(self, seed: int | None = None) -> tuple:
        """Prepare the next episode, cycling tasks after `episodes_per_task`."""
        if not hasattr(self, "_current_task_idx"):
            self._current_task_idx = 0
            self._episodes_completed_for_current_task = 0
            first = self.task_ids[0]
            if self.task_id != first:
                self._cleanup_task_resources()
                self.setup_metaworld_env(first)
            logger.info(
                f"[Task 1/{self.num_tasks_in_suite}] {self.env_name}: {self.task_description}"
            )
        else:
            self._episodes_completed_for_current_task += 1
            if self._episodes_completed_for_current_task >= self.episodes_per_task:
                self._cleanup_task_resources()
                self._current_task_idx = (self._current_task_idx + 1) % self.num_tasks_in_suite
                self._episodes_completed_for_current_task = 0
                self.setup_metaworld_env(self.task_ids[self._current_task_idx])
                logger.info(
                    f"[Task {self._current_task_idx + 1}/{self.num_tasks_in_suite}] "
                    f"{self.env_name}: {self.task_description}"
                )

        return self.reset(seed=seed)

    def _cleanup_task_resources(self):
        if getattr(self, "env", None) is not None:
            try:
                self.env.close()
            except Exception as e:  # noqa: BLE001 - teardown is best-effort
                logger.warning(f"Error closing environment: {e}")
            self.env = None
        self.last_frame = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def render(self) -> Any | list[Any]:
        """Return the frame cached during _process_input (EGL contexts are
        expensive to re-enter mid-rollout)."""
        return self.last_frame

    def close(self):
        self._cleanup_task_resources()

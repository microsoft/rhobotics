"""
Dataset Environment and Mirror Policy for transform/normalization verification.

This module provides two components that work together to verify the full
``PolicyInterface`` pipeline (normalize → policy → denormalize) is faithful:

``DatasetEnvironment``
    Loads raw (untransformed) samples from a ``LeRobotDataset``.  Each
    "episode" corresponds to one dataset sample.  The raw observation dict
    — **including the action chunk** — is handed to the policy so that the
    normalize → mirror → denormalize roundtrip can be checked.

``MirrorPolicy``
    A trivial policy that extracts the ``"action"`` tensor from the
    observation dict and returns it unchanged.  When plugged into
    ``PolicyInterface.get_action_chunk``, it turns the pipeline into an
    identity test: the only thing that should change the actions is
    normalize + denormalize, so any difference between input and output
    reveals a bug in the transform chain.

Example usage::

    from rho.environment.dataset_env import (
        DatasetEnvironmentConfig,
        DatasetEnvironment,
        MirrorPolicy,
    )
    from rho.eval.policy_interface import PolicyInterface, PolicyInterfaceConfig
    from rho.environment.env import evaluate_policy

    env_cfg = DatasetEnvironmentConfig(
        repo_id="my_robot_dataset",
        root_dir="/path/to/lerobot_dataset",
        chunk_size=32,
        observation_mapping={
            "observation.images.agentview_image": "observation.image.0",
            "observation.images.left_wrist_image": "observation.image.1",
            "observation.images.right_wrist_image": "observation.image.2",
            "observation.language_instruction": "task",
            "observation.ee_quat_pos": "observation.state",
            "action.ee_quat_pos": "action",
        },
    )
    env = DatasetEnvironment(env_cfg)

    mirror = MirrorPolicy(chunk_size=32, action_key="action")

    pi_cfg = PolicyInterfaceConfig(
        data_config=my_data_config,      # carries stats / normalization
        policy=mirror,
        device="cpu",
    )
    pi = PolicyInterface(pi_cfg)

    results = evaluate_policy(
        env, pi, num_episodes=10, max_steps=32,
    )
    env.close()   # prints summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rho.environment.env import EnvironmentConfig, EnvironmentWrapper, ResetReturn, StepReturn

logger = logging.getLogger(__name__)


# =====================================================================
# Mirror Policy
# =====================================================================


class MirrorPolicy:
    """A policy that echoes back the action tensor found in the observation.

    When used with ``PolicyInterface.get_action_chunk`` the data flow is:

    1. ``get_action_chunk`` normalizes the observation dict (including
       the ``"action"`` key).
    2. ``sample_actions`` simply returns that normalized action tensor.
    3. ``get_action_chunk`` denormalizes the returned tensor.

    If the transform pipeline is correct, the denormalized output will
    exactly match the raw action that the ``DatasetEnvironment``
    originally loaded.

    Args:
        chunk_size: Expected action-chunk length (only used for sanity
            checks / logging).
        action_key: Key under which the action tensor lives in the
            observation dict passed to ``sample_actions``.
        device: Torch device string.  The mirror policy is stateless,
            so this is only kept for compatibility with
            ``PolicyInterfaceConfig``.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        action_key: str = "action",
        device: str = "cpu",
    ):
        self.chunk_size = chunk_size
        self.action_key = action_key
        self.device = device

    # -- public interface expected by PolicyInterface ------------------

    def sample_actions(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return the action tensor from *batch* without modification.

        Args:
            batch: The (already-normalised) observation dict.  Must
                contain ``self.action_key`` with shape
                ``(batch_size, chunk_size, action_dim)``.

        Returns:
            ``{"actions": actions}`` — the same format every policy
            uses so ``PolicyInterface.get_action_chunk`` can call
            ``process_action`` on it.
        """
        if self.action_key not in batch:
            raise KeyError(
                f"MirrorPolicy: action key '{self.action_key}' not "
                f"found in observation dict.  Available keys: "
                f"{list(batch.keys())}"
            )

        actions = batch[self.action_key]

        # Defensive: ensure the tensor has 3 dims (B, T, D)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)

        return {"actions": actions}

    def reset(self) -> None:
        """No-op — the mirror policy is stateless."""
        pass

    def eval(self) -> MirrorPolicy:
        """No-op — compatibility with ``nn.Module.eval()``."""
        return self

    def to(self, device) -> MirrorPolicy:
        """No-op — compatibility with ``nn.Module.to()``."""
        self.device = str(device)
        return self


# =====================================================================
# Dataset Environment
# =====================================================================


@dataclass
@EnvironmentConfig.register_subclass("DatasetEnvironment")
class DatasetEnvironmentConfig(EnvironmentConfig):
    """Config for the dataset-replay environment.

    The observation mapping should be the *dataset-level* mapping (the
    same one your dataset YAML uses) so that raw column names are
    translated to the canonical keys the ``PolicyInterface`` expects.

    Important: the mapping **must** include the action key (e.g.
    ``"action.ee_quat_pos": "action"``), because the action chunk is
    kept in the observation dict so that ``MirrorPolicy`` (or a real
    policy) can see it.
    """

    name: str = "DatasetEnvironment"

    # LeRobotDataset location
    repo_id: str = "lerobot/pusht"
    root_dir: str | None = None
    tolerance_s: float = 0.1

    # Action chunk
    chunk_size: int = 16
    action_key: str = "action"  # canonical key (after mapping)

    # max_episode_steps is used by evaluate_policy; for dataset replay
    # each episode is one action chunk, so default to chunk_size.
    max_episode_steps: int = 300

    # Dataset-level observation mapping (raw → canonical)
    observation_mapping: dict[str, str] = field(
        default_factory=dict,
    )

    # Iteration control
    start_index: int = 0
    max_episodes: int | None = None


class DatasetEnvironment(EnvironmentWrapper):
    """Replay raw dataset samples through the PolicyInterface pipeline.

    Each *episode* corresponds to one sample from the underlying
    ``LeRobotDataset``.  On ``next_episode`` / ``reset``:

    * The raw sample is loaded.
    * The observation mapping is applied.
    * The ground-truth action chunk is stashed for later comparison
      **and also kept in the observation dict** so the policy (or
      ``MirrorPolicy``) can see it.
    * Observations are converted to the ``(batch=1, seq, *feat)``
      tensor format that ``evaluate_policy`` expects.

    On each ``step(action)`` call the single-timestep predicted action
    is compared to the corresponding timestep of the stashed GT chunk.
    After ``chunk_size`` steps the episode terminates.
    """

    def __init__(self, config: DatasetEnvironmentConfig) -> None:
        super().__init__(config)
        self.config = config
        self.n_envs = 1
        self._is_vectorized = False
        self.episodes_per_task = None  # set by evaluate_policy

        from rho.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        # --- resolve raw action key (before mapping) -----------------
        self._obs_mapping = config.observation_mapping or {}
        self._reverse_mapping = {v: k for k, v in self._obs_mapping.items()}
        raw_action_key = self._reverse_mapping.get(
            config.action_key,
            config.action_key,
        )

        # --- dataset metadata (fps) ---------------------------------
        self.ds_meta = LeRobotDatasetMetadata(
            config.repo_id,
            root=config.root_dir,
        )
        fps = self.ds_meta.fps

        delta_timestamps = {
            raw_action_key: [i / fps for i in range(config.chunk_size)],
        }

        logger.info(
            "DatasetEnvironment: loading %s (root=%s), fps=%s, chunk=%d, raw_action_key=%s",
            config.repo_id,
            config.root_dir,
            fps,
            config.chunk_size,
            raw_action_key,
        )

        # --- raw dataset (NO transforms) ----------------------------
        self.dataset = LeRobotDataset(
            repo_id=config.repo_id,
            root=config.root_dir,
            delta_timestamps=delta_timestamps,
            tolerance_s=config.tolerance_s,
        )
        logger.info(
            "DatasetEnvironment: %d samples available",
            len(self.dataset),
        )

        # --- state ---------------------------------------------------
        self.current_index: int = config.start_index
        self.current_step_in_chunk: int = 0
        self.current_gt_actions: torch.Tensor | None = None
        self.current_obs: dict[str, torch.Tensor] | None = None

        # --- error tracking ------------------------------------------
        self.action_errors: list[dict[str, Any]] = []
        self.episodes_completed: int = 0

    # =================================================================
    # observation / action helpers
    # =================================================================

    def _apply_obs_mapping(
        self,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Rename raw keys → canonical keys."""
        if not self._obs_mapping:
            return sample
        return {self._obs_mapping.get(k, k): v for k, v in sample.items()}

    def _extract_gt_actions(
        self,
        sample: dict[str, Any],
    ) -> torch.Tensor:
        """Copy (not pop) the action chunk from *sample*.

        The action is **kept** in the sample so it will appear in the
        observation dict handed to the policy.
        """
        ak = self.config.action_key
        if ak not in sample:
            raise KeyError(f"Action key '{ak}' not in sample after mapping. Keys: {list(sample.keys())}")
        actions = sample[ak]  # keep in dict
        if isinstance(actions, np.ndarray):
            return torch.from_numpy(actions).float()
        if isinstance(actions, torch.Tensor):
            return actions.float()
        raise TypeError(f"Expected ndarray or Tensor for actions, got {type(actions)}")

    def _process_input(
        self,
        raw_obs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Convert mapped sample to ``(batch=1, …)`` tensor dict.

        All tensors get an ``unsqueeze(0)`` for the batch dimension.
        Metadata scalars are silently skipped.
        """
        _skip = frozenset(
            {
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
                "episode_data_index_from",
                "episode_data_index_to",
            }
        )
        processed: dict[str, Any] = {}
        for key, value in raw_obs.items():
            if key in _skip:
                continue
            if isinstance(value, torch.Tensor):
                processed[key] = value.float().unsqueeze(0)
            elif isinstance(value, np.ndarray):
                processed[key] = torch.from_numpy(value).float().unsqueeze(0)
            elif isinstance(value, str):
                processed[key] = [value]
            elif isinstance(value, (int, float, bool)):
                continue  # metadata scalar
            else:
                processed[key] = value
        return processed

    def _process_output(
        self,
        action: torch.Tensor,
    ) -> np.ndarray:
        """Strip batch dim and convert to numpy for error computation."""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        return action

    # =================================================================
    # error computation
    # =================================================================

    @staticmethod
    def _compute_action_error(
        predicted: np.ndarray | torch.Tensor,
        ground_truth: np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        """Per-timestep MSE / MAE / max-error."""
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()

        predicted = np.atleast_1d(predicted).astype(np.float64)
        ground_truth = np.atleast_1d(ground_truth).astype(np.float64)

        diff = predicted - ground_truth
        return {
            "mse": float(np.mean(diff**2)),
            "mae": float(np.mean(np.abs(diff))),
            "max_error": float(np.max(np.abs(diff))),
            "per_dim_error": np.abs(diff).tolist(),
        }

    # =================================================================
    # core EnvironmentWrapper interface
    # =================================================================

    def _load_sample(
        self,
        index: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Load one sample, apply mapping, stash GT, return obs."""
        raw = self.dataset[index]
        mapped = self._apply_obs_mapping(raw)

        # Stash GT (action stays in *mapped* for the policy to see)
        self.current_gt_actions = self._extract_gt_actions(mapped)

        obs = self._process_input(mapped)
        self.current_obs = obs
        self.current_step_in_chunk = 0

        info = {
            "dataset_index": index,
            "gt_actions_shape": tuple(self.current_gt_actions.shape),
        }
        return obs, info

    def reset(self, seed: int | None = None) -> ResetReturn:
        if seed is not None:
            self.current_index = (self.config.start_index + seed) % len(self.dataset)
        else:
            self.current_index = self.config.start_index

        self.episodes_completed = 0
        self.action_errors = []
        return self._load_sample(self.current_index)

    def next_episode(self, seed: int | None = None) -> ResetReturn:
        if self.current_index >= len(self.dataset):
            self.current_index = 0
        return self._load_sample(self.current_index)

    def step(self, action: torch.Tensor) -> StepReturn:
        """Compare one predicted action step against GT."""
        action_np = self._process_output(action)
        gt = self.current_gt_actions[self.current_step_in_chunk]

        error = self._compute_action_error(action_np, gt)
        self.action_errors.append(
            {
                "dataset_index": self.current_index,
                "chunk_step": self.current_step_in_chunk,
                **error,
                "predicted": action_np.copy(),
                "ground_truth": gt.cpu().numpy().copy(),
            }
        )

        # ---- breakpoint hook: set a breakpoint on the `pass` below ----
        if error["mse"] > 1e-6:
            pass  # <-- SET BREAKPOINT HERE to inspect `error`, `action_np`, `gt`

        self.current_step_in_chunk += 1
        chunk_done = self.current_step_in_chunk >= self.config.chunk_size

        reward = -error["mse"]
        terminated = chunk_done
        truncated = False

        info: dict[str, Any] = {
            "is_success": error["mse"] < 1e-6,
            "action_mse": error["mse"],
            "action_mae": error["mae"],
            "action_max_error": error["max_error"],
        }

        if chunk_done:
            self.episodes_completed += 1
            self.current_index += 1

        dataset_exhausted = self.current_index >= len(self.dataset)
        max_reached = (
            self.config.max_episodes is not None and self.episodes_completed >= self.config.max_episodes
        )
        if dataset_exhausted or max_reached:
            terminated = True

        return self.current_obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the most recent GT vs predicted EEF pose as a 3-D plot.

        Only produces a frame when the action dimension is 16
        (``ee_quat_pos`` format).  Returns ``None`` otherwise.
        """
        if not self.action_errors:
            return None

        last = self.action_errors[-1]
        gt = last["ground_truth"]
        pred = last["predicted"]

        if gt.shape[-1] != 16:
            return None

        from rho.utils.eef_viz import render_eef_frame

        step = last["chunk_step"]
        ds_idx = last["dataset_index"]
        mse = last["mse"]
        title = f"Sample {ds_idx}  Step {step}  MSE={mse:.6f}"
        return render_eef_frame(gt, pred, title=title)

    def close(self) -> None:
        """Log aggregate error summary."""
        summary = self.get_error_summary()
        if summary["num_steps"] == 0:
            logger.info("DatasetEnvironment: no steps recorded.")
            return

        logger.info("=" * 70)
        logger.info("DatasetEnvironment — Transform Verification Summary")
        logger.info("=" * 70)
        logger.info(
            "  Episodes evaluated  : %d",
            summary["num_episodes"],
        )
        logger.info(
            "  Action steps compared: %d",
            summary["num_steps"],
        )
        logger.info(
            "  Mean MSE            : %.8f",
            summary["mean_mse"],
        )
        logger.info(
            "  Std  MSE            : %.8f",
            summary["std_mse"],
        )
        logger.info(
            "  Max  MSE            : %.8f",
            summary["max_mse"],
        )
        logger.info(
            "  Mean MAE            : %.8f",
            summary["mean_mae"],
        )
        logger.info(
            "  Max single-dim err  : %.8f",
            summary["max_single_dim_error"],
        )
        if summary["mean_mse"] < 1e-6:
            logger.info("  ✅  Pipeline is faithful — actions reconstructed.")
        elif summary["mean_mse"] < 1e-3:
            logger.info("  ⚠️  Small numerical drift — likely acceptable.")
        else:
            logger.warning("  ❌  Significant error — check transforms!")
        logger.info("=" * 70)

    def get_error_summary(self) -> dict[str, Any]:
        """Aggregate error statistics over all recorded steps."""
        if not self.action_errors:
            return {"num_steps": 0, "num_episodes": 0}

        mses = np.array([e["mse"] for e in self.action_errors])
        maes = np.array([e["mae"] for e in self.action_errors])
        maxs = np.array([e["max_error"] for e in self.action_errors])

        return {
            "num_steps": len(self.action_errors),
            "num_episodes": self.episodes_completed,
            "mean_mse": float(mses.mean()),
            "std_mse": float(mses.std()),
            "max_mse": float(mses.max()),
            "min_mse": float(mses.min()),
            "mean_mae": float(maes.mean()),
            "max_single_dim_error": float(maxs.max()),
            "per_step_errors": self.action_errors,
        }

import logging
from collections import deque
from dataclasses import dataclass, field  # noqa: I001
from typing import Any

import numpy as np
import torch

from rho.datasets.data_config import DataConfig
from rho.policies.base import PreTrainedPolicy

logger = logging.getLogger(__name__)


def _move_transforms_to_device(transform, device):
    """Recursively move a transform (or Compose of transforms) to *device*.

    Handles:
    - ``torchvision.transforms.Compose`` (iterates children)
    - ``nn.Module`` and other objects with a ``.to()`` method
    - ``TransformWrapper`` (moves the inner ``.transform``)
    - Closures returned by ``DataConfig.get_transforms`` (peeks into
      the captured ``transform_list`` and moves each element)
    """
    from torchvision.transforms import Compose

    from rho.datasets.data_config import TransformWrapper

    if isinstance(transform, Compose):
        transform.transforms = [_move_transforms_to_device(t, device) for t in transform.transforms]
        return transform
    if isinstance(transform, TransformWrapper):
        transform.transform = _move_transforms_to_device(transform.transform, device)
        return transform
    if hasattr(transform, "to"):
        return transform.to(device)
    # Handle closures (e.g. composed_transform from get_transforms)
    if callable(transform) and hasattr(transform, "__closure__") and transform.__closure__:
        for cell in transform.__closure__:
            try:
                cell_contents = cell.cell_contents
                if isinstance(cell_contents, list):
                    for i, item in enumerate(cell_contents):
                        cell_contents[i] = _move_transforms_to_device(item, device)
            except ValueError:
                pass
    return transform


def debug_print_and_save_obs(obs, debug_dir="debug_obs_images"):
    import os

    import matplotlib.pyplot as plt

    os.makedirs(debug_dir, exist_ok=True)
    logger.debug("Observation details before sample_actions:")
    for k, v in obs.items():
        if k.startswith("observation.image"):
            # Assume shape is (B, C, H, W) in RGB format
            logger.debug(f"{k}: shape={v.shape}")
            # Save PNG
            img = v
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # Expecting (B, C, H, W)
            for i in range(img.shape[0]):
                img_i = img[i]  # (C, H, W)
                if img_i.shape[0] == 3:
                    img_i = np.transpose(img_i, (1, 2, 0))  # (H, W, C) for RGB
                plt.imsave(
                    os.path.join(debug_dir, f"{k.replace('.', '_')}_sample{i}.png"), np.clip(img_i, 0, 1)
                )
        elif k == "observation.state":
            v_np = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            logger.debug(
                f"{k}: shape={v.shape}, mean={v_np.mean():.4f}, min={v_np.min():.4f}, max={v_np.max():.4f}"
            )


def validate_rtc_horizons(inference_delay: int, execution_horizon: int, chunk_size: int) -> None:
    """Validate that the RTC horizon parameters are self-consistent.

    Raises ``ValueError`` when ``inference_delay + execution_horizon > chunk_size`` because
    the rollout queue would then retain actions from an older chunk while the cached
    ``prev_action_chunk`` has already advanced, causing RTC to blend against actions the
    robot is not actually executing.

    Also emits ``logger.warning`` for two degenerate-but-non-fatal configurations:

    * ``execution_horizon >= chunk_size``: re-inference occurs on essentially every
      timestep so RTC provides no latency benefit.
    * ``inference_delay >= chunk_size - execution_horizon``: the soft-mask decay region
      (Section 3.2 of arXiv:2506.07339) is empty and the method reduces to the naive
      hard-masking baseline.
    """
    if execution_horizon >= chunk_size:
        logger.warning(
            "RTC: execution_horizon (%d) >= chunk_size (%d). Re-inference will occur on essentially "
            "every timestep, so RTC provides no latency benefit. Consider reducing execution_horizon.",
            execution_horizon,
            chunk_size,
        )

    if inference_delay >= chunk_size - execution_horizon:
        logger.warning(
            "RTC: inference_delay (%d) >= chunk_size (%d) - execution_horizon (%d) = %d. "
            "Every overlapping row of the soft mask sits in the i < d branch of Eq. 5 and receives "
            "weight 1.0, so the exponential decay region is empty and the method reduces to the "
            "naive hard-masking baseline that soft masking exists to improve on.",
            inference_delay,
            chunk_size,
            execution_horizon,
            chunk_size - execution_horizon,
        )

    if inference_delay + execution_horizon > chunk_size:
        raise ValueError(
            f"RTC: inference_delay ({inference_delay}) + execution_horizon ({execution_horizon}) = "
            f"{inference_delay + execution_horizon} > chunk_size ({chunk_size}). "
            "RTC would otherwise blend against actions the robot is not executing because the rollout "
            "queue retains actions from an older chunk while prev_action_chunk has already advanced. "
            "Fix: reduce execution_horizon or inference_delay so their sum fits within chunk_size."
        )


def _parse_num_actions_executed(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("num_actions_executed must be an integer scalar.")
        value = value.item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("num_actions_executed must be an integer scalar.")
        value = value.item()

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("num_actions_executed must be an integer scalar.")
    return int(value)


def _validate_guidance_schedule(value: str) -> None:
    if value not in ("paper", "constant"):
        raise ValueError(f"guidance_schedule must be 'paper' or 'constant', got {value!r}.")


@dataclass
class PolicyInterfaceConfig:
    """Configuration for environment interactions"""

    data_config: DataConfig = None
    input_transforms = None
    output_transforms = None
    normalize_inputs = None

    observation_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "observation.state": "observation.state",
            "observation.image.0": "observation.image.0",
            "observation.image.1": "observation.image.1",
            "observation.image.2": "observation.image.2",
            "task": "task",
            "action": "action",
        }
    )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    policy: PreTrainedPolicy = None

    eval_mode: str = "standard"  # Options: 'standard', 'rtc'
    inference_delay: int = None  # Number of steps policy inference takes
    execution_horizon: int | None = None  # Actions executed between inferences; defaults to n_action_steps
    beta: int = None  # Weighting parameter for RTC update vs standard update
    guidance_schedule: str = "paper"  # Guidance coefficient schedule: 'paper' or 'constant'

    def __post_init__(self):
        assert self.eval_mode in ["standard", "rtc"], "eval_mode must be 'standard' or 'rtc'"
        _validate_guidance_schedule(self.guidance_schedule)
        if "task" not in self.observation_mapping.values():
            self.observation_mapping["task"] = "task"
        if "action" not in self.observation_mapping.values():
            self.observation_mapping["action"] = "action"


class PolicyInterface:
    def __init__(self, cfg: PolicyInterfaceConfig):
        _validate_guidance_schedule(cfg.guidance_schedule)
        self.policy = cfg.policy
        self.data_config = cfg.data_config
        self.device = cfg.device
        self.input_transforms = cfg.input_transforms
        self.output_transforms = cfg.output_transforms
        self.normalize_inputs = cfg.normalize_inputs
        self.observation_mapping = cfg.observation_mapping
        self.eval_mode = cfg.eval_mode
        self.inference_delay = cfg.inference_delay
        self.beta = cfg.beta
        self.guidance_schedule = cfg.guidance_schedule
        self.horizon = self.policy.config.chunk_size
        self.execution_horizon = (
            cfg.execution_horizon if cfg.execution_horizon is not None else self.policy.config.n_action_steps
        )
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive.")
        if self.eval_mode == "standard" and self.execution_horizon > self.horizon:
            raise ValueError("Standard execution_horizon must not exceed chunk_size.")

        # Most recent action chunk after output transforms, in the absolute
        # action representation consumed by the environment. Before RTC reuses
        # the remaining actions, process_observation transforms them into the
        # current request's policy frame. This is required for state-relative
        # deltas and per-timestep ACTIONCHUNK normalization: cached model-space
        # values from old index e+i are not comparable to new index i.
        self.prev_action_chunk: torch.Tensor | None = None

        if self.eval_mode == "rtc":
            logger.info("RTC mode enabled: applying necessary config adjustments.")
            if hasattr(self.policy.model, "flow_model"):
                self.policy.model.flow_model.action_expert.enable_gradient_checkpointing = False
            else:
                self.policy.model.action_expert.enable_gradient_checkpointing = False

            assert self.inference_delay is not None and self.beta is not None, (
                "RTC mode requires inference_delay and beta to be set."
            )
            try:
                validate_rtc_horizons(self.inference_delay, self.execution_horizon, self.horizon)
            except ValueError as exc:
                if cfg.execution_horizon is None:
                    raise ValueError(
                        f"{exc} execution_horizon was not set, so policy.config.n_action_steps "
                        f"({self.policy.config.n_action_steps}) was used. Set execution_horizon "
                        "explicitly in the evaluation or serving config."
                    ) from exc
                raise

        if self.data_config is not None:
            logger.info("Initializing data config within PolicyInterfaceConfig.")
            if self.input_transforms is None:
                logger.debug(f"data_config.transform_mapping: {self.data_config.transform_mapping}")
                if self.data_config.transform_mapping is not None:
                    for _k, _v in self.data_config.transform_mapping.items():
                        _vl = _v if isinstance(_v, (list, tuple)) else [_v]
                        for _t in _vl:
                            logger.debug(f"  transform_mapping['{_k}']: {type(_t).__name__} -> {_t}")
                else:
                    logger.warning("transform_mapping is None in data_config!")
                self.input_transforms = self.data_config.get_transforms(remap=False, training=False)
                logger.debug(f"input_transforms after get_transforms: {self.input_transforms}")
            if self.output_transforms is None:
                self.output_transforms = self.data_config.get_action_denormalization()

        # Move output_transforms to the target device so denormalization
        # stats live on the same device as the action tensors.
        if self.input_transforms is not None:
            self.input_transforms = _move_transforms_to_device(self.input_transforms, self.device)
        if self.output_transforms is not None:
            self.output_transforms = _move_transforms_to_device(self.output_transforms, self.device)

        assert self.observation_mapping is not None, (
            "observation_mapping must contain all keys you want to pass in to the policy."
        )
        assert self.policy is not None, "policy must be provided to PolicyInterfaceConfig."

        # obs history logic
        self.delta_indices_dict = self.policy.config.delta_indices_dict

        self.obs_queue = {}
        for key, delta_indices in self.delta_indices_dict.items():
            if delta_indices is not None and len(delta_indices) > 1 and not key.startswith("action"):
                self.obs_queue[key] = deque(maxlen=int(max(abs(np.array(delta_indices))) + 1))
            elif "image" not in key and not key.startswith("action"):
                self.obs_queue[key] = deque(maxlen=1)

        # Rank (number of dims, excluding batch) of a single observation frame
        # for each queued key. Used by process_obs_queue to detect whether an
        # incoming tensor already carries an explicit temporal axis. Some
        # environments (e.g. TabletopSim) provide observations as
        # (batch, seq_len, *feature_shape) while others (e.g. UR5) provide them
        # as (batch, *feature_shape) with no temporal axis.
        self._feature_ranks = {}
        feature_dict = getattr(self.policy.config, "feature_dict", None) or {}
        for key in self.obs_queue:
            feature = feature_dict.get(key)
            if feature is not None and getattr(feature, "shape", None) is not None:
                self._feature_ranks[key] = len(feature.shape)
            else:
                # Fallback when feature metadata is unavailable: images are
                # (C, H, W) -> rank 3, everything else (state/env) -> rank 1.
                self._feature_ranks[key] = 3 if "image" in key else 1

    def reset(self):
        """Reset policy interface state between episodes.

        Clears the observation history queue so that stale GPU tensors
        from previous episodes are freed and don't leak into the next episode.
        """
        print("RESETTING OBS QUEUE")
        for key in self.obs_queue:
            self.obs_queue[key].clear()
        # Drop the stored RTC action chunk so remaining-action blending does not
        # leak across episode boundaries.
        self.prev_action_chunk = None

    def remap_observation(self, obs: dict[str, torch.Tensor | str]) -> dict[str, Any]:
        """Remap observation keys based on observation_mapping.

        Converts observation keys from the environment/client format to the
        policy-expected format using the configured observation_mapping.

        Args:
            obs: Dictionary of current observations where:
                - observation keys: torch.Tensor of shape (batch_size, seq_len, *obs_shape)
                - 'action' key (if present): torch.Tensor of shape (batch_size, seq_len, action_dim)
                    should be the prev actions from the client
                - Images: torch.Tensor of shape (batch_size, C, H, W), values in [0, 1], RGB
                - Text (e.g., "task"): List of strings

        Returns:
            dict: remapped_observation where keys are from observation_mapping values and tensors with
                dicts have keys from observation_mapping values and tensors with
                same shapes as input.
        """
        assert self.observation_mapping is not None, (
            "observation_mapping must contain all keys you want to pass in to the policy."
        )

        remapped_observation = {}
        for key, value in obs.items():
            if key in self.observation_mapping:
                remapped_observation[self.observation_mapping[key]] = value
            else:
                remapped_observation[key] = value

        return remapped_observation

    def process_obs_queue(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert dict of observation deques to a dict of concatenated tensors and merge with current obs.

        Uses delta_indices_dict from policy.cfg to determine which timesteps to select
        for each observation key.

        Args:
            obs: A dictionary with current observation tensors.

        Returns:
            A dictionary with observation keys and concatenated tensors of shape
            (batch_size, n_timesteps, *obs_shape) where n_timesteps is determined by
            the delta_indices for each key. For every processed key, a companion
            ``{key}_is_pad`` boolean tensor of shape (batch_size, n_timesteps) is
            added, mirroring the ``LeRobotDataset`` getitem behaviour: an entry is
            True when the requested timestep falls before the start of the
            available history and is therefore filled with the oldest observation.
        """

        pad_updates: dict[str, torch.Tensor] = {}
        for key in obs:
            if key in self.obs_queue:
                # first append current obs to the queues, preserving the batch dimension
                values = obs[key]
                if values.ndim < 2:
                    continue

                # Detect whether the incoming tensor carries an explicit temporal
                # axis. A single frame is (batch, *feature_shape); with a temporal
                # axis it is (batch, seq_len, *feature_shape). We disambiguate by
                # comparing the tensor rank against the known per-frame feature
                # rank. Some environments (e.g. TabletopSim) supply the temporal
                # axis; others (e.g. UR5) supply a single current frame.
                feature_rank = self._feature_ranks[key]
                if values.ndim == feature_rank + 2:
                    # (batch, seq_len, *feature_shape): append each timestep,
                    # oldest first so the latest ends up at the end of the queue.
                    frames = [values[:, t : t + 1] for t in reversed(range(values.shape[1]))]
                else:
                    # (batch, *feature_shape) with no temporal axis (or an
                    # unexpected rank): treat as a single current frame and add
                    # the temporal axis ourselves -> (batch, 1, *feature_shape).
                    frames = [values.unsqueeze(1)]

                for frame in frames:
                    self.obs_queue[key].append(frame)  # (batch_size, 1, *obs_shape)

                # then sample from the queue
                delta_indices = self.delta_indices_dict[key]

                buffer_list = list(self.obs_queue[key])
                buffer_len = len(buffer_list)
                sampled_history = []
                pad_flags = []

                for delta_idx in delta_indices:
                    # delta_idx is negative or zero (e.g., -27, -24, ..., 0)
                    # Convert to positive index from current timestep
                    # Current timestep is at index buffer_len - 1
                    actual_idx = (buffer_len - 1) + delta_idx

                    if actual_idx < 0:
                        # Pad with oldest available observation
                        sampled_history.append(buffer_list[0])
                        pad_flags.append(True)
                    else:
                        sampled_history.append(buffer_list[actual_idx])
                        pad_flags.append(False)

                # Concatenate along seq dim: (batch_size, history_len, *obs_shape)
                obs[key] = torch.cat(sampled_history, dim=1)

                # Emit an is_pad mask matching the dataset getitem, so policies can
                # mask out padded (repeated boundary) observations at inference just
                # as they do during training. The pad status only depends on the
                # buffer length, so it is shared across the batch dimension.
                pad_mask = torch.tensor(pad_flags, dtype=torch.bool, device=obs[key].device)
                pad_updates[f"{key}_is_pad"] = (
                    pad_mask.unsqueeze(0).expand(obs[key].shape[0], -1).contiguous()
                )

        obs.update(pad_updates)
        return obs

    def process_observation(self, obs, process_action: bool = False):
        """Process observations for policy input.

        Full preprocessing pipeline:
        1. Remaps observation keys via observation_mapping
        2. Processes observation queue to select relevant timesteps (if provided)
        3. Applies input image transforms (augmentation)
        4. Applies input normalization

        Args:
            obs: Dictionary of current observations where:
                - State keys: torch.Tensor of shape (batch_size, seq_len, state_dim)
                - Image keys: torch.Tensor of shape (batch_size, C, H, W)
                - Previous actions: torch.Tensor of shape (batch_size, seq_len, action_dim) (optional)
                - Task: List of strings

        Returns:
            dict: Processed observations ready for policy.sample_actions(), with
                normalized values and selected temporal indices.
        """

        obs = self.remap_observation(obs)

        if (self.eval_mode == "rtc" or process_action) and "action" in obs and len(obs["action"]) > 0:
            prev_action = obs["action"]  # (seq_len, action_dim)
            original_prev_action_size = prev_action.shape[-2]  # seq_len

            # pad action to chunk size if needed
            if prev_action.shape[-2] < self.horizon:
                action_dim = prev_action.shape[-1]
                pad_len = self.horizon - prev_action.shape[-2]
                padding = torch.zeros(
                    prev_action.shape[0],
                    pad_len,
                    action_dim,
                    device=self.device,
                )
                prev_action = torch.cat([prev_action, padding], dim=-2)
            obs["action"] = (
                prev_action.unsqueeze(0) if len(prev_action.shape) == 2 else prev_action
            )  # (1, chunk_size, action_dim)
        elif "action" in obs:
            obs.pop("action")  # remove prev action if not in RTC mode

        obs = self.process_obs_queue(obs)
        obs = self.input_transforms(obs)

        if (self.eval_mode == "rtc" or process_action) and "action" in obs:
            obs["action"] = obs["action"][:, :original_prev_action_size, :] if "action" in obs else None

        return obs

    def process_action(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Process policy action output before sending to environment.

        This method applies output transformations such as denormalization
        to the action tensor produced by the policy.

        Args:
            obs: A dictionary where keys are observation names and values are tensors
                 of shape (batch_size, 1, *obs_shape).
            action: A tensor representing the action output from the policy.

        Returns:
            A tensor representing the processed action ready for the environment.
        """
        obs["action"] = action
        return self.output_transforms(obs)["action"]

    def get_action_chunk(
        self,
        obs: dict[str, torch.Tensor],
        noise=None,
    ) -> torch.Tensor:
        """Get action chunk from policy given observations.

        Main entry point for policy inference. Processes observations,
        runs the policy, and returns denormalized actions.

        Args:
            obs: Dictionary of observations where:
                - State keys: torch.Tensor of shape (batch_size, num_obs, state_dim)
                - Image keys: torch.Tensor of shape (batch_size, num_obs, C, H, W)
                - Task: List of strings
                - Action (optional, for RTC mode): torch.Tensor of shape
                    (batch_size, chunk_size, action_dim)
                - num_actions_executed (optional, for RTC mode): int count of how
                    many actions from the previously predicted chunk the client
                    has already executed. The remaining (not-yet-executed)
                    actions are recovered by indexing into the internally stored
                    policy-space chunk and used as the RTC previous actions,
                    superseding any client-provided "action" tensor.
                - _reset_ (optional): bool-like flag. If True, clears internal
                    observation history buffers before processing this call.
            noise: Optional initial noise tensor. When provided,
                seeds the flow/diffusion policy's denoising process instead
                of sampling from N(0, I). Shape: (B, C, noise_dim) or a numpy
                array of the same shape.

        Returns:
            torch.Tensor: Denormalized action chunk of shape
                (batch_size, chunk_size, action_dim) ready for execution.
        """
        # Remap observation keys to policy-expected format and move to device
        # Keys in observation_mapping get remapped, others pass through as-is

        # Support episode boundary signaling from clients.
        # If reset is true, clear temporal history buffers before ingesting obs.
        reset_requested = obs.pop("_reset_", False)
        if isinstance(reset_requested, torch.Tensor):
            reset_requested = bool(reset_requested.any().item())
        elif isinstance(reset_requested, np.ndarray):
            reset_requested = bool(reset_requested.any())
        else:
            reset_requested = bool(reset_requested)
        if reset_requested:
            self.reset()

        # RTC: the client tells us how many actions from the *previously*
        # predicted chunk it has already executed. The still-pending
        # ("remaining") actions are recovered by indexing into the stored
        # absolute chunk (see ``self.prev_action_chunk``). They are then passed
        # through the input transforms with the current observation so RTC
        # guidance compares actions in one policy coordinate frame.
        num_actions_executed = obs.pop("num_actions_executed", None)
        if num_actions_executed is not None:
            num_actions_executed = _parse_num_actions_executed(num_actions_executed)
            if self.eval_mode == "rtc":
                cached_chunk_length = (
                    self.prev_action_chunk.shape[-2] if self.prev_action_chunk is not None else 0
                )
                if not 0 <= num_actions_executed <= cached_chunk_length:
                    raise ValueError(
                        "num_actions_executed must be between 0 and the cached action chunk length "
                        f"({cached_chunk_length}), got {num_actions_executed}."
                    )

        # Decide whether to source RTC previous actions from the stored absolute
        # chunk. Ignore client-provided actions and re-transform the stored
        # remainder against the current observation.
        use_stored_prev = (
            self.eval_mode == "rtc"
            and num_actions_executed is not None
            and self.prev_action_chunk is not None
        )
        remaining_absolute_actions = None
        if use_stored_prev:
            obs.pop("action", None)
            remaining_absolute_actions = self.prev_action_chunk[:, num_actions_executed:, :]
            if remaining_absolute_actions.shape[-2] > 0:
                obs["action"] = remaining_absolute_actions.detach().clone()

        # Move observation tensors to the same device as the policy
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                obs[key] = value.to(self.device)

        obs = self.process_observation(obs)

        # Convert the initial noise to a device tensor if provided.
        noise_tensor = None
        if noise is not None:
            if isinstance(noise, np.ndarray):
                noise_tensor = torch.from_numpy(noise).to(self.device)
            else:
                noise_tensor = noise.to(self.device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor.unsqueeze(0)

        remaining_actions = None
        if self.eval_mode == "standard":
            with torch.no_grad():
                action_chunk = self.policy.sample_actions(obs, noise=noise_tensor)
        elif self.eval_mode == "rtc":
            # Stored absolute actions, when present, have already been converted
            # into current-frame, normalized policy actions.
            remaining_actions = obs.get("action", None)

            action_chunk = self.policy.sample_actions_rtc(
                obs,
                inference_delay=self.inference_delay,
                prev_actions=remaining_actions,
                beta=self.beta,
                execution_horizon=self.execution_horizon,
                guidance_schedule=self.guidance_schedule,
                noise=noise_tensor,
            )
        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        action = self.process_action(obs, action_chunk["actions"])

        # Cache absolute actions. Re-running these through process_observation
        # on the next request aligns state-relative deltas and ACTIONCHUNK stats
        # with new chunk indices before RTC guidance.
        if self.eval_mode == "rtc":
            self.prev_action_chunk = action.detach().clone()

        return action


def main():
    """
    Test PolicyInterfaceConfig with a pretrained checkpoint and ground truth dataset.

    This function:
    1. Loads an EvalConfig from command-line args or defaults
    2. Creates a policy and dataset from the EvalConfig
    3. Runs policy inference on ground truth observations
    4. Plots predicted actions vs ground truth actions

    Usage:
        python -m rho.common.policy_interface --checkpoint <path_to_checkpoint>
    """
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    from rho.datasets.lerobot_dataset import LeRobotDataset
    from rho.eval.eval_config import EvalConfig
    from rho.policies import make_policy
    from rho.utils import init_logging

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test PolicyInterfaceConfig with ground truth data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint bundle, legacy .pt file, or checkpoints directory",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory for the ground truth LeRobot dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate from the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output plots (default: checkpoint_folder/policy_interface_test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory for the dataset (used to select dataset when training with multidataset)",
    )
    # RTC-specific arguments
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="standard",
        choices=["standard", "rtc"],
        help="Evaluation mode: 'standard' or 'rtc' (Real-Time Control)",
    )
    parser.add_argument(
        "--inference_delay",
        type=int,
        default=3,
        help="Number of steps policy inference takes (required for RTC mode)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="Weighting parameter for RTC update vs standard update (required for RTC mode)",
    )
    parser.add_argument(
        "--guidance_schedule",
        type=str,
        default="paper",
        choices=["paper", "constant"],
        help="Guidance coefficient schedule for RTC: 'paper' (Eq. 1/4) or 'constant' (always beta)",
    )

    args = parser.parse_args()

    # Initialize logging
    init_logging(console_level=args.log_level)

    logger.info("=" * 80)
    logger.info("PolicyInterfaceConfig Test with Ground Truth Data")
    logger.info("=" * 80)

    # Create EvalConfig from checkpoint
    logger.info(f"Loading EvalConfig from checkpoint: {args.checkpoint}")
    eval_cfg = EvalConfig(
        pretrained_checkpoint=args.checkpoint,
        log_level=args.log_level,
        dataset_root_dir=args.dataset_root_dir,
    )

    # Set up output directory
    if args.output_dir is None:
        output_dir = Path(eval_cfg.checkpoint_folder).parent / "policy_interface_test"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create policy from EvalConfig
    logger.info("Creating policy from config...")
    policy = make_policy(eval_cfg.policy)
    policy.load_from_pretrained(eval_cfg.pretrained_checkpoint)
    policy.eval()
    policy.to(args.device)
    logger.info(f"Policy loaded: {type(policy).__name__}")
    logger.info(f"  Device: {policy.device}")
    logger.info(f"  Chunk size: {eval_cfg.policy.chunk_size}")

    # Load ground truth dataset
    logger.info(f"Loading ground truth dataset from: {args.dataset_root}")
    dataset = LeRobotDataset(repo_id="test", root=args.dataset_root, video_backend="pyav")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    eval_cfg.dataset.observation_mapping["task"] = "task"  # ensure task key is mapped

    # Create PolicyInterfaceConfig
    logger.info("Creating PolicyInterfaceConfig...")
    logger.info(f"  Eval mode: {args.eval_mode}")
    if args.eval_mode == "rtc":
        logger.info(f"  Inference delay: {args.inference_delay}")
        logger.info(f"  Beta: {args.beta}")
        logger.info(f"  Guidance schedule: {args.guidance_schedule}")

    policy_interface_cfg = PolicyInterfaceConfig(
        data_config=eval_cfg.dataset,
        policy=policy,
        device=args.device,
        eval_mode=args.eval_mode,
        inference_delay=args.inference_delay if args.eval_mode == "rtc" else None,
        beta=args.beta if args.eval_mode == "rtc" else None,
        guidance_schedule=args.guidance_schedule,
        observation_mapping=eval_cfg.dataset.observation_mapping
        or {
            "observation.state": "observation.state",
            "observation.image.0": "observation.image.0",
            "task": "task",
            "action": "action",
        },
    )

    # Create PolicyInterface from config
    policy_interface = PolicyInterface(policy_interface_cfg)

    # Process first num_samples from the dataset
    num_samples = min(args.num_samples, len(dataset))
    logger.info(f"Evaluating on {num_samples} samples...")

    all_gt_actions = []
    all_pred_actions = []

    if args.eval_mode == "rtc":
        # RTC mode: predict chunks, execute from them, re-predict when inference_delay steps remain
        current_chunk = None  # Current action chunk being executed
        chunk_idx = 0  # Current position within the chunk
        chunk_size = policy.config.chunk_size

        for idx in range(num_samples):
            sample = dataset[idx]

            # Add batch dimension to observations
            obs = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim >= 3:
                        obs[key] = value.unsqueeze(0).to(args.device)
                    else:
                        obs[key] = value.unsqueeze(0).unsqueeze(0).to(args.device)
                else:
                    obs[key] = [value]

            # Get ground truth actions
            gt_actions = obs.get("action.joint_position")
            if gt_actions is not None:
                all_gt_actions.append(gt_actions.cpu())

            # Check if we need to predict a new chunk
            steps_remaining = chunk_size - chunk_idx if current_chunk is not None else 0
            need_new_chunk = current_chunk is None or steps_remaining <= args.inference_delay

            if need_new_chunk:
                # Pass remaining actions from current chunk for RTC blending
                # e.g., at step 24 of a 32-step chunk, pass actions[24:32]
                remaining_actions = None
                if current_chunk is not None and steps_remaining > 0:
                    if current_chunk.ndim == 3:
                        remaining_actions = current_chunk[
                            :, chunk_idx:, :
                        ]  # (1, steps_remaining, action_dim)
                    else:
                        remaining_actions = current_chunk
                    obs["action"] = remaining_actions
                    logger.debug(
                        f"  Step {idx}: Passing {steps_remaining} remaining actions for RTC blending"
                    )

                # Predict new chunk
                with torch.no_grad():
                    pred_output = policy_interface.get_action_chunk(obs)

                new_chunk = pred_output
                current_chunk = (
                    torch.cat((remaining_actions, new_chunk[:, steps_remaining:, :]), dim=1)
                    if remaining_actions is not None
                    else new_chunk
                )
                chunk_idx = 0
                logger.debug(f"  Step {idx}: Predicted new chunk")

            # Get action from current chunk
            if current_chunk is not None:
                if current_chunk.ndim == 3:
                    action = current_chunk[:, chunk_idx, :].cpu()
                else:
                    action = current_chunk.cpu()
                all_pred_actions.append(action)
                chunk_idx += 1

            if (idx + 1) % max(1, num_samples // 10) == 0:
                logger.info(f"  Processed {idx + 1}/{num_samples} samples")
    else:
        # Standard mode: predict at every timestep
        for idx in range(num_samples):
            sample = dataset[idx]

            # Add batch dimension to observations
            obs = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim >= 3:
                        obs[key] = value.unsqueeze(0).to(args.device)
                    else:
                        obs[key] = value.unsqueeze(0).unsqueeze(0).to(args.device)
                else:
                    obs[key] = [value]

            # Get ground truth actions
            gt_actions = obs.get("action.joint_position")
            if gt_actions is not None:
                all_gt_actions.append(gt_actions.cpu())

            # Run policy inference
            with torch.no_grad():
                pred_output = policy_interface.get_action_chunk(obs)

            if isinstance(pred_output, dict):
                pred_actions = pred_output.get("actions", pred_output.get("action"))
            else:
                pred_actions = pred_output

            if pred_actions is not None:
                # Take only the first action from the chunk for comparison
                if pred_actions.ndim == 3:
                    all_pred_actions.append(pred_actions[:, 0, :].cpu())
                else:
                    all_pred_actions.append(pred_actions.cpu())

            if (idx + 1) % max(1, num_samples // 10) == 0:
                logger.info(f"  Processed {idx + 1}/{num_samples} samples")

    # Stack all actions
    if all_gt_actions and all_pred_actions:
        gt_actions_stacked = torch.cat(all_gt_actions, dim=0)  # (N, chunk_size, action_dim)
        pred_actions_stacked = torch.cat(all_pred_actions, dim=0)  # (N, action_dim)

        # For GT, take the first action step to match predictions
        gt_actions_plot = gt_actions_stacked[:, 0, :] if gt_actions_stacked.ndim == 3 else gt_actions_stacked

        action_dim = gt_actions_plot.shape[-1]
        logger.info(f"Ground truth actions shape: {gt_actions_plot.shape}")
        logger.info(f"Predicted actions shape: {pred_actions_stacked.shape}")

        # Compute per-dimension MSE
        mse_per_dim = ((gt_actions_plot - pred_actions_stacked) ** 2).mean(dim=0)
        total_mse = mse_per_dim.mean().item()
        logger.info(f"Total MSE: {total_mse:.6f}")
        for d in range(action_dim):
            logger.info(f"  Dimension {d} MSE: {mse_per_dim[d].item():.6f}")

        # Plot actions vs ground truth
        fig, axes = plt.subplots(min(action_dim, 16), 1, figsize=(12, 3 * min(action_dim, 16)), squeeze=False)

        for d in range(min(action_dim, 16)):
            ax = axes[d, 0]
            ax.plot(gt_actions_plot[:, d].numpy(), label="Ground Truth", alpha=0.8, linewidth=2)
            ax.plot(pred_actions_stacked[:, d].numpy(), label="Predicted", alpha=0.8, linewidth=2)
            ax.set_ylabel(f"Dim {d}")
            ax.legend(loc="upper right")
            ax.set_title(f"Action Dimension {d} (MSE: {mse_per_dim[d].item():.4f})")
            ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("Sample Index")
        plt.suptitle(
            f"Predicted vs Ground Truth Actions\nTotal MSE: {total_mse:.6f}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        plot_path = output_dir / f"actions_comparison_{args.eval_mode}_{args.beta}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to: {plot_path}")
        plt.close()

        # Also create a scatter plot for each dimension
        num_scatter_dims = min(action_dim, 16)
        num_cols = 4
        num_rows = (num_scatter_dims + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows), squeeze=False)
        for d in range(num_scatter_dims):
            row, col = d // num_cols, d % num_cols
            ax = axes[row, col]
            ax.scatter(
                gt_actions_plot[:, d].numpy(),
                pred_actions_stacked[:, d].numpy(),
                alpha=0.5,
                s=20,
            )
            # Plot diagonal line
            lims = [
                min(gt_actions_plot[:, d].min().item(), pred_actions_stacked[:, d].min().item()),
                max(gt_actions_plot[:, d].max().item(), pred_actions_stacked[:, d].max().item()),
            ]
            ax.plot(lims, lims, "r--", alpha=0.8, label="Perfect Prediction")
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Dimension {d}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for d in range(num_scatter_dims, num_rows * num_cols):
            row, col = d // num_cols, d % num_cols
            axes[row, col].set_visible(False)

        plt.suptitle("Ground Truth vs Predicted Scatter Plots", fontsize=14, fontweight="bold")
        plt.tight_layout()

        scatter_path = output_dir / "actions_scatter.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        logger.info(f"Scatter plot saved to: {scatter_path}")
        plt.close()

        # Save metrics to file
        metrics_path = output_dir / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("PolicyInterfaceConfig Test Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Eval mode: {args.eval_mode}\n")
            if args.eval_mode == "rtc":
                f.write(f"Inference delay: {args.inference_delay}\n")
                f.write(f"Beta: {args.beta}\n")
            f.write("\n")
            f.write(f"Total MSE: {total_mse:.6f}\n")
            f.write("Per-dimension MSE:\n")
            for d in range(action_dim):
                f.write(f"  Dimension {d}: {mse_per_dim[d].item():.6f}\n")
        logger.info(f"Metrics saved to: {metrics_path}")

    else:
        logger.warning("No actions collected for comparison. Check dataset and policy configuration.")

    logger.info("=" * 80)
    logger.info("PolicyInterfaceConfig test completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

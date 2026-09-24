#!/usr/bin/env python3
"""
RoboEval Environment Configuration and Wrapper Classes.

This module provides specialized environment configurations and wrappers for the
RoboEval benchmark suite, extending the base environment classes to handle
RoboEval-specific observation spaces, actions, and rendering.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rho.environment import EnvironmentConfig, EnvironmentWrapper, register_environment

logger = logging.getLogger(__name__)

from roboeval.action_modes import JointPositionActionMode  # noqa: E402
from roboeval.demonstrations.utils import Metadata  # noqa: E402
from roboeval.envs.lift_pot import LiftPotPositionAndOrientation  # noqa: E402
from roboeval.envs.lift_tray import LiftTrayPositionAndOrientation  # noqa: E402
from roboeval.envs.manipulation import (  # noqa: E402
    CubeHandoverPositionAndOrientation,
    StackTwoBlocksPositionAndOrientation,
)
from roboeval.envs.pack_objects import PackBoxPositionAndOrientation  # noqa: E402
from roboeval.envs.rotate_utility_objects import RotateValvePositionAndOrientation  # noqa: E402
from roboeval.envs.stack_books import (  # noqa: E402
    PickSingleBookFromTablePositionAndOrientation,
    StackSingleBookShelfPositionAndOrientation,
)
from roboeval.robots.configs.panda import BimanualPanda  # noqa: E402
from roboeval.utils.observation_config import CameraConfig, ObservationConfig  # noqa: E402

# directly from task descriptions table from https://github.com/Robo-Eval/RoboEval
TASK_DESCRIPTIONS = {
    "lift_pot": "Grip the kitchen pot by its handles and raise it above the table.",
    "lift_tray": ("Grasp the breakfast tray with the two grippers and lift it clear of the source table."),
    "pack_box": (
        "Have each arm interact with the two-flap packing box"
        " and close both flaps until the opening is fully covered."
    ),
    "pick_single_book_from_table": "Grip the target book on the table and lift it up.",
    "rotate_valve": "Rotate each valve counterclockwise.",
    "stack_single_book_shelf": (
        "Pick up the book from the table and place it in contact with one of the shelves."
    ),
    "stack_two_blocks": "Manipulate two cubes placed on the table so that they are stacked.",
    "cube_handover": "Pass a cube between the robot's two arms.",
}

TASK_ENVS = {
    "lift_pot": LiftPotPositionAndOrientation,
    "lift_tray": LiftTrayPositionAndOrientation,
    "pack_box": PackBoxPositionAndOrientation,
    "pick_single_book_from_table": PickSingleBookFromTablePositionAndOrientation,
    "rotate_valve": RotateValvePositionAndOrientation,
    "stack_single_book_shelf": StackSingleBookShelfPositionAndOrientation,
    "stack_two_blocks": StackTwoBlocksPositionAndOrientation,
    "cube_handover": CubeHandoverPositionAndOrientation,
}


def flatten_dict(observation: dict) -> dict:
    """Flatten nested observation dictionary."""
    flattened = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            value = flatten_dict(value)
            for sub_key, sub_value in value.items():
                flattened[f"{key}.{sub_key}"] = sub_value
        else:
            flattened[key] = value
    return flattened


def convert_rpy_to_quat(rpy):
    """Convert roll-pitch-yaw angles to quaternion."""
    r = Rotation.from_euler("xyz", rpy)
    quat = r.as_quat()  # Returns in (x, y, z, w) format
    return quat


def convert_quat_to_rpy(quat):
    """Convert quaternion to roll-pitch-yaw angles."""
    r = Rotation.from_quat(quat)
    rpy = r.as_euler("xyz")
    return rpy


def convert_rpy_to_6d(rpy):
    """
    Convert xyz Euler angles to 6D rotation representation.
    6D = first two columns of rotation matrix concatenated.
    """
    rot_mat = Rotation.from_euler("xyz", rpy).as_matrix()
    return np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])


def convert_6d_to_rotation_matrix(d6):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Proper implementation following Zhou et al. 2019.
    """
    eps = 1e-8
    d6 = np.asarray(d6)

    a1 = d6[0:3]
    a2 = d6[3:6]

    # First basis vector
    b1 = a1 / (np.linalg.norm(a1) + eps)

    # Make second orthogonal to first
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + eps)

    # Third via cross product (guarantees right-handed)
    b3 = np.cross(b1, b2)

    rot_mat = np.stack([b1, b2, b3], axis=1)
    return rot_mat


def convert_6d_to_rpy(d6):
    """
    Convert 6D representation back to xyz Euler angles.
    """
    rot_mat = convert_6d_to_rotation_matrix(d6)
    return Rotation.from_matrix(rot_mat).as_euler("xyz")


def convert_bimanual_ee_rpy_to_quat(ee_rpy):
    """Convert bimanual EE RPY to separate 6D and quaternion representations."""
    left_rpy = ee_rpy[:6]  # x, y, z, roll, pitch, yaw, gripper for left arm
    right_rpy = ee_rpy[6:12]  # x, y, z, roll, pitch, yaw, gripper for right arm

    left_quat = convert_rpy_to_quat(left_rpy[3:6])  # Convert left arm RPY to quaternion
    right_quat = convert_rpy_to_quat(right_rpy[3:6])  # Convert right arm RPY to quaternion

    return np.concatenate((left_rpy[:3], left_quat, right_rpy[:3], right_quat))


def convert_bimanual_ee_rpy_to_6d(ee_rpy):
    """Convert bimanual EE RPY to separate 6D representations."""
    left_rpy = ee_rpy[:6]  # x, y, z, roll, pitch, yaw, gripper for left arm
    right_rpy = ee_rpy[6:12]  # x, y, z, roll, pitch, yaw, gripper for right arm

    left_6d = convert_rpy_to_6d(left_rpy[3:6])  # Convert left arm RPY to 6D
    right_6d = convert_rpy_to_6d(right_rpy[3:6])  # Convert right arm RPY to 6D

    return np.concatenate((left_rpy[:3], left_6d, right_rpy[:3], right_6d))


def convert_bimanual_ee_quat_to_rpy(ee_quat):
    """Convert bimanual EE quaternion to separate RPY representations."""
    left_quat = ee_quat[3:7]  # Quaternion for left arm
    right_quat = ee_quat[10:14]  # Quaternion for right arm

    left_rpy = convert_quat_to_rpy(left_quat)  # Convert left arm quaternion to RPY
    right_rpy = convert_quat_to_rpy(right_quat)  # Convert right arm quaternion to RPY

    return np.concatenate((ee_quat[:3], left_rpy, ee_quat[7:10], right_rpy))


def convert_bimanual_ee_6d_to_rpy(ee_6d):
    """Convert bimanual EE 6D to separate RPY representations."""
    left_6d = ee_6d[3:9]  # 6D for left arm
    right_6d = ee_6d[12:18]  # 6D for right arm

    left_rpy = convert_6d_to_rpy(left_6d)  # Convert left arm 6D to RPY
    right_rpy = convert_6d_to_rpy(right_6d)  # Convert right arm 6D to RPY

    return np.concatenate((ee_6d[:3], left_rpy, ee_6d[9:12], right_rpy))


def insert_gripper(values, gripper_values):
    """Insert gripper values into the appropriate positions in the array."""
    dim = len(values)
    return np.concatenate(
        (values[: dim // 2], [gripper_values[0]], values[dim // 2 : dim], [gripper_values[1]])
    )


@EnvironmentConfig.register_subclass("roboeval")
@dataclass
class RoboevalEnvConfig(EnvironmentConfig):
    """Configuration for RoboEval environments."""

    # Available task options: lift_pot, lift_tray, pack_box,
    # pick_single_book_from_table, rotate_valve,
    # stack_single_book_shelf, stack_two_blocks, cube_handover
    task_name: str = "lift_pot"

    # Override defaults for RoboEval
    name: str = "roboeval"
    max_episode_steps: int = 250

    action_space: str = "joint_pos"  # ee_6d_pos, ee_rpy_pos, ee_quat_pos
    seed: int = 12345
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # RoboEval observation mapping (env keys -> policy keys)
    observation_mapping: dict[str, str] | None = None

    # Default task description for the environment
    # Default to lift_pot, will be overridden in reset()
    default_task: str = TASK_DESCRIPTIONS["lift_pot"]


@register_environment("roboeval")
class RoboevalWrapper(EnvironmentWrapper):
    """
    Environment wrapper for RoboEval benchmark environments.

    This wrapper handles the specific requirements of RoboEval environments,
    including initialization with task configurations, proper observation processing,
    and rendering capabilities.

    All observations are returned as tensors with shape (batch, seq_len, **feature_size).
    Actions are accepted as tensors with shape (batch, action_dim).
    """

    def __init__(self, config: RoboevalEnvConfig):
        """
        Initialize the RoboEval environment wrapper.

        Args:
            config: RoboevalEnvConfig containing RoboEval-specific settings
        """
        super().__init__(config=config)
        self.config = config
        self.task_name = config.task_name
        self.action_space = config.action_space
        self.device = config.device

        # State tracking
        self.done = np.zeros(1, dtype=bool)
        self.total_reward = np.zeros(1, dtype=np.float64)
        self.last_frame: np.ndarray | None = None

        self.task_description = TASK_DESCRIPTIONS[self.task_name]
        self.sim = TASK_ENVS[self.task_name](
            action_mode=JointPositionActionMode(
                floating_base=True,
                absolute=True,
                floating_dofs=[],
                ee=self.action_space.startswith("ee"),
            ),
            render_mode="rgb_array",
            control_frequency=20,
            robot_cls=BimanualPanda,
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name="head", rgb=True, depth=False, resolution=(224, 224)),
                    CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(224, 224)),
                    CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=(224, 224)),
                    # CameraConfig(
                    #     name="external", rgb=True, depth=False,
                    #     resolution=(480, 640),
                    #     pos=np.array([2.0, 0, 1.5]),
                    #     quat=np.array([0.5610, 0.4305, 0.4305, 0.5610]),
                    # )
                ],
            ),
        )

        self.robot = Metadata.from_env(self.sim).get_robot()

    @property
    def is_vectorized(self) -> bool:
        """Return True if using vectorized environments."""
        return False  # RoboEval doesn't support vectorization

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return 1

    @property
    def n_envs(self) -> int:
        """Alias for num_envs for compatibility."""
        return 1

    def _process_output(self, action: torch.Tensor) -> np.ndarray:
        """Convert policy action tensor to environment numpy format.

        Note: Denormalization is handled by PolicyInterface, not here.

        Args:
            action: Action tensor from the policy, shape (batch, action_dim)

        Returns:
            Numpy array action for the environment
        """
        if isinstance(action, torch.Tensor):
            action = action.float().cpu().numpy()

        # Remove batch dimension if present
        if action.ndim == 2 and action.shape[0] == 1:
            action = action.squeeze(0)

        # for roboeval, action should be all arm commands and then gripper as opposed to training
        left = action[: action.shape[0] // 2]
        right = action[action.shape[0] // 2 :]
        action = np.concatenate([left[:-1], right[:-1], left[-1:], right[-1:]])

        if self.action_space.startswith("ee"):
            # For end-effector control, we may need to convert between the rotation types
            if self.action_space == "ee_quat_pos":
                rpy_action = convert_bimanual_ee_quat_to_rpy(
                    action[:-2]
                )  # Convert back to RPY for the environment
                action = np.concatenate((rpy_action, action[-2:]))  # Reinsert gripper values
            elif self.action_space == "ee_6d_pos":
                rpy_action = convert_bimanual_ee_6d_to_rpy(
                    action[:-2]
                )  # Convert back to RPY for the environment
                action = np.concatenate((rpy_action, action[-2:]))  # Reinsert gripper values

        return action.astype(np.float32)

    def _process_input(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw RoboEval observations to policy tensor format.

        Note: Normalization is handled by PolicyInterface, not here.

        Args:
            raw_obs: Raw observation dictionary from the environment

        Returns:
            Dict with torch tensors of shape (batch, seq_len, **feature_size):
                - Image keys: (batch=1, seq_len=1, C, H, W), values in [0, 1]
                - State keys: (batch=1, seq_len=1, feature_dim)
                - Task: List of strings
        """
        # Flatten nested observation structure
        processed_obs = {"observation": raw_obs}
        processed_obs = flatten_dict(processed_obs)

        left_joints = raw_obs["proprioception"][:7]
        right_joints = raw_obs["proprioception"][9:16]

        if self.action_space.startswith("ee"):
            ee_rpy = self.robot.forward_kinematics(np.concatenate((left_joints, right_joints))).flatten()
            if self.action_space == "ee_rpy_pos":
                processed_obs["observation.state"] = insert_gripper(
                    ee_rpy, raw_obs["proprioception_grippers"]
                )
            elif self.action_space == "ee_quat_pos":
                ee_quat = convert_bimanual_ee_rpy_to_quat(ee_rpy)
                processed_obs["observation.state"] = insert_gripper(
                    ee_quat, raw_obs["proprioception_grippers"]
                )
            elif self.action_space == "ee_6d_pos":
                ee_6d = convert_bimanual_ee_rpy_to_6d(ee_rpy)
                processed_obs["observation.state"] = insert_gripper(ee_6d, raw_obs["proprioception_grippers"])
        else:
            processed_obs["observation.state"] = insert_gripper(
                np.concatenate((left_joints, right_joints)), raw_obs["proprioception_grippers"]
            )

        # Apply observation mapping if configured
        if self.config.observation_mapping is not None:
            remapped_obs = {}
            for key, value in processed_obs.items():
                new_key = self.config.observation_mapping.get(key, key)
                remapped_obs[new_key] = value
            processed_obs = remapped_obs

        # Convert numpy arrays to tensors with proper shapes
        for key, value in list(processed_obs.items()):
            if not isinstance(value, np.ndarray):
                continue

            tensor_value = torch.from_numpy(value.copy()).float().to(self.device)

            if key.startswith("observation.image"):
                # Images: (H, W, C) -> (batch=1, seq_len=1, C, H, W)
                tensor_value = tensor_value / 255.0  # Normalize to [0, 1]
                if tensor_value.ndim == 3:
                    # (H, W, C) -> (C, H, W) -> (1, 1, C, H, W)
                    tensor_value = tensor_value.unsqueeze(0).unsqueeze(0)

                processed_obs[key] = tensor_value
            elif tensor_value.ndim == 1:
                # State: (dim,) -> (batch=1, seq_len=1, dim)
                processed_obs[key] = tensor_value.unsqueeze(0).unsqueeze(0)
            else:
                processed_obs[key] = tensor_value

        # Add task description
        processed_obs["task"] = [self.task_description]

        # Store frame for video recording
        self._store_video_frame(raw_obs)

        return processed_obs

    def _store_video_frame(self, obs: dict[str, Any]):
        """Store frame for video recording from observation."""
        # Try common image keys
        self.last_frame = obs["rgb_head"]
        # self.last_frame = obs["rgb_external"]

    def reset(self, seed: int | None = None) -> tuple:
        """Reset the environment.

        Returns observations as tensors with shape (batch, seq_len, **feature_size).

        Args:
            seed: Optional random seed

        Returns:
            Tuple of (obs, info) where obs has tensor values
        """
        # Reset state tracking
        self.done = np.zeros(1, dtype=bool)
        self.total_reward = np.zeros(1, dtype=np.float64)

        # Reset environment
        raw_obs, info = self.sim.reset()

        # Process observations to tensor format
        obs = self._process_input(raw_obs)

        return obs, info

    def step(self, action: torch.Tensor) -> tuple:
        """Take a step in the environment.

        Args:
            action: Action tensor of shape (batch, action_dim)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
            where obs has tensors with shape (batch, seq_len, **feature_size)
        """
        # Convert action from tensor to numpy
        action_np = self._process_output(action)

        # Step the environment
        raw_obs, reward, done, truncated, info = self.sim.step(action_np)

        # Process observations to tensor format
        obs = self._process_input(raw_obs)

        # Extract reward and done status
        reward = np.array([reward], dtype=np.float64)
        done = np.array([done], dtype=bool)  # Success if reward > 0

        # Update state tracking
        self.total_reward += reward
        self.done = self.done | done

        info.update({"is_success": done})

        return obs, reward, done, np.array([False]), info

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB image array of shape (H, W, 3), or last stored frame
        """
        return self.last_frame.transpose(1, 2, 0) if self.last_frame is not None else None

    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self, "sim") and self.sim is not None and hasattr(self.sim, "close"):
            self.sim.close()

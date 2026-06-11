import warnings

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# IK dependencies
try:
    from roboticstoolbox import DHRobot, RevoluteDH
    from spatialmath import SE3

    IK_AVAILABLE = True
except ImportError:
    IK_AVAILABLE = False
    warnings.warn(
        "Robotics Toolbox not available. IK functionality will be disabled. "
        "Install with: pip install roboticstoolbox-python spatialmath-python",
        stacklevel=2,
    )

# ===================== IK/FK Functions =====================

# Define UR5e robot model using DH parameters (found from https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)


def create_ur5e_model():
    """Create UR5e kinematic model using DH parameters"""
    # UR5e DH parameters (a, alpha, d, theta_offset)
    # These are approximate - you should verify with official UR5e specs
    links = [
        RevoluteDH(d=0.1625, a=0, alpha=np.pi / 2),  # Joint 1 (base)
        RevoluteDH(d=0, a=-0.425, alpha=0),  # Joint 2 (shoulder)
        RevoluteDH(d=0, a=-0.3922, alpha=0),  # Joint 3 (elbow)
        RevoluteDH(d=0.1333, a=0, alpha=np.pi / 2),  # Joint 4 (wrist1)
        RevoluteDH(d=0.0997, a=0, alpha=-np.pi / 2),  # Joint 5 (wrist2)
        RevoluteDH(d=0.0996, a=0, alpha=0),  # Joint 6 (wrist3)
    ]
    return DHRobot(links, name="UR5e")


def compute_eef_poses_for_dual_arm(robot_joint_positions):
    """
    Compute EEF poses from joint positions. Supports both single-arm (7D) and dual-arm (14D).

    Args:
        robot_joint_positions: Array of joint positions, shape (N, 7) for single arm
                               or (N, 14) for dual arm.

    Returns:
        Array of EEF poses:
          - Single arm: (N, 8) = [pose(7), gripper(1)]
          - Dual arm:   (N, 16) = [left_pose(7), left_gripper(1), right_pose(7), right_gripper(1)]
    """
    eef_poses = []

    for joint_pos in robot_joint_positions:
        if joint_pos.shape[-1] == 7:
            # Single arm: [6_joints, gripper]
            joints = joint_pos[:6]
            gripper = joint_pos[6]
            eef = joint_positions_to_eef_pose(joints)  # [x,y,z,roll,pitch,yaw]
            combined_eef = np.concatenate([eef, [gripper]])
        elif joint_pos.shape[-1] == 14:
            # Dual arm: [left_6_joints, left_gripper, right_6_joints, right_gripper]
            left_joints = joint_pos[:6]
            left_gripper = joint_pos[6]
            right_joints = joint_pos[7:13]
            right_gripper = joint_pos[13]

            left_eef = joint_positions_to_eef_pose(left_joints)
            right_eef = joint_positions_to_eef_pose(right_joints)

            combined_eef = np.concatenate([left_eef, [left_gripper], right_eef, [right_gripper]])
        else:
            raise ValueError(
                f"Unexpected joint_positions dim: {joint_pos.shape[-1]}."
                f" Expected 7 (single arm) or 14 (dual arm)."
            )

        eef_poses.append(combined_eef)

    return np.array(eef_poses)


def joint_positions_to_eef_pose(joint_positions):
    """
    Convert joint positions to end-effector pose

    Args:
        joint_positions: array of shape (6,) with joint angles in radians

    Returns:
        poses: array of shape (6,) containing [x, y, z, roll, pitch, yaw]
    """
    ur5e = create_ur5e_model()

    transform = ur5e.fkine(joint_positions)  # 4x4 transformation matrix
    position = transform.t  # translation vector [x, y, z]

    # Extract orientation as roll, pitch, yaw
    rpy = transform.rpy()

    position = np.array(position, dtype=np.float32)
    rpy = np.array(rpy, dtype=np.float32)

    return np.concatenate([position, rpy])


def joint_positions_to_eef_quat_pose(joint_positions):
    """Convert joint positions to end-effector pose with quaternion orientation.

    Extracts quaternion directly from the FK rotation matrix, avoiding
    the lossy joints → euler → quat path.

    Args:
        joint_positions: array of shape (6,) with joint angles in radians

    Returns:
        pose: array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
    """
    ur5e = create_ur5e_model()

    transform = ur5e.fkine(joint_positions)  # 4x4 transformation matrix
    position = np.array(transform.t, dtype=np.float32)  # [x, y, z]
    quat = Rotation.from_matrix(transform.R).as_quat()  # [qx, qy, qz, qw]
    quat = np.array(quat, dtype=np.float32)

    return np.concatenate([position, quat])


def compute_eef_quat_poses(robot_joint_positions):
    """Compute EEF poses with quaternion orientation directly from joint positions.

    Uses rotation matrix → quaternion directly, avoiding euler intermediate.
    Supports both single-arm (7D) and dual-arm (14D).

    Args:
        robot_joint_positions: Array of joint positions, shape (N, 7) for single arm
                               or (N, 14) for dual arm.

    Returns:
        Array of EEF poses:
          - Single arm: (N, 8) = [x,y,z,qx,qy,qz,qw, gripper]
          - Dual arm:   (N, 16) = [left(8), right(8)]
    """
    eef_poses = []

    for joint_pos in robot_joint_positions:
        if joint_pos.shape[-1] == 7:
            # Single arm: [6_joints, gripper]
            joints = joint_pos[:6]
            gripper = joint_pos[6]
            eef = joint_positions_to_eef_quat_pose(joints)  # [x,y,z,qx,qy,qz,qw]
            combined_eef = np.concatenate([eef, [gripper]])
        elif joint_pos.shape[-1] == 14:
            # Dual arm
            left_joints = joint_pos[:6]
            left_gripper = joint_pos[6]
            right_joints = joint_pos[7:13]
            right_gripper = joint_pos[13]

            left_eef = joint_positions_to_eef_quat_pose(left_joints)
            right_eef = joint_positions_to_eef_quat_pose(right_joints)

            combined_eef = np.concatenate([left_eef, [left_gripper], right_eef, [right_gripper]])
        else:
            raise ValueError(
                f"Unexpected joint_positions dim: {joint_pos.shape[-1]}."
                f" Expected 7 (single arm) or 14 (dual arm)."
            )

        eef_poses.append(combined_eef)

    return np.array(eef_poses)


def eef_pose_to_joint_positions(eef_pose, current_joints=None, method="LM"):
    # TODO: IK not fully working, error of ~0.15 radians on orientation
    """
    Convert end-effector pose to joint positions using inverse kinematics

    Args:
        eef_pose: array of shape (6,) containing [x, y, z, roll, pitch, yaw]
        current_joints: array of shape (6,) with current joint angles for warm start (optional)
        method: IK method to use ('LM' for Levenberg-Marquardt, 'NR' for Newton-Raphson)

    Returns:
        joint_angles: array of shape (6,) with joint angles in radians, or None if IK failed
    """
    if not IK_AVAILABLE:
        raise ImportError("Robotics Toolbox not available")

    ur5e = create_ur5e_model()

    # Parse input pose
    # Input is [x, y, z, roll, pitch, yaw]
    position = eef_pose[:3]
    rpy = eef_pose[3:6]
    transform = SE3.Trans(position) * SE3.RPY(rpy, order="xyz")

    # Set up IK solver parameters
    ik_kwargs = {
        "mask": [1, 1, 1, 1, 1, 1],  # [x, y, z, rx, ry, rz] - all enabled
        "ilimit": 1000,  # iteration limit
        "tol": 1e-6,  # tolerance
    }

    # Add initial guess if provided
    if current_joints is not None:
        ik_kwargs["q0"] = current_joints

    try:
        # Perform inverse kinematics
        if method == "LM":
            sol = ur5e.ikine_LM(transform, **ik_kwargs)
        elif method == "NR":
            sol = ur5e.ikine_NR(transform, **ik_kwargs)
        else:
            raise ValueError(f"Unknown IK method: {method}")

        if sol.success:
            return np.array(sol.q, dtype=np.float32)
        else:
            warnings.warn(f"IK failed to converge. Residual: {sol.residual}", stacklevel=2)
            return None

    except Exception as e:
        warnings.warn(f"IK solver encountered an error: {e}", stacklevel=2)
        return None


def transform_delta_joints_to_absolute_joints(action, joint_positions):
    """
    Transform delta joint actions to absolute joint positions.

    Args:
        action: tensor containing delta joint positions (1, action_chunk_size, 14)
        joint_positions: tensor containing current joint positions (1, 1, 14)

    Returns:
        absolute_joint_positions: tensor containing absolute joint positions
    """
    cumulative_action = torch.cumsum(action, dim=-2 if action.ndim > 2 else -1)
    absolute_positions = joint_positions + cumulative_action
    return absolute_positions


def transform_absolute_ee_pose_to_absolute_joints(action, joint_positions):
    """
    Transform absolute end effector poses to absolute joint positions.

    Args:
        action: tensor containing absolute end effector poses (1, action_chunk_size, 14)
        joint_positions: tensor containing current joint positions (1, 1, 14)

    Returns:
        absolute_joint_positions: tensor containing absolute joint positions
    """

    action_seq = action.squeeze(0)  # (action_chunk_size, 14)
    current_joints = joint_positions.squeeze(0).squeeze(0)  # (14,)

    absolute_joints = torch.zeros_like(action_seq)

    prev_joints = current_joints.cpu().numpy()

    # iterate through each timestep in the action sequence and solve IK using previous joints as warm start
    for i in range(action_seq.shape[0]):
        target_pose = action_seq[i].cpu().numpy()  # (14,)

        # Solve IK for left arm (first 6 DOF)
        joint_states_left = eef_pose_to_joint_positions(target_pose[:6], prev_joints[:6])

        # Solve IK for right arm (DOF 7-12, skipping gripper at index 6)
        joint_states_right = eef_pose_to_joint_positions(target_pose[7:13], prev_joints[7:13])

        if joint_states_left is None or joint_states_right is None:
            raise RuntimeError(f"Inverse kinematics failed at timestep {i}")

        # Combine results: [left_arm_joints, left_gripper, right_arm_joints, right_gripper]
        joint_result = np.concatenate(
            [
                joint_states_left,  # 6 left arm joints
                [target_pose[6]],  # left gripper
                joint_states_right,  # 6 right arm joints
                [target_pose[13]],  # right gripper
            ]
        )

        # Store result
        absolute_joints[i] = torch.tensor(joint_result, dtype=torch.float32)

        # Use this solution as starting point for next iteration
        prev_joints = joint_result

    return absolute_joints.to(action.device)


def transform_delta_ee_pose_to_absolute_joints(action, joint_positions):
    """
    Transform delta end effector poses to absolute joint positions

    Args:
        action: tensor containing delta ee poses (1, action_chunk_size, 14)
        joint_positions: tensor containing current joint positions

    Returns:
        absolute_joint_positions: tensor containing absolute joint positions
    """
    if not IK_AVAILABLE:
        raise ImportError("Robotics Toolbox not available")

    # Get current end effector poses using forward kinematics
    joint_positions = joint_positions.squeeze(0).squeeze(0)  # (14,)
    current_eef_left = joint_positions_to_eef_pose(joint_positions[:6].cpu().numpy())
    current_eef_right = joint_positions_to_eef_pose(joint_positions[7:13].cpu().numpy())
    current_eef_pose = torch.tensor(
        np.concatenate(
            [current_eef_left, [joint_positions[6].item()], current_eef_right, [joint_positions[13].item()]]
        ),
        dtype=torch.float32,
        device=action.device,
    )

    # Compute cumulative delta poses and add to current pose
    cumulative_deltas = torch.cumsum(action, dim=-2 if action.ndim > 2 else -1)
    absolute_poses = current_eef_pose + cumulative_deltas  # (1, 8, 14)

    return transform_absolute_ee_pose_to_absolute_joints(
        absolute_poses, joint_positions.unsqueeze(0).unsqueeze(0)
    )

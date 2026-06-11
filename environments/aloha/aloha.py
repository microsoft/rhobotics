import logging
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np
import pinocchio as pin
import torch

from rho.common.rotation_helpers import matrix_to_rotation_6d_np, rotation_6d_to_matrix_np
from rho.common.types import ActionType
from rho.environment import register_environment
from rho.environment.env import EnvironmentConfig
from rho.server.serve_policy import Server

logger = logging.getLogger(__name__)


@EnvironmentConfig.register_subclass("aloha_server")
@dataclass
class AlohaServerConfig(EnvironmentConfig):
    """Configuration for AlohaServer environment."""

    name: str = "aloha_server"
    port: int = 5000
    device: str = "cuda"

    input_action_type: ActionType = ActionType.POSITION  # "joint_position" or "ee_quat_pos" or "ee_rpy_pos"
    output_action_type: ActionType = ActionType.POSITION  # "joint_position" or "ee_quat_pos" or "ee_rpy_pos"

    # this will be set by the training data config action type
    policy_action_type: ActionType = ActionType.POSITION  # "joint_position" or "ee_quat_pos" or "ee_rpy_pos"


@register_environment("aloha_server")
class AlohaServer(Server):
    def __init__(self, config: AlohaServerConfig) -> None:
        super().__init__(config)

        self.device = config.device

        self.image_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

        self.input_action_type = config.input_action_type
        self.output_action_type = config.output_action_type
        self.policy_action_type = config.policy_action_type

        if self.policy_action_type == ActionType.EE_EULER_POS:
            import warnings

            warnings.warn(
                "EE_EULER_POS action type is deprecated and will be removed in a future version. "
                "Use EE_6D_POS for gimbal-lock-free rotation representation. "
                "See: rho/common/rotation_helpers.py for details.",
                FutureWarning,
                stacklevel=2,
            )

        # Pinocchio IK setup (used when policy_action_type is not POSITION and output_action_type is POSITION)
        self._pin_model = None
        self._pin_data = None
        self._pin_ee_frame_id = None
        self._ik_seed_left = None
        self._ik_seed_right = None
        self._ik_urdf_path = Path(__file__).resolve().parent / "aloha_vx300s.urdf"
        self.use_pinocchio_ik = False
        self._last_joint_position = None  # used for action chunk safety check
        self._joint_delta_limit = np.deg2rad(30.0)  # max joint change per step
        logger.info(
            "AlohaServer initialized with input_action_type=%s, policy_action_type=%s, output_action_type=%s",
            self.input_action_type,
            self.policy_action_type,
            self.output_action_type,
        )
        if self.policy_action_type != ActionType.POSITION and self.output_action_type == ActionType.POSITION:
            self.use_pinocchio_ik = True
        if self.use_pinocchio_ik:
            self._ik_seed_left = np.zeros(6)
            self._ik_seed_right = np.zeros(6)
            self._init_pinocchio_ik()

    def _init_pinocchio_ik(self) -> None:
        if not self._ik_urdf_path.exists():
            return
        self._pin_model = pin.buildModelFromUrdf(str(self._ik_urdf_path))
        self._pin_data = self._pin_model.createData()
        self._pin_ee_frame_id = self._pin_model.getFrameId("gripper_prop_link")

    def _solve_ik_se3(
        self,
        oMt: "pin.SE3",  # noqa: N803
        q_seed: np.ndarray,
        q_ref: np.ndarray | None = None,
        max_iters: int = 200,
        tol: float = 1e-12,
        damping: float = 1e-3,
        joint_reg_weight: float = 0.05,
        reg_decay_threshold: float = 1e-4,
        max_joint_step: float = 0.1,
    ) -> np.ndarray:
        """Damped-least-squares IK with adaptive joint-space regularization.

        Each iteration solves the regularized normal equations:

            (J^T J + (λ² + α²) I) dq = J^T e + α² (q_ref - q)

        where e is the SE(3) log-space error between the current and target
        end-effector pose, λ is the damping factor, and α is the joint
        regularization weight.

        The joint regularization biases the solution toward *q_ref*,
        preventing ~180° flips on forearm_roll / wrist_rotate that pure
        DLS produces.  To avoid the regularizer fighting final task-space
        convergence, α² is decayed linearly to zero as the error norm
        drops below *reg_decay_threshold*.

        The step dq is *scaled* (not element-wise clipped) when any
        component exceeds *max_joint_step*, preserving the descent
        direction.  Joint limits are enforced by clamping q after each
        update.

        Args:
            oMt: Desired end-effector pose as a pinocchio SE3 object.
            q_seed: Warm-start joint configuration (typically the
                previous IK solution or current robot state).
            q_ref: Reference configuration the regularizer pulls toward.
                Defaults to *q_seed* if None.  For action chunks, pass
                the previous step's IK result.
            max_iters: Maximum number of solver iterations.
            tol: Convergence tolerance on the SE(3) log-space error norm.
            damping: λ — Levenberg-Marquardt-style damping.  Prevents
                ill-conditioning near kinematic singularities.
            joint_reg_weight: α — weight of the joint-space regularizer.
                Higher values keep joints closer to q_ref at the cost of
                slower task-space convergence.  0 disables the bias.
            reg_decay_threshold: Error norm below which the joint
                regularizer begins to fade linearly to zero.  Above this
                threshold the regularizer is at full strength.
            max_joint_step: Maximum absolute joint change per iteration
                (radians).  The entire dq vector is scaled down when any
                component exceeds this limit.

        Returns:
            q: Joint configuration that achieves the target pose (within
                tolerance), or the best approximation if max_iters is
                reached.
        """
        if self._pin_model is None or self._pin_data is None or self._pin_ee_frame_id is None:
            raise RuntimeError("Pinocchio IK not initialized. Check URDF path and pinocchio install.")

        q = np.asarray(q_seed, dtype=np.float64).reshape(-1).copy()
        nq = self._pin_model.nq
        if q.shape[0] != nq:
            raise RuntimeError(f"IK seed has size {q.shape[0]} but model expects nq={nq}")

        q_ref = q.copy() if q_ref is None else np.asarray(q_ref, dtype=np.float64).reshape(-1).copy()

        alpha2 = joint_reg_weight**2
        lambda2 = damping**2
        I_nq = np.eye(nq)  # noqa: N806
        lower = self._pin_model.lowerPositionLimit
        upper = self._pin_model.upperPositionLimit

        i = 0
        for _ in range(max_iters):
            pin.forwardKinematics(self._pin_model, self._pin_data, q)
            pin.updateFramePlacements(self._pin_model, self._pin_data)

            oMf = self._pin_data.oMf[self._pin_ee_frame_id]  # noqa: N806
            err = pin.log(oMf.inverse() * oMt).vector
            err_norm = np.linalg.norm(err)

            if err_norm < tol:
                break

            # Decay joint regularization as error shrinks so it does not
            # fight final task-space convergence.  Full strength above
            # err, fades linearly to zero.
            reg_scale = min(1.0, err_norm / reg_decay_threshold)
            cur_alpha2 = alpha2 * reg_scale

            J = pin.computeFrameJacobian(  # noqa: N806
                self._pin_model, self._pin_data, q, self._pin_ee_frame_id, pin.LOCAL
            )

            lhs = J.T @ J + (lambda2 + cur_alpha2) * I_nq
            rhs = J.T @ err + cur_alpha2 * (q_ref - q)
            dq = np.linalg.solve(lhs, rhs)

            # Scale (not clip) dq to preserve step direction
            if max_joint_step > 0:
                max_abs = np.max(np.abs(dq))
                if max_abs > max_joint_step:
                    dq *= max_joint_step / max_abs

            q = q + dq
            q = np.clip(q, lower, upper)
            i += 1

        logger.debug(f"IK solved in {i} iterations with final error norm {err_norm:.6e}")

        return q

    def _solve_ik_dls(
        self,
        target_xyzrpy: np.ndarray,
        q_seed: np.ndarray,
        q_ref: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Thin wrapper around _solve_ik_se3 for RPY targets.

        Converts [x, y, z, roll, pitch, yaw] to a pinocchio SE3 object
        and delegates to the core solver.
        """
        target_pos = target_xyzrpy[:3]
        target_rpy = np.asarray(target_xyzrpy[3:], dtype=np.float64)
        target_rot = pin.rpy.rpyToMatrix(target_rpy[0], target_rpy[1], target_rpy[2])
        oMt = pin.SE3(target_rot, target_pos)  # noqa: N806
        return self._solve_ik_se3(oMt, q_seed, q_ref=q_ref, **kwargs)

    def _solve_ik_6d(
        self,
        target_xyz_rot6d: np.ndarray,
        q_seed: np.ndarray,
        q_ref: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Thin wrapper around _solve_ik_se3 for 6D rotation targets.

        Converts [x, y, z, rot6d(6)] to a pinocchio SE3 object
        and delegates to the core solver.
        """
        target_pos = target_xyz_rot6d[:3]
        rot6d = target_xyz_rot6d[3:9]
        target_rot = rotation_6d_to_matrix_np(rot6d)
        oMt = pin.SE3(target_rot, target_pos)  # noqa: N806
        return self._solve_ik_se3(oMt, q_seed, q_ref=q_ref, **kwargs)

    def process_input(self, input) -> dict:
        """should return observation dict"""
        """obs['action'] should be what the client sent as previous action"""
        """sequence length of each observation tensor should be the history that the client sends"""
        logger.debug("Processing input observation")
        obs = self.dict_to_torch(input, self.device)

        raw_joint_position = None
        if self.use_pinocchio_ik and "joint_position" in obs:
            raw_joint_position = obs["joint_position"]

        jp_np = (
            raw_joint_position.detach().cpu().numpy()
            if isinstance(raw_joint_position, torch.Tensor)
            else np.asarray(raw_joint_position)
        )
        jp_flat = jp_np.reshape(-1)
        logger.debug("raw_joint_position - \nleft_arm:  %s \nright_arm: %s", jp_flat[:7], jp_flat[7:14])
        obs = self.convert_observation_state_type_to_policy_type(obs)
        if self.policy_action_type == ActionType.EE_EULER_POS:
            ee_state = (
                obs["joint_position"].detach().cpu().numpy()
                if isinstance(obs["joint_position"], torch.Tensor)
                else np.asarray(obs["joint_position"])
            )
            ee_flat = ee_state.reshape(-1)
            logger.debug("ee_euler_pos - \nleft_arm:  %s \nright_arm: %s", ee_flat[:7], ee_flat[7:14])
        elif self.policy_action_type == ActionType.EE_6D_POS:
            ee_state = (
                obs["joint_position"].detach().cpu().numpy()
                if isinstance(obs["joint_position"], torch.Tensor)
                else np.asarray(obs["joint_position"])
            )
            ee_flat = ee_state.reshape(-1)
            logger.debug("ee_6d_pos - \nleft_arm:  %s \nright_arm: %s", ee_flat[:10], ee_flat[10:20])

        for key in obs:
            if key in self.image_keys:
                # TODO[dean]: need to confirm image input from aloha is in BGR (I think it is)
                img = obs[key][:, :, [2, 1, 0]]  # BGR to RGB (180,320,3)
                img = img.permute(2, 0, 1) / 255.0  # to (3,180,320) and scale to [0,1]
                obs[key] = img.unsqueeze(0)  # add batch dimension

            # TODO[dean]: need to confirm adding batch and sequence
            # dimensions is correct for aloha state inputs
            elif isinstance(obs[key], torch.Tensor):
                if obs[key].dtype == torch.float64:
                    obs[key] = obs[key].float()
                if len(obs[key].shape) == 2:
                    obs[key] = obs[key].unsqueeze(0)  # add batch dimension
                elif len(obs[key].shape) == 1:
                    obs[key] = obs[key].unsqueeze(0).unsqueeze(0)  # add batch and sequence dimensions

        if self.use_pinocchio_ik:
            if raw_joint_position is None:
                raise RuntimeError("IK seeds require joint_position in input observations.")
            if isinstance(raw_joint_position, torch.Tensor):
                jp_np = raw_joint_position.detach().cpu().numpy()
            else:
                jp_np = np.asarray(raw_joint_position)
            jp_flat = jp_np.reshape(-1)
            if jp_flat.shape[0] != 14:
                raise RuntimeError(
                    f"Expected joint_position with 14 values for IK seed, got shape {jp_np.shape}"
                )
            self._ik_seed_left = jp_flat[:6].copy()
            self._ik_seed_right = jp_flat[7:13].copy()

        # Save current joint position for action chunk safety check
        self._last_joint_position = jp_flat.copy()

        return obs

    def process_output(self, actions):
        # Remove batch dimension if present and convert to numpy
        if actions.ndim > 2 and actions.shape[0] == 1:
            actions = actions.squeeze(0)

        actions = actions.cpu().to(torch.float32).numpy()

        if self.policy_action_type != self.output_action_type:
            first_action = actions[0] if actions.ndim > 1 else actions
            logger.debug(
                "pre-conversion action (first step) - \nleft_arm:  %s \nright_arm: %s",
                first_action[:7],
                first_action[7:14],
            )

        t_start = time()
        actions = self.convert_policy_action_type_to_client_type(actions)
        t_end = time()
        logger.debug("Action conversion took %.6f seconds", t_end - t_start)

        if self.policy_action_type != self.output_action_type:
            first_action = actions[0] if actions.ndim > 1 else actions
            logger.debug(
                "post-conversion action (first step) - \nleft_arm:  %s \nright_arm: %s",
                first_action[:7],
                first_action[7:14],
            )

        # Safety check: ensure no joint jumps more than 15 degrees between consecutive steps
        self._validate_action_chunk(actions)

        return actions

    def _validate_action_chunk(self, actions: np.ndarray) -> None:
        """Check that no joint moves more than _joint_delta_limit rad between
        consecutive action steps. The first step is compared against the current
        observation stored in _last_joint_position."""
        if actions.ndim != 2 or actions.shape[1] != 14:
            return  # nothing to validate

        # Joint indices only (exclude grippers at 6 and 13)
        joint_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

        prev = self._last_joint_position  # (14,) from process_input or None
        for i in range(actions.shape[0]):
            step = actions[i]
            if prev is not None and len(prev) == len(step):
                for j in joint_indices:
                    delta = abs(step[j] - prev[j])
                    if delta > self._joint_delta_limit:
                        raise RuntimeError(
                            f"Action chunk safety violation: joint index {j} jumps "
                            f"{np.degrees(delta):.2f}° (limit {np.degrees(self._joint_delta_limit):.2f}°) "
                            f"at action step {i}. "
                            f"prev={prev[j]:.6f} rad, next={step[j]:.6f} rad. "
                            f"Full prev: {prev}\nFull step: {step}"
                        )
            prev = step

    def convert_observation_state_type_to_policy_type(self, obs) -> dict:
        """
        convert_observation_state_type_to_policy_type converts the observation ActionType
        sent by the client to the ActionType expected by the policy. For example, if the client sends
        observations in the "joint_position" format and the policy expects "ee_quat_pos" format,
        this function will convert the joint positions to end-effector quaternion positions.

        :param obs: The observation dictionary containing the state information from the client.
        """
        if self.input_action_type != ActionType.POSITION:
            raise NotImplementedError(
                f"Conversion from input action type: {self.input_action_type}"
                " to policy action type not implemented"
            )
        if self.policy_action_type not in [
            ActionType.POSITION,
            ActionType.EE_EULER_POS,
            ActionType.EE_6D_POS,
        ]:
            raise NotImplementedError(
                f"Conversion from input action type: {self.input_action_type}"
                f" to policy action type: {self.policy_action_type} not implemented"
            )

        if self.input_action_type == self.policy_action_type:
            return obs
        elif (
            self.input_action_type == ActionType.POSITION
            and self.policy_action_type == ActionType.EE_EULER_POS
        ):
            logger.debug("Converting input joint positions to end-effector euler positions for policy...")
            return self.convert_joint_position_to_ee_euler_pos(obs)
        elif (
            self.input_action_type == ActionType.POSITION and self.policy_action_type == ActionType.EE_6D_POS
        ):
            logger.debug("Converting input joint positions to end-effector 6D positions for policy...")
            return self.convert_joint_position_to_ee_6d_pos(obs)
        else:
            raise NotImplementedError(
                f"Conversion from input action type: {self.input_action_type}"
                f" to policy action type: {self.policy_action_type} not implemented"
            )

    def convert_policy_action_type_to_client_type(self, actions) -> np.ndarray:
        """
        Convert the policy output ActionType to the ActionType expected by the client.

        For example, if the policy outputs actions in "ee_quat_pos" format but the client expects
        "joint_position" format, this function will convert the EE quaternion positions to joint positions.

        :param actions: The action tensor output by the policy.
        """
        if self.policy_action_type not in [
            ActionType.POSITION,
            ActionType.EE_EULER_POS,
            ActionType.EE_6D_POS,
        ]:
            raise NotImplementedError(
                f"Conversion from policy action type: {self.policy_action_type}"
                " to client action type not implemented"
            )
        if self.output_action_type != ActionType.POSITION:
            raise NotImplementedError(
                f"Conversion from policy action type: {self.policy_action_type}"
                f" to client action type: {self.output_action_type} not implemented"
            )

        if self.policy_action_type == self.output_action_type:
            return actions
        elif (
            self.policy_action_type == ActionType.EE_EULER_POS
            and self.output_action_type == ActionType.POSITION
        ):
            logger.debug(
                "Converting policy output end-effector euler positions to joint positions for client..."
            )
            return self.convert_ee_euler_pos_to_joint_position(actions)
        elif (
            self.policy_action_type == ActionType.EE_6D_POS and self.output_action_type == ActionType.POSITION
        ):
            logger.debug(
                "Converting policy output end-effector 6D positions to joint positions for client..."
            )
            return self.convert_ee_6d_pos_to_joint_position(actions)
        else:
            raise NotImplementedError(
                f"Conversion from policy action type: {self.policy_action_type}"
                f" to client action type: {self.output_action_type} not implemented"
            )

    def convert_joint_position_to_ee_euler_pos(self, obs) -> dict:
        """Convert joint_position (14,) to ee_euler_pos for both arms.

        Input:  obs["joint_position"] – tensor of shape (1, 14)
                Layout: [left_arm(0:6), left_gripper(6), right_arm(7:13), right_gripper(13)]

        Output: obs["joint_position"] replaced with tensor of shape (1, 14)
                Layout per arm: [x, y, z, roll, pitch, yaw, gripper]
        """
        joint_pos = obs["joint_position"]  # (1, 14)

        # Left arm: joints 0-5, gripper at index 6
        left_joints = joint_pos[0, :6].cpu().numpy().astype(np.float64)
        left_gripper = joint_pos[0, 6]

        # Right arm: joints 7-12, gripper at index 13
        right_joints = joint_pos[0, 7:13].cpu().numpy().astype(np.float64)
        right_gripper = joint_pos[0, 13]

        if self._pin_model is None or self._pin_data is None or self._pin_ee_frame_id is None:
            raise RuntimeError("Pinocchio FK not initialized. Check URDF path and pinocchio install.")

        # Forward kinematics → (x, y, z) + rotation matrix (Pinocchio)
        pin.forwardKinematics(self._pin_model, self._pin_data, left_joints)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        left_oMf = self._pin_data.oMf[self._pin_ee_frame_id]  # noqa: N806
        left_pos = left_oMf.translation.copy()
        # left_rpy = ScipyRotation.from_matrix(left_oMf.rotation).as_euler('XYZ')
        left_rpy = pin.rpy.matrixToRpy(left_oMf.rotation).copy()

        pin.forwardKinematics(self._pin_model, self._pin_data, right_joints)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        right_oMf = self._pin_data.oMf[self._pin_ee_frame_id]  # noqa: N806
        right_pos = right_oMf.translation.copy()
        # right_rpy = ScipyRotation.from_matrix(right_oMf.rotation).as_euler('XYZ')
        right_rpy = pin.rpy.matrixToRpy(right_oMf.rotation).copy()

        # Assemble: [x, y, z, roll, pitch, yaw, gripper] per arm
        left_ee = np.array([*left_pos, *left_rpy, left_gripper.cpu().numpy()])
        right_ee = np.array([*right_pos, *right_rpy, right_gripper.cpu().numpy()])

        ee_state = np.concatenate([left_ee, right_ee])  # (14,)
        obs["joint_position"] = (
            torch.from_numpy(ee_state).to(dtype=joint_pos.dtype, device=joint_pos.device).unsqueeze(0)
        )  # (1, 14)

        return obs

    def convert_joint_position_to_ee_6d_pos(self, obs) -> dict:
        """Convert joint_position (14,) to ee_6d_pos for both arms.

        Input:  obs["joint_position"] – tensor of shape (1, 14)
                Layout: [left_arm(0:6), left_gripper(6), right_arm(7:13), right_gripper(13)]

        Output: obs["joint_position"] replaced with tensor of shape (1, 20)
                Layout per arm: [x, y, z, rot6d(6), gripper]
        """
        joint_pos = obs["joint_position"]  # (1, 14)

        # Left arm: joints 0-5, gripper at index 6
        left_joints = joint_pos[0, :6].cpu().numpy().astype(np.float64)
        left_gripper = joint_pos[0, 6]

        # Right arm: joints 7-12, gripper at index 13
        right_joints = joint_pos[0, 7:13].cpu().numpy().astype(np.float64)
        right_gripper = joint_pos[0, 13]

        if self._pin_model is None or self._pin_data is None or self._pin_ee_frame_id is None:
            raise RuntimeError("Pinocchio FK not initialized. Check URDF path and pinocchio install.")

        # Forward kinematics → (x, y, z) + rotation matrix (Pinocchio)
        pin.forwardKinematics(self._pin_model, self._pin_data, left_joints)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        left_oMf = self._pin_data.oMf[self._pin_ee_frame_id]  # noqa: N806
        left_pos = left_oMf.translation.copy()
        left_rot6d = matrix_to_rotation_6d_np(left_oMf.rotation)

        pin.forwardKinematics(self._pin_model, self._pin_data, right_joints)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        right_oMf = self._pin_data.oMf[self._pin_ee_frame_id]  # noqa: N806
        right_pos = right_oMf.translation.copy()
        right_rot6d = matrix_to_rotation_6d_np(right_oMf.rotation)

        # Assemble: [x, y, z, rot6d(6), gripper] per arm
        left_ee = np.array([*left_pos, *left_rot6d, left_gripper.cpu().numpy()])
        right_ee = np.array([*right_pos, *right_rot6d, right_gripper.cpu().numpy()])

        ee_state = np.concatenate([left_ee, right_ee])  # (20,)
        obs["joint_position"] = (
            torch.from_numpy(ee_state).to(dtype=joint_pos.dtype, device=joint_pos.device).unsqueeze(0)
        )  # (1, 20)

        return obs

    def convert_ee_euler_pos_to_joint_position(self, actions) -> np.ndarray:
        if actions is None:
            return actions
        if self._ik_seed_left is None or self._ik_seed_right is None:
            raise RuntimeError("IK seeds not initialized. Pinocchio IK setup may have failed.")

        # Expected: actions shape (chunk_size, 14)
        actions_np = np.asarray(actions)
        if actions_np.ndim != 2 or actions_np.shape[1] != 14:
            raise ValueError(f"Expected actions shape (N, 14), got {actions_np.shape}")

        q_ref_left = self._ik_seed_left
        q_ref_right = self._ik_seed_right

        out = np.zeros_like(actions_np)
        for i in range(actions_np.shape[0]):
            step = actions_np[i]

            # Left: [x,y,z,roll,pitch,yaw,gripper]
            left_target = step[:6]
            left_gripper = step[6]

            # Right: [x,y,z,roll,pitch,yaw,gripper]
            right_target = step[7:13]
            right_gripper = step[13]

            left_q = self._solve_ik_dls(left_target, self._ik_seed_left, q_ref=q_ref_left)
            right_q = self._solve_ik_dls(right_target, self._ik_seed_right, q_ref=q_ref_right)

            self._ik_seed_left = left_q
            self._ik_seed_right = right_q

            out[i, :6] = left_q
            out[i, 6] = left_gripper
            out[i, 7:13] = right_q
            out[i, 13] = right_gripper

        return out

    def convert_ee_6d_pos_to_joint_position(self, actions) -> np.ndarray:
        """Convert policy 6D EE actions (N, 20) to joint positions (N, 14)."""
        if actions is None:
            return actions
        if self._ik_seed_left is None or self._ik_seed_right is None:
            raise RuntimeError("IK seeds not initialized. Pinocchio IK setup may have failed.")

        # Expected: actions shape (chunk_size, 20)
        actions_np = np.asarray(actions)
        if actions_np.ndim != 2 or actions_np.shape[1] != 20:
            raise ValueError(f"Expected actions shape (N, 20), got {actions_np.shape}")

        q_ref_left = self._ik_seed_left
        q_ref_right = self._ik_seed_right

        out = np.zeros((actions_np.shape[0], 14), dtype=actions_np.dtype)
        for i in range(actions_np.shape[0]):
            step = actions_np[i]

            # Left: [x,y,z,rot6d(6),gripper]
            left_target = step[:9]
            left_gripper = step[9]

            # Right: [x,y,z,rot6d(6),gripper]
            right_target = step[10:19]
            right_gripper = step[19]

            left_q = self._solve_ik_6d(left_target, self._ik_seed_left, q_ref=q_ref_left)
            right_q = self._solve_ik_6d(right_target, self._ik_seed_right, q_ref=q_ref_right)

            self._ik_seed_left = left_q
            self._ik_seed_right = right_q

            out[i, :6] = left_q
            out[i, 6] = left_gripper
            out[i, 7:13] = right_q
            out[i, 13] = right_gripper

        return out


def _test_pinocchio_ik(server: AlohaServer, trials: int = 5, tol: float = 1e-4) -> None:
    if pin is None:
        print("pinocchio not installed; skipping IK test")
        return
    if server._pin_model is None or server._pin_data is None or server._pin_ee_frame_id is None:
        print("Pinocchio model not initialized; skipping IK test")
        return

    model = server._pin_model
    data = server._pin_data
    frame_id = server._pin_ee_frame_id

    rng = np.random.default_rng(0)
    lower = model.lowerPositionLimit[: model.nq]
    upper = model.upperPositionLimit[: model.nq]

    for i in range(trials):
        q = rng.uniform(lower, upper)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[frame_id]  # noqa: N806

        target_xyz = oMf.translation
        # target_rpy = ScipyRotation.from_matrix(oMf.rotation).as_euler('XYZ')
        target_rpy = pin.rpy.matrixToRpy(oMf.rotation)
        target = np.concatenate([target_xyz, target_rpy])

        q_sol = server._solve_ik_dls(target, q_seed=q)

        pin.forwardKinematics(model, data, q_sol)
        pin.updateFramePlacements(model, data)
        oMf_sol = data.oMf[frame_id]  # noqa: N806
        err = pin.log(oMf_sol.inverse() * oMf).vector
        err_norm = float(np.linalg.norm(err))

        orig_q_str = np.array2string(q, precision=4, separator=", ")
        target_str = np.array2string(target, precision=4, separator=", ")
        sol_q_str = np.array2string(q_sol, precision=4, separator=", ")
        sol_xyz = oMf_sol.translation
        # sol_rpy = ScipyRotation.from_matrix(oMf_sol.rotation).as_euler('XYZ')
        sol_rpy = pin.rpy.matrixToRpy(oMf_sol.rotation)
        sol_pose = np.concatenate([sol_xyz, sol_rpy])
        sol_pose_str = np.array2string(sol_pose, precision=4, separator=", ")

        status = "OK" if err_norm < tol else "FAIL"
        print(f"Trial {i + 1}/{trials}: pose_err={err_norm:.6f} [{status}]")
        print(f"  fk_pin:  {target_str}")
        print(f"  q_orig:  {orig_q_str}")
        print(f"  q_ik:    {sol_q_str}")
        print(f"  fk_ik:   {sol_pose_str}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick FK/IK test for Aloha Pinocchio model")
    parser.add_argument(
        "--urdf",
        type=str,
        default=str(Path(__file__).resolve().parent / "aloha_vx300s.urdf"),
        help="Path to the minimal URDF file",
    )
    parser.add_argument("--trials", type=int, default=5, help="Number of random IK trials")
    parser.add_argument("--tol", type=float, default=1e-4, help="Pose error tolerance")
    args = parser.parse_args()

    cfg = AlohaServerConfig()
    cfg.policy_action_type = ActionType.EE_EULER_POS
    server = AlohaServer(cfg)
    server._ik_urdf_path = Path(args.urdf)
    server._init_pinocchio_ik()

    _test_pinocchio_ik(server, trials=args.trials, tol=args.tol)

from dataclasses import dataclass

import torch

from environments.ur5e.calculate_action_variants import compute_eef_poses_for_dual_arm, compute_eef_quat_poses
from rho.common.types import ActionType
from rho.environment import register_environment
from rho.environment.env import EnvironmentConfig
from rho.server.serve_policy import Server


@EnvironmentConfig.register_subclass("ur5e_server")
@dataclass
class UR5eServerConfig(EnvironmentConfig):
    """Configuration for TabletopSim environments."""

    name: str = "ur5e_server"
    port: int = 5000
    device: str = "cuda"

    input_action_type: ActionType = ActionType.POSITION  # "joint_position" or "ee_quat_pos" or "ee_rpy_pos"
    output_action_type: ActionType = (
        ActionType.EE_QUAT_POS_XYZW
    )  # "joint_position" or "ee_quat_pos" or "ee_rpy_pos"

    # this will be set by the training data config action type
    policy_action_type: ActionType = ActionType.POSITION  # "joint_position" or "ee_quat_pos" or "ee_rpy_pos"


@register_environment("ur5e_server")
class UR5eServer(Server):
    def __init__(self, config: UR5eServerConfig) -> None:
        super().__init__(config)

        self.device = config.device

        self.image_keys = ["zed_scene_bgr", "zed_wrist_left_bgr", "zed_wrist_right_bgr"]

        # Type of actions being returned by the client
        self.input_action_type = config.input_action_type

        # Type of actions being sent to the client
        self.output_action_type = config.output_action_type
        self.policy_action_type = config.policy_action_type

    def process_input(self, input) -> tuple:
        """should return observation dict"""
        """obs['action'] should be what the client sent as previous action"""
        """sequence length of each observation tensor should be the history that the client sends"""
        obs = self.dict_to_torch(input, self.device)

        # UR5e server sends tactile history as 'tactile_history' and tcp forces history
        # as 'tcp_forces_history'
        if "tactile_history" in obs:
            if len(obs["tactile_history"].shape) == 3 and obs["tactile_history"].shape[1] == 1:
                # reshape tactile history from (seq_len, 1, tactile_dim) to (seq_len, tactile_dim)
                obs["tactile_history"] = obs["tactile_history"].squeeze(1)
            obs["tactile"] = obs.pop("tactile_history")
        if "tcp_forces_history" in obs:
            if len(obs["tcp_forces_history"].shape) == 3 and obs["tcp_forces_history"].shape[1] == 1:
                # reshape tcp forces history from (seq_len, 1, tcp_forces_dim) to (seq_len, tcp_forces_dim)
                obs["tcp_forces_history"] = obs["tcp_forces_history"].squeeze(1)
            obs["actual_tcp_forces"] = obs.pop("tcp_forces_history")

        self.convert_observation_state_type_to_policy_type(obs)

        for key in obs:
            if key in self.image_keys:
                img = obs[key][:, :, [2, 1, 0]]  # BGR to RGB (180,320,3)
                img = img.permute(2, 0, 1).float() / 255.0
                obs[key] = img.unsqueeze(0)

            elif isinstance(obs[key], torch.Tensor):
                if len(obs[key].shape) == 2:
                    obs[key] = obs[key].unsqueeze(0).float()
                elif len(obs[key].shape) == 1:
                    obs[key] = obs[key].unsqueeze(0).unsqueeze(0).float()

        return obs

    def process_output(self, actions):
        # Remove batch dimension if present
        if actions.ndim > 2 and actions.shape[0] == 1:
            actions = actions.squeeze(0)

        actions = actions.cpu().to(torch.float32).numpy()

        actions = self.convert_policy_action_type_to_client_type(actions)

        return actions

    def convert_observation_state_type_to_policy_type(self, obs):
        """Convert observation state type (ex. joints, ee_quat_pos, ee_rpy_pos) to policy type."""

        assert self.input_action_type == ActionType.POSITION, (
            "UR5e server only supports joint_position as input action type."
        )
        if self.policy_action_type == ActionType.EE_EULER_POS:
            # FK to get rpy pos — stored under 'ee_state' key; raw joints kept in 'joint_positions'.
            ee_euler = compute_eef_poses_for_dual_arm(obs["joint_positions"].clone().cpu().numpy())
            obs["ee_state"] = torch.tensor(ee_euler, device=self.device, dtype=torch.float32)
        elif self.policy_action_type == ActionType.EE_QUAT_POS_XYZW:
            # FK to quaternion directly (rotation matrix → quat).
            # Converted ee_quat stored in 'ee_state' key; raw joints kept in 'joint_positions'.
            ee_quat = compute_eef_quat_poses(obs["joint_positions"].clone().cpu().numpy())
            obs["ee_state"] = torch.tensor(ee_quat, device=self.device, dtype=torch.float32)
        elif self.input_action_type != self.policy_action_type:
            raise NotImplementedError(
                f"Conversion from {self.input_action_type} to {self.policy_action_type} not implemented."
            )

    def convert_policy_action_type_to_client_type(self, actions):
        """Convert policy action type (ex. joints, ee_quat_pos, ee_rpy_pos) to client type."""

        # # UR5 controller ALWAYS wants ee_rpy as output
        # assert self.output_action_type == ActionType.EE_EULER_POS, (
        #     "UR5e server only supports ee_euler_pos as output action type."
        # )
        # if self.policy_action_type == ActionType.POSITION:
        #     # do forward kinematics to get rpy pos for client
        #     ee_rpy_actions = compute_eef_poses_for_dual_arm(actions)
        # else:
        #     assert self.policy_action_type == ActionType.EE_EULER_POS, (
        #         "UR5e server only supports joint_position and ee_rpy_pos as policy action types."
        #     )
        #     ee_rpy_actions = actions

        # return ee_rpy_actions

        assert self.output_action_type == self.policy_action_type, (
            "UR5e server only supports output action type to be the same as"
            " policy action type to avoid conversion."
        )

        return actions

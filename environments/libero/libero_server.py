"""
LIBERO Server Environment for websocket-based policy serving.

This module defines a LiberoServer class that processes observations from a
remote LIBERO simulation client and returns actions from the policy.
It follows the same pattern as AlohaServer / UR5eServer.

The client sends observations with keys:
    - "agentview": uint8 image (H, W, 3) – agent camera (already rotated 180°)
    - "wrist": uint8 image (H, W, 3) – wrist camera (already rotated 180°)
    - "state": float64 array (1, 8) – robot state [eef_pos, axisangle, gripper]
    - "task": list of strings – natural-language task description

The server converts these into the tensor format expected by the policy
and returns an action chunk.
"""

from dataclasses import dataclass

import torch

from rho.common.types import ActionType
from rho.environment import register_environment
from rho.environment.env import EnvironmentConfig
from rho.server.serve_policy import Server


@EnvironmentConfig.register_subclass("libero_server")
@dataclass
class LiberoServerConfig(EnvironmentConfig):
    """Configuration for LiberoServer websocket environment."""

    name: str = "libero_server"
    port: int = 7000
    device: str = "cuda"

    # LIBERO uses 7D actions (6 DoF + gripper) – no FK/IK conversion needed
    policy_action_type: ActionType = ActionType.POSITION


@register_environment("libero_server")
class LiberoServer(Server):
    """Websocket server environment for LIBERO simulation clients.

    Processes raw observations from the LIBERO client into the tensor format
    expected by the policy, and converts policy action outputs back to numpy
    arrays for the client.
    """

    def __init__(self, config: LiberoServerConfig) -> None:
        super().__init__(config)
        self.device = config.device
        self.image_keys = ["agentview", "wrist"]
        self.policy_action_type = config.policy_action_type

    def process_input(self, input) -> dict:
        """Convert client observations to policy-ready tensor format.

        Expected input dict from client:
            - "agentview": uint8 ndarray (H, W, 3) RGB
            - "wrist": uint8 ndarray (H, W, 3) RGB
            - "state": float64 ndarray (1, 8)
            - "task": list[str]

        Returns:
            dict with:
            - image keys: float tensor (1, C, H, W) in [0, 1] RGB
            - "state": float tensor (1, 1, 8)
            - "task": list[str]
        """
        obs = self.dict_to_torch(input, self.device)

        for key in list(obs.keys()):
            if key in self.image_keys:
                # Client sends RGB uint8 (H, W, 3) – convert to (1, C, H, W) float [0,1]
                img = obs[key].float()
                img = img.permute(2, 0, 1) / 255.0  # (C, H, W)
                obs[key] = img.unsqueeze(0)  # (1, C, H, W)

            elif key == "task":
                # Pass through task strings as-is
                pass

            elif isinstance(obs[key], torch.Tensor):
                # State / other numeric tensors: ensure (batch, seq, dim) shape
                if len(obs[key].shape) == 2:
                    obs[key] = obs[key].unsqueeze(0).float()  # (1, seq, dim)
                elif len(obs[key].shape) == 1:
                    obs[key] = obs[key].unsqueeze(0).unsqueeze(0).float()  # (1, 1, dim)

        return obs

    def process_output(self, actions):
        """Convert policy action tensor to numpy array for the client.

        Args:
            actions: torch.Tensor (1, chunk_size, action_dim) or (chunk_size, action_dim)

        Returns:
            np.ndarray of shape (chunk_size, action_dim)
        """
        if actions.ndim > 2 and actions.shape[0] == 1:
            actions = actions.squeeze(0)

        return actions.cpu().to(torch.float32).numpy()

    def convert_observation_state_type_to_policy_type(self, state):
        """Convert observation state representation to the policy's state type.

        LIBERO already uses POSITION type for both client and policy, so this
        is a no-op conversion.
        """
        assert self.policy_action_type == ActionType.POSITION, (
            "LiberoServer assumes POSITION action type for state conversions."
        )
        return state

    def convert_policy_action_type_to_client_type(self, actions):
        """Convert policy action representation to the client's action type.

        Since LIBERO uses POSITION actions end-to-end, this is a no-op.
        """
        assert self.policy_action_type == ActionType.POSITION, (
            "LiberoServer assumes POSITION action type for policy->client actions."
        )
        return actions

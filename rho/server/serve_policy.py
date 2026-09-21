#!/usr/bin/env python3
"""
Evaluation script for pretrained checkpoints using EnvironmentWrapper

This script loads a checkpoint created by the accelerate-based training script
and evaluates the policy in the cfg.environment using the EnvironmentWrapper
"""

import logging
import socket

import draccus
import numpy as np
import torch

import rho.server.open_pi_server as open_pi_server
from rho.environment import EnvironmentConfig, make_environment
from rho.eval.eval_config import EvalConfig
from rho.eval.policy_interface import PolicyInterface
from rho.policies import make_policy
from rho.utils import init_logging

logger = logging.getLogger(__name__)


class Server:
    """Base server class for real-time robot policy serving.

    This class defines the interface for processing observations from a robot client,
    running them through a policy, and returning actions. Subclasses should implement
    the abstract methods for specific robot configurations.

    Attributes:
        config: EnvironmentConfig containing server settings like port, action types, etc.
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.policy_action_type = config.policy_action_type

    def process_input(self, input) -> dict:
        """Process raw input from client into policy-ready observation format.

        Takes the raw observation data from the robot client and converts it into
        the format expected by the policy network.

        Args:
            input: Either a dict containing just observations, or a dict with keys:
                - "obs": Current observation dictionary
                - "prev_actions": Previous action tensor (optional)
                - "obs_queue": Queue of past observations for temporal models (optional)

        Returns:
            Observation dictionary passed to PolicyInterface.get_action_chunk(),
            including image/state tensors and any reset or RTC metadata.
        """
        raise NotImplementedError

    def process_output(self, actions):
        """Convert policy output actions to client-expected format.

        Args:
            actions: torch.Tensor of shape (1, chunk_size, action_dim) from policy.

        Returns:
            np.ndarray: Actions as CPU numpy array of shape (chunk_size, action_dim),
                ready to send to the robot client.
        """
        raise NotImplementedError

    def convert_observation_state_type_to_policy_type(self, obs):
        """Convert observation state type (ex. joints, ee_quat_pos, ee_rpy_pos) to policy type."""
        raise NotImplementedError

    def convert_policy_action_type_to_client_type(self, actions):
        """Convert policy action type (ex. joints, ee_quat_pos, ee_rpy_pos) to client type."""
        raise NotImplementedError

    def dict_to_torch(self, obs, device):
        new_obs = {}
        for key in obs:
            if isinstance(obs[key], torch.Tensor):
                new_obs[key] = obs[key].to(device)
            elif isinstance(obs[key], dict):  # do we accept nested dicts?
                new_obs[key] = self.dict_to_torch(obs[key], device)
            elif isinstance(obs[key], np.ndarray):
                new_obs[key] = torch.from_numpy(obs[key].copy()).to(device)
            else:
                new_obs[key] = obs[key]
        return new_obs


@draccus.wrap()
def eval(cfg: EvalConfig) -> None:
    # setup policy interface and load pretrained policy
    # 1. Create environment from EvalConfig
    init_logging(console_level=cfg.log_level)

    # 2. Load the pretrained base policy from provided checkpoint.
    logger.info("Creating policy from config...")
    base_policy = make_policy(cfg.policy)
    base_policy.load_from_pretrained(cfg.pretrained_checkpoint)
    base_policy.eval()

    logger.info(f"Base policy loaded: {type(base_policy).__name__}")
    logger.info(f"   Device: {base_policy.device}")

    logger.info("Creating policy interface...")
    cfg.policy_interface_cfg.policy = base_policy
    cfg.policy_interface_cfg.data_config = cfg.dataset

    policy_interface = PolicyInterface(cfg.policy_interface_cfg)

    # find key in policy_interface.data_config.observation_mapping that maps to "action"
    cfg.environment.policy_action_type = policy_interface.data_config.action_type
    cfg.environment.device = policy_interface.device

    env = make_environment(cfg.environment)
    logger.info(f"Environment initialized: {type(env).__name__}")

    # 2. Start serving model
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = open_pi_server.WebsocketPolicyServer(
        policy_interface=policy_interface,
        env=env,
        host="0.0.0.0",  # nosec B104
        port=cfg.environment.port,
    )

    server.serve_forever()


if __name__ == "__main__":
    eval()  # nosec B307

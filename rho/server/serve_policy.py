#!/usr/bin/env python3
"""
Evaluation script for pretrained checkpoints using EnvironmentWrapper

This script loads a checkpoint created by the accelerate-based training script
and evaluates the policy in the cfg.environment using the EnvironmentWrapper
"""

import contextlib
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

    def process_input(self, input) -> tuple:
        """Process raw input from client into policy-ready observation format.

        Takes the raw observation data from the robot client and converts it into
        the format expected by the policy network.

        Args:
            input: Either a dict containing just observations, or a dict with keys:
                - "obs": Current observation dictionary
                - "prev_actions": Previous action tensor (optional)
                - "obs_queue": Queue of past observations for temporal models (optional)

        Returns:
            tuple: (observation_dict, obs_queue, prev_actions) where:
                - observation_dict: Dict with:
                    - Image keys: torch.Tensor of shape (batch_size, C, H, W),
                      scaled to [0, 1], in RGB format
                    - State keys: torch.Tensor of shape (batch_size, 1, feature_dim)
                - obs_queue: Optional queue of past observations
                - prev_actions: Optional tensor of previous actions
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


def create_trainer(cfg: EvalConfig, *, policy=None, policy_interface=None, env=None):
    """Build and start the HIL trainer named by ``cfg.trainer_type``.

    Returns the trainer object (with ``.stop()`` for shutdown). The trainer
    runs its receive loop on a background thread so the websocket server can
    keep handling inference requests in the foreground.

    The "dsrl" trainer needs the loaded base policy plus the constructed
    policy_interface/env to share preprocessing with the inference server.
    "debug" needs nothing beyond the experience port.
    """
    if cfg.trainer_type == "debug":
        from rho.hil.trainers.debug_trainer import start_debug_trainer

        return start_debug_trainer(
            experience_port=cfg.experience_port,
            blocking=False,
        )
    if cfg.trainer_type == "dsrl":
        from rho.hil.trainers.dsrl_trainer import start_dsrl_trainer
        from rho.policies.dsrl.dsrl_config import DSRLConfig

        dsrl_cfg = cfg.dsrl if cfg.dsrl is not None else DSRLConfig()
        return start_dsrl_trainer(
            config=dsrl_cfg,
            base_policy=policy,
            experience_port=cfg.experience_port,
            blocking=False,
            policy_interface=policy_interface,
            env=env,
        )
    if cfg.trainer_type == "flowdagger":
        from rho.hil.trainers.flowdagger_trainer import start_flowdagger_trainer
        from rho.policies.dsrl.flowdagger_config import FlowDAggerConfig

        fd_cfg = cfg.flowdagger if cfg.flowdagger is not None else FlowDAggerConfig()
        return start_flowdagger_trainer(
            config=fd_cfg,
            base_policy=policy,
            experience_port=cfg.experience_port,
            blocking=False,
            policy_interface=policy_interface,
            env=env,
        )
    raise ValueError(f"Unknown trainer_type='{cfg.trainer_type}'. Supported: 'debug', 'dsrl', 'flowdagger'.")


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

    # If a wrapper trainer config is set (dsrl / flowdagger), wrap the loaded
    # base policy in the matching inference-side wrapper. The wrapper is what
    # the websocket server serves; the trainer keeps a reference to the base
    # for inverse_noise_map / inversion calls.
    served_policy = base_policy
    if cfg.flowdagger is not None:
        from rho.policies.dsrl.flowdagger_policy import FlowDAggerPolicy

        policy_cfg = cfg.flowdagger.to_policy_config()
        served_policy = FlowDAggerPolicy(policy_cfg, base_policy=base_policy)
        served_policy.eval()
        logger.info("Wrapped base policy in FlowDAggerPolicy (in-process)")
    elif cfg.dsrl is not None:
        from rho.policies.dsrl.dsrl_policy import DSRLPolicy

        policy_cfg = cfg.dsrl.to_policy_config()
        served_policy = DSRLPolicy(policy_cfg, base_policy=base_policy)
        served_policy.eval()
        logger.info("Wrapped base policy in DSRLPolicy (in-process)")

    # PolicyInterface reads base-policy config (chunk_size, n_action_steps,
    # delta_indices_dict) at __init__ time only. Construct it against the
    # base, then swap in the wrapper so sample_actions calls route through
    # it. (Wrapper has matching chunk geometry but its config is a different
    # dataclass.)
    logger.info("Creating policy interface...")
    cfg.policy_interface_cfg.policy = base_policy
    cfg.policy_interface_cfg.data_config = cfg.dataset

    policy_interface = PolicyInterface(cfg.policy_interface_cfg)

    if served_policy is not base_policy:
        policy_interface.policy = served_policy

    # find key in policy_interface.data_config.observation_mapping that maps to "action"
    cfg.environment.policy_action_type = policy_interface.data_config.action_type
    cfg.environment.device = policy_interface.device

    env = make_environment(cfg.environment)
    logger.info(f"Environment initialized: {type(env).__name__}")

    # Start HIL trainer (if requested) after the policy + env are ready so the
    # wrapper trainers can share the base policy reference; the bind happens
    # before serve_forever so a port conflict surfaces immediately.
    trainer = None
    if cfg.train:
        logger.info(f"Starting HIL trainer '{cfg.trainer_type}' in background...")
        trainer = create_trainer(cfg, policy=base_policy, policy_interface=policy_interface, env=env)

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

    try:
        server.serve_forever()
    finally:
        if trainer is not None:
            with contextlib.suppress(Exception):
                trainer.stop()


if __name__ == "__main__":
    eval()  # nosec B307

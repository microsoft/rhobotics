#!/usr/bin/env python3
"""
Evaluation script for pretrained checkpoints using EnvironmentWrapper

This script loads a checkpoint created by the accelerate-based training script
and evaluates the policy in the cfg.environment using the EnvironmentWrapper
"""

import logging

from aloha import AlohaServer  # noqa: E402, F401

from rho.server.serve_policy import eval

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    eval()  # nosec B307

#!/usr/bin/env python3
"""Train Rho policies with RoboEval datasets and environments."""

from env import RoboevalEnvConfig, require_roboeval_data_root  # noqa: F401

from rho.train import main as train


def main() -> None:
    require_roboeval_data_root()
    train()


if __name__ == "__main__":
    main()

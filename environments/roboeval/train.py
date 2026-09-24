#!/usr/bin/env python3
"""Train Rho policies with RoboEval datasets and environments."""

from env import RoboevalEnvConfig  # noqa: F401

from rho.train import main as train


def main() -> None:
    train()


if __name__ == "__main__":
    main()

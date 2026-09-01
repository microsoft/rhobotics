#!/usr/bin/env python3
"""Evaluate Rho policies in RoboEval."""

from env import RoboevalEnvConfig, require_roboeval_data_root  # noqa: F401

from rho.eval.eval import eval


def main() -> None:
    require_roboeval_data_root()
    eval()


if __name__ == "__main__":
    main()

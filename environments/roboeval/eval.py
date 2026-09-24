#!/usr/bin/env python3
"""Evaluate Rho policies in RoboEval."""

from env import RoboevalEnvConfig  # noqa: F401

from rho.eval.eval import eval


def main() -> None:
    eval()


if __name__ == "__main__":
    main()

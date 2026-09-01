#!/usr/bin/env python3
"""Evaluate Rho policies in the LIBERO benchmark."""

from env import LiberoEnvConfig  # noqa: F401

from rho.eval.eval import eval


def main() -> None:
    eval()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train Rho policies with LIBERO datasets and environments."""

from env import LiberoEnvConfig  # noqa: F401

from rho.train import main

if __name__ == "__main__":
    main()

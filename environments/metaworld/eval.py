#!/usr/bin/env python3
"""Evaluate Rho policies on the MetaWorld MT50 benchmark."""

import os

# MuJoCo picks its GL backend at first render; default to EGL so this works headless.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

from env import MetaworldEnvConfig  # noqa: F401,E402

from rho.eval.eval import eval  # noqa: E402


def main() -> None:
    eval()


if __name__ == "__main__":
    main()

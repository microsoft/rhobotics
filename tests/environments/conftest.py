"""Shared setup for environment tests.

The LIBERO simulator is usually installed only inside the LIBERO container.
The wrapper's pure-Python logic (config validation, task cycling and
initial-state selection) does not need the simulator, so when ``libero`` is not
installed we register minimal stand-in modules that let
``environments.libero.env`` be imported. The stand-ins raise if anything tries
to use the simulator, so tests that need it fail loudly instead of passing
against a fake. When the real package is installed, nothing is stubbed.
"""

import importlib.util
import sys
import types

LIBERO_STUBBED = importlib.util.find_spec("libero") is None


def _simulator_unavailable(*_args, **_kwargs):
    raise RuntimeError("The LIBERO simulator is not installed; this test environment uses a stub.")


class _UnavailableEnv:
    def __init__(self, *args, **kwargs):
        _simulator_unavailable()


if LIBERO_STUBBED:
    libero = types.ModuleType("libero")
    libero.__rho_test_stub__ = True
    libero_libero = types.ModuleType("libero.libero")
    libero_libero.benchmark = types.SimpleNamespace(get_benchmark_dict=_simulator_unavailable)
    libero_libero.get_libero_path = _simulator_unavailable
    libero_envs = types.ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = _UnavailableEnv
    libero_envs.SubprocVectorEnv = _UnavailableEnv
    libero.libero = libero_libero
    libero_libero.envs = libero_envs
    sys.modules.update(
        {
            "libero": libero,
            "libero.libero": libero_libero,
            "libero.libero.envs": libero_envs,
        }
    )

"""Shared CPU fixtures for the optional LIBERO and RoboEval simulators.

Import-only stand-ins let tests exercise the real wrappers without physics
dependencies. Constructing an unavailable simulator component raises.
"""

import importlib.util
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

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


def _missing_roboeval(*args, **kwargs):
    raise RuntimeError("The RoboEval simulator is not installed.")


if importlib.util.find_spec("roboeval") is None:
    # Only the symbols imported by the wrapper are needed at collection time.
    for module_name, symbols in {
        "roboeval": (),
        "roboeval.action_modes": ("JointPositionActionMode",),
        "roboeval.demonstrations": (),
        "roboeval.demonstrations.utils": ("Metadata",),
        "roboeval.envs": (),
        "roboeval.envs.lift_pot": ("LiftPotPositionAndOrientation",),
        "roboeval.envs.lift_tray": ("LiftTrayPositionAndOrientation",),
        "roboeval.envs.manipulation": (
            "CubeHandoverPositionAndOrientation",
            "StackTwoBlocksPositionAndOrientation",
        ),
        "roboeval.envs.pack_objects": ("PackBoxPositionAndOrientation",),
        "roboeval.envs.rotate_utility_objects": ("RotateValvePositionAndOrientation",),
        "roboeval.envs.stack_books": (
            "PickSingleBookFromTablePositionAndOrientation",
            "StackSingleBookShelfPositionAndOrientation",
        ),
        "roboeval.robots": (),
        "roboeval.robots.configs": (),
        "roboeval.robots.configs.panda": ("BimanualPanda",),
        "roboeval.utils": (),
        "roboeval.utils.observation_config": ("CameraConfig", "ObservationConfig"),
    }.items():
        module = types.ModuleType(module_name)
        for symbol in symbols:
            setattr(module, symbol, _missing_roboeval)
        sys.modules[module_name] = module
        if "." in module_name:
            parent, name = module_name.rsplit(".", 1)
            setattr(sys.modules[parent], name, module)


@pytest.fixture
def roboeval_wrapper():
    from environments.roboeval.env import RoboevalEnvConfig, RoboevalWrapper

    config = RoboevalEnvConfig(device="cpu")
    wrapper = RoboevalWrapper.__new__(RoboevalWrapper)
    wrapper.config = config
    wrapper.device = config.device
    wrapper.action_space = config.action_space
    wrapper.task_name = config.task_name
    wrapper.task_description = config.default_task
    wrapper.sim = Mock()
    wrapper.done = np.zeros(1, dtype=bool)
    wrapper.total_reward = np.zeros(1, dtype=np.float64)
    wrapper.last_frame = None
    return wrapper

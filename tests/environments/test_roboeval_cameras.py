from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from environments.roboeval import env


@pytest.mark.parametrize("camera_names", [None, ["right_wrist", "left_wrist"], []])
def test_simulator_receives_configured_cameras(monkeypatch, camera_names):
    cfg = env.RoboevalEnvConfig()
    if camera_names is not None:
        cfg.camera_names = camera_names
    simulator = Mock()
    monkeypatch.setitem(env.TASK_ENVS, cfg.task_name, simulator)
    monkeypatch.setattr(env, "Metadata", Mock())
    monkeypatch.setattr(env, "JointPositionActionMode", SimpleNamespace)
    monkeypatch.setattr(env, "CameraConfig", SimpleNamespace)
    monkeypatch.setattr(env, "ObservationConfig", SimpleNamespace)

    env.RoboevalWrapper(cfg)

    cameras = simulator.call_args.kwargs["observation_config"].cameras
    expected = ["head", "left_wrist", "right_wrist"] if camera_names is None else camera_names
    assert [camera.name for camera in cameras] == expected
    assert all(camera.rgb and not camera.depth and camera.resolution == (224, 224) for camera in cameras)


def test_wrist_only_observations_and_video(roboeval_wrapper):
    wrapper = roboeval_wrapper
    wrapper.config.camera_names = ["right_wrist", "left_wrist"]
    wrapper.config.observation_mapping = {"observation.rgb_right_wrist": "observation.image.0"}
    right = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
    left = np.full_like(right, 127)
    raw = {
        "proprioception": np.zeros(18),
        "proprioception_grippers": np.zeros(2),
        "rgb_left_wrist": left,
        "rgb_right_wrist": right,
    }

    obs = wrapper._process_input(raw)

    np.testing.assert_allclose(obs["observation.image.0"][0, 0].numpy(), right / 255)
    np.testing.assert_array_equal(wrapper.render(), right.transpose(1, 2, 0))

    del raw["rgb_right_wrist"]
    wrapper._process_input(raw)
    np.testing.assert_array_equal(wrapper.render(), left.transpose(1, 2, 0))

    del raw["rgb_left_wrist"]
    wrapper._process_input(raw)
    assert wrapper.render() is None

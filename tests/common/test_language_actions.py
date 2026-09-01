import math

import torch

from rho.common.constants import ACTION, LANGUAGE_ACTION_FRAME, LANGUAGE_ACTION_TARGET, OBSERVATION_STATE
from rho.common.language_actions import (
    discretize_state_for_lap_prompt,
    summarize_ee_6d_language_actions,
    summarize_eef_rpy_language_actions,
)
from rho.common.task_encoding import decode_task_bytes
from rho.common.transforms import LanguageActionTarget


def test_summarize_eef_rpy_language_actions_uses_physical_units():
    actions = torch.tensor([[0.052, -0.024, 0.011, 0.0, -math.pi / 18.0, 0.0, 1.0]])

    text = summarize_eef_rpy_language_actions(actions)[0]

    assert text == "move forward 5 cm, move up 1 cm, move right 2 cm, tilt forward 10 degrees, open gripper"


def test_summarize_ee_6d_language_actions_converts_rotation_layout():
    identity_rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    actions = torch.tensor([[0.05, -0.02, 0.01, *identity_rot6d, 0.0]])

    text = summarize_ee_6d_language_actions(actions)[0]

    assert text == "move forward 5 cm, move up 1 cm, move right 2 cm, close gripper"


def test_language_action_target_encodes_chunk_net_text():
    identity_rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    batch = {
        ACTION: torch.tensor(
            [
                [
                    [0.01, 0.0, 0.0, *identity_rot6d, 0.0],
                    [0.05, -0.02, 0.01, *identity_rot6d, 1.0],
                ]
            ]
        )
    }

    out = LanguageActionTarget(eef_frame_prob=0.0)(batch)
    decoded = decode_task_bytes(out[LANGUAGE_ACTION_TARGET])

    assert decoded == ["move forward 5 cm, move up 1 cm, move right 2 cm, open gripper"]
    assert decode_task_bytes(out[LANGUAGE_ACTION_FRAME]) == ["robot base frame"]


def test_language_action_target_can_use_end_effector_frame():
    identity_rot6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    batch = {
        OBSERVATION_STATE: torch.tensor([[0.0, 0.0, 0.0, *identity_rot6d, 0.0]]),
        ACTION: torch.tensor([[[0.05, -0.02, 0.01, *identity_rot6d, 1.0]]]),
    }

    out = LanguageActionTarget(eef_frame_prob=1.0)(batch)

    assert decode_task_bytes(out[LANGUAGE_ACTION_TARGET]) == [
        "move forward 5 cm, move down 1 cm, move left 2 cm, open gripper"
    ]
    assert decode_task_bytes(out[LANGUAGE_ACTION_FRAME]) == ["end-effector frame"]


def test_discretize_state_for_lap_prompt_clamps_to_256_bins():
    state = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    text = discretize_state_for_lap_prompt(state, max_dims=5)[0]

    assert text == "0 0 127 255 255"

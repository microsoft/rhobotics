"""LAP-style language-action formatting utilities."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from rho.common.rotation_helpers import convert_ee_6d_to_ee_rpy, rotation_6d_to_matrix


def _round_to_nearest(value: float, n: int = 5) -> int:
    return int(round(value / n) * n)


def _format_single_arm_eef_rpy(row: Tensor, *, include_rotation: bool = True) -> str:
    dx_cm = int(round(float(row[0].item()) * 100.0))
    dy_cm = int(round(float(row[1].item()) * 100.0))
    dz_cm = int(round(float(row[2].item()) * 100.0))

    parts: list[str] = []
    if dx_cm > 0:
        parts.append(f"move forward {abs(dx_cm)} cm")
    elif dx_cm < 0:
        parts.append(f"move back {abs(dx_cm)} cm")
    if dz_cm > 0:
        parts.append(f"move up {abs(dz_cm)} cm")
    elif dz_cm < 0:
        parts.append(f"move down {abs(dz_cm)} cm")
    if dy_cm > 0:
        parts.append(f"move left {abs(dy_cm)} cm")
    elif dy_cm < 0:
        parts.append(f"move right {abs(dy_cm)} cm")

    if include_rotation and row.numel() >= 6:
        droll = _round_to_nearest(abs(float(row[3].item()) * 180.0 / math.pi), 5)
        dpitch = _round_to_nearest(abs(float(row[4].item()) * 180.0 / math.pi), 5)
        dyaw = _round_to_nearest(abs(float(row[5].item()) * 180.0 / math.pi), 5)
        if row[3].item() > 0 and droll != 0:
            parts.append(f"tilt left {droll} degrees")
        elif row[3].item() < 0 and droll != 0:
            parts.append(f"tilt right {droll} degrees")
        if row[4].item() > 0 and dpitch != 0:
            parts.append(f"tilt back {dpitch} degrees")
        elif row[4].item() < 0 and dpitch != 0:
            parts.append(f"tilt forward {dpitch} degrees")
        if row[5].item() > 0 and dyaw != 0:
            parts.append(f"rotate counterclockwise {dyaw} degrees")
        elif row[5].item() < 0 and dyaw != 0:
            parts.append(f"rotate clockwise {dyaw} degrees")

    gripper = float(row[-1].item())
    parts.append("open gripper" if gripper >= 0.5 else "close gripper")
    return ", ".join(parts)


def summarize_eef_rpy_language_actions(actions: Tensor, *, include_rotation: bool = True) -> list[str]:
    """Format EEF RPY actions as LAP language-action strings.

    Args:
        actions: Tensor shaped ``(B, D)`` or ``(D,)``. Per arm layout is
            ``[x, y, z, roll, pitch, yaw, gripper]`` in meters/radians.

    Returns:
        One language-action string per batch item.
    """
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if actions.shape[-1] % 7 != 0:
        raise ValueError(f"EEF RPY action dimension must be a multiple of 7, got {actions.shape[-1]}")

    actions = actions.detach().to(torch.float32).cpu()
    num_arms = actions.shape[-1] // 7
    out: list[str] = []
    for row in actions:
        arm_text = [
            _format_single_arm_eef_rpy(row[i * 7 : (i + 1) * 7], include_rotation=include_rotation)
            for i in range(num_arms)
        ]
        if num_arms == 1:
            out.append(arm_text[0])
        elif num_arms == 2:
            out.append(f"Left arm: {arm_text[0]}. Right arm: {arm_text[1]}")
        else:
            out.append(". ".join(f"Arm {i + 1}: {text}" for i, text in enumerate(arm_text)))
    return out


def summarize_ee_6d_language_actions(actions: Tensor, *, include_rotation: bool = True) -> list[str]:
    """Format EE 6D pose actions as LAP language-action strings.

    Per arm input layout is ``[x, y, z, rot6d(6), gripper]`` in meters.
    Rotation is converted to RPY radians before text formatting.
    """
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if actions.shape[-1] % 10 != 0:
        raise ValueError(f"EE 6D action dimension must be a multiple of 10, got {actions.shape[-1]}")
    rpy = convert_ee_6d_to_ee_rpy(actions.detach().to(torch.float32).cpu(), with_ee_gripper=True)
    return summarize_eef_rpy_language_actions(rpy, include_rotation=include_rotation)


def ee_6d_actions_to_eef_rpy(actions: Tensor, state: Tensor) -> Tensor:
    """Convert base-frame EE 6D net actions to end-effector-frame EEF RPY."""
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    actions = actions.detach().to(torch.float32).cpu()
    state = state.detach().to(torch.float32).cpu()
    if actions.shape[-1] % 10 != 0:
        raise ValueError(f"EE 6D action dimension must be a multiple of 10, got {actions.shape[-1]}")
    if state.shape[-1] < actions.shape[-1]:
        raise ValueError(f"State dim {state.shape[-1]} is smaller than action dim {actions.shape[-1]}")

    base_rpy = convert_ee_6d_to_ee_rpy(actions, with_ee_gripper=True)
    out = base_rpy.clone()
    num_arms = actions.shape[-1] // 10
    for arm in range(num_arms):
        action_6d_start = arm * 10
        action_rpy_start = arm * 7
        state_start = arm * 10
        base_to_eef = rotation_6d_to_matrix(state[:, state_start + 3 : state_start + 9]).transpose(-1, -2)

        delta_pos_base = actions[:, action_6d_start : action_6d_start + 3].unsqueeze(-1)
        delta_pos_eef = torch.matmul(base_to_eef, delta_pos_base).squeeze(-1)
        delta_pos_eef[:, 1] *= -1
        delta_pos_eef[:, 2] *= -1
        out[:, action_rpy_start : action_rpy_start + 3] = delta_pos_eef

        delta_rot_base = base_rpy[:, action_rpy_start + 3 : action_rpy_start + 6]
        delta_rotmat_base = torch.stack(
            [rotation_6d_to_matrix(convert_ee_rpy_to_rot6d(row)) for row in delta_rot_base],
            dim=0,
        )
        delta_rotmat_eef = torch.matmul(
            torch.matmul(base_to_eef, delta_rotmat_base),
            base_to_eef.transpose(-1, -2),
        )
        delta_rot_eef = torch.stack([rotmat_to_rpy(mat) for mat in delta_rotmat_eef], dim=0)
        delta_rot_eef[:, 1] *= -1
        delta_rot_eef[:, 2] *= -1
        out[:, action_rpy_start + 3 : action_rpy_start + 6] = delta_rot_eef
    return out


def convert_ee_rpy_to_rot6d(rpy: Tensor) -> Tensor:
    """Convert one RPY vector to a dummy EE 6D row for reuse with roma helpers."""
    from rho.common.rotation_helpers import convert_ee_rpy_to_ee_6d

    row = torch.cat([torch.zeros(3, dtype=rpy.dtype), rpy, torch.zeros(1, dtype=rpy.dtype)])
    return convert_ee_rpy_to_ee_6d(row, with_ee_gripper=True)[3:9]


def rotmat_to_rpy(matrix: Tensor) -> Tensor:
    """Convert one rotation matrix to RPY via the existing 6D converter."""
    from rho.common.rotation_helpers import matrix_to_rotation_6d

    rot6d = matrix_to_rotation_6d(matrix)
    row = torch.cat([torch.zeros(3, dtype=rot6d.dtype), rot6d, torch.zeros(1, dtype=rot6d.dtype)])
    return convert_ee_6d_to_ee_rpy(row, with_ee_gripper=True)[3:6]


def discretize_state_for_lap_prompt(
    state: Tensor,
    *,
    bins: int = 256,
    max_dims: int | None = None,
) -> list[str]:
    """Discretize normalized state vectors into LAP prompt text."""
    if state.ndim == 3:
        state = state[:, -1]
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if max_dims is not None:
        state = state[..., :max_dims]
    state = state.detach().to(torch.float32)
    edges = torch.linspace(-1.0, 1.0, bins + 1, device=state.device)[:-1]
    discretized = torch.bucketize(state, edges) - 1
    discretized = discretized.clamp(0, bins - 1).to(torch.int64).cpu()
    return [" ".join(str(int(v.item())) for v in row) for row in discretized]

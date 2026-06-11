# ---------------------------------------------------------------------------
# Action Transform Utility Functions for Different Rotation Representations
# All rotation operations use roma and stay on GPU
# roma uses (x, y, z, w) quaternion format internally
# Public API supports both xyzw and wxyz quaternion conventions
# ---------------------------------------------------------------------------

import numpy as np
import roma  # pip install roma
import torch

# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def normalize_angles(angles: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [-pi, pi] range."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)


def rotation_6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt."""
    shape = rot6d.shape[:-1]
    flat = rot6d.reshape(-1, 6)
    two_cols = torch.stack([flat[:, :3], flat[:, 3:6]], dim=-1)
    return roma.special_gramschmidt(two_cols).reshape(*shape, 3, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6) first two columns."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def normalize_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
    """Normalize 6D rotation to ensure orthonormality."""
    return matrix_to_rotation_6d(rotation_6d_to_matrix(rot6d))


# ---------------------------------------------------------------------------
# NumPy equivalents (for serving / IK paths that operate on CPU numpy arrays)
# ---------------------------------------------------------------------------


def rotation_6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix (numpy).

    Input:  (..., 6) -- [col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]
    Output: (..., 3, 3) rotation matrix

    Uses Gram-Schmidt orthonormalization, matching roma.special_gramschmidt.
    """
    eps = 1e-8
    shape = rot6d.shape[:-1]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)

    # Orthogonalize second column against first, then normalize
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + eps)

    # Third column via cross product
    b3 = np.cross(b1, b2, axis=-1)

    matrix = np.stack([b1, b2, b3], axis=-1)
    assert np.all(np.linalg.det(matrix) > 0), "det(R) must be positive (proper rotation)"
    return matrix.reshape(*shape, 3, 3)


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D rotation representation (numpy).

    Input:  (..., 3, 3) rotation matrix
    Output: (..., 6) -- first two columns flattened [col1, col2]
    """
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


# ---------------------------------------------------------------------------
# Quaternion convention swizzling (wxyz <-> xyzw)
# ---------------------------------------------------------------------------


def _swizzle_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from (w, x, y, z) to (x, y, z, w)."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def _swizzle_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from (x, y, z, w) to (w, x, y, z)."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _swizzle_ee_wxyz_to_xyzw(actions: torch.Tensor, with_gripper: bool) -> torch.Tensor:
    """Swizzle all quat components in an EE action tensor from wxyz to xyzw.
    Layout per arm: [pos(3), quat(4), gripper(1)] or [quat(4)]."""
    dim_per_arm = 8 if with_gripper else 4
    num_arms = actions.shape[-1] // dim_per_arm
    out = actions.clone()
    for arm in range(num_arms):
        s = arm * dim_per_arm
        q_start = s + 3 if with_gripper else s
        q = out[..., q_start : q_start + 4]
        out[..., q_start : q_start + 4] = _swizzle_wxyz_to_xyzw(q)
    return out


def _swizzle_ee_xyzw_to_wxyz(actions: torch.Tensor, with_gripper: bool) -> torch.Tensor:
    """Swizzle all quat components in an EE action tensor from xyzw to wxyz.
    Layout per arm: [pos(3), quat(4), gripper(1)] or [quat(4)]."""
    dim_per_arm = 8 if with_gripper else 4
    num_arms = actions.shape[-1] // dim_per_arm
    out = actions.clone()
    for arm in range(num_arms):
        s = arm * dim_per_arm
        q_start = s + 3 if with_gripper else s
        q = out[..., q_start : q_start + 4]
        out[..., q_start : q_start + 4] = _swizzle_xyzw_to_wxyz(q)
    return out


# ---------------------------------------------------------------------------
# Rotation representation converters (all use matrix as intermediate)
# ---------------------------------------------------------------------------

_TO_MATRIX = {
    "quat": lambda x: roma.unitquat_to_rotmat(x),
    "rpy": lambda x: roma.euler_to_rotmat("xyz", x),
    "6d": rotation_6d_to_matrix,
}

_FROM_MATRIX = {
    "quat": lambda m: roma.rotmat_to_unitquat(m),
    "rpy": lambda m: roma.rotmat_to_euler("xyz", m),
    "6d": matrix_to_rotation_6d,
}

_ROT_DIM = {"quat": 4, "rpy": 3, "6d": 6}


# ---------------------------------------------------------------------------
# EE format conversion (quat/rpy <-> 6d)
# ---------------------------------------------------------------------------


def _convert_ee_rot(actions: torch.Tensor, in_type: str, out_type: str, with_gripper: bool) -> torch.Tensor:
    """Generic EE rotation format conversion."""
    in_rot_dim, out_rot_dim = _ROT_DIM[in_type], _ROT_DIM[out_type]
    in_dim = (3 + in_rot_dim + 1) if with_gripper else in_rot_dim
    out_dim = (3 + out_rot_dim + 1) if with_gripper else out_rot_dim

    shape, dim = actions.shape, actions.shape[-1]
    assert dim % in_dim == 0
    num_arms = dim // in_dim
    flat = actions.reshape(-1, dim)

    to_mat, from_mat = _TO_MATRIX[in_type], _FROM_MATRIX[out_type]
    result = []
    for i in range(num_arms):
        s = i * in_dim
        if with_gripper:
            pos, grip = flat[:, s : s + 3], flat[:, s + 3 + in_rot_dim : s + in_dim]
            rot_out = from_mat(to_mat(flat[:, s + 3 : s + 3 + in_rot_dim]))
            result.append(torch.cat([pos, rot_out, grip], dim=-1))
        else:
            result.append(from_mat(to_mat(flat[:, s : s + in_rot_dim])))
    return torch.cat(result, dim=-1).reshape(*shape[:-1], num_arms * out_dim)


def convert_ee_quat_to_ee_6d(actions: torch.Tensor, with_ee_gripper: bool = True) -> torch.Tensor:
    return _convert_ee_rot(actions, "quat", "6d", with_ee_gripper)


def convert_ee_rpy_to_ee_6d(actions: torch.Tensor, with_ee_gripper: bool = True) -> torch.Tensor:
    return _convert_ee_rot(actions, "rpy", "6d", with_ee_gripper)


def convert_ee_6d_to_ee_quat(actions: torch.Tensor, with_ee_gripper: bool = True) -> torch.Tensor:
    return _convert_ee_rot(actions, "6d", "quat", with_ee_gripper)


def convert_ee_6d_to_ee_rpy(actions: torch.Tensor, with_ee_gripper: bool = True) -> torch.Tensor:
    return _convert_ee_rot(actions, "6d", "rpy", with_ee_gripper)


# wxyz wrappers: swizzle to xyzw, run existing converter, swizzle back if output is quat
def convert_ee_quat_wxyz_to_ee_6d(actions: torch.Tensor, with_ee_gripper: bool = True) -> torch.Tensor:
    return convert_ee_quat_to_ee_6d(_swizzle_ee_wxyz_to_xyzw(actions, with_ee_gripper), with_ee_gripper)


def convert_ee_6d_to_ee_quat_wxyz(actions: torch.Tensor, with_ee_gripper: bool = True) -> torch.Tensor:
    out = convert_ee_6d_to_ee_quat(actions, with_ee_gripper)
    return _swizzle_ee_xyzw_to_wxyz(out, with_ee_gripper)


# ---------------------------------------------------------------------------
# Position delta/absolute (simple subtraction/addition)
# ---------------------------------------------------------------------------


def compute_delta_pos(
    actions: torch.Tensor, state: torch.Tensor, relative_to_state: bool, slices: list | None = None
) -> torch.Tensor:
    """Compute position deltas. (batch, chunk, dim) format."""
    delta = torch.zeros_like(actions)
    if slices is None:
        if relative_to_state:
            return actions - state.unsqueeze(1)
        delta[..., 0, :] = actions[..., 0, :] - state
        delta[..., 1:, :] = actions[..., 1:, :] - actions[..., :-1, :]
    else:
        for s in slices:
            if relative_to_state:
                delta[..., s] = actions[..., s] - state.unsqueeze(1)[..., s]
            else:
                delta[..., 0, s] = actions[..., 0, s] - state[..., s]
                delta[..., 1:, s] = actions[..., 1:, s] - actions[..., :-1, s]
    return delta


def compute_absolute_pos(
    delta: torch.Tensor, state: torch.Tensor, relative_to_state: bool, slices: list | None = None
) -> torch.Tensor:
    """Compute absolute positions from deltas. (batch, chunk, dim) format."""
    if slices is None:
        if relative_to_state:
            return state.unsqueeze(1) + delta
        return state.unsqueeze(1) + torch.cumsum(delta, dim=1)
    absolute = torch.zeros_like(delta)
    for s in slices:
        if relative_to_state:
            absolute[..., s] = state.unsqueeze(1)[..., s] + delta[..., s]
        else:
            absolute[..., s] = state[..., s].unsqueeze(1) + torch.cumsum(delta[..., s], dim=1)
    return absolute


# ---------------------------------------------------------------------------
# Rotation delta/absolute (unified via matrix representation)
# ---------------------------------------------------------------------------


def _compute_rotation_delta_mat(
    actions: torch.Tensor, state: torch.Tensor, rot_slice: slice, relative_to_state: bool, rot_type: str
) -> torch.Tensor:
    """Compute rotation delta via matrix representation."""
    batch, time = actions.shape[0], actions.shape[1]
    rot_dim = _ROT_DIM[rot_type]
    to_mat, from_mat = _TO_MATRIX[rot_type], _FROM_MATRIX[rot_type]

    rot_actions = actions[..., rot_slice].reshape(-1, rot_dim).float()
    rot_state = state[..., rot_slice].float()
    mat_actions = to_mat(rot_actions)
    mat_state = to_mat(rot_state)

    if relative_to_state:
        mat_state_exp = mat_state.unsqueeze(1).expand(-1, time, -1, -1).reshape(-1, 3, 3)
        mat_delta = torch.bmm(mat_state_exp.transpose(-1, -2), mat_actions)
    else:
        rot_prev = torch.cat([rot_state.unsqueeze(1), actions[..., :-1, rot_slice]], dim=1)
        mat_prev = to_mat(rot_prev.reshape(-1, rot_dim))
        mat_delta = torch.bmm(mat_prev.transpose(-1, -2), mat_actions)

    result = from_mat(mat_delta).reshape(batch, time, rot_dim)
    return normalize_angles(result) if rot_type == "rpy" else result


def _compute_rotation_absolute_mat(
    delta: torch.Tensor,
    state: torch.Tensor,
    rot_slice: slice,
    relative_to_state: bool,
    rot_type: str,
    do_normalize: bool = True,
) -> torch.Tensor:
    """Compute absolute rotation from deltas via matrix representation."""
    batch, time = delta.shape[0], delta.shape[1]
    rot_dim = _ROT_DIM[rot_type]
    to_mat, from_mat = _TO_MATRIX[rot_type], _FROM_MATRIX[rot_type]

    rot_deltas = delta[..., rot_slice].reshape(-1, rot_dim).float()
    rot_state = state[..., rot_slice].float()
    mat_deltas = to_mat(rot_deltas)
    mat_state = to_mat(rot_state)

    if relative_to_state:
        mat_state_exp = mat_state.unsqueeze(1).expand(-1, time, -1, -1).reshape(-1, 3, 3)
        mat_abs = torch.bmm(mat_state_exp, mat_deltas)
    else:
        mat_abs = torch.zeros(batch, time, 3, 3, device=delta.device, dtype=delta.dtype)
        mat_curr = mat_state
        mat_deltas_r = mat_deltas.reshape(batch, time, 3, 3)
        for t in range(time):
            mat_curr = torch.bmm(mat_curr, mat_deltas_r[:, t])
            mat_abs[:, t] = mat_curr
        mat_abs = mat_abs.reshape(-1, 3, 3)

    result = from_mat(mat_abs).reshape(batch, time, rot_dim)
    if rot_type == "rpy":
        return normalize_angles(result)
    if do_normalize:
        if rot_type == "quat":
            return normalize_quaternion(result)
        if rot_type == "6d":
            return normalize_rot6d(result.reshape(-1, 6)).reshape(batch, time, 6)
    return result


# ---------------------------------------------------------------------------
# Quaternion-specific (faster path using quaternion multiplication)
# ---------------------------------------------------------------------------


def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions). xyzw format."""
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def _compute_rotation_delta_quat(
    actions: torch.Tensor, state: torch.Tensor, rot_slice: slice, relative_to_state: bool
) -> torch.Tensor:
    """Faster quaternion delta using direct multiplication."""
    batch, time = actions.shape[0], actions.shape[1]
    rot_actions = actions[..., rot_slice].float()
    rot_state = state[..., rot_slice].float()

    if relative_to_state:
        rot_state_inv = _quat_inverse(rot_state.unsqueeze(1).expand(-1, time, -1))
        rot_delta = roma.quat_product(rot_state_inv.reshape(-1, 4), rot_actions.reshape(-1, 4))
    else:
        rot_prev = torch.cat([rot_state.unsqueeze(1), rot_actions[:, :-1]], dim=1)
        rot_delta = roma.quat_product(_quat_inverse(rot_prev).reshape(-1, 4), rot_actions.reshape(-1, 4))
    return rot_delta.reshape(batch, time, 4)


def _compute_rotation_absolute_quat(
    delta: torch.Tensor,
    state: torch.Tensor,
    rot_slice: slice,
    relative_to_state: bool,
    do_normalize: bool = True,
) -> torch.Tensor:
    """Faster quaternion absolute using direct multiplication."""
    batch, time = delta.shape[0], delta.shape[1]
    rot_deltas = delta[..., rot_slice].float()
    rot_state = state[..., rot_slice].float()

    if relative_to_state:
        rot_state_exp = rot_state.unsqueeze(1).expand(-1, time, -1)
        rot_abs = roma.quat_product(rot_state_exp.reshape(-1, 4), rot_deltas.reshape(-1, 4)).reshape(
            batch, time, 4
        )
    else:
        rot_abs = torch.zeros(batch, time, 4, device=delta.device, dtype=delta.dtype)
        rot_curr = rot_state
        for t in range(time):
            rot_curr = roma.quat_product(rot_curr, rot_deltas[:, t])
            rot_abs[:, t] = rot_curr
    return normalize_quaternion(rot_abs) if do_normalize else rot_abs


# ---------------------------------------------------------------------------
# Public API: compute_delta/absolute_ee_{rpy,quat,6d}_pos
# ---------------------------------------------------------------------------


def _compute_ee_delta_or_abs(
    data: torch.Tensor,
    state: torch.Tensor,
    relative_to_state: bool,
    rot_type: str,
    with_gripper: bool,
    is_delta: bool,
) -> torch.Tensor:
    """Unified delta/absolute computation for all EE rotation types."""
    rot_dim = _ROT_DIM[rot_type]
    dim = (3 + rot_dim + 1) if with_gripper else rot_dim
    num_arms = data.shape[-1] // dim
    result = torch.zeros_like(data)

    pos_fn = compute_delta_pos if is_delta else compute_absolute_pos

    for arm in range(num_arms):
        start = arm * dim
        rot_slice = slice(start + 3, start + 3 + rot_dim) if with_gripper else slice(start, start + rot_dim)

        if with_gripper:
            pos_slice = slice(start, start + 3)
            grip_slice = slice(start + 3 + rot_dim, start + dim)
            pos_grip = pos_fn(data, state, relative_to_state, [pos_slice, grip_slice])
            result[..., pos_slice] = pos_grip[..., pos_slice]
            result[..., grip_slice] = pos_grip[..., grip_slice]

        if rot_type == "quat":
            if is_delta:
                result[..., rot_slice] = _compute_rotation_delta_quat(
                    data, state, rot_slice, relative_to_state
                )
            else:
                result[..., rot_slice] = _compute_rotation_absolute_quat(
                    data, state, rot_slice, relative_to_state
                )
        else:
            if is_delta:
                result[..., rot_slice] = _compute_rotation_delta_mat(
                    data, state, rot_slice, relative_to_state, rot_type
                )
            else:
                result[..., rot_slice] = _compute_rotation_absolute_mat(
                    data, state, rot_slice, relative_to_state, rot_type
                )

    return result


def compute_delta_ee_rpy_pos(
    actions: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    return _compute_ee_delta_or_abs(actions, state, relative_to_state, "rpy", with_ee_gripper, True)


def compute_absolute_ee_rpy_pos(
    delta: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    return _compute_ee_delta_or_abs(delta, state, relative_to_state, "rpy", with_ee_gripper, False)


def compute_delta_ee_quat_pos(
    actions: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    return _compute_ee_delta_or_abs(actions, state, relative_to_state, "quat", with_ee_gripper, True)


def compute_absolute_ee_quat_pos(
    delta: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    return _compute_ee_delta_or_abs(delta, state, relative_to_state, "quat", with_ee_gripper, False)


def compute_delta_ee_6d_pos(
    actions: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    return _compute_ee_delta_or_abs(actions, state, relative_to_state, "6d", with_ee_gripper, True)


def compute_absolute_ee_6d_pos(
    delta: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    return _compute_ee_delta_or_abs(delta, state, relative_to_state, "6d", with_ee_gripper, False)


# ---------------------------------------------------------------------------
# wxyz wrappers: swizzle to xyzw, delegate, swizzle back
# ---------------------------------------------------------------------------


def compute_delta_ee_quat_wxyz_pos(
    actions: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    actions_xyzw = _swizzle_ee_wxyz_to_xyzw(actions, with_ee_gripper)
    state_xyzw = _swizzle_ee_wxyz_to_xyzw(state.unsqueeze(1) if state.dim() == 2 else state, with_ee_gripper)
    if state.dim() == 2:
        state_xyzw = state_xyzw.squeeze(1)
    delta = compute_delta_ee_quat_pos(actions_xyzw, state_xyzw, relative_to_state, with_ee_gripper)
    return _swizzle_ee_xyzw_to_wxyz(delta, with_ee_gripper)


def compute_absolute_ee_quat_wxyz_pos(
    delta: torch.Tensor, state: torch.Tensor, relative_to_state: bool, with_ee_gripper: bool = True
) -> torch.Tensor:
    delta_xyzw = _swizzle_ee_wxyz_to_xyzw(delta, with_ee_gripper)
    state_xyzw = _swizzle_ee_wxyz_to_xyzw(state.unsqueeze(1) if state.dim() == 2 else state, with_ee_gripper)
    if state.dim() == 2:
        state_xyzw = state_xyzw.squeeze(1)
    absolute = compute_absolute_ee_quat_pos(delta_xyzw, state_xyzw, relative_to_state, with_ee_gripper)
    return _swizzle_ee_xyzw_to_wxyz(absolute, with_ee_gripper)

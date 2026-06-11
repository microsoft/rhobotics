"""Comprehensive tests for rho.common.rotation_helpers module.

Tests cover:
- Basic utilities: normalize_angles, normalize_quaternion, rotation_6d_to_matrix, etc.
- Quaternion convention swizzling (wxyz <-> xyzw)
- EE action tensor swizzling
- Rotation representation converters (quat/rpy/6d via matrix intermediate)
- EE format conversion (quat/rpy <-> 6d)
- Position delta/absolute computation
- Rotation delta/absolute computation (matrix-based and quaternion-fast paths)
- Public API: compute_delta/absolute_ee_{rpy,quat,6d}_pos
- wxyz wrapper functions
- Round-trip consistency and numerical edge cases
"""

import math

import pytest
import roma
import torch

from rho.common.rotation_helpers import (
    _compute_rotation_absolute_mat,
    _compute_rotation_absolute_quat,
    _compute_rotation_delta_mat,
    _compute_rotation_delta_quat,
    _quat_inverse,
    _swizzle_ee_wxyz_to_xyzw,
    _swizzle_ee_xyzw_to_wxyz,
    _swizzle_wxyz_to_xyzw,
    _swizzle_xyzw_to_wxyz,
    compute_absolute_ee_6d_pos,
    compute_absolute_ee_quat_pos,
    compute_absolute_ee_quat_wxyz_pos,
    compute_absolute_ee_rpy_pos,
    compute_absolute_pos,
    compute_delta_ee_6d_pos,
    compute_delta_ee_quat_pos,
    compute_delta_ee_quat_wxyz_pos,
    compute_delta_ee_rpy_pos,
    compute_delta_pos,
    convert_ee_6d_to_ee_quat,
    convert_ee_6d_to_ee_quat_wxyz,
    convert_ee_6d_to_ee_rpy,
    convert_ee_quat_to_ee_6d,
    convert_ee_quat_wxyz_to_ee_6d,
    convert_ee_rpy_to_ee_6d,
    matrix_to_rotation_6d,
    normalize_angles,
    normalize_quaternion,
    normalize_rot6d,
    rotation_6d_to_matrix,
)
from tests.utils import DEVICE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_quaternion_xyzw():
    """Identity quaternion in (x, y, z, w) format."""
    return torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE)


@pytest.fixture
def identity_quaternion_wxyz():
    """Identity quaternion in (w, x, y, z) format."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)


@pytest.fixture
def identity_rotation_matrix():
    """3x3 identity rotation matrix."""
    return torch.eye(3, device=DEVICE)


@pytest.fixture
def identity_rot6d():
    """Identity rotation in 6D representation (first two columns of identity)."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=DEVICE)


@pytest.fixture
def sample_90deg_z_quat_xyzw():
    """90-degree rotation about Z axis as quaternion (xyzw)."""
    angle = math.pi / 2
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)], device=DEVICE)


@pytest.fixture
def sample_ee_action_quat_single_arm():
    """Sample EE action tensor: [pos(3), quat(4), gripper(1)] for 1 arm, batch=2, chunk=3."""
    torch.manual_seed(42)
    actions = torch.randn(2, 3, 8, device=DEVICE)
    # Normalize the quaternion part
    actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
    return actions


@pytest.fixture
def sample_ee_state_quat_single_arm():
    """Sample EE state tensor: [pos(3), quat(4), gripper(1)] for 1 arm, batch=2."""
    torch.manual_seed(0)
    state = torch.randn(2, 8, device=DEVICE)
    state[..., 3:7] = state[..., 3:7] / state[..., 3:7].norm(dim=-1, keepdim=True)
    return state


# ---------------------------------------------------------------------------
# Test: normalize_angles
# ---------------------------------------------------------------------------


class TestNormalizeAngles:
    """Test the normalize_angles utility function."""

    def test_angles_already_in_range(self):
        """Test that angles already in [-pi, pi] are unchanged (except ±pi boundary)."""
        angles = torch.tensor([-1.0, 0.0, 1.0, math.pi / 2], device=DEVICE)
        result = normalize_angles(angles)
        torch.testing.assert_close(result, angles, atol=1e-6, rtol=1e-6)

    def test_angles_outside_range_positive(self):
        """Test normalization of angles > pi."""
        angles = torch.tensor([2 * math.pi, 3 * math.pi, 5 * math.pi / 2], device=DEVICE)
        result = normalize_angles(angles)
        expected = torch.tensor([0.0, -math.pi, math.pi / 2], device=DEVICE)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_angles_outside_range_negative(self):
        """Test normalization of angles < -pi."""
        angles = torch.tensor([-2 * math.pi, -3 * math.pi, -5 * math.pi / 2], device=DEVICE)
        result = normalize_angles(angles)
        # atan2 maps to (-pi, pi]; -3*pi wraps to pi (not -pi)
        expected = torch.tensor([0.0, math.pi, -math.pi / 2], device=DEVICE)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_angle(self):
        """Test that zero angle remains zero."""
        angles = torch.tensor([0.0], device=DEVICE)
        result = normalize_angles(angles)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_batch_normalization(self):
        """Test normalization with batched tensors."""
        angles = torch.tensor([[4 * math.pi, -4 * math.pi], [math.pi / 4, 7 * math.pi / 4]], device=DEVICE)
        result = normalize_angles(angles)
        expected = torch.tensor([[0.0, 0.0], [math.pi / 4, -math.pi / 4]], device=DEVICE)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: normalize_quaternion
# ---------------------------------------------------------------------------


class TestNormalizeQuaternion:
    """Test the normalize_quaternion utility function."""

    def test_unit_quaternion_unchanged(self, identity_quaternion_xyzw):
        """Test that a unit quaternion is unchanged after normalization."""
        result = normalize_quaternion(identity_quaternion_xyzw)
        torch.testing.assert_close(result, identity_quaternion_xyzw, atol=1e-6, rtol=1e-6)

    def test_non_unit_quaternion_normalized(self):
        """Test that a non-unit quaternion becomes unit length."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        result = normalize_quaternion(q)
        assert torch.norm(result).item() == pytest.approx(1.0, abs=1e-5)

    def test_near_zero_quaternion(self):
        """Test that near-zero quaternion does not produce NaN."""
        q = torch.tensor([1e-10, 1e-10, 1e-10, 1e-10], device=DEVICE)
        result = normalize_quaternion(q)
        assert not torch.isnan(result).any()

    def test_batched_quaternions(self):
        """Test normalization of batched quaternions."""
        q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]], device=DEVICE)
        result = normalize_quaternion(q)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=DEVICE)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_negative_scaling(self):
        """Test that negative scaling still normalizes correctly."""
        q = torch.tensor([-2.0, 0.0, 0.0, 0.0], device=DEVICE)
        result = normalize_quaternion(q)
        assert torch.norm(result).item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test: rotation_6d_to_matrix / matrix_to_rotation_6d
# ---------------------------------------------------------------------------


class TestRotation6DMatrix:
    """Test 6D rotation <-> matrix conversions."""

    def test_identity_6d_to_matrix(self, identity_rot6d, identity_rotation_matrix):
        """Test that identity 6D rotation converts to identity matrix."""
        result = rotation_6d_to_matrix(identity_rot6d)
        torch.testing.assert_close(result, identity_rotation_matrix, atol=1e-6, rtol=1e-6)

    def test_identity_matrix_to_6d(self, identity_rotation_matrix, identity_rot6d):
        """Test that identity matrix converts to identity 6D rotation."""
        result = matrix_to_rotation_6d(identity_rotation_matrix)
        torch.testing.assert_close(result, identity_rot6d, atol=1e-6, rtol=1e-6)

    def test_roundtrip_6d_to_matrix_to_6d(self):
        """Test round-trip: 6D -> matrix -> 6D preserves representation."""
        torch.manual_seed(123)
        # Generate a valid rotation matrix via roma and extract 6D
        rot_mat = roma.random_rotmat(5, device=DEVICE)
        rot6d = matrix_to_rotation_6d(rot_mat)
        reconstructed_mat = rotation_6d_to_matrix(rot6d)
        reconstructed_6d = matrix_to_rotation_6d(reconstructed_mat)
        torch.testing.assert_close(rot6d, reconstructed_6d, atol=1e-5, rtol=1e-5)

    def test_roundtrip_matrix_to_6d_to_matrix(self):
        """Test round-trip: matrix -> 6D -> matrix preserves rotation."""
        torch.manual_seed(456)
        rot_mat = roma.random_rotmat(5, device=DEVICE)
        rot6d = matrix_to_rotation_6d(rot_mat)
        reconstructed = rotation_6d_to_matrix(rot6d)
        torch.testing.assert_close(rot_mat, reconstructed, atol=1e-5, rtol=1e-5)

    def test_output_is_valid_rotation_matrix(self):
        """Test that output of rotation_6d_to_matrix is a valid SO(3) matrix."""
        torch.manual_seed(789)
        rot6d = torch.randn(10, 6, device=DEVICE)
        mat = rotation_6d_to_matrix(rot6d)
        # Check orthogonality: R^T R = I
        rtranspose_r = torch.bmm(mat.transpose(-1, -2), mat)
        identity = torch.eye(3, device=DEVICE).unsqueeze(0).expand(10, -1, -1)
        torch.testing.assert_close(rtranspose_r, identity, atol=1e-5, rtol=1e-5)
        # Check determinant = 1
        dets = torch.det(mat)
        torch.testing.assert_close(dets, torch.ones(10, device=DEVICE), atol=1e-5, rtol=1e-5)

    def test_batched_6d(self):
        """Test 6D conversion with higher-dimensional batched input."""
        torch.manual_seed(111)
        rot6d = torch.randn(3, 4, 6, device=DEVICE)
        mat = rotation_6d_to_matrix(rot6d)
        assert mat.shape == (3, 4, 3, 3)
        reconstructed = matrix_to_rotation_6d(mat)
        # Round-trip via normalized 6D
        normalized_6d = normalize_rot6d(rot6d.reshape(-1, 6)).reshape(3, 4, 6)
        torch.testing.assert_close(reconstructed, normalized_6d, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: normalize_rot6d
# ---------------------------------------------------------------------------


class TestNormalizeRot6D:
    """Test the normalize_rot6d function."""

    def test_already_normalized(self, identity_rot6d):
        """Test that already normalized 6D rotation is unchanged."""
        result = normalize_rot6d(identity_rot6d)
        torch.testing.assert_close(result, identity_rot6d, atol=1e-6, rtol=1e-6)

    def test_unnormalized_becomes_valid(self):
        """Test that arbitrary 6D vectors become valid rotation representations."""
        torch.manual_seed(222)
        rot6d = torch.randn(5, 6, device=DEVICE)
        normalized = normalize_rot6d(rot6d)
        # Going through matrix and back should yield the same result
        mat = rotation_6d_to_matrix(normalized)
        re_extracted = matrix_to_rotation_6d(mat)
        torch.testing.assert_close(normalized, re_extracted, atol=1e-5, rtol=1e-5)

    def test_idempotent(self):
        """Test that normalizing twice gives the same result."""
        torch.manual_seed(333)
        rot6d = torch.randn(5, 6, device=DEVICE)
        once = normalize_rot6d(rot6d)
        twice = normalize_rot6d(once)
        torch.testing.assert_close(once, twice, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test: Quaternion convention swizzling
# ---------------------------------------------------------------------------


class TestQuaternionSwizzling:
    """Test wxyz <-> xyzw quaternion convention swizzling."""

    def test_wxyz_to_xyzw(self):
        """Test converting (w,x,y,z) to (x,y,z,w)."""
        wxyz = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        result = _swizzle_wxyz_to_xyzw(wxyz)
        expected = torch.tensor([2.0, 3.0, 4.0, 1.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_xyzw_to_wxyz(self):
        """Test converting (x,y,z,w) to (w,x,y,z)."""
        xyzw = torch.tensor([2.0, 3.0, 4.0, 1.0], device=DEVICE)
        result = _swizzle_xyzw_to_wxyz(xyzw)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_roundtrip_wxyz_xyzw(self):
        """Test that wxyz->xyzw->wxyz is identity."""
        torch.manual_seed(0)
        q = torch.randn(5, 4, device=DEVICE)
        roundtrip = _swizzle_xyzw_to_wxyz(_swizzle_wxyz_to_xyzw(q))
        torch.testing.assert_close(roundtrip, q)

    def test_roundtrip_xyzw_wxyz(self):
        """Test that xyzw->wxyz->xyzw is identity."""
        torch.manual_seed(1)
        q = torch.randn(5, 4, device=DEVICE)
        roundtrip = _swizzle_wxyz_to_xyzw(_swizzle_xyzw_to_wxyz(q))
        torch.testing.assert_close(roundtrip, q)

    def test_batched_swizzle(self):
        """Test swizzling with higher-dimensional batch."""
        torch.manual_seed(2)
        q = torch.randn(3, 4, 4, device=DEVICE)  # (3, 4, 4)
        result = _swizzle_wxyz_to_xyzw(q)
        assert result.shape == q.shape
        # w component (index 0) should move to index 3
        torch.testing.assert_close(q[..., 0], result[..., 3])
        torch.testing.assert_close(q[..., 1:4], result[..., 0:3])

    def test_identity_quaternion_conversion(self, identity_quaternion_wxyz, identity_quaternion_xyzw):
        """Test swizzling the identity quaternion."""
        result = _swizzle_wxyz_to_xyzw(identity_quaternion_wxyz)
        torch.testing.assert_close(result, identity_quaternion_xyzw)
        result_back = _swizzle_xyzw_to_wxyz(identity_quaternion_xyzw)
        torch.testing.assert_close(result_back, identity_quaternion_wxyz)


# ---------------------------------------------------------------------------
# Test: EE action tensor swizzling
# ---------------------------------------------------------------------------


class TestEESwizzling:
    """Test swizzling quaternions within EE action tensors."""

    def test_single_arm_with_gripper(self):
        """Test swizzling a single-arm EE action with gripper."""
        # [pos(3), quat_wxyz(4), gripper(1)]
        actions = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3, 0.9], device=DEVICE).unsqueeze(0)
        result = _swizzle_ee_wxyz_to_xyzw(actions, with_gripper=True)
        # pos and gripper should be unchanged
        torch.testing.assert_close(result[0, :3], actions[0, :3])
        torch.testing.assert_close(result[0, 7:8], actions[0, 7:8])
        # quat should be swizzled: wxyz -> xyzw
        expected_quat = torch.tensor([0.1, 0.2, 0.3, 0.5], device=DEVICE)
        torch.testing.assert_close(result[0, 3:7], expected_quat)

    def test_single_arm_without_gripper(self):
        """Test swizzling a single-arm EE action without gripper (quat only)."""
        actions = torch.tensor([0.5, 0.1, 0.2, 0.3], device=DEVICE).unsqueeze(0)
        result = _swizzle_ee_wxyz_to_xyzw(actions, with_gripper=False)
        expected = torch.tensor([0.1, 0.2, 0.3, 0.5], device=DEVICE).unsqueeze(0)
        torch.testing.assert_close(result, expected)

    def test_dual_arm_with_gripper(self):
        """Test swizzling a dual-arm EE action with gripper."""
        # 2 arms: [pos(3), quat_wxyz(4), grip(1), pos(3), quat_wxyz(4), grip(1)]
        arm1 = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3, 0.9], device=DEVICE)
        arm2 = torch.tensor([4.0, 5.0, 6.0, 0.7, 0.4, 0.5, 0.6, 0.8], device=DEVICE)
        actions = torch.cat([arm1, arm2]).unsqueeze(0)
        result = _swizzle_ee_wxyz_to_xyzw(actions, with_gripper=True)
        # Check arm 1 quat
        torch.testing.assert_close(result[0, 3:7], torch.tensor([0.1, 0.2, 0.3, 0.5], device=DEVICE))
        # Check arm 2 quat
        torch.testing.assert_close(result[0, 11:15], torch.tensor([0.4, 0.5, 0.6, 0.7], device=DEVICE))

    def test_roundtrip_ee_swizzle(self):
        """Test that swizzling EE actions wx->xyzw->wxyz is identity."""
        torch.manual_seed(42)
        actions = torch.randn(3, 5, 16, device=DEVICE)  # dual arm, with gripper
        roundtrip = _swizzle_ee_xyzw_to_wxyz(
            _swizzle_ee_wxyz_to_xyzw(actions, with_gripper=True), with_gripper=True
        )
        torch.testing.assert_close(roundtrip, actions)

    def test_swizzle_does_not_modify_input(self):
        """Test that swizzling creates a new tensor, not modifying in place."""
        torch.manual_seed(10)
        actions = torch.randn(2, 3, 8, device=DEVICE)
        original = actions.clone()
        _swizzle_ee_wxyz_to_xyzw(actions, with_gripper=True)
        torch.testing.assert_close(actions, original)


# ---------------------------------------------------------------------------
# Test: _quat_inverse
# ---------------------------------------------------------------------------


class TestQuatInverse:
    """Test the quaternion inverse (conjugate) function."""

    def test_identity_inverse(self, identity_quaternion_xyzw):
        """Test that inverse of identity quaternion is identity."""
        result = _quat_inverse(identity_quaternion_xyzw)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_inverse_negates_xyz(self):
        """Test that inverse negates the xyz components and keeps w."""
        q = torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE)
        result = _quat_inverse(q)
        expected = torch.tensor([-0.1, -0.2, -0.3, 0.4], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_inverse_product_is_identity(self):
        """Test that q * q_inv ≈ identity quaternion."""
        torch.manual_seed(42)
        q = torch.randn(5, 4, device=DEVICE)
        q = q / q.norm(dim=-1, keepdim=True)
        q_inv = _quat_inverse(q)
        product = roma.quat_product(q, q_inv)
        # Quaternions are double-cover, so product might be ±identity
        # Check |w| ≈ 1 and xyz ≈ 0
        torch.testing.assert_close(
            product[..., :3].abs(), torch.zeros(5, 3, device=DEVICE), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(product[..., 3].abs(), torch.ones(5, device=DEVICE), atol=1e-5, rtol=1e-5)

    def test_batched_inverse(self):
        """Test inverse with batched quaternions."""
        q = torch.tensor([[0.1, 0.2, 0.3, 0.9], [-0.5, 0.5, -0.5, 0.5]], device=DEVICE)
        result = _quat_inverse(q)
        assert result.shape == q.shape
        torch.testing.assert_close(result[..., 3], q[..., 3])
        torch.testing.assert_close(result[..., :3], -q[..., :3])


# ---------------------------------------------------------------------------
# Test: Rotation representation converters (_convert_ee_rot)
# ---------------------------------------------------------------------------


class TestConvertEERot:
    """Test the generic EE rotation format conversion."""

    def test_quat_to_6d_with_gripper(self):
        """Test converting EE quat to 6D with gripper."""
        torch.manual_seed(42)
        actions = torch.randn(2, 8, device=DEVICE)  # [pos(3), quat(4), grip(1)]
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        result = convert_ee_quat_to_ee_6d(actions, with_ee_gripper=True)
        assert result.shape == (2, 10)  # pos(3) + 6d(6) + grip(1)

    def test_quat_to_6d_without_gripper(self):
        """Test converting EE quat to 6D without gripper (quat only)."""
        torch.manual_seed(42)
        q = torch.randn(3, 4, device=DEVICE)
        q = q / q.norm(dim=-1, keepdim=True)
        result = convert_ee_quat_to_ee_6d(q, with_ee_gripper=False)
        assert result.shape == (3, 6)

    def test_rpy_to_6d_with_gripper(self):
        """Test converting EE rpy to 6D with gripper."""
        torch.manual_seed(42)
        actions = torch.randn(2, 7, device=DEVICE)  # [pos(3), rpy(3), grip(1)]
        result = convert_ee_rpy_to_ee_6d(actions, with_ee_gripper=True)
        assert result.shape[-1] == 10  # pos(3) + 6d(6) + grip(1)

    def test_6d_to_quat_with_gripper(self):
        """Test converting EE 6D to quat with gripper."""
        torch.manual_seed(42)
        actions = torch.randn(2, 10, device=DEVICE)  # [pos(3), 6d(6), grip(1)]
        result = convert_ee_6d_to_ee_quat(actions, with_ee_gripper=True)
        assert result.shape[-1] == 8  # pos(3) + quat(4) + grip(1)

    def test_6d_to_rpy_with_gripper(self):
        """Test converting EE 6D to rpy with gripper."""
        torch.manual_seed(42)
        actions = torch.randn(2, 10, device=DEVICE)  # [pos(3), 6d(6), grip(1)]
        result = convert_ee_6d_to_ee_rpy(actions, with_ee_gripper=True)
        assert result.shape[-1] == 7  # pos(3) + rpy(3) + grip(1)

    def test_roundtrip_quat_6d_with_gripper(self):
        """Test round-trip: quat -> 6D -> quat preserves action."""
        torch.manual_seed(42)
        actions = torch.randn(4, 8, device=DEVICE)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        intermediate = convert_ee_quat_to_ee_6d(actions, with_ee_gripper=True)
        recovered = convert_ee_6d_to_ee_quat(intermediate, with_ee_gripper=True)
        # Pos and gripper should be exact
        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recovered[..., 7:], actions[..., 7:], atol=1e-5, rtol=1e-5)
        # Quaternions can be ±equivalent
        q_orig = actions[..., 3:7]
        q_rec = recovered[..., 3:7]
        # Check that they represent the same rotation
        dot = (q_orig * q_rec).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-4, rtol=1e-4)

    def test_roundtrip_rpy_6d_with_gripper(self):
        """Test round-trip: rpy -> 6D -> rpy preserves action."""
        torch.manual_seed(42)
        actions = torch.randn(4, 7, device=DEVICE)
        # Use small angles to avoid gimbal lock
        actions[..., 3:6] = actions[..., 3:6] * 0.5
        intermediate = convert_ee_rpy_to_ee_6d(actions, with_ee_gripper=True)
        recovered = convert_ee_6d_to_ee_rpy(intermediate, with_ee_gripper=True)
        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recovered[..., 6:], actions[..., 6:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recovered[..., 3:6], actions[..., 3:6], atol=1e-4, rtol=1e-4)

    def test_dual_arm_conversion(self):
        """Test conversion with dual-arm action tensor."""
        torch.manual_seed(42)
        actions = torch.randn(2, 16, device=DEVICE)  # 2 arms * 8 dims
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        actions[..., 11:15] = actions[..., 11:15] / actions[..., 11:15].norm(dim=-1, keepdim=True)
        result = convert_ee_quat_to_ee_6d(actions, with_ee_gripper=True)
        assert result.shape[-1] == 20  # 2 arms * (pos(3) + 6d(6) + grip(1))

    def test_wxyz_to_6d_roundtrip(self):
        """Test round-trip: quat_wxyz -> 6D -> quat_wxyz."""
        torch.manual_seed(42)
        actions = torch.randn(3, 8, device=DEVICE)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        intermediate = convert_ee_quat_wxyz_to_ee_6d(actions, with_ee_gripper=True)
        recovered = convert_ee_6d_to_ee_quat_wxyz(intermediate, with_ee_gripper=True)
        # Position and gripper should match
        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(recovered[..., 7:], actions[..., 7:], atol=1e-5, rtol=1e-5)
        # Quaternions: check rotation equivalence
        q_orig = actions[..., 3:7]
        q_rec = recovered[..., 3:7]
        dot = (q_orig * q_rec).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test: compute_delta_pos / compute_absolute_pos
# ---------------------------------------------------------------------------


class TestPositionDeltaAbsolute:
    """Test position delta and absolute computation."""

    def test_delta_relative_to_state(self):
        """Test delta computation relative to state."""
        actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=DEVICE)  # (1, 2, 2)
        state = torch.tensor([[0.5, 1.0]], device=DEVICE)  # (1, 2)
        delta = compute_delta_pos(actions, state, relative_to_state=True)
        expected = torch.tensor([[[0.5, 1.0], [2.5, 3.0]]], device=DEVICE)
        torch.testing.assert_close(delta, expected)

    def test_delta_sequential(self):
        """Test sequential delta computation (not relative to state)."""
        actions = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]]], device=DEVICE)
        state = torch.tensor([[0.0, 0.0]], device=DEVICE)
        delta = compute_delta_pos(actions, state, relative_to_state=False)
        expected = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 3.0]]], device=DEVICE)
        torch.testing.assert_close(delta, expected)

    def test_absolute_relative_to_state(self):
        """Test absolute computation relative to state (inverse of delta)."""
        state = torch.tensor([[1.0, 2.0]], device=DEVICE)
        delta = torch.tensor([[[0.5, 1.0], [2.5, 3.0]]], device=DEVICE)
        absolute = compute_absolute_pos(delta, state, relative_to_state=True)
        expected = torch.tensor([[[1.5, 3.0], [3.5, 5.0]]], device=DEVICE)
        torch.testing.assert_close(absolute, expected)

    def test_absolute_sequential(self):
        """Test sequential absolute computation via cumsum."""
        state = torch.tensor([[0.0, 0.0]], device=DEVICE)
        delta = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 3.0]]], device=DEVICE)
        absolute = compute_absolute_pos(delta, state, relative_to_state=False)
        expected = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]]], device=DEVICE)
        torch.testing.assert_close(absolute, expected)

    def test_roundtrip_delta_absolute_relative(self):
        """Test that delta -> absolute -> delta is identity (relative_to_state)."""
        torch.manual_seed(42)
        actions = torch.randn(2, 5, 3, device=DEVICE)
        state = torch.randn(2, 3, device=DEVICE)
        delta = compute_delta_pos(actions, state, relative_to_state=True)
        recovered = compute_absolute_pos(delta, state, relative_to_state=True)
        torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)

    def test_roundtrip_delta_absolute_sequential(self):
        """Test that delta -> absolute -> delta is identity (sequential)."""
        torch.manual_seed(42)
        actions = torch.randn(2, 5, 3, device=DEVICE)
        state = torch.randn(2, 3, device=DEVICE)
        delta = compute_delta_pos(actions, state, relative_to_state=False)
        recovered = compute_absolute_pos(delta, state, relative_to_state=False)
        torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)

    def test_delta_with_slices(self):
        """Test delta computation with specific slices."""
        actions = torch.tensor([[[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]]], device=DEVICE)
        state = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
        delta = compute_delta_pos(actions, state, relative_to_state=True, slices=[slice(0, 2)])
        # Only first two dims should have deltas, third should be zeros
        torch.testing.assert_close(delta[..., 0:2], actions[..., 0:2] - state[:, None, 0:2])
        torch.testing.assert_close(delta[..., 2], torch.zeros(1, 2, device=DEVICE))

    def test_absolute_with_slices(self):
        """Test absolute computation with specific slices."""
        delta = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]], device=DEVICE)
        state = torch.tensor([[0.5, 1.0, 5.0]], device=DEVICE)
        absolute = compute_absolute_pos(delta, state, relative_to_state=True, slices=[slice(0, 2)])
        torch.testing.assert_close(absolute[..., 0], delta[..., 0] + state[:, None, 0])
        torch.testing.assert_close(absolute[..., 1], delta[..., 1] + state[:, None, 1])

    def test_batch_consistency(self):
        """Test that batched computation matches per-sample computation."""
        torch.manual_seed(42)
        actions = torch.randn(4, 3, 5, device=DEVICE)
        state = torch.randn(4, 5, device=DEVICE)
        delta_batched = compute_delta_pos(actions, state, relative_to_state=False)
        for i in range(4):
            delta_single = compute_delta_pos(actions[i : i + 1], state[i : i + 1], relative_to_state=False)
            torch.testing.assert_close(delta_batched[i : i + 1], delta_single, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test: Rotation delta/absolute (matrix-based)
# ---------------------------------------------------------------------------


class TestRotationDeltaAbsoluteMatrix:
    """Test rotation delta/absolute computation via matrix representation."""

    @pytest.mark.parametrize("rot_type", ["quat", "rpy", "6d"])
    def test_identity_delta_relative_to_state(self, rot_type):
        """Test that identical actions and state produce identity delta."""
        rot_dim = {"quat": 4, "rpy": 3, "6d": 6}[rot_type]
        if rot_type == "quat":
            rot_val = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE)
        elif rot_type == "rpy":
            rot_val = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        else:
            rot_val = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=DEVICE)

        state = rot_val.unsqueeze(0)  # (1, rot_dim)
        actions = rot_val.unsqueeze(0).unsqueeze(0).expand(1, 3, -1)  # (1, 3, rot_dim)
        rot_slice = slice(0, rot_dim)

        delta = _compute_rotation_delta_mat(
            actions, state, rot_slice, relative_to_state=True, rot_type=rot_type
        )
        assert delta.shape == (1, 3, rot_dim)

        if rot_type == "quat":
            # Delta should be identity quaternion
            expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE).expand(1, 3, -1)
            dot = (delta * expected).sum(dim=-1).abs()
            torch.testing.assert_close(dot, torch.ones(1, 3, device=DEVICE), atol=1e-4, rtol=1e-4)
        elif rot_type == "rpy":
            # Delta should be zero angles
            torch.testing.assert_close(delta, torch.zeros(1, 3, 3, device=DEVICE), atol=1e-4, rtol=1e-4)
        else:
            # Delta should be identity 6D
            expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=DEVICE).expand(1, 3, -1)
            torch.testing.assert_close(delta, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("rot_type", ["quat", "rpy", "6d"])
    def test_roundtrip_delta_absolute_relative(self, rot_type):
        """Test that delta -> absolute -> check matches original (relative_to_state)."""
        torch.manual_seed(42)
        rot_dim = {"quat": 4, "rpy": 3, "6d": 6}[rot_type]
        batch, time = 2, 4

        if rot_type == "quat":
            rot_vals = torch.randn(batch, time, 4, device=DEVICE)
            rot_vals = rot_vals / rot_vals.norm(dim=-1, keepdim=True)
            state_rot = torch.randn(batch, 4, device=DEVICE)
            state_rot = state_rot / state_rot.norm(dim=-1, keepdim=True)
        elif rot_type == "rpy":
            rot_vals = torch.randn(batch, time, 3, device=DEVICE) * 0.5
            state_rot = torch.randn(batch, 3, device=DEVICE) * 0.5
        else:
            mat = roma.random_rotmat(batch * time, device=DEVICE).reshape(batch, time, 3, 3)
            rot_vals = matrix_to_rotation_6d(mat.reshape(-1, 3, 3)).reshape(batch, time, 6)
            state_mat = roma.random_rotmat(batch, device=DEVICE)
            state_rot = matrix_to_rotation_6d(state_mat)

        rot_slice = slice(0, rot_dim)
        delta = _compute_rotation_delta_mat(rot_vals, state_rot, rot_slice, True, rot_type)
        recovered = _compute_rotation_absolute_mat(delta, state_rot, rot_slice, True, rot_type)

        if rot_type == "quat":
            dot = (rot_vals * recovered).sum(dim=-1).abs()
            torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)
        elif rot_type == "rpy":
            diff = normalize_angles(rot_vals - recovered)
            torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-3, rtol=1e-3)
        else:
            mat_orig = rotation_6d_to_matrix(rot_vals.reshape(-1, 6))
            mat_rec = rotation_6d_to_matrix(recovered.reshape(-1, 6))
            torch.testing.assert_close(mat_orig, mat_rec, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("rot_type", ["quat", "rpy", "6d"])
    def test_roundtrip_delta_absolute_sequential(self, rot_type):
        """Test round-trip delta/absolute with sequential (not relative_to_state)."""
        torch.manual_seed(99)
        rot_dim = {"quat": 4, "rpy": 3, "6d": 6}[rot_type]
        batch, time = 2, 3

        if rot_type == "quat":
            rot_vals = torch.randn(batch, time, 4, device=DEVICE)
            rot_vals = rot_vals / rot_vals.norm(dim=-1, keepdim=True)
            state_rot = torch.randn(batch, 4, device=DEVICE)
            state_rot = state_rot / state_rot.norm(dim=-1, keepdim=True)
        elif rot_type == "rpy":
            rot_vals = torch.randn(batch, time, 3, device=DEVICE) * 0.3
            state_rot = torch.randn(batch, 3, device=DEVICE) * 0.3
        else:
            mat = roma.random_rotmat(batch * time, device=DEVICE).reshape(batch, time, 3, 3)
            rot_vals = matrix_to_rotation_6d(mat.reshape(-1, 3, 3)).reshape(batch, time, 6)
            state_mat = roma.random_rotmat(batch, device=DEVICE)
            state_rot = matrix_to_rotation_6d(state_mat)

        rot_slice = slice(0, rot_dim)
        delta = _compute_rotation_delta_mat(rot_vals, state_rot, rot_slice, False, rot_type)
        recovered = _compute_rotation_absolute_mat(delta, state_rot, rot_slice, False, rot_type)

        if rot_type == "quat":
            dot = (rot_vals * recovered).sum(dim=-1).abs()
            torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)
        elif rot_type == "rpy":
            diff = normalize_angles(rot_vals - recovered)
            torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-3, rtol=1e-3)
        else:
            mat_orig = rotation_6d_to_matrix(rot_vals.reshape(-1, 6))
            mat_rec = rotation_6d_to_matrix(recovered.reshape(-1, 6))
            torch.testing.assert_close(mat_orig, mat_rec, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Rotation delta/absolute (quaternion fast path)
# ---------------------------------------------------------------------------


class TestRotationDeltaAbsoluteQuat:
    """Test rotation delta/absolute computation using quaternion multiplication."""

    def test_identity_delta(self, identity_quaternion_xyzw):
        """Test delta when actions equal state produces identity rotation."""
        state = identity_quaternion_xyzw.unsqueeze(0)
        actions = identity_quaternion_xyzw.unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        rot_slice = slice(0, 4)
        delta = _compute_rotation_delta_quat(actions, state, rot_slice, relative_to_state=True)
        # Should be identity quaternion (0,0,0,1)
        expected = identity_quaternion_xyzw.unsqueeze(0).expand(1, 3, -1)
        dot = (delta * expected).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(1, 3, device=DEVICE), atol=1e-5, rtol=1e-5)

    def test_roundtrip_relative(self):
        """Test delta -> absolute round-trip (relative_to_state) via quat path."""
        torch.manual_seed(42)
        batch, time = 3, 5
        actions = torch.randn(batch, time, 4, device=DEVICE)
        actions = actions / actions.norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 4, device=DEVICE)
        state = state / state.norm(dim=-1, keepdim=True)
        rot_slice = slice(0, 4)

        delta = _compute_rotation_delta_quat(actions, state, rot_slice, True)
        recovered = _compute_rotation_absolute_quat(delta, state, rot_slice, True)

        dot = (actions * recovered).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-4, rtol=1e-4)

    def test_roundtrip_sequential(self):
        """Test delta -> absolute round-trip (sequential) via quat path."""
        torch.manual_seed(99)
        batch, time = 2, 4
        actions = torch.randn(batch, time, 4, device=DEVICE)
        actions = actions / actions.norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 4, device=DEVICE)
        state = state / state.norm(dim=-1, keepdim=True)
        rot_slice = slice(0, 4)

        delta = _compute_rotation_delta_quat(actions, state, rot_slice, False)
        recovered = _compute_rotation_absolute_quat(delta, state, rot_slice, False)

        dot = (actions * recovered).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-4, rtol=1e-4)

    def test_consistency_with_matrix_path(self):
        """Test that quat fast path matches matrix-based path."""
        torch.manual_seed(42)
        batch, time = 2, 3
        actions = torch.randn(batch, time, 4, device=DEVICE)
        actions = actions / actions.norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 4, device=DEVICE)
        state = state / state.norm(dim=-1, keepdim=True)
        rot_slice = slice(0, 4)

        delta_quat = _compute_rotation_delta_quat(actions, state, rot_slice, True)
        delta_mat = _compute_rotation_delta_mat(actions, state, rot_slice, True, "quat")

        # Both should represent the same rotation
        dot = (delta_quat * delta_mat).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Public API - compute_delta/absolute_ee_{quat,rpy,6d}_pos
# ---------------------------------------------------------------------------


class TestPublicAPIDeltaAbsolute:
    """Test the public API functions for EE delta/absolute computation."""

    @pytest.mark.parametrize("relative_to_state", [True, False])
    def test_roundtrip_ee_quat_with_gripper(self, relative_to_state):
        """Test round-trip for EE quat actions with gripper."""
        torch.manual_seed(42)
        batch, time = 2, 4
        actions = torch.randn(batch, time, 8, device=DEVICE)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 8, device=DEVICE)
        state[..., 3:7] = state[..., 3:7] / state[..., 3:7].norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_pos(actions, state, relative_to_state, with_ee_gripper=True)
        recovered = compute_absolute_ee_quat_pos(delta, state, relative_to_state, with_ee_gripper=True)

        # Position and gripper should match
        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(recovered[..., 7:], actions[..., 7:], atol=1e-4, rtol=1e-4)
        # Quaternion should represent same rotation
        dot = (actions[..., 3:7] * recovered[..., 3:7]).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("relative_to_state", [True, False])
    def test_roundtrip_ee_rpy_with_gripper(self, relative_to_state):
        """Test round-trip for EE rpy actions with gripper."""
        torch.manual_seed(42)
        batch, time = 2, 3
        actions = torch.randn(batch, time, 7, device=DEVICE)
        actions[..., 3:6] *= 0.3  # small angles to avoid gimbal lock
        state = torch.randn(batch, 7, device=DEVICE)
        state[..., 3:6] *= 0.3

        delta = compute_delta_ee_rpy_pos(actions, state, relative_to_state, with_ee_gripper=True)
        recovered = compute_absolute_ee_rpy_pos(delta, state, relative_to_state, with_ee_gripper=True)

        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(recovered[..., 6:], actions[..., 6:], atol=1e-4, rtol=1e-4)
        diff = normalize_angles(recovered[..., 3:6] - actions[..., 3:6])
        torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("relative_to_state", [True, False])
    def test_roundtrip_ee_6d_with_gripper(self, relative_to_state):
        """Test round-trip for EE 6D actions with gripper."""
        torch.manual_seed(42)
        batch, time = 2, 3
        # Build valid 6D rotations
        mat = roma.random_rotmat(batch * time, device=DEVICE).reshape(batch, time, 3, 3)
        rot6d = matrix_to_rotation_6d(mat.reshape(-1, 3, 3)).reshape(batch, time, 6)
        pos = torch.randn(batch, time, 3, device=DEVICE)
        grip = torch.randn(batch, time, 1, device=DEVICE)
        actions = torch.cat([pos, rot6d, grip], dim=-1)

        state_mat = roma.random_rotmat(batch, device=DEVICE)
        state_rot6d = matrix_to_rotation_6d(state_mat)
        state_pos = torch.randn(batch, 3, device=DEVICE)
        state_grip = torch.randn(batch, 1, device=DEVICE)
        state = torch.cat([state_pos, state_rot6d, state_grip], dim=-1)

        delta = compute_delta_ee_6d_pos(actions, state, relative_to_state, with_ee_gripper=True)
        recovered = compute_absolute_ee_6d_pos(delta, state, relative_to_state, with_ee_gripper=True)

        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(recovered[..., 9:], actions[..., 9:], atol=1e-4, rtol=1e-4)
        # Compare 6D via rotation matrix
        mat_orig = rotation_6d_to_matrix(actions[..., 3:9].reshape(-1, 6))
        mat_rec = rotation_6d_to_matrix(recovered[..., 3:9].reshape(-1, 6))
        torch.testing.assert_close(mat_orig, mat_rec, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("relative_to_state", [True, False])
    def test_roundtrip_ee_quat_without_gripper(self, relative_to_state):
        """Test round-trip for EE quat actions without gripper (quat only)."""
        torch.manual_seed(42)
        batch, time = 2, 3
        actions = torch.randn(batch, time, 4, device=DEVICE)
        actions = actions / actions.norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 4, device=DEVICE)
        state = state / state.norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_pos(actions, state, relative_to_state, with_ee_gripper=False)
        recovered = compute_absolute_ee_quat_pos(delta, state, relative_to_state, with_ee_gripper=False)

        dot = (actions * recovered).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)

    def test_dual_arm_ee_quat_roundtrip(self):
        """Test round-trip for dual-arm EE quat with gripper."""
        torch.manual_seed(42)
        batch, time = 2, 3
        actions = torch.randn(batch, time, 16, device=DEVICE)  # 2 arms * 8
        for arm_start in [0, 8]:
            q = actions[..., arm_start + 3 : arm_start + 7]
            actions[..., arm_start + 3 : arm_start + 7] = q / q.norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 16, device=DEVICE)
        for arm_start in [0, 8]:
            q = state[..., arm_start + 3 : arm_start + 7]
            state[..., arm_start + 3 : arm_start + 7] = q / q.norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_pos(actions, state, True, with_ee_gripper=True)
        recovered = compute_absolute_ee_quat_pos(delta, state, True, with_ee_gripper=True)

        # Check both arms
        for arm_start in [0, 8]:
            torch.testing.assert_close(
                recovered[..., arm_start : arm_start + 3],
                actions[..., arm_start : arm_start + 3],
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                recovered[..., arm_start + 7 : arm_start + 8],
                actions[..., arm_start + 7 : arm_start + 8],
                atol=1e-4,
                rtol=1e-4,
            )
            dot = (
                (actions[..., arm_start + 3 : arm_start + 7] * recovered[..., arm_start + 3 : arm_start + 7])
                .sum(dim=-1)
                .abs()
            )
            torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: wxyz wrappers
# ---------------------------------------------------------------------------


class TestWxyzWrappers:
    """Test compute_delta/absolute_ee_quat_wxyz_pos wrapper functions."""

    @pytest.mark.parametrize("relative_to_state", [True, False])
    def test_roundtrip_wxyz_with_gripper(self, relative_to_state):
        """Test round-trip for wxyz EE quat with gripper."""
        torch.manual_seed(42)
        batch, time = 2, 3
        actions = torch.randn(batch, time, 8, device=DEVICE)
        # Normalize quat in wxyz format (w at index 3)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 8, device=DEVICE)
        state[..., 3:7] = state[..., 3:7] / state[..., 3:7].norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_wxyz_pos(actions, state, relative_to_state, with_ee_gripper=True)
        recovered = compute_absolute_ee_quat_wxyz_pos(delta, state, relative_to_state, with_ee_gripper=True)

        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(recovered[..., 7:], actions[..., 7:], atol=1e-4, rtol=1e-4)
        dot = (actions[..., 3:7] * recovered[..., 3:7]).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)

    def test_wxyz_vs_xyzw_consistency(self):
        """Test that wxyz wrapper gives same rotation as manually swizzling + xyzw API."""
        torch.manual_seed(42)
        batch, time = 2, 3
        actions_wxyz = torch.randn(batch, time, 8, device=DEVICE)
        actions_wxyz[..., 3:7] = actions_wxyz[..., 3:7] / actions_wxyz[..., 3:7].norm(dim=-1, keepdim=True)
        state_wxyz = torch.randn(batch, 8, device=DEVICE)
        state_wxyz[..., 3:7] = state_wxyz[..., 3:7] / state_wxyz[..., 3:7].norm(dim=-1, keepdim=True)

        # Use wxyz wrapper
        delta_wxyz = compute_delta_ee_quat_wxyz_pos(actions_wxyz, state_wxyz, True, with_ee_gripper=True)

        # Manually swizzle and use xyzw API
        actions_xyzw = _swizzle_ee_wxyz_to_xyzw(actions_wxyz, with_gripper=True)
        state_xyzw = _swizzle_ee_wxyz_to_xyzw(state_wxyz.unsqueeze(1), with_gripper=True).squeeze(1)
        delta_xyzw = compute_delta_ee_quat_pos(actions_xyzw, state_xyzw, True, with_ee_gripper=True)
        delta_manual_wxyz = _swizzle_ee_xyzw_to_wxyz(delta_xyzw, with_gripper=True)

        # Position and gripper should match exactly
        torch.testing.assert_close(delta_wxyz[..., :3], delta_manual_wxyz[..., :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(delta_wxyz[..., 7:], delta_manual_wxyz[..., 7:], atol=1e-5, rtol=1e-5)
        # Quat delta should represent same rotation
        dot = (delta_wxyz[..., 3:7] * delta_manual_wxyz[..., 3:7]).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Edge cases and numerical stability
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_single_timestep(self):
        """Test with a single timestep (chunk_size=1)."""
        torch.manual_seed(42)
        batch = 3
        actions = torch.randn(batch, 1, 8, device=DEVICE)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 8, device=DEVICE)
        state[..., 3:7] = state[..., 3:7] / state[..., 3:7].norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_pos(actions, state, True, with_ee_gripper=True)
        recovered = compute_absolute_ee_quat_pos(delta, state, True, with_ee_gripper=True)
        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-4, rtol=1e-4)

    def test_batch_size_one(self):
        """Test with batch_size=1."""
        torch.manual_seed(42)
        actions = torch.randn(1, 5, 8, device=DEVICE)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        state = torch.randn(1, 8, device=DEVICE)
        state[..., 3:7] = state[..., 3:7] / state[..., 3:7].norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_pos(actions, state, True, with_ee_gripper=True)
        recovered = compute_absolute_ee_quat_pos(delta, state, True, with_ee_gripper=True)
        dot = (actions[..., 3:7] * recovered[..., 3:7]).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(1, 5, device=DEVICE), atol=1e-3, rtol=1e-3)

    def test_180_degree_rotation(self):
        """Test delta computation with 180-degree rotation difference."""
        # State: identity
        state = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
        # Action: 180-degree rotation about Z
        action = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], device=DEVICE)
        rot_slice = slice(0, 4)
        delta = _compute_rotation_delta_quat(action, state, rot_slice, relative_to_state=True)
        recovered = _compute_rotation_absolute_quat(delta, state, rot_slice, relative_to_state=True)
        dot = (action * recovered).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(1, 1, device=DEVICE), atol=1e-4, rtol=1e-4)

    def test_small_rotation(self):
        """Test delta computation with very small rotation difference."""
        eps = 1e-4
        state = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=DEVICE)
        action = torch.tensor([[[eps, 0.0, 0.0, 1.0]]], device=DEVICE)
        action = action / action.norm(dim=-1, keepdim=True)
        rot_slice = slice(0, 4)
        delta = _compute_rotation_delta_quat(action, state, rot_slice, relative_to_state=True)
        recovered = _compute_rotation_absolute_quat(delta, state, rot_slice, relative_to_state=True)
        dot = (action * recovered).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(1, 1, device=DEVICE), atol=1e-4, rtol=1e-4)

    def test_zero_position_delta(self):
        """Test that zero delta produces original state."""
        state = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)
        delta = torch.zeros(1, 5, 3, device=DEVICE)
        absolute = compute_absolute_pos(delta, state, relative_to_state=True)
        expected = state.unsqueeze(1).expand(1, 5, 3)
        torch.testing.assert_close(absolute, expected)

    def test_large_batch_and_chunk(self):
        """Test with larger batch and chunk sizes for numerical stability."""
        torch.manual_seed(42)
        batch, time = 16, 20
        actions = torch.randn(batch, time, 8, device=DEVICE)
        actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
        state = torch.randn(batch, 8, device=DEVICE)
        state[..., 3:7] = state[..., 3:7] / state[..., 3:7].norm(dim=-1, keepdim=True)

        delta = compute_delta_ee_quat_pos(actions, state, True, with_ee_gripper=True)
        recovered = compute_absolute_ee_quat_pos(delta, state, True, with_ee_gripper=True)
        torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-4, rtol=1e-4)
        dot = (actions[..., 3:7] * recovered[..., 3:7]).sum(dim=-1).abs()
        torch.testing.assert_close(dot, torch.ones(batch, time, device=DEVICE), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Output shapes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    """Test that all functions produce correct output shapes."""

    @pytest.mark.parametrize(
        "func,in_dim,out_dim,with_gripper",
        [
            (convert_ee_quat_to_ee_6d, 8, 10, True),
            (convert_ee_quat_to_ee_6d, 4, 6, False),
            (convert_ee_rpy_to_ee_6d, 7, 10, True),
            (convert_ee_rpy_to_ee_6d, 3, 6, False),
            (convert_ee_6d_to_ee_quat, 10, 8, True),
            (convert_ee_6d_to_ee_quat, 6, 4, False),
            (convert_ee_6d_to_ee_rpy, 10, 7, True),
            (convert_ee_6d_to_ee_rpy, 6, 3, False),
        ],
    )
    def test_conversion_output_shape(self, func, in_dim, out_dim, with_gripper):
        """Test that conversion functions produce correct output shapes."""
        torch.manual_seed(42)
        actions = torch.randn(3, in_dim, device=DEVICE)
        if "quat" in func.__name__ and "to" in func.__name__:
            # Normalize quaternion components if input contains quats
            if with_gripper:
                actions[..., 3:7] = actions[..., 3:7] / actions[..., 3:7].norm(dim=-1, keepdim=True)
            else:
                actions = actions / actions.norm(dim=-1, keepdim=True)
        result = func(actions, with_ee_gripper=with_gripper)
        assert result.shape == (3, out_dim)

    @pytest.mark.parametrize(
        "delta_func,abs_func,dim,with_gripper",
        [
            (compute_delta_ee_quat_pos, compute_absolute_ee_quat_pos, 8, True),
            (compute_delta_ee_rpy_pos, compute_absolute_ee_rpy_pos, 7, True),
            (compute_delta_ee_6d_pos, compute_absolute_ee_6d_pos, 10, True),
            (compute_delta_ee_quat_pos, compute_absolute_ee_quat_pos, 4, False),
        ],
    )
    def test_delta_absolute_output_shape(self, delta_func, abs_func, dim, with_gripper):
        """Test that delta/absolute functions preserve shape."""
        torch.manual_seed(42)
        batch, time = 3, 5
        actions = torch.randn(batch, time, dim, device=DEVICE)
        state = torch.randn(batch, dim, device=DEVICE)
        # Normalize quaternions if needed
        if dim in [8, 4]:
            q_start = 3 if dim == 8 else 0
            q_end = 7 if dim == 8 else 4
            actions[..., q_start:q_end] = actions[..., q_start:q_end] / actions[..., q_start:q_end].norm(
                dim=-1, keepdim=True
            )
            state[..., q_start:q_end] = state[..., q_start:q_end] / state[..., q_start:q_end].norm(
                dim=-1, keepdim=True
            )

        delta = delta_func(actions, state, True, with_ee_gripper=with_gripper)
        assert delta.shape == (batch, time, dim)

        absolute = abs_func(delta, state, True, with_ee_gripper=with_gripper)
        assert absolute.shape == (batch, time, dim)


# ---------------------------------------------------------------------------
# Test: Device and dtype handling
# ---------------------------------------------------------------------------


class TestDeviceDtype:
    """Test that functions preserve device and dtype."""

    def test_output_device_matches_input(self):
        """Test that output tensors are on the same device as input."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        result = normalize_quaternion(q)
        assert result.device == q.device

    def test_float32_preserved(self):
        """Test that float32 dtype is preserved."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=DEVICE)
        result = normalize_quaternion(q)
        assert result.dtype == torch.float32

    def test_normalize_angles_dtype(self):
        """Test that normalize_angles preserves dtype."""
        angles = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=DEVICE)
        result = normalize_angles(angles)
        assert result.dtype == torch.float32

    def test_6d_conversion_dtype(self):
        """Test that 6D conversion preserves dtype."""
        rot6d = torch.randn(3, 6, dtype=torch.float32, device=DEVICE)
        mat = rotation_6d_to_matrix(rot6d)
        assert mat.dtype == torch.float32
        back = matrix_to_rotation_6d(mat)
        assert back.dtype == torch.float32

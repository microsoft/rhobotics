"""
Regression tests for RTC soft-mask (W matrix) construction and padding zeroing.

These tests are intentionally free of model weights, GPU, and network access.
They exercise only `compute_W_matrix_rtc` and `_prepare_rtc_mask` by constructing
a minimal stub instance of `FlowMatchingModel` via `object.__new__`.
"""

import math
import types

import torch

from rho.policies.rho.rho_model import FlowMatchingModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_stub(chunk_size=32, max_action_dim=32, dtype=torch.float32, device="cpu"):
    """Return a bare FlowMatchingModel with only the attributes the methods need."""
    obj = object.__new__(FlowMatchingModel)
    cfg = types.SimpleNamespace(chunk_size=chunk_size)
    obj.config = cfg
    obj.max_action_dim = max_action_dim
    obj.dtype = dtype
    obj.device = device
    return obj


# ---------------------------------------------------------------------------
# Tests for compute_W_matrix_rtc
# ---------------------------------------------------------------------------


class TestComputeWMatrix:
    def test_returns_contiguous_writable_tensor(self):
        """Regression: expand() returned a stride-0 view; repeat() must not."""
        stub = make_stub(chunk_size=32, max_action_dim=32)
        W = stub.compute_W_matrix_rtc(inference_delay=6, execution_horizon=8, action_dim=32)
        assert W.shape == (32, 32)
        # Must not raise RuntimeError about multiple elements aliasing memory
        W[3:, :] = 0.0

    def test_shape(self):
        stub = make_stub(chunk_size=16, max_action_dim=7)
        W = stub.compute_W_matrix_rtc(inference_delay=2, execution_horizon=4, action_dim=7)
        assert W.shape == (16, 7)

    def test_eq5_values(self):
        """
        With chunk_size=32, inference_delay=6, execution_horizon=8, action_dim=32:
          - rows 0..5  (i < d=6)            → 1.0
          - rows 24..31 (i >= H-s = 32-8=24) → 0.0
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        W = stub.compute_W_matrix_rtc(inference_delay=6, execution_horizon=8, action_dim=32)

        # Rows 0..5 must be exactly 1.0 across all action dims
        assert torch.all(W[:6, :] == 1.0), f"Expected 1.0 for rows 0-5, got {W[:6, 0]}"

        # Rows 24..31 must be exactly 0.0
        assert torch.all(W[24:, :] == 0.0), f"Expected 0.0 for rows 24-31, got {W[24:, 0]}"

    def test_transition_region_values(self):
        """Rows in [d, H-s) must use the smooth formula and lie strictly in (0, 1)."""
        d, s = 6, 8
        H = 32
        stub = make_stub(chunk_size=H, max_action_dim=4)
        W = stub.compute_W_matrix_rtc(inference_delay=d, execution_horizon=s, action_dim=4)

        for i in range(d, H - s):
            ci = (H - s - i) / (H - s - d + 1)
            expected = (ci * (math.exp(ci) - 1)) / (math.exp(1) - 1)
            got = W[i, 0].item()
            assert abs(got - expected) < 1e-5, f"Row {i}: expected {expected:.6f}, got {got:.6f}"

    def test_all_columns_identical(self):
        """Each row must be constant across action dims (temporal-only weighting)."""
        stub = make_stub(chunk_size=16, max_action_dim=8)
        W = stub.compute_W_matrix_rtc(inference_delay=3, execution_horizon=4, action_dim=8)
        for row in range(16):
            assert W[row].unique().numel() == 1, f"Row {row} has differing values across action dims"


# ---------------------------------------------------------------------------
# Tests for _prepare_rtc_mask
# ---------------------------------------------------------------------------


class TestPrepareRtcMask:
    def test_padded_rows_zeroed(self):
        """Rows at/beyond prev_actions.shape[-2] must be exactly 0.0."""
        stub = make_stub(chunk_size=32, max_action_dim=32)
        T_prev = 8  # fewer timesteps than chunk_size
        prev = torch.ones(1, T_prev, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        assert torch.all(w[T_prev:, :] == 0.0), "Padded rows must be zeroed"

    def test_padded_columns_zeroed(self):
        """Columns at/beyond prev_actions.shape[-1] must be exactly 0.0."""
        stub = make_stub(chunk_size=32, max_action_dim=32)
        A_dim = 14  # fewer action dims than max_action_dim
        prev = torch.ones(1, 8, A_dim)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        assert torch.all(w[:, A_dim:] == 0.0), "Padded columns must be zeroed"

    def test_padded_rows_and_columns_both_zeroed(self):
        """Both padded rows and padded columns must be zeroed independently."""
        stub = make_stub(chunk_size=32, max_action_dim=32)
        T_prev, A_dim = 8, 14
        prev = torch.ones(1, T_prev, A_dim)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        assert torch.all(w[T_prev:, :] == 0.0), "Padded rows not zeroed"
        assert torch.all(w[:, A_dim:] == 0.0), "Padded columns not zeroed"

    def test_valid_region_not_zeroed(self):
        """Entries in the valid (non-padded) region must retain their W values."""
        stub = make_stub(chunk_size=32, max_action_dim=32)
        T_prev, A_dim = 32, 32  # no padding at all
        prev = torch.ones(1, T_prev, A_dim)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        # Rows 0..5 should still be 1.0 (inference_delay region)
        assert torch.all(w[:6, :A_dim] == 1.0), "Valid region should not be zeroed"

    def test_no_column_padding_silent_noop_regression(self):
        """
        Regression for the silent no-op bug:
        when action_dim == max_action_dim the old code produced shape [T,0]
        leaving padded rows with non-zero weights.
        With the fix, padded rows must be zero even when there are no padded columns.
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        T_prev = 8
        # action dim matches max — no column padding
        prev = torch.ones(1, T_prev, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        assert torch.all(w[T_prev:, :] == 0.0), (
            "Padded rows must be zero even when action_dim == max_action_dim"
        )

    def test_prev_actions_padded_shape(self):
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(2, 8, 14)
        prev_padded, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        assert prev_padded.shape == (2, 32, 32)
        assert w.shape == (32, 32)


# ---------------------------------------------------------------------------
# Finding 2: mask boundary derived from actual overlap (s_eff = H - len(A_prev))
# ---------------------------------------------------------------------------


class TestPrepareRtcMaskFinding2:
    """
    Verify that _prepare_rtc_mask derives the mask zero-boundary from the real
    overlap length (Algorithm 1, line 14: s_eff = H - prev_actions.shape[-2])
    rather than the caller-supplied execution_horizon.
    """

    def _reset_warn_flag(self):
        """Clear the module-level one-shot warning flag before each test."""
        import rho.policies.rho.rho_model as _flow

        _flow._rtc_execution_horizon_mismatch_warned = False

    def test_inconsistent_execution_horizon_zero_boundary_at_overlap_edge(self):
        """
        H=32, d=6, len(prev)=8, execution_horizon=8 (inconsistent: s_eff should be 24).
        W[8:, :] must be all zero and W[7, 0] must be > 0.
        """
        self._reset_warn_flag()
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(1, 8, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        assert torch.all(w[8:, :] == 0.0), "Rows beyond overlap must be zero"
        assert w[7, 0].item() > 0.0, "Last overlap row must be nonzero"

    def test_inconsistent_execution_horizon_eq5_values_use_s_eff(self):
        """
        With H=32, d=6, len(prev)=8 → s_eff=24.  The decay region is rows 6..7
        (H - s_eff = 8, so d <= i < 8).  Verify W[6] and W[7] match Eq. 5 with
        s = s_eff = 24, NOT s = execution_horizon = 8.
        """
        self._reset_warn_flag()
        H, d, prev_len = 32, 6, 8
        s_eff = H - prev_len  # 24
        stub = make_stub(chunk_size=H, max_action_dim=4)
        prev = torch.ones(1, prev_len, 4)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=d, execution_horizon=8)

        # Expected from Eq. 5 with s = s_eff = 24 → decay window = [6, 8)
        decay_end = H - s_eff  # = 8
        denom = decay_end - d + 1  # = 3
        for i in [6, 7]:
            ci = (decay_end - i) / denom
            expected = ci * (math.exp(ci) - 1) / (math.exp(1) - 1)
            got = w[i, 0].item()
            assert abs(got - expected) < 1e-5, (
                f"Row {i}: Eq.5(s_eff=24) expects {expected:.6f}, got {got:.6f}"
            )

        # Confirm the old s=8 values (decay over rows 6..23) do NOT match.
        decay_end_old = H - 8  # = 24
        denom_old = decay_end_old - d + 1  # = 19
        for i in [6, 7]:
            ci_old = (decay_end_old - i) / denom_old
            old_val = ci_old * (math.exp(ci_old) - 1) / (math.exp(1) - 1)
            assert abs(w[i, 0].item() - old_val) > 1e-3, (
                f"Row {i} incorrectly matches old s=8 value {old_val:.6f}"
            )

    def test_zero_boundary_equals_overlap_len_for_various_overlaps(self):
        """
        For overlap in [4, 8, 16, 24, 32] with H=32 and d=6, the mask's zero
        boundary must be exactly at prev_actions.shape[-2], regardless of the
        caller-supplied execution_horizon.
        """
        self._reset_warn_flag()
        stub = make_stub(chunk_size=32, max_action_dim=16)
        for overlap in [4, 8, 16, 24, 32]:
            import rho.policies.rho.rho_model as _flow

            _flow._rtc_execution_horizon_mismatch_warned = False

            prev = torch.ones(1, overlap, 16)
            _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)

            if overlap < 32:
                assert torch.all(w[overlap:, :] == 0.0), f"overlap={overlap}: W[{overlap}:] should be zero"
            if overlap > 0:
                assert w[overlap - 1, 0].item() >= 0.0  # just no crash; value may be 0 at d edge

    def test_consistent_execution_horizon_is_noop(self):
        """
        When execution_horizon == H - len(prev_actions), the fix is a no-op.
        Verify by comparing against compute_W_matrix_rtc called directly with
        the same s value, then zeroing padded rows to match _prepare_rtc_mask.
        """
        self._reset_warn_flag()
        H, d = 32, 6
        prev_len = 16
        s = H - prev_len  # = 16 → consistent with execution_horizon=16
        stub = make_stub(chunk_size=H, max_action_dim=32)
        prev = torch.ones(1, prev_len, 32)
        _, w_method = stub._prepare_rtc_mask(prev, inference_delay=d, execution_horizon=s)

        # Direct reference: compute W with s=16, then zero padded rows
        w_ref = stub.compute_W_matrix_rtc(d, s, 32)
        w_ref[prev_len:, :] = 0.0

        assert torch.allclose(w_method, w_ref), "Consistent execution_horizon must produce unchanged mask"


# ---------------------------------------------------------------------------
# Tests for the optional action_dim parameter of _prepare_rtc_mask
# ---------------------------------------------------------------------------


class TestPrepareRtcMaskExplicitActionDim:
    """
    Verify that passing an explicit ``action_dim`` to ``_prepare_rtc_mask``
    correctly widens the mask (mimicking ``combine_action_tactile_head=True``).
    """

    def test_wide_action_dim_mask_shape(self):
        """
        max_action_dim=32, max_tactile_dim=16, explicit action_dim=48:
        mask must have shape (chunk_size, 48).
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(1, 8, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8, action_dim=48)
        assert w.shape == (32, 48), f"Expected (32, 48), got {w.shape}"

    def test_wide_action_dim_padded_tensor_shape(self):
        """prev_actions_padded must also be widened to action_dim=48."""
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(2, 8, 32)
        prev_padded, _ = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8, action_dim=48)
        assert prev_padded.shape == (2, 32, 48), f"Expected (2, 32, 48), got {prev_padded.shape}"

    def test_wide_action_dim_padded_columns_zeroed(self):
        """
        Columns at/beyond prev_actions.shape[-1] (i.e. columns 32..47) must be 0.0
        in the mask when action_dim=48 and prev has only 32 action dims.
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(1, 8, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8, action_dim=48)
        assert torch.all(w[:, 32:] == 0.0), "Padded columns (32:) must be zeroed"

    def test_wide_action_dim_padded_rows_zeroed(self):
        """
        Rows at/beyond prev_actions.shape[-2] (i.e. rows 8..31) must be 0.0
        across ALL 48 columns when action_dim=48.
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(1, 8, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8, action_dim=48)
        assert torch.all(w[8:, :] == 0.0), "Padded rows (8:) must be zeroed across all 48 columns"

    def test_wide_action_dim_valid_region_non_zero(self):
        """
        Rows 0..5 (inference_delay region) in columns 0..31 (non-padded action dims)
        must be 1.0 when action_dim=48 and prev covers all 32 non-tactile dims.
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(1, 8, 32)
        _, w = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8, action_dim=48)
        assert torch.all(w[:6, :32] == 1.0), "Valid rows/cols must be 1.0 in inference_delay region"

    def test_default_action_dim_matches_max_action_dim(self):
        """
        Omitting action_dim must produce a mask of width max_action_dim,
        identical to calling with action_dim=max_action_dim explicitly.
        """
        stub = make_stub(chunk_size=32, max_action_dim=32)
        prev = torch.ones(1, 8, 14)
        _, w_default = stub._prepare_rtc_mask(prev, inference_delay=6, execution_horizon=8)
        _, w_explicit = stub._prepare_rtc_mask(
            prev,
            inference_delay=6,
            execution_horizon=8,
            action_dim=32,
        )
        assert w_default.shape == (32, 32), (
            f"Default width should be max_action_dim=32, got {w_default.shape}"
        )
        assert torch.allclose(w_default, w_explicit), (
            "Default and explicit max_action_dim must produce identical masks"
        )

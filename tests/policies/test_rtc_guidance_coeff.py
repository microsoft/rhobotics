"""
Tests for FlowMatchingModel._rtc_guidance_coeff.

The helper is a pure staticmethod — no weights, GPU, or network needed.
Exercises the paper schedule (Eqs. 1 and 4 of arXiv:2506.07339), the
constant schedule (preserved pre-fix behaviour), and error handling.
"""

import pytest

from rho.policies.rho.rho_model import FlowMatchingModel

# Shortcut: staticmethod requires no instance.
_coeff = FlowMatchingModel._rtc_guidance_coeff

# ---------------------------------------------------------------------------
# Expected coefficients for beta=40, n_steps=10 (tau = 0.0 .. 0.9)
# ---------------------------------------------------------------------------

BETA = 40.0
N_STEPS = 10
# Loop: tau_repo = 1 + (-1/10)*step  → after flip  tau_paper = step/10
TAU_VALUES = [step / N_STEPS for step in range(N_STEPS)]  # 0.0, 0.1, ..., 0.9

EXPECTED = {
    0.0: 40.0000,
    0.1: 9.1111,
    0.2: 4.2500,
    0.3: 2.7619,
    0.4: 2.1667,
    0.5: 2.0000,
    0.6: 2.1667,
    0.7: 2.7619,
    0.8: 4.2500,
    0.9: 9.1111,
}

TOL = 1e-4


class TestPaperScheduleTable:
    """Verify each table entry from the task spec to 1e-4 tolerance."""

    @pytest.mark.parametrize("tau,expected", EXPECTED.items())
    def test_table_value(self, tau, expected):
        got = _coeff(tau, BETA, "paper")
        assert abs(got - expected) <= TOL, f"tau={tau}: expected {expected:.4f}, got {got:.6f}"


class TestPaperScheduleShape:
    """The unclipped paper coefficient is symmetric around tau=0.5."""

    @pytest.mark.parametrize("tau", [0.1, 0.2, 0.3, 0.4])
    def test_symmetric_around_half(self, tau):
        assert _coeff(tau, BETA, "paper") == pytest.approx(_coeff(1.0 - tau, BETA, "paper"))

    def test_minimum_at_half(self):
        values = [_coeff(tau, BETA, "paper") for tau in TAU_VALUES[1:]]
        assert _coeff(0.5, BETA, "paper") == min(values)


class TestTauZero:
    """tau=0 must return exactly beta (guard against division by zero)."""

    def test_tau_zero_returns_beta(self):
        assert _coeff(0.0, BETA, "paper") == BETA

    def test_tau_negative_returns_beta(self):
        assert _coeff(-0.1, BETA, "paper") == BETA


class TestClipping:
    """A small beta must clip all coefficients so none exceeds beta."""

    def test_small_beta_never_exceeded(self):
        small_beta = 0.25
        for tau in TAU_VALUES:
            val = _coeff(tau, small_beta, "paper")
            assert val <= small_beta + 1e-9, f"tau={tau}: coefficient {val} exceeds beta={small_beta}"

    def test_small_beta_clips_at_tau_zero(self):
        small_beta = 0.25
        assert _coeff(0.0, small_beta, "paper") == small_beta


class TestConstantSchedule:
    """schedule='constant' must return beta unchanged for every tau."""

    @pytest.mark.parametrize("tau", TAU_VALUES)
    def test_constant_returns_beta(self, tau):
        assert _coeff(tau, BETA, "constant") == BETA, (
            f"constant schedule should return beta={BETA} for tau={tau}"
        )

    def test_constant_tau_zero_returns_beta(self):
        assert _coeff(0.0, BETA, "constant") == BETA


class TestInvalidSchedule:
    """Unknown schedule strings must raise ValueError."""

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown guidance_schedule"):
            _coeff(0.5, BETA, "unknown")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _coeff(0.5, BETA, "")

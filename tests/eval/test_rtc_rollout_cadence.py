"""
Arithmetic model of the RTC rollout-loop trigger condition (Finding 5 fix).

This test does NOT import anything from ``rho`` — it models only the queue
cadence logic in ``evaluate_policy`` as pure arithmetic, so it can run on any
machine with just Python/pytest.

What is covered
---------------
* Starting from a full queue of ``chunk_size`` entries, popping one action per
  step and re-planning when ``num_executed_since_inference >= execution_horizon``
  (the fixed condition), the inter-inference interval equals ``execution_horizon``.
* At the moment re-inference fires the queue contains
  ``chunk_size - execution_horizon`` entries, so ``len(prev_actions)`` handed to
  the policy equals ``chunk_size - execution_horizon``.  This is the value
  ``_prepare_rtc_mask`` uses to derive s_eff, so the mask's decay region aligns
  with the real overlap.

What is NOT covered
-------------------
* The actual ``evaluate_policy`` function (needs an environment, policy
  interface, and metrics plumbing — impractical without heavy dependencies).
* Splice correctness (``old_in_flight``, ``action_chunk[:, inference_delay:]``).
* The ``elif len(action_queue) == 0`` branch (first-step fill).
* ``_prepare_rtc_mask`` internals (covered by test_rtc_masking.py).
"""

from collections import deque

import pytest


def _simulate_rtc_cadence(chunk_size: int, execution_horizon: int, inference_delay: int, n_steps: int):
    """
    Simulate the RTC branch of evaluate_policy's per-step loop in pure Python.

    Returns a list of (step, queue_len_at_trigger) pairs — one per re-inference
    event — not counting the initial fill (elif branch).
    """
    assert inference_delay + execution_horizon <= chunk_size, (
        "Invalid config: inference_delay + execution_horizon must not exceed chunk_size"
    )

    # Simulate the initial fill (elif branch): queue starts full.
    queue: deque = deque(range(chunk_size))
    num_executed = 0
    events: list[tuple[int, int]] = []  # (step, queue_len_when_triggered)

    for step in range(n_steps):
        # Mirror the fixed if-condition from evaluate_policy.
        if num_executed >= execution_horizon and len(queue) > 0:
            queue_len_at_trigger = len(queue)
            events.append((step, queue_len_at_trigger))

            # Splice: keep inference_delay actions, refill rest from new chunk.
            old_in_flight = list(queue)[:inference_delay]
            queue.clear()
            queue.extend(old_in_flight)
            # New chunk contributes chunk_size - inference_delay fresh actions.
            for _ in range(chunk_size - inference_delay):
                queue.append(object())
            num_executed = 0

        elif len(queue) == 0:
            # Should not occur in a well-configured rollout; guard for safety.
            for _ in range(chunk_size):
                queue.append(object())
            num_executed = 0

        queue.popleft()
        num_executed += 1

    return events


# ---------------------------------------------------------------------------
# Parameterised cases: (chunk_size, execution_horizon, inference_delay)
# ---------------------------------------------------------------------------

CASES = [
    (50, 8, 6),  # paper example from the finding-5 spec
    (32, 8, 6),  # test_rtc_masking.py default config
    (16, 4, 2),  # smaller chunk
    (20, 5, 5),  # inference_delay == execution_horizon
    (10, 3, 2),  # minimal
]


@pytest.mark.parametrize("chunk_size,execution_horizon,inference_delay", CASES)
def test_inter_inference_interval_equals_execution_horizon(chunk_size, execution_horizon, inference_delay):
    """Re-inference fires every execution_horizon steps after the first fill."""
    n_steps = chunk_size * 6  # run long enough for several re-inference events
    events = _simulate_rtc_cadence(chunk_size, execution_horizon, inference_delay, n_steps)

    assert len(events) >= 3, "Too few re-inference events to verify cadence"

    for i in range(1, len(events)):
        gap = events[i][0] - events[i - 1][0]
        assert gap == execution_horizon, (
            f"chunk_size={chunk_size}, execution_horizon={execution_horizon}, "
            f"inference_delay={inference_delay}: "
            f"expected gap {execution_horizon}, got {gap} between events {i - 1} and {i}"
        )


@pytest.mark.parametrize("chunk_size,execution_horizon,inference_delay", CASES)
def test_queue_len_at_trigger_equals_chunk_minus_horizon(chunk_size, execution_horizon, inference_delay):
    """
    When re-inference fires the queue holds chunk_size - execution_horizon entries.
    That is what the policy receives as len(prev_actions), and it determines the
    mask's zero boundary via s_eff = chunk_size - len(prev_actions).
    """
    n_steps = chunk_size * 6
    events = _simulate_rtc_cadence(chunk_size, execution_horizon, inference_delay, n_steps)
    expected_queue_len = chunk_size - execution_horizon

    for step, queue_len in events:
        assert queue_len == expected_queue_len, (
            f"chunk_size={chunk_size}, execution_horizon={execution_horizon}, "
            f"inference_delay={inference_delay}: "
            f"at step {step} expected queue_len {expected_queue_len}, got {queue_len}"
        )


@pytest.mark.parametrize("chunk_size,execution_horizon,inference_delay", CASES)
def test_s_eff_matches_execution_horizon(chunk_size, execution_horizon, inference_delay):
    """
    s_eff = chunk_size - len(prev_actions) must equal execution_horizon.
    This is the invariant that makes the mask boundary line up with the real
    re-inference cadence and that the finding-2 mismatch warning checks for.
    """
    n_steps = chunk_size * 6
    events = _simulate_rtc_cadence(chunk_size, execution_horizon, inference_delay, n_steps)

    for step, queue_len in events:
        s_eff = chunk_size - queue_len
        assert s_eff == execution_horizon, (
            f"chunk_size={chunk_size}, execution_horizon={execution_horizon}: "
            f"s_eff={s_eff} != execution_horizon={execution_horizon} at step {step}"
        )

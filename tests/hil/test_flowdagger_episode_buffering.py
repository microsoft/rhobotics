"""Logic tests for FlowDAggerTrainer's episode-buffering / dual-buffer behavior.

Heavy model construction is stubbed so this runs on CPU without checkpoints
or a base policy. We exercise:
  - Successful episodes → both buffers populated as expected
  - Failed (success=False) and unknown (success=None) episodes → nothing committed
  - E-stop simulation (new episode_id without a prior done) → previous dropped
  - autonomous_subsample_every gates how many autonomous transitions land
  - Mixed-batch sampling honors intervention_sample_ratio
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repo importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rho.hil.experience import Transition
from rho.hil.trainers.flowdagger_trainer import FlowDAggerTrainer, _DAggerBuffer
from rho.policies.dsrl.flowdagger_config import FlowDAggerConfig

IMAGE_KEYS = ["observation.image.agentview", "observation.image.left_wrist"]


def _make_trainer(
    autonomous_target: str = "sampled_noise",
    intervention_sample_ratio: float = 0.5,
    autonomous_subsample_every: int = 1,
    min_interventions_to_start: int = 1,
    buffer_capacity: int = 1000,
    autonomous_buffer_capacity: int = 1000,
) -> FlowDAggerTrainer:
    """Build a trainer with the heavy construction skipped and the model-
    dependent methods stubbed. Buffers and pending state are wired by hand.
    """
    cfg = FlowDAggerConfig(
        image_keys=IMAGE_KEYS,
        image_size=8,  # small for fast tests
        num_cameras=2,
        state_dim=4,
        include_state=True,
        noise_action_steps=2,
        noise_action_dim=4,
        buffer_capacity=buffer_capacity,
        autonomous_buffer_capacity=autonomous_buffer_capacity,
        intervention_sample_ratio=intervention_sample_ratio,
        autonomous_subsample_every=autonomous_subsample_every,
        autonomous_target=autonomous_target,
        min_interventions_to_start=min_interventions_to_start,
        bc_batch_size=8,
    )

    trainer = object.__new__(FlowDAggerTrainer)
    trainer.config = cfg
    trainer.device = "cpu"

    trainer.intervention_buffer = _DAggerBuffer(
        capacity=cfg.buffer_capacity,
        image_keys=cfg.image_keys,
        image_size=cfg.image_size,
        state_dim=cfg.state_dim,
        noise_dim=cfg.noise_dim,
    )
    trainer.autonomous_buffer = _DAggerBuffer(
        capacity=cfg.autonomous_buffer_capacity,
        image_keys=cfg.image_keys,
        image_size=cfg.image_size,
        state_dim=cfg.state_dim,
        noise_dim=cfg.noise_dim,
    )

    # Stats / state expected by _ingest / _commit_episode.
    trainer.total_transitions = 0
    trainer.total_interventions = 0
    trainer.total_inversions = 0
    trainer.total_updates = 0
    trainer.total_episodes_committed = 0
    trainer.total_episodes_dropped = 0
    trainer.recent_bc_losses = []
    trainer.recent_inversion_mse = []
    trainer._interventions_since_last_update = 0
    trainer._pending_intervened = []
    trainer._pending_autonomous = []
    trainer._current_episode_id = None
    trainer._current_episode_saw_success = False
    trainer._autonomous_subsample_counter = 0
    trainer._warned_missing_success = False
    trainer._buffer_input_dumped = True  # silence the one-shot dump in tests

    # Eval-block state (mirrors __init__; tests don't enter eval mode, so the
    # defaults just route _ingest down the normal training path).
    trainer._in_eval_block = False
    trainer._eval_current_episode_id = None
    trainer._eval_current_episode_steps = 0
    trainer._eval_block_episodes = 0
    trainer._eval_block_successes = 0
    trainer._eval_block_steps = []

    # Stub the heavy methods — they would otherwise need the base policy /
    # the env / the policy interface.
    def _stub_processed_obs(transition, action_to_invert):
        return {"_stub": True, "_action": action_to_invert}

    def _stub_images(_processed_obs):
        return {k: np.zeros((3, cfg.image_size, cfg.image_size), dtype=np.float32) for k in cfg.image_keys}

    def _stub_state(_processed_obs):
        return np.zeros(cfg.state_dim, dtype=np.float32)

    def _stub_process_obs_for_training(obs_dict):
        return {"_stub_training_obs": True}

    def _stub_run_inversion(pending, dest_buffer):
        # Drop a deterministic fake w into the buffer per item so we can check
        # routing. The test cares about buffer sizes and routing, not w values.
        for item in pending:
            w = np.full(cfg.noise_dim, 0.5, dtype=np.float32)
            dest_buffer.add(item["images"], item["state"], w)
            trainer.total_inversions += 1

    trainer._prepare_obs_for_inversion = _stub_processed_obs  # type: ignore[assignment]
    trainer._extract_images = _stub_images  # type: ignore[assignment]
    trainer._extract_state = _stub_state  # type: ignore[assignment]
    trainer._process_obs_for_training = _stub_process_obs_for_training  # type: ignore[assignment]
    trainer._run_inversion = _stub_run_inversion  # type: ignore[assignment]
    return trainer


def _make_transition(
    intervened: bool,
    done: bool,
    success=None,
    episode_id: int = 1,
    noise_dim: int = 8,
):
    return Transition(
        obs={
            "observation.image.agentview": np.zeros((128, 128, 3), dtype=np.uint8),
            "observation.image.left_wrist": np.zeros((128, 128, 3), dtype=np.uint8),
            "observation.state": np.zeros(7, dtype=np.float32),
        },
        action=np.zeros(7, dtype=np.float32),
        reward=0.0,
        next_obs={},
        done=done,
        intervened=intervened,
        intervention_action=np.zeros(7, dtype=np.float32) if intervened else None,
        noise=np.full(noise_dim, 0.7, dtype=np.float32),  # populated for all (sim DSRL agent)
        success=success,
        episode_id=episode_id,
    )


def test_successful_episode_commits_both_buffers():
    tr = _make_trainer()
    # 3 autonomous + 2 intervention + terminal-with-success
    for _ in range(3):
        tr._ingest(_make_transition(intervened=False, done=False, episode_id=1))
    for _ in range(2):
        tr._ingest(_make_transition(intervened=True, done=False, episode_id=1))
    tr._ingest(_make_transition(intervened=False, done=True, success=True, episode_id=1))

    # Terminal step is itself non-intervened in this case → counts toward autonomous.
    assert tr.intervention_buffer.size == 2, (
        f"intervention buffer should hold 2, got {tr.intervention_buffer.size}"
    )
    assert tr.autonomous_buffer.size == 4, (
        f"autonomous buffer should hold 4 (3 mid-ep + terminal), got {tr.autonomous_buffer.size}"
    )
    assert tr.total_episodes_committed == 1
    assert tr.total_episodes_dropped == 0


def test_failed_episode_drops_everything():
    tr = _make_trainer()
    for _ in range(3):
        tr._ingest(_make_transition(intervened=False, done=False, episode_id=2))
    for _ in range(2):
        tr._ingest(_make_transition(intervened=True, done=False, episode_id=2))
    tr._ingest(_make_transition(intervened=False, done=True, success=False, episode_id=2))

    assert tr.intervention_buffer.size == 0
    assert tr.autonomous_buffer.size == 0
    assert tr.total_episodes_committed == 0
    assert tr.total_episodes_dropped == 1


def test_unknown_success_drops_episode_and_warns():
    tr = _make_trainer()
    tr._ingest(_make_transition(intervened=True, done=False, episode_id=3))
    tr._ingest(_make_transition(intervened=False, done=True, success=None, episode_id=3))

    assert tr.intervention_buffer.size == 0
    assert tr.autonomous_buffer.size == 0
    assert tr.total_episodes_dropped == 1
    assert tr._warned_missing_success is True


def test_estop_simulation_drops_previous():
    """No terminal arrives; the next episode's transition has a new episode_id.
    The unterminated previous episode should be dropped as implicit failure."""
    tr = _make_trainer()
    # Episode A — accumulates but never terminates.
    for _ in range(4):
        tr._ingest(_make_transition(intervened=True, done=False, episode_id=10))
    assert len(tr._pending_intervened) == 4

    # Episode B starts — new episode_id arrives.
    tr._ingest(_make_transition(intervened=False, done=False, episode_id=11))

    # Episode A should be dropped, episode B should be the new current.
    assert tr.total_episodes_dropped == 1
    assert tr._current_episode_id == 11
    assert tr.intervention_buffer.size == 0
    # The first transition of B was autonomous → it's already in pending.
    assert len(tr._pending_autonomous) == 1
    assert len(tr._pending_intervened) == 0


def test_autonomous_subsample_every():
    tr = _make_trainer(autonomous_subsample_every=4)
    for _ in range(11):
        tr._ingest(_make_transition(intervened=False, done=False, episode_id=20))
    tr._ingest(_make_transition(intervened=True, done=False, episode_id=20))  # ensure success-gate passes
    tr._ingest(_make_transition(intervened=False, done=True, success=True, episode_id=20))

    # With subsample_every=4 over 12 non-intervened steps (11 + the terminal),
    # we expect ~3 to be kept (counter increments at indices 1..12; kept where
    # counter % 4 == 0, i.e. indices 4, 8, 12 → 3 kept).
    assert tr.autonomous_buffer.size == 3, (
        f"expected 3 autonomous samples (every-4 over 12), got {tr.autonomous_buffer.size}"
    )


def test_mixed_batch_sampling_ratio():
    tr = _make_trainer(intervention_sample_ratio=0.5, min_interventions_to_start=1)
    # Stuff distinct sentinel w-values into the two buffers so we can identify
    # which buffer each sample came from.
    int_w = np.full(tr.config.noise_dim, -1.0, dtype=np.float32)
    auto_w = np.full(tr.config.noise_dim, +1.0, dtype=np.float32)
    images = {
        k: np.zeros((3, tr.config.image_size, tr.config.image_size), dtype=np.float32) for k in IMAGE_KEYS
    }
    state = np.zeros(tr.config.state_dim, dtype=np.float32)
    for _ in range(64):
        tr.intervention_buffer.add(images, state, int_w)
        tr.autonomous_buffer.add(images, state, auto_w)

    _, _, w_batch = tr._sample_mixed_batch()
    bsz = tr.config.bc_batch_size
    n_int = int(np.sum(w_batch[:, 0] < 0))
    n_auto = int(np.sum(w_batch[:, 0] > 0))
    assert n_int + n_auto == bsz
    # Expect roughly 50/50 with batch_size=8: exactly 4/4 by ratio rounding.
    assert n_int == 4 and n_auto == 4, f"expected 4/4 split, got {n_int} int + {n_auto} auto"


def test_mixed_batch_falls_back_when_autonomous_empty():
    tr = _make_trainer(intervention_sample_ratio=0.5)
    int_w = np.full(tr.config.noise_dim, -1.0, dtype=np.float32)
    images = {
        k: np.zeros((3, tr.config.image_size, tr.config.image_size), dtype=np.float32) for k in IMAGE_KEYS
    }
    state = np.zeros(tr.config.state_dim, dtype=np.float32)
    for _ in range(32):
        tr.intervention_buffer.add(images, state, int_w)

    _, _, w_batch = tr._sample_mixed_batch()
    bsz = tr.config.bc_batch_size
    assert int(np.sum(w_batch[:, 0] < 0)) == bsz, "should fall back to interventions-only"


def test_sampled_noise_skipped_when_transition_noise_is_none():
    """If autonomous_target='sampled_noise' but the robot didn't populate
    transition.noise, the autonomous transition is silently dropped."""
    tr = _make_trainer(autonomous_target="sampled_noise")
    # Send autonomous transitions with noise=None
    t = _make_transition(intervened=False, done=False, episode_id=30)
    t.noise = None
    tr._ingest(t)
    tr._ingest(_make_transition(intervened=True, done=False, episode_id=30))
    # Terminal
    t_end = _make_transition(intervened=False, done=True, success=True, episode_id=30)
    t_end.noise = None
    tr._ingest(t_end)
    assert tr.intervention_buffer.size == 1
    assert tr.autonomous_buffer.size == 0  # both autonomous had noise=None


if __name__ == "__main__":
    fns = [
        test_successful_episode_commits_both_buffers,
        test_failed_episode_drops_everything,
        test_unknown_success_drops_episode_and_warns,
        test_estop_simulation_drops_previous,
        test_autonomous_subsample_every,
        test_mixed_batch_sampling_ratio,
        test_mixed_batch_falls_back_when_autonomous_empty,
        test_sampled_noise_skipped_when_transition_noise_is_none,
    ]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")

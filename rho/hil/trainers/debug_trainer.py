"""
Debug Trainer for HIL pipeline testing.

Receives experience transitions via ZMQ and prints statistics without running
any RL training. Used to verify the full robot-to-server streaming pipeline.

Usage:
    # From serve_policy.py with --train --trainer_type=debug
    # Or standalone:
    from rho.hil.trainers.debug_trainer import start_debug_trainer
    trainer = start_debug_trainer(experience_port=5555, blocking=True)
"""

import threading
import time
from pathlib import Path

from rho.hil.experience import ExperienceReceiver


class DebugTrainer:
    """
    Receives experience from robot via ZMQ and prints statistics.
    No training -- used for pipeline testing and monitoring.

    Trainer interface: start(), stop(), train()
    """

    def __init__(self, experience_port: int = 5555, log_dir: str = "logs/debug_trainer"):
        self.experience_receiver = ExperienceReceiver(port=experience_port)
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Stats
        self.total_count = 0
        self.intervention_count = 0
        self.done_count = 0
        self.running = False

        # Per-window stats for throughput
        self._window_start = 0.0
        self._window_count = 0

        # Shape validation (logged once on first transition)
        self._shapes_logged = False

        # TensorBoard writer (lazy init)
        self._writer = None

    @staticmethod
    def _log_field_shapes(name, value, indent=2):
        """Log shapes of a transition field, handling dicts, arrays, and scalars."""
        prefix = " " * indent
        if isinstance(value, dict):
            print(f"{prefix}{name}: dict with {len(value)} keys")
            for k, v in value.items():
                if hasattr(v, "shape"):
                    print(f"{prefix}  {k}: {v.shape} dtype={getattr(v, 'dtype', '?')}")
                else:
                    print(f"{prefix}  {k}: {type(v).__name__} = {v}")
        elif hasattr(value, "shape"):
            print(f"{prefix}{name}: {value.shape} dtype={getattr(value, 'dtype', '?')}")
        elif value is not None:
            print(f"{prefix}{name}: {type(value).__name__} = {value}")
        else:
            print(f"{prefix}{name}: None")

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                pass
        return self._writer

    def start(self):
        """Start ZMQ receiver."""
        self.experience_receiver.start()
        self.running = True
        self._window_start = time.time()
        self._window_count = 0
        print(f"[DebugTrainer] Started, listening on port {self.experience_receiver.port}")

    def stop(self):
        """Stop ZMQ receiver and close writer."""
        self.running = False
        self.experience_receiver.stop()
        if self._writer is not None:
            self._writer.close()
        print(
            f"[DebugTrainer] Stopped. "
            f"Total: {self.total_count}, "
            f"Interventions: {self.intervention_count}, "
            f"Episodes: {self.done_count}"
        )

    def train(self, num_steps: int | None = None, num_transitions: int | None = None):
        """
        Main receive loop.

        Receives transitions and prints stats every 5 seconds.
        """
        print("[DebugTrainer] Receive loop started")
        log_interval = 5.0
        last_log_time = time.time()

        while self.running:
            # Check termination
            if num_transitions is not None and self.total_count >= num_transitions:
                print(f"[DebugTrainer] Reached {num_transitions} transitions, stopping")
                break

            # Receive batch
            transitions = self.experience_receiver.receive_batch(max_batch=256, timeout_ms=100)

            for t in transitions:
                self.total_count += 1
                self._window_count += 1

                if t.intervened:
                    self.intervention_count += 1
                if t.done:
                    self.done_count += 1
                    success = getattr(t, "success", None)
                    print(
                        f"[DebugTrainer] Episode {self.done_count} ended | "
                        f"reward: {t.reward} | success: {success} | "
                        f"steps since last: {self.total_count - getattr(self, '_last_done_at', 0)}"
                    )
                    self._last_done_at = self.total_count

                # Print shapes on first transition
                if not self._shapes_logged:
                    print("[DebugTrainer] First transition received:")
                    self._log_field_shapes("obs", t.obs)
                    self._log_field_shapes("action", t.action)
                    self._log_field_shapes("next_obs", t.next_obs)
                    if t.intervention_action is not None:
                        self._log_field_shapes("intervention_action", t.intervention_action)
                    print(f"  done: {t.done}  intervened: {t.intervened}")
                    self._shapes_logged = True

                # TensorBoard logging
                writer = self._get_writer()
                if writer is not None:
                    writer.add_scalar("debug/total_transitions", self.total_count, self.total_count)
                    writer.add_scalar("debug/interventions", self.intervention_count, self.total_count)
                    writer.add_scalar("debug/episodes", self.done_count, self.total_count)
                    if self.total_count > 0:
                        writer.add_scalar(
                            "debug/intervention_ratio",
                            self.intervention_count / self.total_count,
                            self.total_count,
                        )

            # Periodic console printing
            now = time.time()
            if now - last_log_time >= log_interval:
                elapsed = now - self._window_start
                throughput = self._window_count / elapsed if elapsed > 0 else 0.0
                ratio = (self.intervention_count / self.total_count * 100) if self.total_count > 0 else 0.0
                print(
                    f"[DebugTrainer] "
                    f"Received: {self.total_count} | "
                    f"Intervened: {self.intervention_count} ({ratio:.1f}%) | "
                    f"{throughput:.1f} trans/s | "
                    f"Episodes: {self.done_count}"
                )
                # Reset window
                self._window_start = now
                self._window_count = 0
                last_log_time = now

        print(f"[DebugTrainer] Receive loop ended. Total: {self.total_count}")


def start_debug_trainer(
    experience_port: int = 5555,
    log_dir: str = "logs/debug_trainer",
    num_transitions: int | None = None,
    blocking: bool = True,
) -> DebugTrainer:
    """
    Create, start, and run a DebugTrainer.

    Args:
        experience_port: ZMQ port to listen on
        log_dir: Directory for TensorBoard logs
        num_transitions: Stop after N transitions (None = run forever)
        blocking: If True, blocks until done. If False, runs in background thread.

    Returns:
        The DebugTrainer instance (call .stop() to shut down)
    """
    print("=" * 60)
    print("Starting Debug Trainer")
    print("=" * 60)
    print(f"  experience_port: {experience_port}")
    print(f"  log_dir: {log_dir}")
    print(f"  blocking: {blocking}")

    trainer = DebugTrainer(
        experience_port=experience_port,
        log_dir=log_dir,
    )

    def _run():
        trainer.start()
        try:
            trainer.train(num_transitions=num_transitions)
        except KeyboardInterrupt:
            print("\n[DebugTrainer] Interrupted by user")
        except Exception as e:
            print(f"\n[DebugTrainer] Error: {e}")
            raise
        finally:
            trainer.stop()

    if blocking:
        _run()
    else:
        trainer._debug_thread = threading.Thread(
            target=_run,
            name="DebugTrainer",
            daemon=True,
        )
        trainer._debug_thread.start()
        print("\n[DebugTrainer] Started in background thread")

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DebugTrainer standalone")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ experience port")
    parser.add_argument("--log-dir", default="logs/debug_trainer", help="TensorBoard log dir")
    parser.add_argument("--num-transitions", type=int, default=None, help="Stop after N transitions")
    args = parser.parse_args()

    start_debug_trainer(
        experience_port=args.port,
        log_dir=args.log_dir,
        num_transitions=args.num_transitions,
        blocking=True,
    )

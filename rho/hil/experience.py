"""
Experience Publisher/Receiver for HIL-SERL distributed training.

Sends experience transitions between robot machine and training machine via ZMQ.
"""

import io
import pickle
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import numpy as np
import zmq


class _TransitionUnpickler(pickle.Unpickler):
    """Remaps robot-side module paths to server-side paths when unpickling."""

    def find_class(self, module: str, name: str):
        # Robot pickles Transition / ControlMessage as hil.data_publisher.*
        if name in ("Transition", "ControlMessage") and "data_publisher" in module:
            module = "rho.hil.experience"
        return super().find_class(module, name)


def _loads(data: bytes):
    return _TransitionUnpickler(io.BytesIO(data)).load()


# NOTE: Keep this Transition dataclass in sync with ur5e_stack/hil/data_publisher.py
# Both sides must serialize/deserialize the same structure over ZMQ.
#
# obs/next_obs can be either:
#   - dict: {"tcp_pose": np.ndarray(8,), "cam_scene": np.ndarray(128,128,3), ...}
#   - np.ndarray: legacy format (raw joint positions)
# The trainer's _ensure_obs_dict() handles both cases.
@dataclass
class Transition:
    """Single transition for experience replay."""

    obs: Any  # dict or np.ndarray — see note above
    action: np.ndarray
    reward: float
    next_obs: Any  # dict or np.ndarray — see note above
    done: bool
    intervened: bool = False
    intervention_action: np.ndarray | None = None
    noise: np.ndarray | None = None  # Noise vector used by DSRL
    timestamp: float = 0.0
    success: bool | None = None  # Set on terminal transition (done=True)
    # Monotonically increasing per rollout; trainer uses it to detect e-stop /
    # missed-terminal episode boundaries.
    episode_id: int = 0
    # Rollout step cap, stamped on the terminal transition. Trainer uses it so
    # an early-discarded eval failure still tallies as a full-length episode.
    max_steps: int = 0


# Control messages multiplex with Transitions on the same ZMQ socket. The
# robot main process pushes operator commands (eval-mode toggle, manual
# checkpoint save) and the subprocess emits them inline with transitions
# so the trainer sees them in causal order with the surrounding data.
@dataclass
class ControlMessage:
    kind: str  # "enter_eval" | "exit_eval" | "save_checkpoint"
    timestamp: float = 0.0


class ExperiencePublisher:
    """
    ZMQ PUSH socket for sending experience to learner.

    Uses a background thread to avoid blocking the control loop.

    Args:
        host: Learner machine IP address
        port: ZMQ port for experience queue
        queue_size: Maximum pending transitions before dropping
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        queue_size: int = 10000,
    ):
        self.host = host
        self.port = port
        self.queue_size = queue_size

        self.queue: Queue = Queue(maxsize=queue_size)  # holds Transition | ControlMessage
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        self.sent_count = 0
        self.dropped_count = 0

    def start(self):
        """Start the publisher thread."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, self.queue_size)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        print(f"[ExperiencePublisher] Connected to tcp://{self.host}:{self.port}")

    def stop(self):
        """Stop the publisher thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

        print(f"[ExperiencePublisher] Stopped. Sent: {self.sent_count}, Dropped: {self.dropped_count}")

    def _run(self):
        """Background thread loop for sending transitions."""
        while self.running:
            try:
                transition = self.queue.get(timeout=0.1)
                data = pickle.dumps(transition)
                self.socket.send(data, zmq.NOBLOCK)
                self.sent_count += 1
            except Empty:
                continue
            except zmq.Again:
                self.dropped_count += 1
            except Exception as e:
                print(f"[ExperiencePublisher] Error: {e}")

    def send_transition(
        self,
        obs,
        action: np.ndarray,
        reward: float,
        next_obs,
        done: bool,
        intervened: bool = False,
        intervention_action: np.ndarray | None = None,
        noise: np.ndarray | None = None,
        success: bool | None = None,
        episode_id: int = 0,
        max_steps: int = 0,
    ):
        """Queue a transition for sending to learner."""
        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            intervened=intervened,
            intervention_action=intervention_action,
            noise=noise,
            timestamp=time.time(),
            success=success,
            episode_id=episode_id,
            max_steps=max_steps,
        )

        try:
            self.queue.put_nowait(transition)
        except Exception:
            self.dropped_count += 1

    def send_control(self, kind: str):
        """Queue a ControlMessage to ship inline with transitions."""
        msg = ControlMessage(kind=kind, timestamp=time.time())
        try:
            self.queue.put_nowait(msg)
        except Exception:
            self.dropped_count += 1

    @property
    def pending_count(self) -> int:
        """Number of transitions pending in queue."""
        return self.queue.qsize()


class ExperienceReceiver:
    """
    ZMQ PULL socket for receiving experience on learner machine.

    Args:
        port: ZMQ port to listen on
    """

    def __init__(self, port: int = 5555):
        self.port = port

        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.running = False
        self.received_count = 0

    def start(self):
        """Start the receiver."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = True

        print(f"[ExperienceReceiver] Listening on tcp://*:{self.port}")

    def stop(self):
        """Stop the receiver."""
        self.running = False
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

        print(f"[ExperienceReceiver] Stopped. Received: {self.received_count}")

    def receive(self, timeout_ms: int = 100) -> Any | None:
        """Receive a single message (Transition or ControlMessage)."""
        if self.socket.poll(timeout_ms):
            data = self.socket.recv()
            msg = _loads(data)
            self.received_count += 1
            return msg
        return None

    def receive_batch(self, max_batch: int = 256, timeout_ms: int = 100) -> list[Any]:
        """Receive a batch of messages (Transition or ControlMessage)."""
        messages: list[Any] = []

        first = self.receive(timeout_ms)
        if first is None:
            return messages

        messages.append(first)

        while len(messages) < max_batch:
            if self.socket.poll(0):
                data = self.socket.recv()
                msg = _loads(data)
                messages.append(msg)
                self.received_count += 1
            else:
                break

        return messages

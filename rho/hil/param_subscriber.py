"""
Parameter Subscriber for HIL-SERL distributed training.

Receives updated policy parameters from training machine via ZMQ PUB/SUB.

Usage:
    subscriber = ParamSubscriber(host="192.168.1.100", port=5556)
    subscriber.start()

    # In inference loop:
    new_params = subscriber.get_latest_params()
    if new_params is not None:
        policy.load_state_dict(new_params)

    # Cleanup:
    subscriber.stop()
"""

import pickle
import threading
import time
from collections.abc import Callable
from typing import Any

import zmq


class ParamSubscriber:
    """
    ZMQ SUB socket for receiving policy parameter updates.

    Uses a background thread to continuously receive updates.
    Call get_latest_params() to retrieve the most recent parameters.

    Args:
        host: Learner machine IP address
        port: ZMQ port for parameter updates
        topic: ZMQ subscription topic (default: "params")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        topic: str = "params",
    ):
        self.host = host
        self.port = port
        self.topic = topic

        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        # Latest params (thread-safe via lock)
        self._latest_params: dict[str, Any] | None = None
        self._params_lock = threading.Lock()
        self._params_version = 0
        self._last_fetched_version = 0

        # Stats
        self.received_count = 0

        # Optional callback
        self._on_update: Callable[[dict[str, Any]], None] | None = None

    def start(self):
        """Start the subscriber thread."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.RCVHWM, 2)  # Only keep latest
        self.socket.connect(f"tcp://{self.host}:{self.port}")

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        print(f"[ParamSubscriber] Connected to tcp://{self.host}:{self.port}")

    def stop(self):
        """Stop the subscriber thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

        print(f"[ParamSubscriber] Stopped. Received: {self.received_count} updates")

    def _run(self):
        """Background thread loop for receiving parameter updates."""
        while self.running:
            try:
                if self.socket.poll(100):  # 100ms timeout
                    message = self.socket.recv_multipart()
                    if len(message) >= 2:
                        # message[0] is topic, message[1] is data
                        params = pickle.loads(message[1])

                        with self._params_lock:
                            self._latest_params = params
                            self._params_version += 1

                        self.received_count += 1

                        # Call optional callback
                        if self._on_update is not None:
                            try:
                                self._on_update(params)
                            except Exception as e:
                                print(f"[ParamSubscriber] Callback error: {e}")

            except Exception as e:
                print(f"[ParamSubscriber] Error: {e}")

    def get_latest_params(self) -> dict[str, Any] | None:
        """
        Get the latest received parameters.

        Returns None if no new parameters since last call.

        Returns:
            Policy state dict or None
        """
        with self._params_lock:
            if self._params_version > self._last_fetched_version:
                self._last_fetched_version = self._params_version
                return self._latest_params
            return None

    def get_params_blocking(self, timeout: float = 10.0) -> dict[str, Any] | None:
        """
        Wait for new parameters (blocking).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Policy state dict or None if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            params = self.get_latest_params()
            if params is not None:
                return params
            time.sleep(0.01)
        return None

    def set_callback(self, callback: Callable[[dict[str, Any]], None]):
        """
        Set callback to be called when new params are received.

        Args:
            callback: Function taking params dict
        """
        self._on_update = callback

    @property
    def params_version(self) -> int:
        """Current parameter version number."""
        return self._params_version

    @property
    def has_new_params(self) -> bool:
        """Check if new params are available without consuming them."""
        with self._params_lock:
            return self._params_version > self._last_fetched_version


class ParamPublisher:
    """
    ZMQ PUB socket for publishing policy parameter updates.

    Used on the learner machine to broadcast updated parameters.

    Args:
        port: ZMQ port for parameter updates
        topic: ZMQ publication topic (default: "params")
    """

    def __init__(
        self,
        port: int = 5556,
        topic: str = "params",
    ):
        self.port = port
        self.topic = topic

        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None

        # Stats
        self.published_count = 0

    def start(self):
        """Start the publisher."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)  # Only buffer latest
        self.socket.bind(f"tcp://*:{self.port}")

        # Give subscribers time to connect
        time.sleep(0.5)

        print(f"[ParamPublisher] Publishing on tcp://*:{self.port}")

    def stop(self):
        """Stop the publisher."""
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

        print(f"[ParamPublisher] Stopped. Published: {self.published_count} updates")

    def publish(self, params: dict[str, Any]):
        """
        Publish new policy parameters.

        Args:
            params: Policy state dict to publish
        """
        data = pickle.dumps(params)
        self.socket.send_multipart([self.topic.encode(), data])
        self.published_count += 1

    def publish_from_agent(self, agent):
        """
        Publish parameters from an agent with get_policy_params() method.

        Args:
            agent: Agent with get_policy_params() method
        """
        params = agent.get_policy_params()
        self.publish(params)

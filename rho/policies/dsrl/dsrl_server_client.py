"""
DSRL Server Client.

Wraps rho_client.WebsocketClientPolicy to add optional initial_noise support.
Used by both DSRLTrainer (batch RL training calls) and DSRLPolicy (inference).

The noise is passed as a special key "__dsrl_initial_noise__" inside the
observation dict.  The server-side open_pi_server.py extracts it and forwards
it to the policy's sample_actions(... noise=...) call, which seeds the flow
ODE instead of sampling from N(0, I).
"""

import numpy as np
from rho_client.websocket_client_policy import WebsocketClientPolicy


class DSRLServerClient:
    """
    Policy server client with noise-injection support for DSRL.

    Args:
        url: WebSocket URL of the policy server, e.g. "ws://localhost:8765".
    """

    _NOISE_KEY = "__dsrl_initial_noise__"

    def __init__(self, url: str):
        self._client = WebsocketClientPolicy(host=url)

    def infer(
        self,
        obs: dict,
        initial_noise: np.ndarray | None = None,
    ) -> dict:
        """
        Query the policy server, optionally injecting initial noise.

        Args:
            obs:            Observation dict (standard robot format).
                            May be batched: each value has leading batch dim B.
            initial_noise:  Optional noise array to seed the flow ODE.
                            Shape: (C, noise_dim)        for a single sample, or
                                   (B, C, noise_dim)     for a batched call.
                            If None, the server samples from N(0, I) as normal.

        Returns:
            Response dict with at least an "action" key.
        """
        payload = dict(obs)
        if initial_noise is not None:
            payload[self._NOISE_KEY] = initial_noise
        return self._client.infer(payload)

    def get_server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    def reset(self) -> None:
        self._client.reset()

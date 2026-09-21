# Rho Client

`rho-client` is a lightweight Python client for Rho websocket policy servers.
It serializes NumPy observations with MessagePack and returns the action
response produced by the remote policy.

## Installation

From the Rho source tree:

```bash
pip install -e rho_client
```

## Usage

```python
from rho_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=9999)
result = policy.infer(observation)
actions = result["action"]
```

Call `reset()` between episodes when the server or policy implementation uses
episode state. It adds `_reset_: true` to the next `infer()` request, without
modifying your observation dictionary. The Rho server clears observation
history and cached RTC actions before processing that request. The reset
remains pending if the request fails, and is sent only once after a successful
response. Discard any client-side queued actions at the same episode boundary;
if using `num_actions_executed`, start the new episode with zero.

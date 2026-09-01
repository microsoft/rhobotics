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
episode state.

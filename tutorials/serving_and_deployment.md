# Serving a Policy for Real-Time Robot Control

This tutorial explains how to deploy a trained RhoAlpha policy as a WebSocket server that a robot client can query for actions in real time.

## Architecture Overview

RhoAlpha uses a WebSocket-based server/client architecture with msgpack serialization:

```
┌──────────────┐    WebSocket (msgpack)    ┌──────────────────┐
│              │  ──── observations ────►   │                  │
│  Robot       │                           │  Policy Server   │
│  Client      │  ◄──── action chunk ────  │                  │
│              │                           │  (GPU machine)   │
└──────────────┘                           └──────────────────┘
```

1. **Server** loads a pretrained checkpoint, wraps it in a `PolicyInterface`, and listens for WebSocket connections.
2. **Client** sends raw observations (images, joint states, language task) as a msgpack-encoded dict.
3. Server passes observations through the environment's `process_input()`, runs inference via `PolicyInterface.get_action_chunk()`, applies `process_output()`, and returns the action chunk.
4. On initial connection, the server sends metadata (`action_type`, `execution_horizon`) so the client knows what to expect.

A `/healthz` HTTP endpoint is available for liveness checks.


## 1. Server Configuration

Server configs are YAML files with three main sections: the checkpoint to load, the environment type, and the observation mapping. Here is a minimal example:

```yaml
# my_server.yaml
pretrained_checkpoint: /path/to/checkpoints/checkpoint_latest.pt

environment:
  type: "my_robot_server"    # Registered environment name
  port: 7000                 # WebSocket port to listen on

policy_interface_cfg:
  observation_mapping:
    joint_position: observation.state        # Maps client key → policy key
    cam_high: observation.image.0
    cam_wrist: observation.image.1
    task: task
```

### Key parameters

- **`pretrained_checkpoint`** — Path to the checkpoint file (or directory containing `checkpoint_latest.pt`). The server automatically loads the `train_config.json` from the parent directory to reconstruct the policy config, dataset config, and transforms.

- **`environment.type`** — The registered name of your server environment class (e.g. `"aloha_server"`, `"ur5e_server"`). This determines which `process_input()` / `process_output()` logic is used.

- **`environment.port`** — The WebSocket port to listen on.

- **`policy_interface_cfg.observation_mapping`** — Maps the key names your client sends to the key names the policy expects. The client can send observations using its own naming convention (e.g. `"joint_position"`, `"cam_high"`) and this mapping translates them to the policy's expected keys (e.g. `"observation.state"`, `"observation.image.0"`).

### Real-Time Control (RTC) mode

For latency-sensitive deployments, RTC mode overlaps inference with execution:

```yaml
policy_interface_cfg:
  eval_mode: rtc
  inference_delay: 8    # Number of action steps inference takes
  beta: 10              # Weighting of RTC update vs standard update
```

### Example configs

See these working configs for reference:

- `environments/aloha/new_server.yaml` — Aloha bimanual robot (3 cameras, joint position observations)
- `environments/ur5e/new_server.yaml` — UR5e dual-arm setup (3 cameras, tactile, action type conversion)


## 2. Creating a Server Environment

To serve a policy on new hardware, create a server environment class that handles the translation between your robot's observation/action format and the policy's expected format.

### Step 1: Define your config and environment class

Create a file (e.g. `environments/my_robot/my_robot.py`) with a config dataclass and a server class:

```python
from dataclasses import dataclass

import numpy as np
import torch

from rho.common.types import ActionType
from rho.environment import register_environment
from rho.environment.env import EnvironmentConfig
from rho.server.serve_policy import Server


@EnvironmentConfig.register_subclass("my_robot_server")
@dataclass
class MyRobotServerConfig(EnvironmentConfig):
    name: str = "my_robot_server"
    port: int = 7000
    device: str = "cuda"
    input_action_type: ActionType = ActionType.POSITION
    output_action_type: ActionType = ActionType.POSITION
    policy_action_type: ActionType = ActionType.POSITION


@register_environment("my_robot_server")
class MyRobotServer(Server):
    def __init__(self, config: MyRobotServerConfig) -> None:
        super().__init__(config)
        self.device = config.device
        self.image_keys = ["cam_high", "cam_wrist"]
        self.input_action_type = config.input_action_type
        self.output_action_type = config.output_action_type
        self.policy_action_type = config.policy_action_type

    def process_input(self, input) -> dict:
        """Convert raw client observations to policy-ready format.

        Responsibilities:
        - Convert numpy arrays to torch tensors on the correct device
        - Convert images from HWC uint8 BGR to CHW float32 RGB in [0, 1]
        - Reshape state vectors to (1, 1, state_dim)
        - Handle any sensor history or preprocessing
        """
        obs = self.dict_to_torch(input, self.device)

        # Convert BGR images to RGB and normalize to [0, 1]
        for key in self.image_keys:
            if key in obs:
                img = obs[key]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                # HWC → CHW
                if img.ndim == 3:
                    img = img.permute(2, 0, 1)
                # BGR → RGB
                img = img.flip(0)
                # Add batch dimension
                obs[key] = img.unsqueeze(0)

        # Reshape state to (1, 1, state_dim)
        if "joint_position" in obs:
            state = obs["joint_position"]
            if state.ndim == 1:
                state = state.unsqueeze(0).unsqueeze(0)
            elif state.ndim == 2:
                state = state.unsqueeze(0)
            obs["joint_position"] = state

        return obs

    def process_output(self, actions) -> np.ndarray:
        """Convert policy output to client-expected format.

        Input: torch.Tensor of shape (1, chunk_size, action_dim)
        Output: np.ndarray of shape (chunk_size, action_dim)
        """
        return actions.squeeze(0).cpu().numpy()
```

The two methods you must implement are:

- **`process_input()`** — Converts raw client observations into the format expected by the policy. This typically involves: converting numpy to torch, normalizing images to `[0, 1]` float RGB CHW format, reshaping state vectors, and handling any sensor-specific preprocessing (e.g. BGR→RGB conversion, tactile history).

- **`process_output()`** — Converts the policy's action tensor back to the format your client expects. At minimum this involves moving to CPU and converting to numpy. It may also include action space conversions (e.g. joint positions to end effector poses via IK).

### Step 2: Create the server entry point

Create a minimal script (e.g. `environments/my_robot/serve_policy.py`):

```python
#!/usr/bin/env python3
from my_robot import MyRobotServer  # noqa: F401 — registers the environment
from rho.server.serve_policy import eval

if __name__ == "__main__":
    eval()
```

The import of `MyRobotServer` is required even though it appears unused — the `@register_environment` decorator registers the class so that the server can instantiate it from the YAML config.

### Step 3: Create the server config

Create a `server.yaml` file as described in section 1.


## 3. Writing a Client

The `rho_client` package provides a `WebsocketClientPolicy` class for connecting to the server.

### Installation

```bash
pip install -e rho_client/
```

### Minimal client example

```python
import numpy as np
from rho_client import websocket_client_policy

# Connect to the server
client = websocket_client_policy.WebsocketClientPolicy(
    host="localhost",
    port=7000,
)

# Server sends metadata on connect
metadata = client.get_server_metadata()
print(f"Action type: {metadata['action_type']}")
print(f"Execution horizon: {metadata['execution_horizon']}")

# Build an observation dict using the raw key names your robot produces.
# The server's observation_mapping translates these to policy keys.
obs = {
    "joint_position": np.array([[0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.0]],
                                dtype=np.float64),
    "cam_high": np.zeros((480, 640, 3), dtype=np.uint8),      # HWC BGR
    "cam_wrist": np.zeros((480, 640, 3), dtype=np.uint8),      # HWC BGR
    "task": ["pick up the red block"],
    "_reset_": 1,  # Signal a new episode (resets action history)
}

# Run inference
result = client.infer(obs)
print(f"Inference time: {result['infer_ms'][0]:.1f} ms")
print(f"Action shape: {result['action'].shape}")  # (chunk_size, action_dim)
```

### Key conventions

- **`_reset_`** — Include `"_reset_": 1` in the first observation of each episode. This tells the `PolicyInterface` to clear its action history and observation queue. Omit it (or set to 0) for subsequent steps.

- **`task`** — The language instruction as a list containing a single string, e.g. `["pick up the red block"]`.

- **Images** — Send as numpy arrays in HWC format (the server's `process_input()` handles any reordering). The channel format (BGR vs RGB) depends on your server environment implementation.

- **State vectors** — Send as numpy arrays. The shape should match what your `process_input()` expects (typically `(1, state_dim)` or `(state_dim,)`).

- **Response** — The response dict contains:
  - `"action"`: numpy array of shape `(chunk_size, action_dim)` — the action chunk to execute.
  - `"infer_ms"`: list with a single float — the inference time in milliseconds.

### Robot control loop

A typical control loop executes a subset of the action chunk before querying for the next one:

```python
execution_horizon = metadata["execution_horizon"]

while running:
    obs = collect_observations()  # Read from your robot's sensors

    result = client.infer(obs)
    actions = result["action"]  # shape: (chunk_size, action_dim)

    # Execute only the first `execution_horizon` steps
    for i in range(execution_horizon):
        send_to_robot(actions[i])
        wait_for_control_cycle()
```

See `environments/ur5e/dummy_client.py` for a complete working example.


## 4. Running the Server

### Start the server

```bash
python environments/my_robot/serve_policy.py \
  --config_path=environments/my_robot/server.yaml
```

### Health check

```bash
curl http://localhost:7000/healthz
# Returns: OK
```

### Docker deployment

When running inside Docker, make sure to expose the server port:

```bash
docker run --gpus all -p 7000:7000 ...
```

### Action type conversions

The server environment can convert between action types. For example, if the policy outputs end effector poses but your robot expects joint positions, your `process_output()` can perform inverse kinematics. The `AlohaServer` includes a full Pinocchio IK implementation as a reference (`environments/aloha/aloha.py`).

Configure the action types in the environment section of your server YAML:

```yaml
environment:
  type: "my_robot_server"
  port: 7000
  input_action_type: "POSITION"           # What the client sends as state
  output_action_type: "EE_QUAT_POS_XYZW"  # What the client expects back
```


## 5. Reference implementations

| Environment | Server class | Key features |
|---|---|---|
| `environments/aloha/` | `AlohaServer` | Pinocchio IK, joint↔EE conversion, gripper-aware 14D bimanual layout, action chunk safety checks (joint delta limits) |
| `environments/ur5e/` | `UR5eServer` | BGR→RGB image handling, dual-arm FK, tactile history processing |

These serve as templates for new hardware integrations. The Aloha implementation is the most full-featured example, while UR5e demonstrates a simpler setup.

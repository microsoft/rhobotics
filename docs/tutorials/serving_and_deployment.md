# Serving Rho for Robot Control

Rho provides a WebSocket policy server and a lightweight Python client. The
server loads a portable Rho checkpoint, receives observations encoded with
MessagePack, and returns an action chunk.

The public repository does not include a production hardware integration.
Before serving on a robot, implement and test an environment adapter for that
robot's observations, action space, timing, and safety constraints.

## Install the server and client

From the repository root:

```bash
pip install -e rho_client
pip install -e ".[server]"
```

The training container installs both packages in this order.

## Create an environment adapter

An adapter converts the robot's wire-format observations into tensors accepted
by the policy and converts policy outputs back into robot commands.

Create `environments/my_robot/server_env.py`:

```python
from dataclasses import dataclass

import numpy as np
import torch

from rho.common.types import ActionType
from rho.environment import EnvironmentConfig, register_environment
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

    def process_input(self, observation: dict) -> dict:
        observation = self.dict_to_torch(observation, self.device)

        for key in ("camera", "wrist_camera"):
            image = observation[key]
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            if image.ndim == 3:
                image = image.permute(2, 0, 1)
            observation[key] = image.unsqueeze(0)

        state = observation["joint_position"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        observation["joint_position"] = state
        return observation

    def process_output(self, actions: torch.Tensor) -> np.ndarray:
        return actions.squeeze(0).detach().cpu().numpy()

    def convert_observation_state_type_to_policy_type(self, observation):
        return observation

    def convert_policy_action_type_to_client_type(self, actions):
        return actions
```

Image channel order, normalization, state shape, and action conversion are
robot-specific. The example assumes RGB images and matching policy/client
action representations. Add explicit validation and safety limits before
connecting physical hardware.

## Register the adapter

Create `environments/my_robot/serve.py`:

```python
from server_env import MyRobotServer  # noqa: F401

from rho.server.serve_policy import eval


if __name__ == "__main__":
    eval()
```

Importing `MyRobotServer` registers both the environment configuration and
runtime implementation before Draccus decodes the YAML configuration.

## Configure the server

The server uses the same `EvalConfig` structure as evaluation. A portable
finetuned checkpoint supplies the policy configuration, feature schema,
dataset configuration, and normalization statistics.

Create `environments/my_robot/server.yaml`:

```yaml
pretrained_checkpoint: /path/to/portable/checkpoint
device: "cuda"
record_videos: false

environment:
  type: "my_robot_server"
  port: 7000

policy_interface_cfg:
  observation_mapping:
    joint_position: observation.state
    camera: observation.images.image
    wrist_camera: observation.images.wrist_image
    task: task
```

The mapping keys are produced by `process_input`; the values must match the
feature names stored in the checkpoint.

For real-time chunking, add:

```yaml
eval_mode: "rtc"
inference_delay: 8
execution_horizon: 16
beta: 10
guidance_schedule: "paper"
```

## Start the server

```bash
python environments/my_robot/serve.py \
  --config_path=environments/my_robot/server.yaml
```

The WebSocket server listens on `environment.port`. Its health endpoint is:

```bash
curl http://localhost:7000/healthz
```

## Query the server

```python
import numpy as np

from rho_client.websocket_client_policy import WebsocketClientPolicy


client = WebsocketClientPolicy(host="localhost", port=7000)
metadata = client.get_server_metadata()

observation = {
    "joint_position": np.zeros(7, dtype=np.float32),
    "camera": np.zeros((480, 640, 3), dtype=np.uint8),
    "wrist_camera": np.zeros((480, 640, 3), dtype=np.uint8),
    "task": ["pick up the object"],
    "_reset_": 1,
}

result = client.infer(observation)
actions = result["action"]

for action in actions[: metadata["execution_horizon"]]:
    send_to_robot(action)
```

Call `client.reset()` between episodes when the policy or server maintains
episode state. The response uses the singular `"action"` key.

## DSRL and FlowDAgger

The same server can wrap the loaded Rho policy for online adaptation. Add one
of the following sections to the server configuration:

```yaml
train: true
trainer_type: "dsrl"
dsrl:
  base_policy_name: "rho"
```

or:

```yaml
train: true
trainer_type: "flowdagger"
flowdagger:
  base_policy_name: "rho"
```

These modes require an environment integration that publishes the appropriate
experience and intervention signals. Validate the complete control loop in
simulation before using online adaptation on hardware.

## Deployment safety

Before commanding a robot:

- Validate observation keys, shapes, units, image channel order, and timestamps.
- Clamp actions to hardware limits and reject discontinuous action chunks.
- Add watchdogs, communication timeouts, emergency-stop handling, and a safe
  fallback command.
- Test checkpoint loading, the health endpoint, metadata, reset behavior, and
  client/server inference without hardware.
- Run the adapter in simulation or shadow mode before enabling actuation.

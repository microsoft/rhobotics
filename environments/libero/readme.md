# LIBERO Networked Evaluation

Evaluate trained checkpoints on the LIBERO simulated environment using a
server/client architecture. The **server** loads the policy and serves it
over websocket; the **client** runs the LIBERO simulator, sends observations,
and receives actions. Both run inside the same Docker container.

## Quick start (phi4mm)

### 1. Launch the container

From the repo root (`phi-4-robotics/`):

```bash
./docker/alku/libero/run_interactive.sh
```
> **Note:** The target container may need to be changed from `alku-libero:20260203` to `msrxworkspace1acr.azurecr.io/phi4robotics/alku-libero:20260203` and additional ports may need to be added or changed.


This starts the `alku-libero` container with GPU access and drops you into a
shell at `/workspace`.

### 2. One-time setup (inside the container)

Install the EGL driver needed for headless MuJoCo rendering and the websocket
client:

```bash
apt-get update && apt-get install -y libnvidia-gl-580-server
pip install -e rho_client/
```

> **Note:** The `libnvidia-gl` version must match the host driver. Check with
> `cat /proc/driver/nvidia/version` — if your driver isn't `580`, adjust the
> package name accordingly (e.g. `libnvidia-gl-550-server`).

### 3. Open two terminals

Use `tmux` (or `docker exec`) to get two shells inside the container.

**Terminal 1 — Start the server:**

```bash
python3 environments/libero/serve_libero.py \
    --config_path environments/libero/configs/serve_libero_phi4mm.yaml
```

**Terminal 2 — Run the client:**

```bash
python3 environments/libero/libero_client.py \
    --host localhost --port 7000 \
    --task_suite libero_spatial \
    --episodes_per_task 10
```

Add `--record_video` to save MP4s of each episode.

## Evaluating Pi0.5

Swap the server config:

> **Note:** Before evaluating pi0.5, install sentencepiece and update transformers according to the [pi0 readme](../../alku/policies/pi0/readme.md).

```bash
# Terminal 1
python3 environments/libero/serve_libero.py \
    --config_path environments/libero/configs/serve_libero_pi05.yaml

# Terminal 2 (same client command)
python3 environments/libero/libero_client.py \
    --host localhost --port 7000 \
    --task_suite libero_spatial \
    --episodes_per_task 10
```

## Client options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Server hostname or IP |
| `--port` | `7000` | Server port |
| `--task_suite` | `libero_spatial` | Suite name (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`) |
| `--task_ids` | all | Specific task IDs to evaluate |
| `--episodes_per_task` | `10` | Episodes per task |
| `--max_steps` | suite default | Max steps per episode |
| `--n_action_steps` | server chunk_size | Actions to execute per inference call |
| `--record_video` | off | Record MP4 videos |
| `--output_dir` | auto | Directory for videos and results JSON |
| `--seed` | `42` | Random seed |

## Remote evaluation

To run the server on a GPU machine and the client on a different machine,
point the client at the server machine's IP. Port `7000` is exposed by the
container's `run_interactive.sh` script.

```bash
# On sim machine (client):
python3 environments/libero/libero_client.py \
    --host <GPU_MACHINE_IP> --port 7000 \
    --task_suite libero_spatial
```

## Running tests

Unit tests for `LiberoServer` live at `tests/environments/test_libero_server.py`.
They run on CPU and don't require EGL or the LIBERO simulator.

Inside the container:

```bash
python3 -m pytest tests/environments/test_libero_server.py -v
```

## Troubleshooting

### EGL errors (`Cannot initialize a EGL device display`)

The LIBERO client requires the NVIDIA EGL driver for headless rendering.
If you see EGL errors, make sure `libnvidia-gl` is installed inside the
container (see [One-time setup](#2-one-time-setup-inside-the-container)).

Verify the driver is working:

```bash
python3 -c "
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
    has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
    camera_names=['agentview', 'robot0_eye_in_hand'])
print('EGL rendering OK')
env.close()
"
```

If it still fails, check:

1. The NVIDIA EGL vendor ICD is registered:
   ```bash
   ls /usr/share/glvnd/egl_vendor.d/
   # Should contain 10_nvidia.json
   ```
2. The EGL library exists:
   ```bash
   find / -name "libEGL_nvidia.so*" 2>/dev/null
   ```

## File structure

```
environments/libero/
├── configs/
│   ├── serve_libero_phi4mm.yaml   # Server config for phi4mm checkpoint
│   └── serve_libero_pi05.yaml    # Server config for pi0.5 checkpoint
├── libero_server.py              # LiberoServer env (process_input/output)
├── serve_libero.py               # Server entry point
├── libero_client.py              # Simulation evaluation client
└── readme.md                     # This file
```

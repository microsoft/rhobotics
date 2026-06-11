#!/usr/bin/env python3
"""
Web-based visualizer proxy for policy server debugging.

Acts as a transparent proxy between a robot client and a policy server,
while streaming all observations and actions to a browser-based dashboard.

Features:
- Live camera image display (auto-detects image keys)
- Proprioceptive state display (table of joint states / EEF poses)
- Action chunk visualization
- Toggle between Joint-State view and EEF Pose view
- In EEF Pose mode: interactive 3D plot showing gripper positions + action path
- Auto-detects arm configuration (single/dual) and orientation format
  (quaternion, RPY, 6DOF) from action dimensions

Usage:
    python -m rho.utils.policy_visualizer --port 8000 --viz-port 8080 \
        --upstream-host localhost --upstream-port 7010

    Then open http://localhost:8080 in a browser.
"""

import argparse
import asyncio
import base64
import http
import io
import logging
import threading
import traceback
from dataclasses import dataclass, field

import numpy as np
import websockets.asyncio.server as _server

from rho_client.msgpack_numpy import Packer, unpackb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("policy_visualizer")


# ---------------------------------------------------------------------------
# Arm / orientation detection
# ---------------------------------------------------------------------------


@dataclass
class ArmConfig:
    """Detected arm configuration."""

    num_arms: int = 2
    orientation_type: str = "unknown"  # "quaternion", "rpy", "6dof"
    pos_dim: int = 3
    orient_dim: int = 4
    grip_dim: int = 1
    per_arm_dim: int = 8
    total_dim: int = 16

    @property
    def label(self) -> str:
        arms = "Dual-arm" if self.num_arms == 2 else "Single-arm"
        return f"{arms} / {self.orientation_type} (dim={self.total_dim})"

    def to_dict(self) -> dict:
        return {
            "num_arms": self.num_arms,
            "orientation_type": self.orientation_type,
            "pos_dim": self.pos_dim,
            "orient_dim": self.orient_dim,
            "grip_dim": self.grip_dim,
            "per_arm_dim": self.per_arm_dim,
            "total_dim": self.total_dim,
            "label": self.label,
        }


def detect_arm_config(action_dim: int) -> ArmConfig:
    """Detect arm configuration from action dimensions.

    Known configurations:
      Single arm:  7 = xyz+quat, 8 = xyz+quat+grip, 10 = xyz+6dof+grip
      Dual arm:   14 = 2*(xyz+quat) or 2*(xyz+rpy+grip), 16 = 2*(xyz+quat+grip), 20 = 2*(xyz+6dof+grip)
    """
    configs = {
        7: (1, "quaternion", 4, 0),
        8: (1, "quaternion", 4, 1),
        10: (1, "6dof", 6, 1),
        14: (2, "quaternion", 4, 0),
        16: (2, "quaternion", 4, 1),
        20: (2, "6dof", 6, 1),
    }

    if action_dim in configs:
        num_arms, orient_type, orient_dim, grip_dim = configs[action_dim]
        per_arm = action_dim // num_arms
        return ArmConfig(num_arms, orient_type, 3, orient_dim, grip_dim, per_arm, action_dim)

    # Fallback heuristics
    if action_dim % 2 == 0:
        per_arm = action_dim // 2
        if per_arm == 7:
            return ArmConfig(2, "rpy", 3, 3, 1, 7, action_dim)
        orient_dim = per_arm - 3 - 1
        orient_type = {3: "rpy", 4: "quaternion", 6: "6dof"}.get(orient_dim, "unknown")
        return ArmConfig(2, orient_type, 3, max(orient_dim, 0), 1, per_arm, action_dim)

    orient_dim = action_dim - 3 - 1
    orient_type = {3: "rpy", 4: "quaternion", 6: "6dof"}.get(orient_dim, "unknown")
    return ArmConfig(1, orient_type, 3, max(orient_dim, 0), 1, action_dim, action_dim)


# ---------------------------------------------------------------------------
# Shared visualization state
# ---------------------------------------------------------------------------


@dataclass
class VizState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    step: int = 0
    last_state: list = field(default_factory=list)
    last_actions: list = field(default_factory=list)
    arm_config: ArmConfig | None = None
    infer_ms: float = 0.0
    images: dict = field(default_factory=dict)  # key -> base64 jpeg
    image_resolutions: dict = field(default_factory=dict)  # key -> "HxW"
    prompt: str = ""
    # Prompt override
    prompt_override: str | None = None  # None = passthrough, str = inject this prompt
    paused: bool = False  # True = halt forwarding while user types new prompt
    # WebSocket subscribers (set of asyncio.Queue)
    _subscribers: set = field(default_factory=set)
    _dashboard_loop: asyncio.AbstractEventLoop | None = None

    def set_dashboard_loop(self, loop: asyncio.AbstractEventLoop):
        self._dashboard_loop = loop

    def subscribe(self, queue: asyncio.Queue):
        self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue):
        self._subscribers.discard(queue)

    def notify(self):
        """Push current state snapshot to all WebSocket subscribers.

        Thread-safe: schedules puts on the dashboard event loop.
        """
        if not self._subscribers or not self._dashboard_loop:
            return
        data = self._snapshot()
        self._dashboard_loop.call_soon_threadsafe(self._dispatch, data)

    def _dispatch(self, data: dict):
        """Runs on the dashboard event loop thread."""
        for q in list(self._subscribers):
            try:
                # Drain stale items so consumer always gets the latest
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def _snapshot(self) -> dict:
        return {
            "step": self.step,
            "state": self.last_state,
            "actions": self.last_actions,
            "arm_config": self.arm_config.to_dict() if self.arm_config else None,
            "infer_ms": self.infer_ms,
            "images": self.images,
            "image_resolutions": self.image_resolutions,
            "prompt": self.prompt,
            "prompt_override": self.prompt_override,
            "paused": self.paused,
        }


viz_state = VizState()


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def encode_image_b64(img: np.ndarray, bgr: bool = False) -> str:
    """Encode HWC image to base64 JPEG.

    Args:
        img: numpy array (H, W, 3) in either RGB or BGR order.
        bgr: if True, input is BGR and will be converted to RGB for display.
    """
    from PIL import Image

    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3 and bgr:
        img = np.ascontiguousarray(img[:, :, ::-1])
    else:
        img = np.ascontiguousarray(img)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def is_image_array(val) -> bool:
    """Heuristic: check if a value looks like an HWC image array."""
    if not isinstance(val, np.ndarray):
        return False
    if val.ndim != 3:
        return False
    # (H, W, C) where C is 3 or 4
    if val.shape[2] not in (3, 4):
        return False
    # Minimum spatial size
    return not (val.shape[0] < 16 or val.shape[1] < 16)


def is_state_array(val) -> bool:
    """Heuristic: check if a value looks like a proprioceptive state vector.

    State vectors are small numeric arrays (1D or 2D with small last dim).
    Distinguishes from images by dimensionality and size.
    """
    if not isinstance(val, np.ndarray):
        return False
    if not np.issubdtype(val.dtype, np.number):
        return False
    flat_size = val.size
    # State vectors are typically 1-100 elements
    if flat_size < 1 or flat_size > 200:
        return False
    # Must be 1D or 2D (batch,dim)
    return val.ndim <= 2


def is_prompt_value(val) -> bool:
    """Heuristic: check if a value looks like a text prompt."""
    if isinstance(val, str):
        return True
    return isinstance(val, list) and len(val) > 0 and isinstance(val[0], str)


# ---------------------------------------------------------------------------
# Proxy websocket server
# ---------------------------------------------------------------------------


class VizProxyServer:
    """Websocket proxy that captures obs/actions for the web dashboard."""

    def __init__(self, port: int, upstream_host: str, upstream_port: int):
        self.port = port
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self._metadata: dict = {}

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.sync.client

        upstream_uri = f"ws://{self.upstream_host}:{self.upstream_port}"
        logger.info(f"Connecting to upstream policy server at {upstream_uri}...")
        upstream_conn = websockets.sync.client.connect(upstream_uri, compression=None, max_size=None)
        metadata_bytes = upstream_conn.recv()
        self._metadata = unpackb(metadata_bytes)
        logger.info(f"Upstream metadata: {self._metadata}")
        upstream_conn.close()

        async with _server.serve(
            self._handler,
            "0.0.0.0",  # nosec B104
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"Viz proxy server listening on 0.0.0.0:{self.port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        import websockets.sync.client

        logger.info(f"Client connected from {websocket.remote_address}")
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        upstream_uri = f"ws://{self.upstream_host}:{self.upstream_port}"
        upstream = websockets.sync.client.connect(upstream_uri, compression=None, max_size=None)
        upstream.recv()  # consume metadata

        try:
            while True:
                # Forward raw bytes in BOTH directions — never re-serialize
                raw_bytes = await websocket.recv()

                # If paused, hold the message until unpaused
                while viz_state.paused:
                    await asyncio.sleep(0.05)

                try:
                    obs = unpackb(raw_bytes)
                    self._capture_observation(obs)
                except Exception:
                    pass  # nosec B110 - visualization failure must not break the proxy

                # If prompt override is active, inject it into the observation
                override = viz_state.prompt_override
                if override is not None:
                    try:
                        obs_to_send = unpackb(raw_bytes)
                        # Replace any prompt/task key found, or add "prompt"
                        injected = False
                        for key in list(obs_to_send.keys()):
                            val = obs_to_send[key]
                            if isinstance(val, str) or (
                                isinstance(val, list) and val and isinstance(val[0], str)
                            ):
                                obs_to_send[key] = override if isinstance(val, str) else [override]
                                injected = True
                        if not injected:
                            obs_to_send["prompt"] = override
                        packer = Packer()
                        raw_bytes = packer.pack(obs_to_send)
                    except Exception:
                        pass  # nosec B110 - if injection fails, forward original

                upstream.send(raw_bytes)
                response_bytes = upstream.recv()

                if isinstance(response_bytes, str):
                    await websocket.send(response_bytes)
                    continue

                try:
                    response = unpackb(response_bytes)
                    self._capture_actions(response)
                except Exception:  # noqa: BLE001
                    pass  # nosec B110

                # Forward raw response bytes — preserves exact serialization
                await websocket.send(response_bytes)

        except websockets.ConnectionClosed:
            logger.info(f"Client {websocket.remote_address} disconnected")
        except Exception:
            logger.exception("Proxy error")
            await websocket.send(traceback.format_exc())
            await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Proxy error")
        finally:
            upstream.close()

    def _capture_observation(self, obs: dict):
        with viz_state.lock:
            viz_state.step += 1

            # Priority-based state detection: prefer keys with state/joint/qpos in name
            state_candidates = {}
            for key, val in obs.items():
                if is_image_array(val):
                    is_bgr = key.startswith("cam")
                    viz_state.images[key] = encode_image_b64(val, bgr=is_bgr)
                    viz_state.image_resolutions[key] = f"{val.shape[0]}x{val.shape[1]}"
                elif is_prompt_value(val):
                    viz_state.prompt = val[0] if isinstance(val, list) else str(val)
                elif isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
                    state_candidates[key] = val

            # Pick the best state key by priority
            if state_candidates:
                # Priority: keys containing "state" > "joint" > "qpos" > smallest 1D array
                def state_priority(k):
                    kl = k.lower()
                    if "state" in kl:
                        return 0
                    if "joint" in kl or "qpos" in kl:
                        return 1
                    # Deprioritize keys that look like history/actions/tactile
                    if any(x in kl for x in ["history", "tactile", "force", "action", "prev"]):
                        return 9
                    return 5

                best_key = min(state_candidates.keys(), key=state_priority)
                val = state_candidates[best_key]
                state = np.asarray(val, dtype=np.float64).flatten()
                viz_state.last_state = state.tolist()

    def _capture_actions(self, response: dict):
        with viz_state.lock:
            actions = response.get("actions", response.get("action"))
            if actions is not None:
                actions = np.asarray(actions, dtype=np.float64)
                viz_state.last_actions = actions.tolist()

                action_dim = actions.shape[-1]
                if viz_state.arm_config is None or viz_state.arm_config.total_dim != action_dim:
                    viz_state.arm_config = detect_arm_config(action_dim)
                    logger.info(f"Detected: {viz_state.arm_config.label}")

            if "policy_timing" in response:
                viz_state.infer_ms = response["policy_timing"].get("infer_ms", 0)
            elif "infer_ms" in response:
                ms = response["infer_ms"]
                viz_state.infer_ms = ms[0] if isinstance(ms, list) else ms

            # Notify WebSocket subscribers after a complete inference cycle
            viz_state.notify()


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ---------------------------------------------------------------------------
# Web dashboard (aiohttp)
# ---------------------------------------------------------------------------


def get_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
<title>Policy Visualizer</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
       background: #1a1a2e; color: #e0e0e0; padding: 12px; }
h1 { color: #64ffda; margin-bottom: 8px; font-size: 1.4em; }
.header { display: flex; align-items: center; gap: 20px; margin-bottom: 12px; flex-wrap: wrap; }
.status { font-size: 0.9em; color: #aaa; }
.status .val { color: #64ffda; font-weight: bold; }
.toggle-group { display: flex; gap: 8px; }
.toggle-btn { padding: 6px 14px; border: 1px solid #444; background: #2a2a3e;
              color: #ccc; cursor: pointer; border-radius: 4px; font-size: 0.85em; }
.toggle-btn.active { background: #64ffda; color: #1a1a2e; border-color: #64ffda; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; margin-bottom: 12px; }
.cam-card { background: #2a2a3e; border-radius: 6px; overflow: hidden; }
.cam-card img { width: 100%; height: auto; display: block; }
.cam-card .label { padding: 4px 8px; font-size: 0.75em; color: #888; text-align: center; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.panel { background: #2a2a3e; border-radius: 6px; padding: 12px; overflow-x: auto; }
.panel h3 { color: #bb86fc; margin-bottom: 8px; font-size: 1em; }
table { width: 100%; border-collapse: collapse; font-size: 0.78em; }
table th, table td { padding: 3px 6px; text-align: right; border-bottom: 1px solid #333; }
table th { color: #888; }
table td { color: #e0e0e0; font-family: monospace; }
.prompt-bar { background: #2a2a3e; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.9em;
             display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.prompt-bar span { color: #64ffda; }
.prompt-bar .override-controls { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.prompt-bar input[type="text"] { background: #1a1a2e; border: 1px solid #444; color: #e0e0e0;
                                  padding: 4px 8px; border-radius: 4px; font-size: 0.9em; width: 280px; }
.prompt-bar button { padding: 4px 10px; border: 1px solid #444; background: #3a3a5e; color: #ccc;
                     cursor: pointer; border-radius: 4px; font-size: 0.8em; }
.prompt-bar button:hover { background: #4a4a6e; }
.prompt-bar button.active { background: #bb86fc; color: #1a1a2e; border-color: #bb86fc; }
.prompt-bar .paused-badge { color: #ff6b6b; font-weight: bold; font-size: 0.8em; }
#plot3d { width: 100%; height: 450px; }
</style>
</head>
<body>
<div class="header">
  <h1>&#x1F916; Policy Visualizer</h1>
  <div class="status">Step: <span class="val" id="step">0</span></div>
  <div class="status">Infer: <span class="val" id="infer-ms">0</span> ms</div>
  <div class="status" id="arm-badge" style="font-size:0.8em;color:#888"></div>
  <div class="status" id="conn-badge" style="font-size:0.8em;color:#888">&#x25cb; Connecting</div>
  <div class="toggle-group">
    <button class="toggle-btn active" id="btn-joints" onclick="setMode('joints')">Joint States</button>
    <button class="toggle-btn" id="btn-eef" onclick="setMode('eef')">EEF Poses</button>
  </div>
</div>

<div class="prompt-bar">
  <span>Prompt:</span> <span id="prompt">&mdash;</span>
  <span class="paused-badge" id="paused-badge" style="display:none">PAUSED</span>
  <div class="override-controls">
    <input type="text" id="override-input" placeholder="Override prompt..." />
    <button id="btn-override" onclick="setOverride()">Override</button>
    <button id="btn-clear-override" onclick="clearOverride()" style="display:none">Clear</button>
  </div>
</div>
<div class="grid" id="cameras"></div>

<div class="panels">
  <div class="panel">
    <h3>Observation State</h3>
    <div id="state-table"></div>
  </div>
  <div class="panel">
    <h3>Action Chunk</h3>
    <div id="action-table"></div>
  </div>
</div>

<div class="panel" id="plot-panel" style="margin-top:12px; display:none;">
  <h3>3D EEF Trajectory</h3>
  <div id="plot3d"></div>
</div>

<script>
let mode = 'joints';
let plotInitialized = false;

function setMode(m) {
    mode = m;
    document.getElementById('btn-joints').classList.toggle('active', m === 'joints');
    document.getElementById('btn-eef').classList.toggle('active', m === 'eef');
    document.getElementById('plot-panel').style.display = m === 'eef' ? 'block' : 'none';
    if (m === 'eef') plotInitialized = false;
}

function fmt(v) { return v == null ? '\\u2014' : parseFloat(v).toFixed(4); }

function getLabels(dim, cfg) {
    if (!cfg) return Array(dim).fill('');
    const ol = {quaternion:['qx','qy','qz','qw'], rpy:['r','p','y'], '6dof':['r1','r2','r3','r4','r5','r6']}[cfg.orientation_type] || [];
    const arms = cfg.num_arms === 2 ? ['L','R'] : [''];
    const labels = [];
    for (const a of arms) {
        const p = a ? a+'_' : '';
        labels.push(p+'x', p+'y', p+'z');
        for (const o of ol) labels.push(p+o);
        if (cfg.grip_dim > 0) labels.push(p+'grip');
    }
    while (labels.length < dim) labels.push('');
    return labels;
}

function renderState(state, cfg) {
    if (!state || !state.length) return '<em>No state</em>';
    const labels = getLabels(state.length, cfg);
    let h = '<table><tr><th>#</th><th>Value</th><th style="text-align:left">Label</th></tr>';
    for (let i = 0; i < state.length; i++)
        h += '<tr><td>'+i+'</td><td>'+fmt(state[i])+'</td><td style="color:#666;text-align:left">'+labels[i]+'</td></tr>';
    return h + '</table>';
}

function renderActions(actions) {
    if (!actions || !actions.length) return '<em>No actions</em>';
    const H = actions.length, D = actions[0].length;
    let h = '<div style="font-size:0.75em;color:#888;margin-bottom:4px">Horizon: '+H+' &times; Dim: '+D+'</div>';
    h += '<table><tr><th>t</th>';
    for (let d = 0; d < Math.min(D,14); d++) h += '<th>'+d+'</th>';
    if (D > 14) h += '<th>...</th>';
    h += '</tr>';
    const rows = [];
    for (let i = 0; i < Math.min(4, H); i++) rows.push(i);
    if (H > 6) rows.push(-1);
    for (let i = Math.max(4, H-2); i < H; i++) rows.push(i);
    for (const r of rows) {
        if (r === -1) { h += '<tr><td colspan="'+(Math.min(D,14)+2)+'" style="text-align:center;color:#555">&#8942;</td></tr>'; continue; }
        h += '<tr><td>'+r+'</td>';
        for (let d = 0; d < Math.min(D,14); d++) h += '<td>'+fmt(actions[r][d])+'</td>';
        if (D > 14) h += '<td>&hellip;</td>';
        h += '</tr>';
    }
    return h + '</table>';
}

function render3D(state, actions, cfg) {
    if (mode !== 'eef' || !cfg || !actions || !actions.length) return;
    const perArm = cfg.per_arm_dim;
    const traces = [];
    const armNames = cfg.num_arms === 2 ? ['Left', 'Right'] : ['Arm'];
    const armColors = ['#64ffda', '#ff6b6b'];
    const offsets = [0, perArm];

    for (let ai = 0; ai < cfg.num_arms; ai++) {
        const off = offsets[ai];
        if (state && state.length > off + 2) {
            traces.push({
                x: [state[off]], y: [state[off+1]], z: [state[off+2]],
                mode: 'markers', type: 'scatter3d',
                marker: {size: 8, color: armColors[ai], symbol: 'diamond'},
                name: armNames[ai] + ' (current)'
            });
        }
        const xs = [], ys = [], zs = [];
        for (let t = 0; t < actions.length; t++) {
            xs.push(actions[t][off]);
            ys.push(actions[t][off+1]);
            zs.push(actions[t][off+2]);
        }
        traces.push({
            x: xs, y: ys, z: zs,
            mode: 'lines+markers', type: 'scatter3d',
            line: {color: armColors[ai], width: 3},
            marker: {size: 2, color: armColors[ai]},
            name: armNames[ai] + ' (planned)'
        });
    }

    const layout = {
        margin: {l:0, r:0, t:30, b:0},
        paper_bgcolor: '#2a2a3e', plot_bgcolor: '#1a1a2e',
        font: {color: '#ccc'},
        scene: {
            xaxis: {title: 'X', gridcolor: '#333', zerolinecolor: '#555'},
            yaxis: {title: 'Y', gridcolor: '#333', zerolinecolor: '#555'},
            zaxis: {title: 'Z', gridcolor: '#333', zerolinecolor: '#555'},
            bgcolor: '#1a1a2e'
        }
    };

    if (!plotInitialized) {
        Plotly.newPlot('plot3d', traces, layout, {responsive: true});
        plotInitialized = true;
    } else {
        Plotly.react('plot3d', traces, layout);
    }
}

function renderCameras(images, resolutions) {
    const el = document.getElementById('cameras');
    const keys = Object.keys(images);
    if (!keys.length) { el.innerHTML = '<div style="color:#555">No images yet</div>'; return; }
    // Update existing img elements in-place to avoid DOM rebuild and scroll jump
    for (const k of keys) {
        let card = document.getElementById('cam-' + k);
        if (!card) {
            card = document.createElement('div');
            card.className = 'cam-card';
            card.id = 'cam-' + k;
            card.innerHTML = '<img/><div class="label"></div>';
            el.appendChild(card);
        }
        const img = card.querySelector('img');
        const newSrc = 'data:image/jpeg;base64,' + images[k];
        if (img.src !== newSrc) img.src = newSrc;
        const res = (resolutions && resolutions[k]) ? ' [' + resolutions[k] + ']' : '';
        card.querySelector('.label').textContent = k + res;
    }
}

async function poll() {
    try {
        const r = await fetch('/api/state');
        const d = await r.json();
        handleUpdate(d);
    } catch(e) { console.error(e); }
}

function handleUpdate(d) {
    document.getElementById('step').textContent = d.step;
    document.getElementById('infer-ms').textContent = d.infer_ms.toFixed(1);
    document.getElementById('arm-badge').textContent = d.arm_config ? d.arm_config.label : '';
    renderCameras(d.images, d.image_resolutions);
    document.getElementById('state-table').innerHTML = renderState(d.state, d.arm_config);
    document.getElementById('action-table').innerHTML = renderActions(d.actions);
    render3D(d.state, d.actions, d.arm_config);

    // Prompt display: show override if active
    const promptEl = document.getElementById('prompt');
    if (d.prompt_override != null) {
        promptEl.textContent = d.prompt_override + ' [OVERRIDE]';
        promptEl.style.color = '#bb86fc';
    } else {
        promptEl.textContent = d.prompt || '\\u2014';
        promptEl.style.color = '#64ffda';
    }
    document.getElementById('paused-badge').style.display = d.paused ? 'inline' : 'none';
}

// --- Prompt override controls ---
async function setOverride() {
    const input = document.getElementById('override-input');
    const text = input.value.trim();
    if (!text) {
        // Pause first so user can type
        await fetch('/api/prompt', {method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({action:'pause'})});
        input.focus();
        document.getElementById('btn-override').textContent = 'Send';
        document.getElementById('btn-override').classList.add('active');
        document.getElementById('btn-override').onclick = sendOverride;
        return;
    }
    sendOverride();
}

async function sendOverride() {
    const input = document.getElementById('override-input');
    const text = input.value.trim();
    if (!text) return;
    await fetch('/api/prompt', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action:'set', prompt: text})});
    document.getElementById('btn-override').textContent = 'Override';
    document.getElementById('btn-override').classList.remove('active');
    document.getElementById('btn-override').onclick = setOverride;
    document.getElementById('btn-clear-override').style.display = 'inline';
}

async function clearOverride() {
    await fetch('/api/prompt', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action:'clear'})});
    document.getElementById('override-input').value = '';
    document.getElementById('btn-clear-override').style.display = 'none';
    document.getElementById('btn-override').textContent = 'Override';
    document.getElementById('btn-override').classList.remove('active');
    document.getElementById('btn-override').onclick = setOverride;
}

// Allow Enter key to submit override
document.getElementById('override-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') sendOverride();
});

// WebSocket streaming with automatic fallback to polling
let ws = null;
let wsConnected = false;
let pollInterval = null;

function connectWs() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = function() {
        wsConnected = true;
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
        document.getElementById('conn-badge').textContent = '\\u25cf WS';
        document.getElementById('conn-badge').style.color = '#64ffda';
    };
    ws.onmessage = function(ev) {
        try { handleUpdate(JSON.parse(ev.data)); } catch(e) { console.error(e); }
    };
    ws.onclose = function() {
        wsConnected = false;
        document.getElementById('conn-badge').textContent = '\\u25cb Poll';
        document.getElementById('conn-badge').style.color = '#ff6b6b';
        // Fallback to polling, retry WS after 3s
        if (!pollInterval) pollInterval = setInterval(poll, 200);
        setTimeout(connectWs, 3000);
    };
    ws.onerror = function() { ws.close(); };
}

// Initial fetch to populate immediately, then connect WS
poll();
connectWs();
</script>
</body>
</html>"""


async def handle_index(request):
    return web.Response(text=get_dashboard_html(), content_type="text/html")


async def handle_api_state(request):
    """Fallback REST endpoint for polling (kept for compatibility)."""
    with viz_state.lock:
        data = viz_state._snapshot()
    return web.json_response(data)


async def handle_ws_stream(request):
    """WebSocket endpoint — pushes state updates to browser in real time."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    queue = asyncio.Queue(maxsize=4)
    viz_state.subscribe(queue)
    logger.info("Browser WebSocket connected")

    try:
        while not ws.closed:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=5.0)
                await ws.send_json(data)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await ws.ping()
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        viz_state.unsubscribe(queue)
        logger.info("Browser WebSocket disconnected")

    return ws


async def handle_prompt_override(request):
    """POST /api/prompt — set or clear prompt override.

    JSON body:
      {"action": "set", "prompt": "new prompt text"}  — pause + set override
      {"action": "clear"}                             — clear override, resume
      {"action": "resume"}                            — resume with current override active
    """
    body = await request.json()
    action = body.get("action", "")

    with viz_state.lock:
        if action == "set":
            viz_state.prompt_override = body.get("prompt", "")
            viz_state.paused = False
            logger.info(f"Prompt override set: '{viz_state.prompt_override}'")
        elif action == "clear":
            viz_state.prompt_override = None
            viz_state.paused = False
            logger.info("Prompt override cleared")
        elif action == "pause":
            viz_state.paused = True
            logger.info("Proxy paused for prompt input")
        elif action == "resume":
            viz_state.paused = False
            logger.info("Proxy resumed")
        else:
            return web.json_response({"error": f"Unknown action: {action}"}, status=400)

    return web.json_response(
        {"ok": True, "prompt_override": viz_state.prompt_override, "paused": viz_state.paused}
    )


def start_web_dashboard(viz_port: int):
    """Start the aiohttp web dashboard in a separate thread."""
    from aiohttp import web as _web

    async def run_app():
        # Register this event loop so cross-thread notify works
        viz_state.set_dashboard_loop(asyncio.get_running_loop())

        app = _web.Application()
        app.router.add_get("/", handle_index)
        app.router.add_get("/api/state", handle_api_state)
        app.router.add_get("/ws", handle_ws_stream)
        app.router.add_post("/api/prompt", handle_prompt_override)
        runner = _web.AppRunner(app)
        await runner.setup()
        site = _web.TCPSite(runner, "0.0.0.0", viz_port)  # nosec B104
        await site.start()
        logger.info(f"Web dashboard available at http://localhost:{viz_port}")
        await asyncio.Event().wait()

    def thread_target():
        asyncio.run(run_app())

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Policy visualizer proxy — transparent bridge with a web-based debug dashboard"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for robot client to connect to (proxy port)"
    )
    parser.add_argument("--viz-port", type=int, default=8080, help="Port for web browser dashboard")
    parser.add_argument("--upstream-host", type=str, default="localhost", help="Upstream policy server host")
    parser.add_argument("--upstream-port", type=int, default=7010, help="Upstream policy server port")
    args = parser.parse_args()

    logger.info(
        f"Starting visualizer proxy: client->:{args.port} "
        f"-> upstream {args.upstream_host}:{args.upstream_port}"
    )
    logger.info(f"Web dashboard will be at http://localhost:{args.viz_port}")

    start_web_dashboard(args.viz_port)

    proxy = VizProxyServer(
        port=args.port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
    )
    proxy.serve_forever()


if __name__ == "__main__":
    try:
        from aiohttp import web  # noqa: F401
    except ImportError as err:
        print("ERROR: aiohttp is required. Install with: pip install aiohttp")
        raise SystemExit(1) from err

    main()

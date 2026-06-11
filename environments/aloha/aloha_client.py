#!/usr/bin/env python3
"""
Aloha robot client for the websocket-based policy server (serve_real.py).

Connects to the policy server via websocket (openpi_client), reads observations
from the real Aloha robot, sends them to the server, receives predicted action
chunks, and executes them on the robot.

Usage:
    # First start the server:
    python3 environments/aloha/serve_real.py --config_path environments/aloha/new_server.yaml

    # Then run this client:
    python3 environments/aloha/aloha_client.py \
        --server-ip <SERVER_IP> --port 7000 \
        --language "Push the red button with the right gripper."

    # Headless mode (immediate continuous inference):
    python3 environments/aloha/aloha_client.py \
        --server-ip <SERVER_IP> --port 7000 --headless \
        --language "Push the red button with the right gripper." \
        --timeout 60

Keyboard / pedal controls (non-headless mode):
    LEFT_PEDAL   – execute a test oscillation move
    MIDDLE_PEDAL – display camera feeds and joint positions
    RIGHT_PEDAL  – toggle continuous inference on/off
    ESC / q      – quit
"""

import argparse
import logging
import time

import cv2
import numpy as np
from interbotix_common_modules.common_robot.robot import robot_shutdown
from openpi_client import websocket_client_policy
from real_aloha_helpers import (
    LEFT_PEDAL,
    MIDDLE_PEDAL,
    RIGHT_PEDAL,
    bringup_robots,
    generate_oscillating_commands,
    opening_ceremony,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_FREQ = 50.0  # Hz
DT = 1.0 / CONTROL_FREQ

DEFAULT_SERVER_HOST = "10.209.224.254"
DEFAULT_SERVER_PORT = 7000

# Camera keys expected by the Aloha server (must match observation_mapping in
# new_server.yaml and AlohaServer.image_keys in aloha.py).
IMAGE_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# Joint names for display (14 joints: 7 left + 7 right)
JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def robot_obs_to_server_obs(
    observation: dict,
    language: str,
    convert_rgb_to_bgr: bool = True,
) -> dict:
    """Convert a real-robot observation into the dict the websocket server expects.

    The AlohaServer.process_input() on the server side assumes:
      - Images are BGR uint8 (H, W, 3) — it converts BGR→RGB internally.
      - ``joint_position`` is a float64 array of shape (1, 14).
      - ``task`` is a list of strings.

    If the robot environment returns RGB images (typical for modern drivers),
    set *convert_rgb_to_bgr=True* so the server's channel swap produces the
    correct RGB output for the policy.

    Args:
        observation: Dict from ``env.get_observation()`` with keys:
            ``qpos``  – (14,) joint positions
            ``images`` – dict of camera name → (H, W, 3) uint8
        language: Natural-language task instruction.
        convert_rgb_to_bgr: If True, swap R↔B channels before sending so
            the server's BGR→RGB conversion yields correct RGB.
    """
    obs: dict = {}

    # Joint state: float64, shape (1, 14)
    qpos = observation["qpos"]
    if isinstance(qpos, np.ndarray):
        qpos = qpos.astype(np.float64)
    obs["joint_position"] = qpos.reshape(1, -1)

    # Camera images
    images = observation.get("images", {})
    for key in IMAGE_KEYS:
        img = images.get(key)
        if img is None:
            logger.warning("Image key '%s' missing from observation – skipping.", key)
            continue
        if convert_rgb_to_bgr:
            img = img[:, :, ::-1].copy()  # RGB → BGR
        obs[key] = img  # uint8 (H, W, 3)

    # Language instruction
    obs["task"] = [language]

    return obs


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def move_chunk(commands: list, env) -> None:
    """Execute a list of joint-position commands on the robot."""
    for cmd in commands:
        env.step(cmd)
        time.sleep(DT)


def run_inference_and_execute(
    env,
    client: websocket_client_policy.WebsocketClientPolicy,
    language: str,
    action_size: int | None = None,
    convert_rgb_to_bgr: bool = True,
) -> None:
    """Get observation → send to server → receive action chunk → execute.

    Args:
        env: Aloha robot environment.
        client: Connected websocket client.
        language: Task instruction string.
        action_size: If set, execute only the first N actions from the chunk.
            Must be < chunk_size.  ``None`` means execute the full chunk.
        convert_rgb_to_bgr: Passed through to ``robot_obs_to_server_obs``.
    """
    logger.info("=" * 60)
    logger.info("RUNNING INFERENCE – prompt: %s", language)
    logger.info("=" * 60)

    # 1. Read observation from robot
    observation = env.get_observation()

    # 2. Convert to server format and send
    server_obs = robot_obs_to_server_obs(observation, language, convert_rgb_to_bgr)
    logger.debug("Sending observation to server …")
    result = client.infer(server_obs)

    # 3. Parse returned action chunk
    actions: np.ndarray = result["action"]  # (chunk_size, action_dim)
    chunk_size = len(actions)
    infer_ms = result.get("infer_ms", [None])[0]

    logger.info(
        "Received action chunk: shape=%s, infer_ms=%s",
        actions.shape,
        f"{infer_ms:.1f}" if infer_ms is not None else "N/A",
    )
    logger.debug("First action:  %s", actions[0])
    logger.debug("Action range:  [%.4f, %.4f]", actions.min(), actions.max())

    # 4. Determine how many actions to execute
    if action_size is not None:
        if action_size <= 0:
            logger.warning("action_size=%d <= 0; executing full chunk.", action_size)
            exec_actions = actions
        elif action_size >= chunk_size:
            adjusted = max(chunk_size - 1, 1)
            logger.warning(
                "action_size (%d) >= chunk_size (%d); adjusting to %d.",
                action_size,
                chunk_size,
                adjusted,
            )
            exec_actions = actions[:adjusted]
        else:
            exec_actions = actions[:action_size]
    else:
        exec_actions = actions

    # 5. Execute on robot
    logger.info("Executing %d / %d actions …", len(exec_actions), chunk_size)
    for i, action in enumerate(exec_actions):
        env.step(action)
        time.sleep(DT)
        if (i + 1) % 10 == 0:
            logger.debug("  Executed %d / %d actions", i + 1, len(exec_actions))

    logger.info("Action chunk executed.")
    logger.info("=" * 60)


def run_inference_rtc(
    env,
    client: websocket_client_policy.WebsocketClientPolicy,
    language: str,
    action_size: int | None = None,
    inference_delay: int = 3,
    convert_rgb_to_bgr: bool = True,
) -> None:
    """Continuous Real-Time Control (RTC) inference loop.

    Sends remaining (un-executed) actions back to the server as ``action``
    so the policy can blend the new prediction with the old one.

    The loop keeps running until the caller breaks out of it.
    """
    current_chunk: np.ndarray | None = None
    chunk_idx = 0
    chunk_size: int | None = None

    step = 0
    while True:
        observation = env.get_observation()
        server_obs = robot_obs_to_server_obs(observation, language, convert_rgb_to_bgr)

        # Determine whether we need a fresh prediction
        if chunk_size is None:
            steps_remaining = 0
        else:
            steps_remaining = (chunk_size - chunk_idx) if current_chunk is not None else 0
        need_new = current_chunk is None or steps_remaining <= inference_delay * 2

        if need_new:
            remaining_actions = None
            if current_chunk is not None and steps_remaining > 0:
                remaining_actions = current_chunk[chunk_idx:]
                server_obs["action"] = remaining_actions
                logger.debug(
                    "Step %d: passing %d remaining actions for RTC blending.",
                    step,
                    steps_remaining,
                )

            result = client.infer(server_obs)
            new_chunk = result["action"]
            if chunk_size is None:
                chunk_size = new_chunk.shape[0]

            if remaining_actions is not None:
                current_chunk = np.concatenate((remaining_actions, new_chunk[steps_remaining:]), axis=0)
            else:
                current_chunk = new_chunk
            chunk_idx = 0
            logger.debug("Step %d: new chunk from server (size=%d).", step, chunk_size)

        # Execute one action
        assert current_chunk is not None
        action = current_chunk[chunk_idx]
        env.step(action)
        chunk_idx += 1
        step += 1
        time.sleep(DT)

        if step % 50 == 0:
            logger.info("RTC step %d", step)

        # Yield to allow KeyboardInterrupt / cv2 key checks
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            logger.info("RTC loop interrupted by user.")
            break


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_image_windows_created: set[str] = set()
_qpos_window_name = "qpos_values"


def view_observation(env) -> dict:
    """Display camera feeds and joint positions in OpenCV windows."""
    observation = env.get_observation()
    images = observation.get("images", {})
    qpos = observation.get("qpos")

    for key in IMAGE_KEYS:
        img = images.get(key)
        if img is None:
            logger.warning("Image '%s' is None – skipping display.", key)
            continue
        if key not in _image_windows_created:
            cv2.namedWindow(key, cv2.WINDOW_NORMAL)
            _image_windows_created.add(key)
        # Ensure BGR for OpenCV display (robot may return RGB)
        display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(key, display)

    # Joint values overlay
    if qpos is not None and len(qpos) >= 14:
        if _qpos_window_name not in _image_windows_created:
            cv2.namedWindow(_qpos_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_qpos_window_name, 640, 480)
            _image_windows_created.add(_qpos_window_name)
        text_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(text_img, "qpos (14):", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        left, right = qpos[:7], qpos[7:14]
        for i, (lv, rv) in enumerate(zip(left, right, strict=False)):
            y = 100 + i * 50
            cv2.putText(
                text_img, f"{i:02d}: {lv: .3f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
            cv2.putText(
                text_img, f"{i + 7:02d}: {rv: .3f}", (340, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )
        cv2.imshow(_qpos_window_name, text_img)

    return observation


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------


def control_loop(
    env,
    client: websocket_client_policy.WebsocketClientPolicy,
    language: str,
    action_size: int | None = None,
    convert_rgb_to_bgr: bool = True,
    max_runtime_s: float | None = None,
    eval_mode: str = "standard",
    inference_delay: int = 3,
) -> None:
    """Interactive control loop with pedal / keyboard input.

    States
    ------
    ACTIVE          – idle, waiting for user input
    TEST_MOVE       – execute a small oscillatory test motion
    VIEW_OBSERVATION – snapshot camera feeds / joint values
    RUN_INFERENCE   – continuously query the server and execute actions
    """
    commands_l = generate_oscillating_commands(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.02239], num_steps=50, step_size=0.005
    )
    commands_r = generate_oscillating_commands(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.02239], num_steps=50, step_size=-0.005
    )
    test_commands = [cl + cr for cl, cr in zip(commands_l, commands_r, strict=False)]

    state = "ACTIVE"
    start_time = time.time()

    while True:
        # Timeout check
        if max_runtime_s is not None and (time.time() - start_time) >= max_runtime_s:
            logger.info("Max runtime reached (%.1f s). Exiting.", max_runtime_s)
            break

        key = cv2.waitKey(int(DT * 1000)) & 0xFF
        if key == 27 or key == ord("q"):
            logger.info("Exiting control loop.")
            break
        elif key == LEFT_PEDAL:
            state = "TEST_MOVE"
        elif key == MIDDLE_PEDAL:
            state = "VIEW_OBSERVATION"
        elif key == RIGHT_PEDAL:
            state = "RUN_INFERENCE" if state != "RUN_INFERENCE" else "ACTIVE"
            logger.info("State → %s", state)

        if state == "TEST_MOVE":
            logger.info("Executing test move …")
            move_chunk(test_commands, env)
            state = "ACTIVE"
        elif state == "VIEW_OBSERVATION":
            view_observation(env)
            state = "ACTIVE"
        elif state == "RUN_INFERENCE":
            try:
                if eval_mode == "rtc":
                    run_inference_rtc(
                        env,
                        client,
                        language,
                        action_size=action_size,
                        inference_delay=inference_delay,
                        convert_rgb_to_bgr=convert_rgb_to_bgr,
                    )
                    # After RTC loop exits (user pressed ESC inside), go back to ACTIVE
                    state = "ACTIVE"
                else:
                    run_inference_and_execute(
                        env,
                        client,
                        language,
                        action_size=action_size,
                        convert_rgb_to_bgr=convert_rgb_to_bgr,
                    )
            except Exception:
                logger.exception("Inference error – returning to ACTIVE.")
                state = "ACTIVE"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aloha websocket inference client (compatible with serve_real.py)",
        add_help=False,
    )
    parser.add_argument("--help", action="help", help="Show this help message and exit.")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="Push the red button with the right gripper.",
        help="Natural-language task instruction for the policy.",
    )
    parser.add_argument(
        "--headless",
        "-H",
        action="store_true",
        help="Run headless (no OpenCV windows). Immediately starts continuous inference.",
    )
    parser.add_argument(
        "--action-size",
        "-a",
        type=int,
        default=None,
        help="Execute only the first N actions from each predicted chunk (must be < chunk size).",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="End the programme after this many seconds. If omitted, run indefinitely.",
    )
    parser.add_argument(
        "--server-ip",
        "-s",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Policy server IP address (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Policy server websocket port (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="standard",
        choices=["standard", "rtc"],
        help="Evaluation mode: 'standard' (chunk-at-a-time) or 'rtc' (real-time control).",
    )
    parser.add_argument(
        "--inference-delay",
        type=int,
        default=3,
        help="Steps the policy inference takes (used in RTC mode only).",
    )
    parser.add_argument(
        "--no-bgr-convert",
        action="store_true",
        help=(
            "Do NOT convert robot RGB images to BGR before sending. "
            "By default the client converts RGB→BGR because AlohaServer.process_input() "
            "assumes BGR input. Set this flag if your robot already outputs BGR."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    convert_rgb_to_bgr = not args.no_bgr_convert

    # ---- Robot bring-up ----
    logger.info("Bringing up Aloha robots …")
    node, env = bringup_robots()

    try:
        opening_ceremony(env.follower_bot_left, env.follower_bot_right)

        # ---- Connect to policy server via websocket ----
        logger.info("Connecting to policy server at %s:%d …", args.server_ip, args.port)
        client = websocket_client_policy.WebsocketClientPolicy(host=args.server_ip, port=args.port)
        logger.info("Connected to policy server!")

        if args.headless:
            # ----------------------------------------------------------
            # Headless: continuous inference until timeout / Ctrl-C
            # ----------------------------------------------------------
            start_time = time.time()
            try:
                if args.eval_mode == "rtc":
                    run_inference_rtc(
                        env,
                        client,
                        args.language,
                        action_size=args.action_size,
                        inference_delay=args.inference_delay,
                        convert_rgb_to_bgr=convert_rgb_to_bgr,
                    )
                else:
                    while True:
                        run_inference_and_execute(
                            env,
                            client,
                            args.language,
                            action_size=args.action_size,
                            convert_rgb_to_bgr=convert_rgb_to_bgr,
                        )
                        if args.timeout is not None and (time.time() - start_time) >= args.timeout:
                            logger.info("Timeout reached (%.1f s). Exiting.", args.timeout)
                            break
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
        else:
            # ----------------------------------------------------------
            # Interactive: pedal / keyboard control loop
            # ----------------------------------------------------------
            cv2.namedWindow("Control Window", cv2.WINDOW_NORMAL)
            ctrl_img = np.zeros((120, 400, 3), dtype=np.uint8)
            cv2.putText(
                ctrl_img, "Pedals / ESC to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            cv2.putText(
                ctrl_img,
                f"Lang: {args.language[:38]}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                ctrl_img,
                f"Mode: {args.eval_mode}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            if args.action_size is not None:
                cv2.putText(
                    ctrl_img,
                    f"ActSz: {args.action_size}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                )
            cv2.imshow("Control Window", ctrl_img)

            control_loop(
                env,
                client,
                language=args.language,
                action_size=args.action_size,
                convert_rgb_to_bgr=convert_rgb_to_bgr,
                max_runtime_s=args.timeout,
                eval_mode=args.eval_mode,
                inference_delay=args.inference_delay,
            )
            cv2.destroyAllWindows()

    finally:
        try:
            robot_shutdown(node)
        except Exception:
            logger.exception("Robot shutdown error.")

    logger.info("Exiting.")


if __name__ == "__main__":
    main()

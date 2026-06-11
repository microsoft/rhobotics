#!/usr/bin/env python3
"""
LIBERO simulation client for websocket-based policy evaluation.

Connects to a policy server (serve_libero.py), runs episodes in the LIBERO
simulator, sends observations, receives action chunks, and reports metrics.

Usage:
    # First start the server in a separate terminal:
    python3 environments/libero/serve_libero.py \
        --config_path environments/libero/configs/serve_libero_phi4mm.yaml

    # Then run this client:
    python3 environments/libero/libero_client.py \
        --host localhost --port 7000 \
        --task_suite libero_spatial \
        --episodes_per_task 10 \
        --max_steps 300

    # With video recording:
    python3 environments/libero/libero_client.py \
        --host localhost --port 7000 \
        --task_suite libero_spatial \
        --episodes_per_task 10 \
        --record_video --output_dir outputs/eval_libero_phi4mm
"""

import argparse
import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from rho_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
RESOLUTION = 256
NUM_WAIT_STEPS = 10  # Settling steps after reset
DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # 6 DoF + gripper open

# Task suite to recommended max_episode_steps
SUITE_MAX_STEPS = {
    "libero_spatial": 300,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def _rotate_image_180(img: np.ndarray) -> np.ndarray:
    """Rotate image 180 degrees (matches LIBERO training preprocessing)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def libero_obs_to_server_obs(raw_obs: dict, task_description: str) -> dict:
    """Convert raw LIBERO observation into the dict the websocket server expects.

    The server's LiberoServer.process_input() expects:
        - Images are RGB uint8 (H, W, 3), already rotated 180°
        - "state" is float64 array of shape (1, 8)
        - "task" is a list of strings

    Args:
        raw_obs: Raw observation dict from LIBERO environment.
        task_description: Natural-language task instruction.

    Returns:
        Dict ready to send via websocket_client_policy.infer().
    """
    # Rotate images 180° (matches training data preprocessing)
    agentview = _rotate_image_180(raw_obs["agentview_image"])
    wrist = _rotate_image_180(raw_obs["robot0_eye_in_hand_image"])

    # Build state vector: (eef_pos[3], axisangle[3], gripper_qpos[2]) = 8
    state = np.concatenate(
        (
            raw_obs["robot0_eef_pos"],
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        )
    ).astype(np.float64)

    return {
        "agentview": agentview,  # uint8 (H, W, 3)
        "wrist": wrist,  # uint8 (H, W, 3)
        "state": state.reshape(1, -1),  # float64 (1, 8)
        "task": [task_description],
    }


def save_video(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    """Save a list of RGB frames as an MP4 video."""
    try:
        import imageio

        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.info(f"Saved video: {path}")
    except ImportError:
        logger.warning(
            "imageio not installed – skipping video save. Install with: pip install imageio[ffmpeg]"
        )


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
def create_libero_env(task_suite_name: str, task_id: int, resolution: int = RESOLUTION, seed: int = 42):
    """Create a single LIBERO environment for a specific task.

    Args:
        task_suite_name: One of libero_spatial, libero_object, libero_goal, libero_10, libero_90.
        task_id: Task index within the suite.
        resolution: Image resolution (default 256).
        seed: Random seed.

    Returns:
        Tuple of (env, task_description, initial_states).
    """
    import torch
    from libero.libero import benchmark, get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_description = task.language

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": CAMERA_NAMES,
    }

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    # Load initial states
    init_states_folder = get_libero_path("init_states")
    init_states_path = os.path.join(init_states_folder, task.problem_folder, task.init_states_file)

    if os.path.exists(init_states_path):
        initial_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    else:
        raise FileNotFoundError(f"Initial states not found: {init_states_path}")

    return env, task_description, initial_states, task_suite


def reset_env(env, initial_states, init_state_id: int = 0) -> dict:
    """Reset the environment and return the settled observation.

    Applies initial state and runs settling steps with dummy actions.

    Returns:
        Raw observation dict from the LIBERO environment.
    """
    env.reset()
    raw_obs = env.set_init_state(initial_states[init_state_id])

    # Let objects settle
    for _ in range(NUM_WAIT_STEPS):
        raw_obs, _, _, _ = env.step(DUMMY_ACTION)

    return raw_obs


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate_task(
    env,
    client,
    task_description: str,
    initial_states,
    num_episodes: int,
    max_steps: int,
    n_action_steps: int | None = None,
    record_video: bool = False,
    output_dir: str | None = None,
    task_id: int = 0,
) -> dict:
    """Run evaluation episodes for a single LIBERO task.

    Args:
        env: LIBERO OffScreenRenderEnv instance.
        client: Connected WebsocketClientPolicy instance.
        task_description: Task language instruction.
        initial_states: Tensor of initial states for reproducible resets.
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        n_action_steps: Number of actions to execute per chunk (default: use server metadata).
        record_video: Whether to record videos of episodes.
        output_dir: Directory for saving videos.
        task_id: Task ID (for logging and filenames).

    Returns:
        Dict with episode_rewards, episode_successes, episode_steps.
    """
    episode_rewards = []
    episode_successes = []
    episode_steps = []

    for ep in range(num_episodes):
        # Reset with different initial states for variety
        init_state_id = ep % len(initial_states)
        raw_obs = reset_env(env, initial_states, init_state_id)

        total_reward = 0.0
        success = False
        frames = [] if record_video else None
        step_count = 0

        logger.info(
            f"  Episode {ep + 1}/{num_episodes} (task {task_id}, "
            f'init_state {init_state_id}) – "{task_description}"'
        )

        while step_count < max_steps:
            # Send observation to server
            server_obs = libero_obs_to_server_obs(raw_obs, task_description)
            t0 = time.monotonic()
            result = client.infer(server_obs)
            infer_ms = (time.monotonic() - t0) * 1000

            # Parse action chunk
            actions = result["action"]  # (chunk_size, action_dim)
            server_infer_ms = result.get("infer_ms", [None])[0]

            if step_count == 0:
                logger.info(
                    f"    Action chunk shape: {actions.shape}, "
                    f"round-trip: {infer_ms:.0f}ms, "
                    f"server infer: {server_infer_ms:.0f}ms"
                    if server_infer_ms
                    else f"    Action chunk shape: {actions.shape}, round-trip: {infer_ms:.0f}ms"
                )

            # Determine how many actions to execute
            exec_steps = n_action_steps if n_action_steps is not None else len(actions)
            exec_steps = min(exec_steps, len(actions), max_steps - step_count)

            # Execute action steps
            for i in range(exec_steps):
                action = actions[i]
                raw_obs, reward, done, info = env.step(action)
                total_reward += reward
                step_count += 1

                # Record frame after each step
                if record_video:
                    frame = _rotate_image_180(raw_obs["agentview_image"])
                    frames.append(frame.copy())

                if done:
                    success = True
                    break

            if success:
                break

        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_steps.append(step_count)

        status = "SUCCESS" if success else "FAIL"
        logger.info(f"    [{status}] reward={total_reward:.2f}, steps={step_count}/{max_steps}")

        # Save video
        if record_video and output_dir and frames:
            video_dir = Path(output_dir) / f"task_{task_id:02d}"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(video_dir / f"ep_{ep:03d}_{status.lower()}.mp4")
            save_video(frames, video_path)

    return {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
    }


def evaluate_suite(
    client,
    task_suite_name: str,
    num_episodes: int,
    max_steps: int,
    n_action_steps: int | None = None,
    record_video: bool = False,
    output_dir: str | None = None,
    seed: int = 42,
    task_ids: list[int] | None = None,
) -> dict:
    """Evaluate across all tasks (or a subset) in a LIBERO suite.

    Args:
        client: Connected WebsocketClientPolicy instance.
        task_suite_name: Name of the LIBERO task suite.
        num_episodes: Number of episodes per task.
        max_steps: Maximum steps per episode.
        n_action_steps: Actions per chunk to execute.
        record_video: Whether to record videos.
        output_dir: Directory for outputs.
        seed: Random seed.
        task_ids: Optional subset of task IDs to evaluate. None = all tasks.

    Returns:
        Dict with per-task and aggregate metrics.
    """
    import torch
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks

    if task_ids is None:
        task_ids = list(range(num_tasks))

    logger.info(
        f"Evaluating {len(task_ids)} tasks from '{task_suite_name}', "
        f"{num_episodes} episodes each, max {max_steps} steps"
    )

    all_results = {}
    per_task_success = {}

    for task_id in task_ids:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Task {task_id + 1}/{num_tasks}")

        env, task_description, initial_states, _ = create_libero_env(task_suite_name, task_id, seed=seed)

        logger.info(f'  Description: "{task_description}"')

        task_results = evaluate_task(
            env=env,
            client=client,
            task_description=task_description,
            initial_states=initial_states,
            num_episodes=num_episodes,
            max_steps=max_steps,
            n_action_steps=n_action_steps,
            record_video=record_video,
            output_dir=output_dir,
            task_id=task_id,
        )

        task_sr = np.mean(task_results["episode_successes"])
        per_task_success[task_id] = task_sr
        all_results[task_id] = task_results

        logger.info(f"  Task {task_id} success rate: {task_sr:.1%}")

        # Clean up task environment
        try:
            env.close()
        except Exception as e:
            logger.warning("Error while closing environment for task %s: %s", task_id, e, exc_info=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate metrics
    all_successes = [s for r in all_results.values() for s in r["episode_successes"]]
    all_rewards = [r for res in all_results.values() for r in res["episode_rewards"]]
    all_steps = [s for r in all_results.values() for s in r["episode_steps"]]

    summary = {
        "task_suite": task_suite_name,
        "num_tasks": len(task_ids),
        "episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "overall_success_rate": float(np.mean(all_successes)),
        "mean_reward": float(np.mean(all_rewards)),
        "mean_steps": float(np.mean(all_steps)),
        "per_task_success_rate": {str(k): float(v) for k, v in per_task_success.items()},
        "per_task_results": {
            str(k): {
                "successes": v["episode_successes"],
                "rewards": v["episode_rewards"],
                "steps": v["episode_steps"],
            }
            for k, v in all_results.items()
        },
    }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LIBERO simulation client for websocket policy evaluation")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server hostname or IP")
    parser.add_argument("--port", type=int, default=7000, help="Policy server port")
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        choices=list(SUITE_MAX_STEPS.keys()),
        help="LIBERO task suite to evaluate",
    )
    parser.add_argument(
        "--task_ids", type=int, nargs="*", default=None, help="Specific task IDs to evaluate (default: all)"
    )
    parser.add_argument("--episodes_per_task", type=int, default=10, help="Number of episodes per task")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max steps per episode (default: suite-specific)"
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Number of actions to execute per chunk (default: from server metadata or full chunk)",
    )
    parser.add_argument("--record_video", action="store_true", help="Record evaluation videos")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory for outputs (videos, results JSON)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Max steps default
    if args.max_steps is None:
        args.max_steps = SUITE_MAX_STEPS.get(args.task_suite, 300)
        logger.info(f"Using default max_steps={args.max_steps} for {args.task_suite}")

    # Output directory
    if args.output_dir is None and args.record_video:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.output_dir = f"outputs/eval_libero_client/{args.task_suite}/{timestamp}"
        logger.info(f"Auto output_dir: {args.output_dir}")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Connect to server
    logger.info(f"Connecting to policy server at {args.host}:{args.port} ...")

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    logger.info(f"Connected. Server metadata: {metadata}")

    # Use server's chunk_size to determine n_action_steps if not specified
    if args.n_action_steps is None and "chunk_size" in metadata:
        # Default to half the chunk size (common receding-horizon pattern)
        args.n_action_steps = metadata["chunk_size"]
        logger.info(f"Using n_action_steps={args.n_action_steps} from server chunk_size")

    # Run evaluation
    logger.info("=" * 60)
    logger.info(f"LIBERO Evaluation: {args.task_suite}")
    logger.info(f"  Episodes per task: {args.episodes_per_task}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Action steps per chunk: {args.n_action_steps or 'full chunk'}")
    logger.info("=" * 60)

    summary = evaluate_suite(
        client=client,
        task_suite_name=args.task_suite,
        num_episodes=args.episodes_per_task,
        max_steps=args.max_steps,
        n_action_steps=args.n_action_steps,
        record_video=args.record_video,
        output_dir=args.output_dir,
        seed=args.seed,
        task_ids=args.task_ids,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Suite: {summary['task_suite']}")
    logger.info(f"  Tasks evaluated: {summary['num_tasks']}")
    logger.info(f"  Episodes per task: {summary['episodes_per_task']}")
    logger.info(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
    logger.info(f"  Mean reward: {summary['mean_reward']:.3f}")
    logger.info(f"  Mean steps: {summary['mean_steps']:.1f}")
    logger.info("")
    logger.info("Per-task success rates:")
    for task_id_str, sr in summary["per_task_success_rate"].items():
        logger.info(f"  Task {task_id_str}: {sr:.1%}")

    # Save results JSON
    if args.output_dir:
        results_path = Path(args.output_dir) / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()

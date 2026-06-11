"""
Dataset client for testing a policy server without a real robot.

Loads a LeRobot dataset, sends observations to a running websocket server,
collects predicted actions, and plots them against ground truth actions.

Robot-specific settings (observation mapping, GT action key, joint names)
are loaded from a YAML config file.

Usage:
    # First start the server, e.g.:
    python3 environments/aloha/serve_real.py --config_path environments/aloha/new_server.yaml

    # Then run this client with a robot config:
    python3 environments/dataset_client.py \
        --config environments/aloha/dataset_client.yaml \
        --dataset_root /datadrive/datasets/aloha-busybox_lerobot_v3 \
        --port 7000 --num_samples 100

    # Or for UR5e:
    python3 environments/dataset_client.py \
        --config environments/ur5e/dataset_client.yaml \
        --dataset_root /data/simran/bimanual_plug_0112/ \
        --port 6000 --num_samples 100
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rho.utils import init_logging
from rho_client import websocket_client_policy

logger = logging.getLogger(__name__)


def load_client_config(config_path: str) -> dict:
    """Load a dataset-client YAML config file.

    Expected keys:
        obs_mapping:  {server_key: dataset_key, ...}
        gt_action_key: str
        joint_names: list[str]  (optional, for plot labels)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "obs_mapping" not in cfg:
        raise ValueError(f"Config {config_path} must contain 'obs_mapping'")
    return cfg


def parse_obs_mapping(mapping_str: str) -> dict:
    """Parse a comma-separated observation mapping string into a dict.

    Format: "server_key=dataset_key,server_key2=dataset_key2,..."
    """
    mapping = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(f"Invalid mapping pair (missing '='): '{pair}'")
        server_key, dataset_key = pair.split("=", 1)
        mapping[server_key.strip()] = dataset_key.strip()
    return mapping


def sample_to_server_obs(sample: dict, obs_mapping: dict) -> dict:
    """Convert a dataset sample into the dict format expected by the websocket server.

    Uses *obs_mapping* (server_key -> dataset_key) to rename and select the
    correct fields from the dataset sample. Values are converted from
    torch tensors to numpy arrays with the shapes/dtypes the server expects.
    """
    obs: dict = {}
    for server_key, dataset_key in obs_mapping.items():
        if dataset_key not in sample:
            continue

        value = sample[dataset_key]

        if isinstance(value, torch.Tensor):
            arr = value.cpu().numpy()
            # Image tensors: ensure (H, W, C) uint8
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # (C, H, W)
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 4 and arr.shape[1] in (1, 3):  # (B, C, H, W)
                arr = np.transpose(arr.squeeze(0), (1, 2, 0))
            # Convert float images to uint8 [0,255]
            if arr.ndim == 3 and arr.dtype in (np.float32, np.float64):
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            # State / force vectors: float64, shape (1, dim)
            if arr.ndim <= 2:
                arr = arr.astype(np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
            obs[server_key] = arr
        elif isinstance(value, str):
            obs[server_key] = [value]
        elif isinstance(value, list):
            obs[server_key] = value
        else:
            obs[server_key] = value
    return obs


def run_inference_loop(
    client,
    dataset,
    indices: list[int],
    obs_mapping: dict,
    gt_action_key: str,
    eval_mode: str,
    inference_delay: int,
):
    """Run inference over *indices* and return (gt_actions, pred_actions, task_instruction).

    Returns:
        gt_actions: list of tensors, each (1, action_dim) or (1, chunk, action_dim)
        pred_actions: list of tensors, each (1, action_dim)
        task_instruction: str or None
    """
    all_gt_actions: list[torch.Tensor] = []
    all_pred_actions: list[torch.Tensor] = []
    task_instruction = None
    num_samples = len(indices)

    current_chunk = None
    chunk_idx = 0
    chunk_size = None  # determined from first server response

    for step, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]

        gt_actions = sample.get(gt_action_key)
        obs = sample_to_server_obs(sample, obs_mapping)

        if task_instruction is None and "task" in sample:
            task_instruction = sample["task"]
            logger.info(f"Task instruction: {task_instruction}")

        if gt_actions is not None and isinstance(gt_actions, torch.Tensor):
            all_gt_actions.append(gt_actions.unsqueeze(0).cpu())

        if eval_mode == "rtc":
            # Decide whether to request a new chunk
            if chunk_size is None:
                steps_remaining = 0
            else:
                steps_remaining = chunk_size - chunk_idx if current_chunk is not None else 0
            need_new_chunk = current_chunk is None or steps_remaining <= inference_delay * 2

            if need_new_chunk:
                remaining_actions = None
                if current_chunk is not None and steps_remaining > 0:
                    remaining_actions = current_chunk[chunk_idx:]
                    obs["action"] = remaining_actions
                    logger.debug(
                        f"  Step {step}: Passing {steps_remaining} remaining actions for RTC blending"
                    )

                result = client.infer(obs)
                new_chunk = result["action"]
                if chunk_size is None:
                    chunk_size = new_chunk.shape[0]

                if remaining_actions is not None:
                    current_chunk = np.concatenate(
                        (remaining_actions, new_chunk[steps_remaining:]),
                        axis=0,
                    )
                else:
                    current_chunk = new_chunk
                chunk_idx = 0
                logger.debug(f"  Step {step}: Predicted new chunk")

        else:
            # Standard mode: infer when chunk exhausted
            if current_chunk is None or chunk_idx >= len(current_chunk):
                result = client.infer(obs)
                current_chunk = result["action"]
                chunk_idx = 0
                logger.debug(f"  Step {step}: Predicted new chunk (size={len(current_chunk)})")

        if current_chunk is not None:
            action = current_chunk[chunk_idx]
            all_pred_actions.append(torch.from_numpy(action).unsqueeze(0))
            chunk_idx += 1

        if (step + 1) % max(1, num_samples // 10) == 0:
            logger.info(f"  Processed {step + 1}/{num_samples} samples")

    return all_gt_actions, all_pred_actions, task_instruction


def plot_results(
    gt_actions_plot: torch.Tensor,
    pred_actions_stacked: torch.Tensor,
    dim_labels: list[str],
    output_dir: Path,
    episode: int,
    eval_mode: str,
    task_instruction: str | None,
):
    """Generate time-series and scatter plots, save metrics."""
    action_dim = gt_actions_plot.shape[-1]

    # Per-dimension MSE
    mse_per_dim = ((gt_actions_plot - pred_actions_stacked) ** 2).mean(dim=0)
    total_mse = mse_per_dim.mean().item()
    logger.info(f"Total MSE: {total_mse:.6f}")
    for d in range(action_dim):
        logger.info(f"  {dim_labels[d]} MSE: {mse_per_dim[d].item():.6f}")

    task_line = f"\nTask: {task_instruction}" if task_instruction else ""

    # ---- Time-series plot ----
    num_plot_dims = min(action_dim, 14)
    fig, axes = plt.subplots(
        num_plot_dims,
        1,
        figsize=(14, 2.5 * num_plot_dims),
        squeeze=False,
    )
    for d in range(num_plot_dims):
        ax = axes[d, 0]
        ax.plot(
            gt_actions_plot[:, d].numpy(),
            label="Ground Truth",
            alpha=0.8,
            linewidth=2,
        )
        ax.plot(
            pred_actions_stacked[:, d].numpy(),
            label="Predicted",
            alpha=0.8,
            linewidth=2,
        )
        ax.set_ylabel(dim_labels[d], fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(
            f"{dim_labels[d]} (MSE: {mse_per_dim[d].item():.4f})",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frame Index")
    plt.suptitle(
        f"Predicted vs Ground Truth Actions (Episode {episode})\n"
        f"Total MSE: {total_mse:.6f}  |  Mode: {eval_mode}"
        f"{task_line}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / f"actions_ep{episode}_{eval_mode}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Time-series plot saved to: {plot_path}")
    plt.close()

    # ---- Scatter plot ----
    num_cols = 4
    num_rows = (num_plot_dims + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(16, 4 * num_rows),
        squeeze=False,
    )
    for d in range(num_plot_dims):
        row, col = d // num_cols, d % num_cols
        ax = axes[row, col]
        ax.scatter(
            gt_actions_plot[:, d].numpy(),
            pred_actions_stacked[:, d].numpy(),
            alpha=0.5,
            s=20,
        )
        lims = [
            min(
                gt_actions_plot[:, d].min().item(),
                pred_actions_stacked[:, d].min().item(),
            ),
            max(
                gt_actions_plot[:, d].max().item(),
                pred_actions_stacked[:, d].max().item(),
            ),
        ]
        ax.plot(lims, lims, "r--", alpha=0.8, label="Perfect")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(dim_labels[d], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for d in range(num_plot_dims, num_rows * num_cols):
        row, col = d // num_cols, d % num_cols
        axes[row, col].set_visible(False)

    plt.suptitle(
        f"GT vs Predicted Scatter (Episode {episode}){task_line}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    scatter_path = output_dir / f"scatter_ep{episode}_{eval_mode}.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    logger.info(f"Scatter plot saved to: {scatter_path}")
    plt.close()

    # ---- Metrics file ----
    metrics_path = output_dir / f"metrics_ep{episode}_{eval_mode}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Total MSE: {total_mse:.6f}\n")
        f.write("Per-dimension MSE:\n")
        for d in range(action_dim):
            f.write(f"  {dim_labels[d]}: {mse_per_dim[d].item():.6f}\n")
    logger.info(f"Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a policy server against a ground truth dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a dataset_client YAML config (e.g. environments/aloha/dataset_client.yaml)",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory for the LeRobot dataset",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: entire episode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dataset_client_test",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="standard",
        choices=["standard", "rtc"],
    )
    parser.add_argument("--inference_delay", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument(
        "--gt_action_key",
        type=str,
        default=None,
        help="Override gt_action_key from the YAML config",
    )
    parser.add_argument(
        "--obs_mapping",
        type=str,
        default=None,
        help="Override obs_mapping as comma-separated server_key=dataset_key pairs",
    )
    args = parser.parse_args()

    init_logging(console_level=args.log_level)

    # Load robot-specific config
    cfg = load_client_config(args.config)

    # Resolve obs_mapping (CLI override wins)
    if args.obs_mapping is not None:
        obs_mapping = parse_obs_mapping(args.obs_mapping)
    else:
        obs_mapping = dict(cfg["obs_mapping"])

    # Resolve gt_action_key (CLI override wins)
    gt_action_key = args.gt_action_key or cfg.get("gt_action_key", "action.joint_position")

    # Optional joint names for plot labels
    joint_names: list[str] | None = cfg.get("joint_names")

    logger.info("=" * 80)
    logger.info("Dataset Client – Server Inference Test")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info("Observation mapping (server_key -> dataset_key):")
    for sk, dk in obs_mapping.items():
        logger.info(f"  {sk}  ->  {dk}")
    logger.info(f"GT action key: {gt_action_key}")

    # Connect to server
    logger.info(f"Connecting to server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info("Connected!")

    # Load dataset
    logger.info(f"Loading dataset from: {args.dataset_root}")
    dataset = LeRobotDataset(
        repo_id="test",
        root=args.dataset_root,
        video_backend="pyav",
    )
    logger.info(f"Dataset loaded: {len(dataset)} total samples")

    # Select episode indices
    logger.info(f"Filtering for episode {args.episode}...")
    ep_col = dataset.hf_dataset["episode_index"]
    episode_indices = [i for i, ep in enumerate(ep_col) if ep == args.episode]
    logger.info(f"Episode {args.episode}: {len(episode_indices)} frames")

    if not episode_indices:
        logger.error(f"Episode {args.episode} not found in dataset!")
        return

    if args.num_samples is not None:
        episode_indices = episode_indices[: args.num_samples]
    num_samples = len(episode_indices)
    logger.info(f"Evaluating {num_samples} samples ...")

    # Run inference
    all_gt, all_pred, task_instruction = run_inference_loop(
        client=client,
        dataset=dataset,
        indices=episode_indices,
        obs_mapping=obs_mapping,
        gt_action_key=gt_action_key,
        eval_mode=args.eval_mode,
        inference_delay=args.inference_delay,
    )

    if not all_gt or not all_pred:
        logger.warning("No actions collected. Check dataset and server config.")
        return

    gt_stacked = torch.cat(all_gt, dim=0)
    pred_stacked = torch.cat(all_pred, dim=0)
    gt_plot = gt_stacked[:, 0, :] if gt_stacked.ndim == 3 else gt_stacked

    action_dim = gt_plot.shape[-1]
    logger.info(f"GT shape: {gt_plot.shape}, Pred shape: {pred_stacked.shape}")

    # Build dimension labels
    if joint_names and len(joint_names) == action_dim:
        dim_labels = joint_names
    else:
        dim_labels = [f"Dim {d}" for d in range(action_dim)]

    plot_results(
        gt_actions_plot=gt_plot,
        pred_actions_stacked=pred_stacked,
        dim_labels=dim_labels,
        output_dir=output_dir,
        episode=args.episode,
        eval_mode=args.eval_mode,
        task_instruction=task_instruction,
    )

    logger.info("=" * 80)
    logger.info("Dataset client test completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

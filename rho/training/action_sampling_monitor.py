import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ActionSamplingMonitor:
    """
    Monitor training progress by sampling actions and comparing with ground truth.
    This helps track training progress and detect convergence/overfitting.
    """

    def __init__(self, policy, monitor_interval: int = 1000, num_samples: int = 16, device: str = "cuda"):
        self.policy = policy
        self.monitor_interval = monitor_interval
        self.num_samples = num_samples
        self.device = device

    def should_monitor(self, step: int) -> bool:
        """Check if we should run monitoring at this step."""
        return step % self.monitor_interval == 0

    @torch.no_grad()
    def monitor_training_progress(
        self, training_dataloader, step: int, wandb_logger=None
    ) -> dict[str, float]:
        """
        Sample actions and compare with ground truth to monitor training progress.

        Args:
            training_dataloader: DataLoader for training data
            step: Current training step
            wandb_logger: WandB logger instance

        Returns:
            Dictionary of monitoring metrics
        """
        if not self.should_monitor(step):
            return {}

        self.policy.eval()

        # Get the actual policy module (handle DDP wrapping)
        policy = self.policy.module if hasattr(self.policy, "module") else self.policy

        # Collect samples for monitoring
        sampled_actions_list = []
        ground_truth_actions_list = []
        n_action_steps = policy.config.n_action_steps

        sample_count = 0
        progress = tqdm(
            total=self.num_samples,
            desc=f"ActionSamplingMonitor (step {step})",
            unit="sample",
            leave=False,
        )
        for batch in training_dataloader:
            if sample_count >= self.num_samples:
                break

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)

            batch_size = batch["action"].shape[0]

            # Get ground truth actions
            ground_truth_actions = batch["action"]

            # Remove action labels, otherwise they get put into the queues
            del batch["action"]

            # Squeeze out time dimension if present
            # TODO: Handle variable-length sequences properly
            batch["observation.state"] = batch["observation.state"].squeeze(1)

            # Sample actions
            policy.reset()
            sampled_actions = []
            for t in range(n_action_steps):
                sampled_action = policy.select_action(batch)
                sampled_actions.append(sampled_action)
                progress.set_postfix(
                    batch=sample_count // max(batch_size, 1), action_step=f"{t + 1}/{n_action_steps}"
                )

            # Store results
            sampled_actions = torch.stack(sampled_actions, dim=1)  # shape (B, n_action_steps, action_dim)
            sampled_actions_list.append(sampled_actions.cpu())
            ground_truth_actions_list.append(ground_truth_actions.cpu())

            sample_count += batch_size
            progress.update(batch_size)

        progress.close()

        if not sampled_actions_list:
            return {}

        # Concatenate all samples
        sampled_actions = torch.cat(sampled_actions_list, dim=0)[: self.num_samples]
        ground_truth_actions = torch.cat(ground_truth_actions_list, dim=0)[: self.num_samples]

        # Remove padding dimension
        ground_truth_actions = ground_truth_actions[:, :n_action_steps, :]

        # Ensure both tensors have the same sequence length
        assert sampled_actions.shape == ground_truth_actions.shape, (
            f"Shape mismatch: sampled {sampled_actions.shape}, ground truth {ground_truth_actions.shape}"
        )

        # Compute metrics
        metrics = self._compute_monitoring_metrics(sampled_actions, ground_truth_actions)

        logger.info(
            "ActionSamplingMonitor @ step %d — MSE=%.6f  sampled(μ=%.4f σ=%.4f)  GT(μ=%.4f σ=%.4f)",
            step,
            metrics["sampling_mse_overall"],
            metrics["sampled_actions_mean"],
            metrics["sampled_actions_std"],
            metrics["ground_truth_mean"],
            metrics["ground_truth_std"],
        )

        # Log to wandb
        if wandb_logger is not None:
            self._log_to_wandb(metrics, step, wandb_logger)

        self.policy.train()  # Return to training mode
        return metrics

    def _compute_monitoring_metrics(
        self, sampled_actions: torch.Tensor, ground_truth_actions: torch.Tensor
    ) -> dict[str, float]:
        """Compute monitoring metrics from sampled and ground truth actions."""

        # Compute MSE
        mse_per_sample = torch.mean((sampled_actions - ground_truth_actions) ** 2, dim=(1, 2))
        overall_mse = torch.mean(mse_per_sample).item()

        # Compute per-dimension MSE
        mse_per_dim = torch.mean((sampled_actions - ground_truth_actions) ** 2, dim=(0, 1))
        per_dim_mse = mse_per_dim.tolist()

        # Compute per-timestep MSE
        mse_per_timestep = torch.mean((sampled_actions - ground_truth_actions) ** 2, dim=(0, 2))
        per_timestep_mse = mse_per_timestep.tolist()

        # Action statistics
        sampled_mean = torch.mean(sampled_actions).item()
        sampled_std = torch.std(sampled_actions).item()
        gt_mean = torch.mean(ground_truth_actions).item()
        gt_std = torch.std(ground_truth_actions).item()

        return {
            "sampling_mse_overall": overall_mse,
            "per_dim_mse": per_dim_mse,
            "per_timestep_mse": per_timestep_mse,
            "sampled_actions_mean": sampled_mean,
            "sampled_actions_std": sampled_std,
            "ground_truth_mean": gt_mean,
            "ground_truth_std": gt_std,
        }

    def _log_to_wandb(self, metrics: dict[str, float], step: int, wandb_logger):
        """Log metrics to wandb with proper organization."""

        # Overall metrics
        overall_metrics = {
            "sampling_monitor/mse_overall": metrics["sampling_mse_overall"],
        }

        # Action statistics
        action_stats = {
            "sampling_monitor/sampled_mean": metrics["sampled_actions_mean"],
            "sampling_monitor/sampled_std": metrics["sampled_actions_std"],
            "sampling_monitor/gt_mean": metrics["ground_truth_mean"],
            "sampling_monitor/gt_std": metrics["ground_truth_std"],
        }

        # Per-dimension MSE
        per_dim_metrics = {}
        for i, dim_mse in enumerate(metrics["per_dim_mse"]):
            per_dim_metrics[f"sampling_monitor/mse_dim_{i:02d}"] = dim_mse

        # Per-timestep MSE
        per_timestep_metrics = {}
        for i, timestep_mse in enumerate(metrics["per_timestep_mse"]):
            per_timestep_metrics[f"sampling_monitor/mse_timestep_{i:02d}"] = timestep_mse

        # Log all metrics
        all_metrics = {**overall_metrics, **action_stats, **per_dim_metrics, **per_timestep_metrics}

        wandb_logger.log(all_metrics, step=step)

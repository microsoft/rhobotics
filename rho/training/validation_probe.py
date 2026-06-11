import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from rho.common.constants import (
    ACTION,
    ACTION_TACTILE,
    OBSERVATION_IMAGE,
    OBSERVATION_LANG,
    OBSERVATION_STATE,
    OBSERVATION_TACTILE,
)

logger = logging.getLogger(__name__)


def generate_action_by_episode(policy, episode, device, num_action_steps=8, denorm_transform=None):
    """
    Generate actions for an episode using a policy, processing in chunks.

    Args:
        policy: Policy object with a .sample_actions(batch) method.
        episode: List of dicts, each dict representing a timestep observation.
            Each dict must contain:
                - "observation.image.*": torch.Tensor, shape (3, H, W) or (C, H, W)
                - "observation.state": torch.Tensor, shape (14,) or np.ndarray, shape (14,)
                - "observation.tactile": torch.Tensor, shape (36,) or np.ndarray, shape (36,)
                - "action": ground truth action, shape (action_dim,)
                - other keys (e.g., metadata, task, etc.)
        device: torch.device, device to move tensors to.
        num_action_steps: int, chunk size for processing.

    Returns:
        actions: np.ndarray, shape (N, action_dim)
            Predicted actions for all timesteps in the episode, where N = num_action_steps * num_chunks.
        gt_actions: np.ndarray, shape (N, action_dim)
            Ground truth actions for all timesteps in the episode, matching actions.
    """
    actions = []
    gt_actions = []
    tactile_actions = []
    gt_tactile_actions = []
    for start in range(0, len(episode), num_action_steps):
        # TODO currently this runs B=1 sequentially, we could speed up by batching multiple chunks
        obs = episode[start]
        gt_action_chunk = torch.from_numpy(np.array(obs[ACTION])).unsqueeze(0).to(device, dtype=torch.float32)

        # Prepare input for policy (first obs in chunk)
        batch = {
            key: (
                value.unsqueeze(0).to(device, dtype=torch.float32)
                if isinstance(value, torch.Tensor) and OBSERVATION_IMAGE in key
                else value.to(device, dtype=torch.float32)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in obs.items()
            if ACTION not in key
        }  # add batch dimension, convert to float32, move to device
        batch[OBSERVATION_LANG] = [batch[OBSERVATION_LANG]]
        batch[OBSERVATION_STATE] = batch[OBSERVATION_STATE].unsqueeze(0)
        if OBSERVATION_TACTILE in batch:
            batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE].unsqueeze(0)

        with torch.no_grad():
            output = policy.sample_actions(batch)
        action = output["actions"]
        tactile_action = output.get("tactile_action", None)

        # Denormalize predicted actions if transform is provided
        if denorm_transform is not None:
            pred_denorm_batch = {ACTION: action, OBSERVATION_STATE: batch[OBSERVATION_STATE]}
            if OBSERVATION_TACTILE in batch:
                pred_denorm_batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE]
            denorm_action = denorm_transform(pred_denorm_batch)[ACTION]

            gt_denorm_batch = {ACTION: gt_action_chunk, OBSERVATION_STATE: batch[OBSERVATION_STATE]}
            if OBSERVATION_TACTILE in batch:
                gt_denorm_batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE]
            denorm_gt_action = denorm_transform(gt_denorm_batch)[ACTION]
        else:
            denorm_action = action
            denorm_gt_action = gt_action_chunk

        if tactile_action is not None:
            gt_tactile_actions.extend([np.array(obs[ACTION_TACTILE])[:num_action_steps, :]])
            tactile_actions.append(
                tactile_action.squeeze(0)[:num_action_steps].to(torch.float32).cpu().numpy()
            )

        actions.append(denorm_action.squeeze(0)[:num_action_steps].to(torch.float32).cpu().numpy())
        gt_actions.append(denorm_gt_action.squeeze(0)[:num_action_steps].cpu().numpy())

    actions = np.concatenate(actions, axis=0)
    gt_actions = np.concatenate(gt_actions, axis=0)
    tactile_actions = np.concatenate(tactile_actions, axis=0) if len(tactile_actions) > 0 else None
    gt_tactile_actions = np.concatenate(gt_tactile_actions, axis=0) if len(gt_tactile_actions) > 0 else None
    return actions, gt_actions, tactile_actions, gt_tactile_actions


def plot_action_means(actions, gt_actions, title="Episode Trajectory", ax=None, dot_interval=8):
    """
    Plot mean action value per timestep for predicted and ground truth actions and return RGB image.

    Args:
        actions: np.ndarray, shape (N, action_dim)
        gt_actions: np.ndarray, shape (N, action_dim)
        title: str, plot title
        ax: matplotlib axis (optional)
        dot_interval: int, interval for highlighting chunk starts
    Returns:
        img: np.ndarray, shape (H, W, 3), dtype=uint8 (RGB)
    """
    timesteps = np.arange(actions.shape[0])
    actions_mean = actions.mean(axis=1)
    gt_actions_mean = gt_actions.mean(axis=1)
    dot_indices = np.arange(0, actions.shape[0], dot_interval)

    created_fig = False
    if ax is None:
        # Use constrained_layout with increased bottom margin
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=False)
        plt.subplots_adjust(bottom=0.15)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(timesteps, actions_mean, "b--", label="Predicted Action Mean")
    ax.plot(timesteps, gt_actions_mean, "k-", label="Ground Truth Action Mean")
    if dot_indices.size > 0:
        ax.scatter(
            dot_indices,
            gt_actions_mean[dot_indices],
            color="orange",
            s=15,
            label="Chunk Start (GT)",
            zorder=3,
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Action Value")
    ax.set_title(title)
    ax.legend()

    # Draw and extract RGB buffer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[:, :, :3]  # Convert RGBA to RGB by taking first 3 channels

    if created_fig:
        plt.close(fig)

    return img


class ValidationProbe:
    """
    A probe for validating policy performance on validation and optionally training datasets.
    This class processes episodes from LeRobot datasets, generates policy predictions,
    and creates visualization metrics for monitoring during training.

    Args:
        val_dataloader: DataLoader containing the validation dataset
        train_dataloader: Optional DataLoader containing the training dataset for comparison
    """

    def __init__(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        max_batches: int = 100,
        train_dataloader: torch.utils.data.DataLoader = None,
        denorm_transform=None,
    ):
        """
        Initialize the ValidationProbe with validation and optional training DataLoaders.

        Args:
            val_dataloader: DataLoader containing the validation dataset
            train_dataloader: Optional DataLoader containing the training dataset for comparison
        """
        self.val_dataloader = val_dataloader
        self.val_dataset = val_dataloader.dataset
        self.val_meta = self.val_dataset.meta
        self.val_num_episodes = len(self.val_meta.episodes)
        self.denorm_transform = denorm_transform
        self.max_batches = max_batches  # how many batches to use for action loss computation

        self.using_train = train_dataloader is not None
        if self.using_train:
            try:
                self.train_dataloader = train_dataloader
                self.train_dataset = train_dataloader.dataset
                self.train_meta = self.train_dataset.meta
                self.train_num_episodes = len(self.train_meta.episodes)
            except AttributeError:
                logger.warning(
                    "Training dataset does not have .meta attribute "
                    "(e.g. MultiDataset). Disabling train comparison."
                )
                self.using_train = False

    def validate_policy_by_episode(self, policy, device):
        """
        Validates policy on validation and optionally training episodes.
        Logs wandb.Image plots of action means.

        Args:
            policy: Policy model to validate (must support eval() and torch.no_grad()).
            device: torch device for inference.

        Returns:
            dict: Validation metrics with wandb.Image plots comparing predicted vs ground truth actions.
        """
        policy.eval()
        validation_metrics = {}

        if self.denorm_transform is not None:
            if hasattr(self.denorm_transform, "transforms"):  # Compose
                for i, t in enumerate(self.denorm_transform.transforms):
                    if isinstance(t, torch.nn.Module):
                        self.denorm_transform.transforms[i] = t.to(device)
            elif isinstance(self.denorm_transform, torch.nn.Module):  # Transform
                self.denorm_transform = self.denorm_transform.to(device)

        # Get unique episode indices in order
        with torch.no_grad():
            validation_actions = []
            validation_predictions = []
            validation_tactile_actions = []
            validation_tactile_predictions = []

            for i, batch in enumerate(self.val_dataloader):
                if i >= self.max_batches:
                    break
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                output = policy.sample_actions(batch)
                pred_actions = output["actions"]

                if self.denorm_transform is not None:
                    denorm_batch = {ACTION: pred_actions, OBSERVATION_STATE: batch[OBSERVATION_STATE]}
                    if OBSERVATION_TACTILE in batch:
                        denorm_batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE]
                    denorm_batch = self.denorm_transform(denorm_batch)
                    pred_actions = denorm_batch[ACTION]

                    gt_denorm_batch = {ACTION: batch[ACTION], OBSERVATION_STATE: batch[OBSERVATION_STATE]}
                    if OBSERVATION_TACTILE in batch:
                        gt_denorm_batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE]
                    gt_denorm_batch = self.denorm_transform(gt_denorm_batch)
                    gt_actions = gt_denorm_batch[ACTION]
                else:
                    gt_actions = batch[ACTION]

                validation_actions.append(gt_actions.cpu())
                validation_predictions.append(pred_actions.cpu())

                if "tactile_action" in output and output["tactile_action"] is not None:
                    validation_tactile_actions.append(batch[ACTION_TACTILE].cpu())
                    validation_tactile_predictions.append(output["tactile_action"].cpu())

            validation_actions = torch.cat(validation_actions, dim=0)
            validation_predictions = torch.cat(validation_predictions, dim=0).float()
            action_mse = F.mse_loss(validation_predictions, validation_actions, reduction="none")
            validation_metrics["validation_action_mse"] = action_mse.mean().item()

            if len(validation_tactile_actions) > 0:
                validation_tactile_actions = torch.cat(validation_tactile_actions, dim=0)
                validation_tactile_predictions = torch.cat(validation_tactile_predictions, dim=0).float()
                tactile_mse = F.mse_loss(
                    validation_tactile_predictions, validation_tactile_actions, reduction="none"
                )
                validation_metrics["validation_tactile_action_mse"] = tactile_mse.mean().item()

            # Generate plots for first 3 validation episodes only
            for episode_idx in range(min(3, self.val_num_episodes)):
                start_idx = self.val_meta.episodes[episode_idx]["dataset_from_index"]
                end_idx = self.val_meta.episodes[episode_idx]["dataset_to_index"]
                episode = [self.val_dataset[i] for i in range(start_idx, end_idx)]

                actions, gt_actions, tactile_actions, gt_tactile_actions = generate_action_by_episode(
                    policy,
                    episode,
                    device,
                    num_action_steps=policy.n_action_steps,
                    denorm_transform=self.denorm_transform,
                )

                img = plot_action_means(
                    actions,
                    gt_actions,
                    title=f"Validation Episode {episode_idx}",
                    dot_interval=policy.n_action_steps,
                )
                validation_metrics[f"ep{episode_idx}_action_means"] = wandb.Image(
                    img, caption=f"Episode {episode_idx} Action Means"
                )

                if tactile_actions is not None:
                    img_tactile = plot_action_means(
                        tactile_actions,
                        gt_tactile_actions,
                        title=f"Validation Episode {episode_idx} Tactile Actions",
                        dot_interval=policy.n_action_steps,
                    )
                    validation_metrics[f"ep{episode_idx}_tactile_action_means"] = wandb.Image(
                        img_tactile, caption=f"Episode {episode_idx} Tactile Action Means"
                    )

            if self.using_train:
                train_actions = []
                train_predictions = []
                train_tactile_actions = []
                train_tactile_predictions = []

                # Compute loss on up to 100 of training dataset
                for i, batch in enumerate(self.train_dataloader):
                    if i >= self.max_batches:
                        break
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    output = policy.sample_actions(batch)
                    pred_actions = output["actions"]

                    if self.denorm_transform is not None:
                        denorm_batch = {ACTION: pred_actions, OBSERVATION_STATE: batch[OBSERVATION_STATE]}
                        if OBSERVATION_TACTILE in batch:
                            denorm_batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE]
                        denorm_batch = self.denorm_transform(denorm_batch)
                        pred_actions = denorm_batch[ACTION]

                        gt_denorm_batch = {ACTION: batch[ACTION], OBSERVATION_STATE: batch[OBSERVATION_STATE]}
                        if OBSERVATION_TACTILE in batch:
                            gt_denorm_batch[OBSERVATION_TACTILE] = batch[OBSERVATION_TACTILE]
                        gt_denorm_batch = self.denorm_transform(gt_denorm_batch)
                        gt_actions = gt_denorm_batch[ACTION]
                    else:
                        gt_actions = batch[ACTION]

                    train_actions.append(gt_actions.cpu())
                    train_predictions.append(pred_actions.cpu())

                    if "tactile_action" in output and output["tactile_action"] is not None:
                        train_tactile_actions.append(batch[ACTION_TACTILE].cpu())
                        train_tactile_predictions.append(output["tactile_action"].cpu())

                train_actions = torch.cat(train_actions, dim=0)
                train_predictions = torch.cat(train_predictions, dim=0).float()
                action_mse = F.mse_loss(train_predictions, train_actions, reduction="none")
                validation_metrics["train_action_mse"] = action_mse.mean().item()

                if len(train_tactile_actions) > 0:
                    train_tactile_actions = torch.cat(train_tactile_actions, dim=0)
                    train_tactile_predictions = torch.cat(train_tactile_predictions, dim=0).float()
                    tactile_mse = F.mse_loss(
                        train_tactile_predictions, train_tactile_actions, reduction="none"
                    )
                    validation_metrics["train_tactile_action_mse"] = tactile_mse.mean().item()

                for episode_idx in range(min(3, self.train_num_episodes)):
                    start_idx = self.train_meta.episodes[episode_idx]["dataset_from_index"]
                    end_idx = self.train_meta.episodes[episode_idx]["dataset_to_index"]
                    episode = [self.train_dataset[i] for i in range(start_idx, end_idx)]

                    actions, gt_actions, tactile_actions, gt_tactile_actions = generate_action_by_episode(
                        policy,
                        episode,
                        device,
                        num_action_steps=policy.n_action_steps,
                        denorm_transform=self.denorm_transform,
                    )

                    img = plot_action_means(
                        actions,
                        gt_actions,
                        title=f"Train Episode {episode_idx}",
                        dot_interval=policy.n_action_steps,
                    )
                    validation_metrics[f"train_ep{episode_idx}_action_means"] = wandb.Image(
                        img, caption=f"Train Episode {episode_idx} Action Means"
                    )

                    if tactile_actions is not None:
                        img_tactile = plot_action_means(
                            tactile_actions,
                            gt_tactile_actions,
                            title=f"Train Episode {episode_idx} Tactile Actions",
                            dot_interval=policy.n_action_steps,
                        )
                        validation_metrics[f"train_ep{episode_idx}_tactile_action_means"] = wandb.Image(
                            img_tactile, caption=f"Train Episode {episode_idx} Tactile Action Means"
                        )

        return validation_metrics

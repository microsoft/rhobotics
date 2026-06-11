"""Replay buffer for DSRL-SAC training.

Stores raw images (uint8) for memory efficiency, state (float32), and noise vectors.
Uses pre-allocated numpy arrays with circular buffer indexing.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DSRLReplayBuffer:
    """Replay buffer optimized for DSRL-SAC with image observations.

    Stores images as uint8 for memory efficiency (converted to float32 on sampling).
    Supports multiple camera views and optional robot state.

    Args:
        capacity: maximum number of transitions
        image_shapes: dict mapping camera key -> (H, W, C) shape
        noise_dim: flattened noise dimension
        state_dim: robot state dimension (0 to disable)
        action_dim: flattened action dimension for NA mode (0 to disable)
    """

    def __init__(
        self,
        capacity: int,
        image_shapes: dict[str, tuple[int, int, int]],
        noise_dim: int,
        state_dim: int = 0,
        action_dim: int = 0,
    ):
        self.capacity = capacity
        self.noise_dim = noise_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_keys = list(image_shapes.keys())
        self.image_shapes = dict(image_shapes)

        # Pre-allocate arrays
        self.images: dict[str, np.ndarray] = {}
        self.next_images: dict[str, np.ndarray] = {}
        for key, (H, W, C) in image_shapes.items():
            self.images[key] = np.zeros((capacity, H, W, C), dtype=np.uint8)
            self.next_images[key] = np.zeros((capacity, H, W, C), dtype=np.uint8)

        self.noise = np.zeros((capacity, noise_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.discount = np.zeros((capacity,), dtype=np.float32)
        self.mask = np.ones((capacity,), dtype=np.float32)

        if state_dim > 0:
            self.state = np.zeros((capacity, state_dim), dtype=np.float32)
            self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        else:
            self.state = None
            self.next_state = None

        if action_dim > 0:
            self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.action = None

        self._size = 0
        self._ptr = 0

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def insert(
        self,
        images: dict[str, np.ndarray],
        next_images: dict[str, np.ndarray],
        noise: np.ndarray,
        reward: float,
        done: bool,
        discount: float,
        state: np.ndarray | None = None,
        next_state: np.ndarray | None = None,
        mask: float = 1.0,
        action: np.ndarray | None = None,
    ):
        """Insert a single transition.

        Args:
            images: dict of camera_key -> (H, W, C) uint8 arrays
            next_images: dict of camera_key -> (H, W, C) uint8 arrays
            noise: (noise_dim,) float32 array
            reward: scalar reward
            done: whether episode ended
            discount: effective discount factor
            state: (state_dim,) optional robot state
            next_state: (state_dim,) optional next robot state
            mask: Bellman backup mask (0.0 at episode end or takeover boundary)
            action: (action_dim,) optional action array (NA mode)
        """
        idx = self._ptr

        for key in self.image_keys:
            self.images[key][idx] = images[key]
            self.next_images[key][idx] = next_images[key]

        self.noise[idx] = noise.flatten()
        self.reward[idx] = reward
        self.done[idx] = float(done)
        self.discount[idx] = discount
        self.mask[idx] = mask

        if self.state is not None and state is not None:
            self.state[idx] = state
        if self.next_state is not None and next_state is not None:
            self.next_state[idx] = next_state
        if self.action is not None and action is not None:
            self.action[idx] = action.flatten()

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cuda") -> dict[str, torch.Tensor]:
        """Sample a random batch and convert to torch tensors.

        Images are converted from uint8 [0,255] to float32 [0,1] and transposed
        to (B, C, H, W) format. Multiple cameras are concatenated along channel dim.

        Args:
            batch_size: number of transitions to sample
            device: torch device to put tensors on

        Returns:
            Dict with keys:
                "images": (B, C_total, H, W) float32 in [0, 1]
                "next_images": (B, C_total, H, W) float32 in [0, 1]
                "noise": (B, noise_dim) float32
                "reward": (B,) float32
                "done": (B,) float32
                "discount": (B,) float32
                "mask": (B,) float32 — Bellman backup mask
                "state": (B, state_dim) float32 (if state_dim > 0)
                "next_state": (B, state_dim) float32 (if state_dim > 0)
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        # Stack camera images and convert to float
        all_images = []
        all_next_images = []
        for key in self.image_keys:
            # (B, H, W, C) uint8 -> (B, C, H, W) float32
            imgs = self.images[key][indices]
            next_imgs = self.next_images[key][indices]
            all_images.append(imgs)
            all_next_images.append(next_imgs)

        # Concatenate cameras along channel dim: (B, H, W, C*num_cameras)
        stacked_images = np.concatenate(all_images, axis=-1)
        stacked_next = np.concatenate(all_next_images, axis=-1)

        # Convert to torch: (B, H, W, C) -> (B, C, H, W), uint8 -> float [0,1]
        batch = {
            "images": torch.from_numpy(stacked_images).permute(0, 3, 1, 2).float().div(255.0).to(device),
            "next_images": torch.from_numpy(stacked_next).permute(0, 3, 1, 2).float().div(255.0).to(device),
            "noise": torch.from_numpy(self.noise[indices]).to(device),
            "reward": torch.from_numpy(self.reward[indices]).to(device),
            "done": torch.from_numpy(self.done[indices]).to(device),
            "discount": torch.from_numpy(self.discount[indices]).to(device),
            "mask": torch.from_numpy(self.mask[indices]).to(device),
        }

        if self.state is not None:
            batch["state"] = torch.from_numpy(self.state[indices]).to(device)
            batch["next_state"] = torch.from_numpy(self.next_state[indices]).to(device)

        return batch

    def save(self, path: str):
        """Save buffer contents to disk.

        Only saves the filled portion of each array to reduce file size.
        Images are kept as uint8 for compact storage.

        Args:
            path: file path (will be saved as .npz)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        n = self._size
        data = {
            "noise": self.noise[:n],
            "reward": self.reward[:n],
            "done": self.done[:n],
            "discount": self.discount[:n],
            "mask": self.mask[:n],
            "_size": np.array(self._size),
            "_ptr": np.array(self._ptr),
        }
        for key in self.image_keys:
            data[f"images_{key}"] = self.images[key][:n]
            data[f"next_images_{key}"] = self.next_images[key][:n]
        if self.state is not None:
            data["state"] = self.state[:n]
            data["next_state"] = self.next_state[:n]
        if self.action is not None:
            data["action"] = self.action[:n]

        np.savez_compressed(path, **data)
        logger.info(f"Saved replay buffer ({n} transitions) to {path}")

    def load(self, path: str):
        """Load buffer contents from disk.

        Args:
            path: file path (.npz)
        """
        data = np.load(path)
        n = int(data["_size"])
        loaded_ptr = int(data["_ptr"])

        # Copy into pre-allocated arrays
        self.noise[:n] = data["noise"]
        self.reward[:n] = data["reward"]
        self.done[:n] = data["done"]
        self.discount[:n] = data["discount"]
        self.mask[:n] = data["mask"]

        for key in self.image_keys:
            self.images[key][:n] = data[f"images_{key}"]
            self.next_images[key][:n] = data[f"next_images_{key}"]

        if self.state is not None and "state" in data:
            self.state[:n] = data["state"]
            self.next_state[:n] = data["next_state"]

        if self.action is not None and "action" in data:
            self.action[:n] = data["action"]

        self._size = n
        self._ptr = loaded_ptr
        logger.info(f"Loaded replay buffer ({n} transitions) from {path}")

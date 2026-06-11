"""DSRL Trainer for HIL pipeline.

Receives experience transitions via ZMQ, trains a DSRL agent in noise space,
and publishes updated actor+encoder parameters back to the robot.

Supports two algorithm variants:
  - "sac": Standard SAC in noise space (no policy server calls during training).
  - "na":  Noise-Aliased (Algorithm 1) — Q^A + Q^W distillation.
           Requires calling π_dp via WebSocket during training.

The trainer:
1. Receives transitions from the robot via ExperienceReceiver
2. Extracts images, state, noise (and actions for NA) from observations
3. Optionally runs inverse_noise_map for intervention recovery
4. Stores transitions in DSRLReplayBuffer
5. Runs SAC/NA updates via DSRLSACAgent
6. Publishes actor+encoder params via ParamPublisher

Usage:
    # From serve_hil.py with --train --trainer_type=dsrl
    # Or standalone:
    from rho.hil.trainers.dsrl_trainer import start_dsrl_trainer
    trainer = start_dsrl_trainer(config=config, base_policy=policy, blocking=True)
"""

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from rho.common.wandb_logging import WandBConfig, WandBLogger
from rho.hil.experience import ExperienceReceiver, Transition
from rho.hil.param_subscriber import ParamPublisher
from rho.policies.dsrl.dsrl_agent import DSRLSACAgent
from rho.policies.dsrl.dsrl_config import DSRLConfig
from rho.policies.dsrl.replay_buffer import DSRLReplayBuffer

if True:  # avoid circular import guard; these are runtime-only
    from rho.eval.policy_interface import PolicyInterface

logger = logging.getLogger(__name__)


def _make_policy_forward_fn(
    server_url: str,
    config: DSRLConfig,
) -> Callable:
    """Create a policy_forward_fn for NA mode using the DSRLServerClient.

    The returned function takes (obs_emb, noise_single) — but since the policy
    server operates on raw observations, the trainer must call this with
    raw transition data. This creates a closure that handles the server call.

    For NA mode, the trainer calls the policy server with batched observations
    and tiled noise to get π_dp(s, tile(w)) -> flattened actions.
    """
    from rho.policies.dsrl.dsrl_server_client import DSRLServerClient

    client = DSRLServerClient(server_url)
    chunk_size = config.action_chunk_size

    def forward_fn(
        batched_obs: dict,
        noise_single: torch.Tensor,
        use_next_obs: bool = False,
    ) -> torch.Tensor:
        """Call π_dp(obs, tile(w_single)) -> flattened actions.

        Args:
            batched_obs: dict of batched observation arrays
            noise_single: (B, noise_dim) per-step noise
            use_next_obs: unused (obs already selected by caller)

        Returns:
            (B, action_chunk_size * action_dim) flattened actions
        """
        B = noise_single.shape[0]
        noise_np = noise_single.detach().cpu().numpy()

        # Tile per-step noise to (B, C, noise_dim)
        noise_tiled = np.tile(noise_np[:, None, :], (1, chunk_size, 1))

        result = client.infer(batched_obs, initial_noise=noise_tiled)
        action = result["action"] if isinstance(result, dict) else result

        action_tensor = torch.from_numpy(action).float() if isinstance(action, np.ndarray) else action.float()

        device = noise_single.device
        return action_tensor.to(device).view(B, -1)

    return forward_fn


class DSRLTrainer:
    """DSRL trainer for human-in-the-loop online RL.

    Trainer interface: start(), stop(), train()

    Args:
        config: DSRLConfig with all hyperparameters
        base_policy: frozen Phi4MM model for inverse_noise_map (optional)
        prompt: text prompt for the base policy (used in inverse_noise_map)
        experience_port: ZMQ port for receiving transitions
        param_port: ZMQ port for publishing parameters
        device: torch device for training
        policy_interface: PolicyInterface for proper observation preprocessing
            (remapping, transforms, normalization) before inverse noise map
        env: Environment server with process_input() for raw-to-tensor conversion
    """

    def __init__(
        self,
        config: DSRLConfig,
        base_policy: Any = None,
        prompt: str = "",
        experience_port: int = 5555,
        param_port: int = 5556,
        device: str = "cuda",
        policy_interface: Optional["PolicyInterface"] = None,
        env: Any | None = None,
    ):
        self.config = config
        self.base_policy = base_policy
        self.prompt = prompt
        self.device = device
        self.policy_interface = policy_interface
        self.env = env

        # HIL infrastructure
        self.experience_receiver = ExperienceReceiver(port=experience_port)
        self.param_publisher = ParamPublisher(port=param_port)

        # For NA mode: create policy server callback
        if config.algorithm == "na":
            self._server_forward_fn = _make_policy_forward_fn(config.policy_server_url, config)
            # Wrap to match the agent's PolicyForwardFn signature:
            # The agent passes (obs_emb, noise) but for NA the server needs
            # raw observations. We store transitions for the current batch
            # and use them in the callback.
            self._current_batch_transitions: list[Transition] | None = None

            def _na_policy_fn(obs_emb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
                # The trainer sets self._current_batch_transitions before calling update
                if self._current_batch_transitions is None:
                    raise RuntimeError("NA policy_forward_fn called without batch context")
                # Build batched obs dict from stored transitions
                transitions = self._current_batch_transitions
                sample_obs = transitions[0].obs
                batched_obs: dict = {}
                for key in sample_obs:
                    arrays = [t.obs[key] for t in transitions]
                    try:
                        batched_obs[key] = np.stack(arrays, axis=0)
                    except ValueError:
                        batched_obs[key] = arrays
                return self._server_forward_fn(batched_obs, noise)

        # DSRL agent
        self.agent = DSRLSACAgent(
            config,
            device=device,  # policy_forward_fn=policy_forward_fn
        )

        # Replay buffer
        image_shapes = {}
        for key in config.image_keys:
            image_shapes[key] = (config.image_size, config.image_size, 3)
        self.replay_buffer = DSRLReplayBuffer(
            capacity=config.replay_buffer_capacity,
            image_shapes=image_shapes,
            noise_dim=config.noise_dim,
            state_dim=config.state_dim if config.include_state else 0,
            # action_dim=config.action_value_dim if config.algorithm == "na" else 0,
        )

        # Logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.total_transitions = 0
        self.total_updates = 0
        self.total_interventions = 0
        self.total_episodes = 0
        self.total_successes = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_interventions = 0
        self._recent_returns: list = []  # last N episode returns for rolling average
        self._recent_lengths: list = []  # last N episode lengths
        self._recent_intervention_rates: list = []  # last N per-episode intervention rates
        self._inverse_map_mse_sum = 0.0
        self._inverse_map_noise_norm_sum = 0.0
        self._inverse_map_count = 0
        self.running = False

        # TensorBoard writer (lazy init)
        self._writer = None

        self._wandb_logger: WandBLogger | None = None
        if config.wandb_enabled:
            # Auto-generate run name and group from experiment config
            # e.g. "rho_dsrl-sac_sparse_pick_drop_pumpkin_0402_1530"
            task_name = Path(config.log_dir).name  # e.g. "dsrl_pick_drop_pumpkin"
            algo_tag = f"dsrl-{config.algorithm}"  # e.g. "dsrl-sac"
            base_model = config.base_policy_name  # e.g. "rhoalpha" or "phi4mm"
            reward_tag = config.reward_type  # e.g. "sparse" or "intervention"
            timestamp = datetime.now().strftime("%m%d_%H%M%S")

            wandb_name = config.wandb_name or (
                f"{base_model}_{algo_tag}_{reward_tag}_{task_name}_{timestamp}"
            )
            wandb_group = config.wandb_group or (f"{base_model}_{algo_tag}_{task_name}")

            wandb_cfg = WandBConfig(
                enabled=True,
                project=config.wandb_project,
                username=config.wandb_entity or None,
                id=wandb_name,
                group=wandb_group,
            )
            self._wandb_logger = WandBLogger(wandb_cfg)

        # Resume from previous run
        if config.resume_checkpoint:
            self.agent.load_checkpoint(config.resume_checkpoint)
            self.total_updates = self.agent.train_step
            logger.info(
                f"[DSRLTrainer] Resumed agent from {config.resume_checkpoint} (step {self.total_updates})"
            )
        if config.resume_buffer:
            self.replay_buffer.load(config.resume_buffer)
            self.total_transitions = self.replay_buffer.size
            logger.info(
                f"[DSRLTrainer] Resumed buffer from {config.resume_buffer} "
                f"({self.replay_buffer.size} transitions)"
            )

        # Precomputed VLM hidden state cache for inverse noise map
        self._cached_hidden_state = None

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                pass
        return self._writer

    def start(self):
        """Start ZMQ receiver and publisher."""
        self.experience_receiver.start()
        self.param_publisher.start()
        self.running = True
        logger.info(
            f"[DSRLTrainer] Started ({self.config.algorithm.upper()}) | "
            f"experience_port={self.experience_receiver.port} | "
            f"param_port={self.param_publisher.port}"
        )

    def _save_checkpoint(self, tag: str):
        """Save agent checkpoint and replay buffer.

        Args:
            tag: identifier appended to filenames (e.g. "step1000", "final_step500")
        """
        ckpt_path = str(self.log_dir / f"checkpoint_{tag}.pt")
        self.agent.save_checkpoint(ckpt_path)
        buf_path = str(self.log_dir / f"buffer_{tag}.npz")
        self.replay_buffer.save(buf_path)

    def stop(self):
        """Stop, save final checkpoint."""
        self.running = False

        # Wait for training loop to exit before cleanup, but only if stop() is
        # being called from a different thread (otherwise we'd deadlock).
        if (
            hasattr(self, "_train_thread")
            and self._train_thread is not None
            and self._train_thread.is_alive()
            and self._train_thread is not threading.current_thread()
        ):
            self._train_thread.join(timeout=5.0)

        # Save final checkpoint + buffer
        self._save_checkpoint(f"final_step{self.total_updates}")

        self.experience_receiver.stop()
        self.param_publisher.stop()
        if self._writer is not None:
            self._writer.close()

        if self._wandb_logger is not None:
            self._wandb_logger.finish()

        logger.info(
            f"[DSRLTrainer] Stopped | transitions={self.total_transitions} | "
            f"updates={self.total_updates} | episodes={self.total_episodes} | "
            f"interventions={self.total_interventions}"
        )

    def _extract_images(self, obs: Any) -> dict[str, np.ndarray]:
        """Extract image arrays from observation dict.

        Returns:
            Dict of camera_key -> (H, W, C) uint8 arrays
        """
        images = {}
        if isinstance(obs, dict):
            for key in self.config.image_keys:
                img = None
                for obs_key in [key, f"image.{key}", f"observation.image.{key}"]:
                    if obs_key in obs:
                        img = obs[obs_key]
                        break
                if img is None and "image" in obs and isinstance(obs["image"], dict):
                    img = obs["image"].get(key)
                if img is None:
                    raise KeyError(
                        f"Could not find image key '{key}' in obs. Available keys: {list(obs.keys())}"
                    )
                if img.dtype != np.uint8:
                    img = (
                        (img * 255).clip(0, 255).astype(np.uint8)
                        if img.max() <= 1.0
                        else img.astype(np.uint8)
                    )
                if img.shape[0] != self.config.image_size or img.shape[1] != self.config.image_size:
                    import cv2

                    img = cv2.resize(img, (self.config.image_size, self.config.image_size))
                images[key] = img
        else:
            raise ValueError(f"Expected dict obs, got {type(obs)}")
        return images

    def _extract_state(self, obs: Any) -> np.ndarray | None:
        """Extract state vector from observation dict."""
        if not self.config.include_state or self.config.state_dim == 0:
            return None
        if isinstance(obs, dict):
            for key in ["state", "tcp_pose", "joint_positions", "observation.state"]:
                if key in obs:
                    return np.asarray(obs[key], dtype=np.float32).flatten()[: self.config.state_dim]
        return None

    def _compute_reward(self, transition: Transition) -> float:
        """Compute reward for a transition based on config."""
        if self.config.reward_type == "sparse":
            if transition.done and getattr(transition, "success", None):
                return self.config.success_reward
            return self.config.step_reward
        elif self.config.reward_type == "intervention":
            if transition.intervened:
                return self.config.intervention_reward
            if transition.done and getattr(transition, "success", None):
                return self.config.success_reward
            return self.config.step_reward
        return self.config.step_reward

    def _process_obs_for_policy(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run raw observation dict through the full preprocessing pipeline.

        Uses env.process_input() for raw-to-tensor conversion (BGR→RGB,
        normalization to [0,1], batch dims, FK conversions) and then
        policy_interface.process_observation() for key remapping and
        input transforms/normalization — mirroring exactly what the base
        policy sees at inference time.

        Args:
            obs_dict: Raw observation dict (numpy arrays, same format as
                what ExperiencePublisher sends)

        Returns:
            Processed observation dict with policy-expected keys and
            properly transformed tensors.
        """
        processed_obs = self.env.process_input(obs_dict)
        processed_obs = self.policy_interface.process_observation(processed_obs, process_action=True)
        return processed_obs

    def _prepare_inverse_map_inputs(self, transition: Transition, images: dict[str, np.ndarray]):
        """Prepare inputs for inverse noise mapping.

        Uses env.process_input() and policy_interface.process_observation()
        to replicate the exact preprocessing the base policy sees at
        inference time, ensuring the inverse noise map operates in the
        same observation space.

        Returns:
            Tuple of (flow_model, image_input, prompt_input, state_val, a_target)
            or None if base_policy is unavailable.
        """
        model = self.base_policy
        if hasattr(model, "model"):
            flow_model = model.model
            if hasattr(flow_model, "flow_model"):
                flow_model = flow_model.flow_model
        else:
            flow_model = model

        # Prepare the intervention action as target
        a_raw = transition.intervention_action
        a_target = torch.from_numpy(np.asarray(a_raw, dtype=np.float32)).float().to(self.device)
        chunk_size = flow_model.config.chunk_size
        if a_target.ndim == 1:
            a_target = a_target.unsqueeze(0).unsqueeze(0).expand(1, chunk_size, -1)
        elif a_target.ndim == 2:
            a_target = a_target.unsqueeze(0)

        # Process observation through the same pipeline the base policy uses
        obs_dict = dict(transition.obs)  # copy to avoid mutating original
        obs_dict["action"] = a_target.cpu().numpy()  # add action to obs for policy processing
        processed_obs = self._process_obs_for_policy(obs_dict)

        batch = dict(processed_obs)
        batch = model.consolidate_images(batch)

        image, image_mask = model.prepare_image(batch)
        prompt = model.prepare_prompt(batch)
        state = model.prepare_state(batch)
        a_target = model.prepare_action(batch)

        return flow_model, image, prompt, state, a_target

    def _run_inverse_noise_map(
        self, transition: Transition, images: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Run inverse noise mapping on an intervention action.

        Converts the human's intervention action to noise space using the
        frozen base policy.

        Returns:
            Recovered noise vector (noise_dim,) or None if base_policy unavailable
        """
        if self.base_policy is None or transition.intervention_action is None:
            return None

        try:
            from rho.hil.noise_inverse_map import inverse_noise_map

            cfg = self.config
            flow_model, image, prompt, state, a_target = self._prepare_inverse_map_inputs(transition, images)

            best_w, best_losses, _ = inverse_noise_map(
                flow_model,
                image,
                prompt,
                state,
                a_target,
                n_restarts=cfg.inverse_map_restarts,
                optimizer_steps=cfg.inverse_map_steps,
                lr=cfg.inverse_map_lr,
                b_W=cfg.noise_magnitude,
            )

            noise = best_w[0, : cfg.noise_action_steps, : cfg.noise_action_dim]
            noise = noise.cpu().numpy().flatten()

            mse = best_losses[0].item()
            noise_norm = np.linalg.norm(noise)
            self._inverse_map_mse_sum += mse
            self._inverse_map_noise_norm_sum += noise_norm
            self._inverse_map_count += 1

            logger.debug(f"Inverse map: MSE={mse:.6f}, ||w||={noise_norm:.4f}")
            return noise

        except Exception as e:
            logger.warning(f"Inverse noise map failed: {e}", exc_info=True)
            return None

    def _process_transition(self, transition: Transition):
        """Convert a Transition to a replay buffer entry.

        Handles image/state extraction, noise extraction or inverse mapping,
        reward computation, and buffer insertion.
        """
        try:
            images = self._extract_images(transition.obs)
            next_images = self._extract_images(transition.next_obs)
            state = self._extract_state(transition.obs)
            next_state = self._extract_state(transition.next_obs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping transition: {e}")
            return

        # Get noise vector
        noise = None
        if hasattr(transition, "noise") and transition.noise is not None:
            noise = np.asarray(transition.noise, dtype=np.float32).flatten()
        elif transition.intervened and self.config.reward_type == "intervention":
            noise = self._run_inverse_noise_map(transition, images)

        if noise is None:
            logger.debug("Skipping transition: no noise vector available")
            return

        # Ensure correct noise shape.
        # The robot may send w_single (noise_action_dim) which needs tiling to
        # (noise_action_steps * noise_action_dim) for the chunk-noise approach.
        expected_dim = self.config.noise_dim
        if noise.shape[0] == self.config.noise_action_dim and noise.shape[0] != expected_dim:
            # w_single received — tile across noise_action_steps
            noise = np.tile(noise, self.config.noise_action_steps)
        elif noise.shape[0] != expected_dim:
            noise = (
                noise[:expected_dim]
                if noise.shape[0] > expected_dim
                else np.pad(noise, (0, expected_dim - noise.shape[0]))
            )

        reward = self._compute_reward(transition)
        discount = self.config.effective_discount

        # Bellman backup mask: 0.0 at episode end, and optionally at
        # intervention boundaries when intervention_stop_reward is enabled.
        mask = 0.0 if transition.done else 1.0
        if self.config.intervention_stop_reward and transition.intervened:
            mask = 0.0

        # For NA mode, extract and flatten the action trajectory
        if self.config.algorithm == "na":
            np.asarray(transition.action, dtype=np.float32).flatten()

        self.replay_buffer.insert(
            images=images,
            next_images=next_images,
            noise=noise,
            reward=reward,
            done=transition.done,
            discount=discount,
            mask=mask,
            state=state,
            next_state=next_state,
            # action=action,
        )

        self.total_transitions += 1
        self._episode_reward += reward
        self._episode_length += 1
        if transition.intervened:
            self.total_interventions += 1
            self._episode_interventions += 1
        if transition.done:
            self.total_episodes += 1
            self._recent_returns.append(self._episode_reward)
            self._recent_lengths.append(self._episode_length)
            ep_int_rate = (
                self._episode_interventions / self._episode_length if self._episode_length > 0 else 0.0
            )
            self._recent_intervention_rates.append(ep_int_rate)
            if len(self._recent_returns) > 20:
                self._recent_returns.pop(0)
            if len(self._recent_lengths) > 20:
                self._recent_lengths.pop(0)
            if len(self._recent_intervention_rates) > 20:
                self._recent_intervention_rates.pop(0)
            if getattr(transition, "success", None):
                self.total_successes += 1
            self._episode_reward = 0.0
            self._episode_length = 0
            self._episode_interventions = 0

    def train(
        self,
        num_steps: int | None = None,
        num_transitions: int | None = None,
        num_episodes: int | None = None,
    ):
        """Main training loop.

        1. Receive transitions via ZMQ
        2. Process and insert into buffer
        3. If buffer large enough: run updates (UTD ratio handled by agent)
        4. Publish actor params every publish_interval
        5. Log every log_interval, checkpoint every checkpoint_interval

        Args:
            num_steps: stop after this many training updates (None = run forever)
            num_transitions: stop after receiving this many transitions (None = run forever)
            num_episodes: stop after this many episodes (None = run forever)
        """
        logger.info(f"[DSRLTrainer] Training loop started ({self.config.algorithm.upper()})")
        cfg = self.config
        last_log_time = time.time()

        while self.running:
            # Check termination
            if num_transitions is not None and self.total_transitions >= num_transitions:
                logger.info(f"Reached {num_transitions} transitions, stopping")
                break
            if num_steps is not None and self.total_updates >= num_steps:
                logger.info(f"Reached {num_steps} training steps, stopping")
                break
            if num_episodes is not None and self.total_episodes >= num_episodes:
                logger.info(f"Reached {num_episodes} episodes, stopping")
                break

            # Receive transitions
            transitions = self.experience_receiver.receive_batch(max_batch=256, timeout_ms=100)
            # if transitions:
            #     print(f"[DEBUG] Received {len(transitions)} transitions")

            for t in transitions:
                self._process_transition(t)

            # Train if buffer is large enough
            if not self.running:
                break
            if self.replay_buffer.size >= cfg.start_training_after:
                batch = self.replay_buffer.sample(cfg.batch_size, device=self.device)
                info = self.agent.update(batch)
                self.total_updates += 1

                # Publish params
                if self.total_updates % cfg.publish_interval == 0:
                    self.param_publisher.publish(self.agent.get_policy_params())

                # TensorBoard logging
                if self.total_updates % cfg.log_interval == 0:
                    writer = self._get_writer()
                    if writer is not None:
                        for k, v in info.items():
                            writer.add_scalar(f"train/{k}", v, self.total_updates)
                        writer.add_scalar("buffer/size", self.replay_buffer.size, self.total_updates)
                        writer.add_scalar("buffer/transitions", self.total_transitions, self.total_updates)
                        writer.add_scalar("buffer/episodes", self.total_episodes, self.total_updates)
                        writer.add_scalar(
                            "buffer/interventions", self.total_interventions, self.total_updates
                        )
                    # WandB logging
                    if self._wandb_logger is not None:
                        metrics = {f"train/{k}": v for k, v in info.items()}
                        metrics.update(
                            {
                                "buffer/size": self.replay_buffer.size,
                                "buffer/transitions": self.total_transitions,
                                "buffer/episodes": self.total_episodes,
                                "buffer/interventions": self.total_interventions,
                            }
                        )
                        if self.total_episodes > 0:
                            metrics["episode/success_rate"] = self.total_successes / self.total_episodes
                        if self._recent_returns:
                            metrics["episode/return_mean"] = sum(self._recent_returns) / len(
                                self._recent_returns
                            )
                            metrics["episode/return_last"] = self._recent_returns[-1]
                            if len(self._recent_returns) > 1:
                                metrics["episode/return_std"] = (
                                    sum(
                                        (r - metrics["episode/return_mean"]) ** 2
                                        for r in self._recent_returns
                                    )
                                    / (len(self._recent_returns) - 1)
                                ) ** 0.5
                        if self._recent_lengths:
                            metrics["episode/length_mean"] = sum(self._recent_lengths) / len(
                                self._recent_lengths
                            )
                            metrics["episode/length_last"] = self._recent_lengths[-1]
                        if self._recent_intervention_rates:
                            metrics["episode/intervention_rate_mean"] = sum(
                                self._recent_intervention_rates
                            ) / len(self._recent_intervention_rates)
                        # Inverse noise map metrics
                        if self._inverse_map_count > 0:
                            metrics["inverse_map/mse_mean"] = (
                                self._inverse_map_mse_sum / self._inverse_map_count
                            )
                            metrics["inverse_map/noise_norm_mean"] = (
                                self._inverse_map_noise_norm_sum / self._inverse_map_count
                            )
                            metrics["inverse_map/count"] = self._inverse_map_count
                            self._inverse_map_mse_sum = 0.0
                            self._inverse_map_noise_norm_sum = 0.0
                            self._inverse_map_count = 0
                        # Throughput
                        elapsed = time.time() - last_log_time
                        if elapsed > 0:
                            metrics["throughput/updates_per_sec"] = cfg.log_interval / elapsed
                            metrics["throughput/utd_effective"] = self.total_updates / max(
                                self.total_transitions, 1
                            )
                        self._wandb_logger.log(metrics, step=self.total_updates)

                # Checkpoint
                if self.total_updates % cfg.checkpoint_interval == 0:
                    self._save_checkpoint(f"step{self.total_updates}")

            # Periodic console log
            now = time.time()
            if now - last_log_time >= 10.0:
                buf_size = self.replay_buffer.size
                int_ratio = (
                    self.total_interventions / self.total_transitions * 100
                    if self.total_transitions > 0
                    else 0.0
                )
                logger.info(
                    f"[DSRLTrainer] ({cfg.algorithm.upper()}) "
                    f"transitions={self.total_transitions} | "
                    f"updates={self.total_updates} | buffer={buf_size} | "
                    f"episodes={self.total_episodes} | "
                    f"interventions={self.total_interventions} ({int_ratio:.1f}%)"
                )
                last_log_time = now

        logger.info(
            f"[DSRLTrainer] Training loop ended | updates={self.total_updates} | "
            f"transitions={self.total_transitions}"
        )


def start_dsrl_trainer(
    config: DSRLConfig,
    base_policy: Any = None,
    prompt: str = "",
    experience_port: int = 5555,
    param_port: int = 5556,
    device: str = "cuda",
    num_steps: int | None = None,
    num_transitions: int | None = None,
    num_episodes: int | None = None,
    blocking: bool = True,
    policy_interface: Optional["PolicyInterface"] = None,
    env: Any | None = None,
) -> DSRLTrainer:
    """Create, start, and run a DSRLTrainer.

    Args:
        config: DSRLConfig with all hyperparameters
        base_policy: frozen Phi4MM model (optional, for inverse_noise_map)
        prompt: text prompt for the base policy
        experience_port: ZMQ port to listen for transitions
        param_port: ZMQ port to publish parameters
        device: torch device for training
        num_steps: stop after this many training updates
        num_transitions: stop after receiving this many transitions
        num_episodes: stop after this many episodes
        blocking: if True, blocks until done. If False, runs in background thread.
        policy_interface: PolicyInterface for observation preprocessing (remapping, transforms)
        env: Environment server with process_input() for raw-to-tensor conversion

    Returns:
        The DSRLTrainer instance (call .stop() to shut down)
    """
    logger.info(
        f"Starting DSRL-{config.algorithm.upper()} Trainer | "
        f"experience_port={experience_port} | param_port={param_port} | "
        f"device={device} | noise_dim={config.noise_dim} | "
        f"encoder={config.encoder_type} | critics={config.num_critics} | "
        f"utd_ratio={config.utd_ratio} | batch_size={config.batch_size}"
    )

    trainer = DSRLTrainer(
        config=config,
        base_policy=base_policy,
        prompt=prompt,
        experience_port=experience_port,
        param_port=param_port,
        device=device,
        policy_interface=policy_interface,
        env=env,
    )

    def _run():
        trainer.start()
        try:
            trainer.train(num_steps=num_steps, num_transitions=num_transitions, num_episodes=num_episodes)
        except KeyboardInterrupt:
            logger.info("[DSRLTrainer] Interrupted by user")
        except Exception as e:
            logger.error(f"[DSRLTrainer] Error: {e}")
            raise
        finally:
            trainer.stop()

    if blocking:
        _run()
    else:
        trainer._train_thread = threading.Thread(
            target=_run,
            name="DSRLTrainer",
            daemon=True,
        )
        trainer._train_thread.start()
        logger.info("[DSRLTrainer] Started in background thread")

    return trainer

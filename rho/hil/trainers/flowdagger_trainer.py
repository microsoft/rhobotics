"""FlowDAggerTrainer: supervised noise-DAgger over a frozen flow base policy.

Receives intervened Transitions from the robot, inverts the human's
intervention_action through the base flow ODE to recover w* (the noise that
would have produced that action), and trains a deterministic noise policy
to regress to w* via MSE. Trained weights are hot-swapped into the
inference-side FlowDAggerPolicy via ZMQ PUB/SUB.

Sibling of DSRLTrainer; reuses the same ExperienceReceiver / ParamPublisher
plumbing, swaps SAC for supervised BC.

Usage:
    from rho.hil.trainers.flowdagger_trainer import start_flowdagger_trainer
    trainer = start_flowdagger_trainer(
        config=cfg, base_policy=policy, policy_interface=pi, env=env,
        blocking=False,
    )
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rho.hil.experience import ControlMessage, ExperienceReceiver, Transition
from rho.hil.noise_inverse_map import inverse_noise_map
from rho.hil.param_subscriber import ParamPublisher
from rho.policies.dsrl.encoder import ObsEncoder
from rho.policies.dsrl.flowdagger_config import FlowDAggerConfig
from rho.policies.dsrl.noise_policy import DeterministicNoisePolicy

logger = logging.getLogger(__name__)


class _DAggerBuffer:
    """Random-eviction buffer for (image-set, state, w*) tuples.

    Stores post-normalize float32 images at image_size in CHW layout so the
    BC update can hand them straight to the encoder. Train-time and
    inference-time encoder inputs are then identical (same env+remap+
    transforms pipeline applied beforehand).

    Eviction: fills sequentially until full, then evicts a uniformly random
    existing slot per new insertion. This gives each existing item an
    exponential survival probability with timescale ~capacity, which mixes
    recent and older transitions better than strict FIFO — important when
    each episode dumps in many highly temporally-correlated frames.
    """

    def __init__(
        self,
        capacity: int,
        image_keys,
        image_size: int,
        state_dim: int,
        noise_dim: int,
    ):
        self.capacity = capacity
        self.image_keys = list(image_keys)
        self.images = {
            k: np.zeros((capacity, 3, image_size, image_size), dtype=np.float32) for k in self.image_keys
        }
        self.state = np.zeros((capacity, state_dim), dtype=np.float32) if state_dim > 0 else None
        self.noise = np.zeros((capacity, noise_dim), dtype=np.float32)
        self.size = 0

    def add(self, images: dict[str, np.ndarray], state: np.ndarray | None, w: np.ndarray):
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = int(np.random.randint(0, self.capacity))
        for k in self.image_keys:
            self.images[k][idx] = images[k]
        if self.state is not None and state is not None:
            self.state[idx] = state[: self.state.shape[1]]
        self.noise[idx] = w[: self.noise.shape[1]]

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        images = {k: self.images[k][idx] for k in self.image_keys}
        state = self.state[idx] if self.state is not None else None
        noise = self.noise[idx]
        return images, state, noise


class FlowDAggerTrainer:
    """Supervised noise-DAgger trainer."""

    def __init__(
        self,
        config: FlowDAggerConfig,
        base_policy: Any,
        prompt: str = "",
        experience_port: int = 5555,
        param_port: int = 5556,
        device: str = "cuda",
        policy_interface: Any | None = None,
        env: Any | None = None,
    ):
        if base_policy is None:
            raise ValueError(
                "FlowDAggerTrainer requires base_policy for inverse_noise_map. "
                "Pass the loaded RhoAlphaPolicy / pi0 policy from serve_policy.eval()."
            )
        self.config = config
        self.base_policy = base_policy
        self.prompt = prompt
        self.device = device
        self.policy_interface = policy_interface
        self.env = env

        self.experience_receiver = ExperienceReceiver(port=experience_port)
        self.param_publisher = ParamPublisher(port=param_port)

        in_channels = 3 * config.num_cameras
        self.encoder = ObsEncoder(
            encoder_type=config.encoder_type,
            in_channels=in_channels,
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            state_dim=config.state_dim if config.include_state else 0,
            norm_type=config.encoder_norm,
            use_spatial_softmax=config.use_spatial_softmax,
            softmax_temperature=config.softmax_temperature,
        ).to(device)

        self.actor = DeterministicNoisePolicy(
            obs_dim=config.obs_dim,
            noise_dim=config.noise_dim,
            hidden_dims=config.hidden_dims,
            use_layer_norm=config.use_layer_norm,
            magnitude=config.noise_magnitude,
        ).to(device)

        # Freeze the encoder when configured (right call for pretrained
        # backbones; the small-CNN default trains end-to-end).
        if config.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            optim_params = list(self.actor.parameters())
        else:
            optim_params = list(self.encoder.parameters()) + list(self.actor.parameters())
        self.optimizer = torch.optim.Adam(
            optim_params,
            lr=config.bc_lr,
            weight_decay=config.weight_decay,
        )

        # Two ring buffers, one per supervision source:
        #   intervention_buffer: noise targets recovered from operator-corrected
        #     actions via inverse_noise_map (the corrective signal).
        #   autonomous_buffer: implicit-approval noise targets from successful
        #     non-intervention transitions. Acts as an anti-drift anchor so the
        #     BC update doesn't overfit a tiny set of corrections and flip the
        #     policy away from its pretrained behavior.
        # Both store (images, state, noise) tuples in the same shape so the BC
        # step can concat samples from both and run a single MSE.
        self.intervention_buffer = _DAggerBuffer(
            capacity=config.buffer_capacity,
            image_keys=config.image_keys,
            image_size=config.image_size,
            state_dim=config.state_dim if config.include_state else 0,
            noise_dim=config.noise_dim,
        )
        self.autonomous_buffer = _DAggerBuffer(
            capacity=config.autonomous_buffer_capacity,
            image_keys=config.image_keys,
            image_size=config.image_size,
            state_dim=config.state_dim if config.include_state else 0,
            noise_dim=config.noise_dim,
        )

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Optional resume: load encoder/actor/optimizer + counters from a
        # prior run's checkpoint. Must come AFTER encoder/actor/optimizer
        # are constructed above, BEFORE the train loop starts.
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

        # Stats
        self.total_transitions = 0
        self.total_interventions = 0
        self.total_inversions = 0
        self.total_updates = 0
        self.total_episodes_committed = 0
        self.total_episodes_dropped = 0
        self.recent_bc_losses: list = []
        self.recent_inversion_mse: list = []
        self._interventions_since_last_update = 0

        # Per-episode pending lists. Transitions accumulate here as they
        # arrive; on episode success they're committed (interventions get
        # inverted, autonomous either stored directly or inverted depending
        # on config); on failure they're dropped.
        # Each item is a dict: {'images': dict, 'state': arr|None,
        #   'noise': arr (if direct-insert), 'processed_obs': dict (if needs inversion)}.
        self._pending_intervened: list = []
        self._pending_autonomous: list = []
        self._current_episode_id: int | None = None
        self._current_episode_saw_success: bool = False
        self._autonomous_subsample_counter: int = 0
        self._warned_missing_success: bool = False

        self._buffer_input_dumped = False
        self.running = False
        self._thread: threading.Thread | None = None

        # Eval block: operator sends ControlMessage(enter_eval / exit_eval)
        # from the robot to evaluate the current policy without contaminating
        # training. While in a block, transitions are tallied (SR / step count)
        # but skipped from buffering/inversion/updates. Toggling instantly
        # changes _in_eval_block — no waiting for the next transition. Tally
        # is per-block (resets on exit), no lifetime accumulation.
        self._in_eval_block: bool = False
        self._eval_current_episode_id: int | None = None
        self._eval_current_episode_steps: int = 0
        self._eval_block_episodes: int = 0
        self._eval_block_successes: int = 0
        self._eval_block_steps: list = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self.experience_receiver.start()
        self.param_publisher.start()
        self.running = True
        logger.info(
            f"[FlowDAggerTrainer] up | experience_port={self.experience_receiver.port} "
            f"param_port={self.param_publisher.port}"
        )

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Drop any unterminated pending episode — it has no success annotation,
        # so by policy we treat it as a failure and discard.
        if self._pending_intervened or self._pending_autonomous:
            self._pending_intervened.clear()
            self._pending_autonomous.clear()
            self.total_episodes_dropped += 1
        # Final checkpoint so a SIGTERM/KeyboardInterrupt doesn't drop the
        # trained noise policy. Wrapped in try/except so cleanup still runs
        # if disk-write fails (out of space, permission, etc.).
        try:
            self._save_checkpoint(f"final_step{self.total_updates}")
        except Exception as e:
            logger.warning(f"[FlowDAggerTrainer] final checkpoint save failed: {e}")
        self.experience_receiver.stop()
        self.param_publisher.stop()
        logger.info(
            f"[FlowDAggerTrainer] stopped | transitions={self.total_transitions} "
            f"interventions={self.total_interventions} updates={self.total_updates} "
            f"episodes_committed={self.total_episodes_committed} "
            f"episodes_dropped={self.total_episodes_dropped}"
        )

    # ── Checkpointing ───────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        """Persist encoder/actor/optimizer + run counters to log_dir.

        Buffer is intentionally not persisted: it can be 10GB+ in the
        steady state, and a small warm-up refill on resume is acceptable.
        """
        path = self.log_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_updates": self.total_updates,
                "total_transitions": self.total_transitions,
                "total_interventions": self.total_interventions,
                "total_inversions": self.total_inversions,
                "total_episodes_committed": self.total_episodes_committed,
                "total_episodes_dropped": self.total_episodes_dropped,
            },
            str(path),
        )
        logger.info(f"[FlowDAggerTrainer] saved checkpoint -> {path}")

    def load_checkpoint(self, path: str):
        """Restore encoder/actor/optimizer + counters from a prior run."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_updates = ckpt.get("total_updates", 0)
        self.total_transitions = ckpt.get("total_transitions", 0)
        self.total_interventions = ckpt.get("total_interventions", 0)
        self.total_inversions = ckpt.get("total_inversions", 0)
        self.total_episodes_committed = ckpt.get("total_episodes_committed", 0)
        self.total_episodes_dropped = ckpt.get("total_episodes_dropped", 0)
        logger.info(
            f"[FlowDAggerTrainer] resumed from {path} "
            f"(updates={self.total_updates}, transitions={self.total_transitions}, "
            f"interventions={self.total_interventions})"
        )

    # ── Transition processing ───────────────────────────────────────────────

    @staticmethod
    def _add_wire_batch_dim(obs_dict: dict[str, Any]) -> dict[str, Any]:
        """Match the inference client's leading batch dim on raw state keys.

        The inference client (robot_policy_base.prepare_obs) prepends a
        leading batch dim to joint_positions / ee_pos_quat before sending
        to the server, so env.process_input -> compute_eef_quat_poses
        iterates (N, 14) safely. The experience-streaming path ships raw
        1D arrays, so normalize here before invoking env.process_input.
        """
        obs_dict = dict(obs_dict)
        for k in ("joint_positions", "ee_pos_quat"):
            v = obs_dict.get(k)
            if isinstance(v, np.ndarray) and v.ndim == 1:
                obs_dict[k] = np.expand_dims(v, axis=0)
        return obs_dict

    def _process_obs_for_training(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """Mirror the inference preprocessing without touching the obs_queue.

        Applies env.process_input (BGR->RGB, dtype, ee_state synthesis) then
        PolicyInterface.remap_observation (key rename) then input_transforms
        (resize + dataset Normalize). Skips process_obs_queue because the
        queue is stateful across calls and would cross-contaminate buffer
        transitions; the noise encoder is single-step anyway.
        """
        pi = self.policy_interface
        obs = self.env.process_input(self._add_wire_batch_dim(obs_dict))
        obs = pi.remap_observation(obs)
        if getattr(pi, "input_transforms", None) is not None:
            obs = pi.input_transforms(obs)
        return obs

    # Kept for the inversion path (it expects the full pipeline including the
    # action-handling branches gated by process_action=True).
    def _process_obs_for_policy(self, obs_dict):
        processed = self.env.process_input(self._add_wire_batch_dim(obs_dict))
        processed = self.policy_interface.process_observation(processed, process_action=True)
        return processed

    @staticmethod
    def _to_chw_image(img_tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """Reduce an image of arbitrary shape to (3, target_size, target_size)."""
        t = img_tensor
        if t.ndim == 5:  # (B, T, C, H, W)
            t = t[:, -1]  # take latest timestep
        if t.ndim == 4:  # (B, C, H, W)
            t = t[0]  # single-step extraction
        elif t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)  # HWC -> CHW
        elif t.ndim != 3:
            raise ValueError(f"unexpected image shape {tuple(t.shape)}")
        t = t.float()
        if t.shape[-1] != target_size or t.shape[-2] != target_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    def _extract_images(self, processed_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        target = self.config.image_size
        out: dict[str, np.ndarray] = {}
        for key in self.config.image_keys:
            if key not in processed_obs:
                raise KeyError(
                    f"canonical image key '{key}' missing from processed obs "
                    f"(have: {list(processed_obs.keys())}). Check serve yaml's "
                    f"policy_interface_cfg.observation_mapping."
                )
            t = processed_obs[key]
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            chw = self._to_chw_image(t, target)
            out[key] = chw.detach().cpu().numpy().astype(np.float32)
        return out

    def _extract_state(self, processed_obs: dict[str, Any]) -> np.ndarray | None:
        if not self.config.include_state or self.config.state_dim == 0:
            return None
        for k in ("observation.state", "state"):
            if k in processed_obs:
                v = processed_obs[k]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                return v[: self.config.state_dim]
        return None

    def _prepare_obs_for_inversion(self, transition: Transition, action_to_invert: np.ndarray):
        """Run a single transition's obs through the inference preprocessing
        pipeline with `action_to_invert` injected as the target action.

        Same path as the legacy intervention-only flow, but parameterized on
        the action so we can also invert a policy-emitted `transition.action`
        (used when autonomous_target='invert_action').
        """
        flow_model = getattr(self.base_policy, "model", self.base_policy)
        flow_model = getattr(flow_model, "flow_model", flow_model)

        a_target = torch.from_numpy(np.asarray(action_to_invert, dtype=np.float32)).float().to(self.device)
        a_target = a_target * self.config.pre_norm_scale
        chunk_size = flow_model.config.chunk_size
        if a_target.ndim == 1:
            a_target = a_target.unsqueeze(0).unsqueeze(0).expand(1, chunk_size, -1)
        elif a_target.ndim == 2:
            a_target = a_target.unsqueeze(0)

        obs_dict = dict(transition.obs)
        obs_dict["action"] = a_target.cpu().numpy()
        return self._process_obs_for_policy(obs_dict)

    def _build_batched_inverse_inputs(self, processed_list: list[dict[str, Any]]):
        """Concatenate per-transition processed obs dicts along the batch dim
        and run the base policy's prepare_* methods on the batched result.

        Matches the JAX reference's _stack_observations + jnp.concatenate
        pattern (hil_dsrl_pi0/flow_matching_inverter.py:36, 994).
        """
        model = self.base_policy
        flow_model = getattr(model, "model", model)
        flow_model = getattr(flow_model, "flow_model", flow_model)

        # Stack each key across the batch dim. Tensors -> torch.cat along
        # dim 0; lists (e.g. "task") -> concatenated python list.
        batched: dict[str, Any] = {}
        for key in processed_list[0]:
            vals = [p[key] for p in processed_list if key in p]
            if not vals:
                continue
            if isinstance(vals[0], torch.Tensor):
                batched[key] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], list):
                merged = []
                for v in vals:
                    merged.extend(v)
                batched[key] = merged
            else:
                batched[key] = vals[0]

        batched = model.consolidate_images(batched)
        image, _ = model.prepare_image(batched)
        prompt = model.prepare_prompt(batched)
        state = model.prepare_state(batched)
        a_target = model.prepare_action(batched)
        return flow_model, image, prompt, state, a_target

    def _run_inversion(self, pending: list, dest_buffer: _DAggerBuffer):
        """Chunked batch-invert pending items into the given destination buffer.

        Each item is a dict with keys:
          'images': dict[str, np.ndarray]  (already extracted, ready for buffer)
          'state':  np.ndarray | None       (already extracted)
          'processed_obs': Dict             (action-conditioned obs for inverter)

        Inversion is grouped into chunks of cfg.inversion_batch_size; each
        chunk = one VLM forward + one denoise/inversion loop. Mirrors the
        JAX reference's _stack_observations + jnp.concatenate(targets) path.
        """
        if not pending:
            return
        cfg = self.config
        bs = max(1, cfg.inversion_batch_size)
        for start in range(0, len(pending), bs):
            chunk = pending[start : start + bs]
            try:
                flow_model, image, prompt, state, a_target = self._build_batched_inverse_inputs(
                    [item["processed_obs"] for item in chunk]
                )
                w_full, losses, _ = inverse_noise_map(
                    flow_model,
                    image,
                    prompt,
                    state,
                    a_target,
                    method=cfg.inversion_method,
                    n_restarts=cfg.inverse_map_restarts,
                    optimizer_steps=cfg.inverse_map_steps,
                    lr=cfg.inverse_map_lr,
                    b_W=cfg.inverse_map_b_W,
                )
            except Exception as e:
                logger.warning(f"batch inverse_noise_map failed (batch={len(chunk)}): {e}", exc_info=True)
                continue

            w_full = w_full.detach()
            losses = losses.detach()
            for i, item in enumerate(chunk):
                w_i = w_full[i, : cfg.noise_action_steps, : cfg.noise_action_dim]
                w_flat = w_i.cpu().numpy().reshape(-1)
                mse = float(losses[i].item()) if i < losses.numel() else float("nan")
                self.total_inversions += 1
                self.recent_inversion_mse.append(mse)
                if len(self.recent_inversion_mse) > 100:
                    self.recent_inversion_mse.pop(0)

                dest_buffer.add(item["images"], item["state"], w_flat)

                if not self._buffer_input_dumped:
                    from rho.policies.dsrl._obs_dump import dump_obs_once

                    dump_obs_once(
                        "trainer_buffer_input",
                        {
                            **{k: item["images"][k] for k in self.config.image_keys},
                            **({"state": item["state"]} if item["state"] is not None else {}),
                        },
                        image_keys=self.config.image_keys,
                        extra_fields={
                            "post_extract": True,
                            "inversion_batch_size": len(chunk),
                        },
                    )
                    self._buffer_input_dumped = True

    def _commit_episode(self, success: bool):
        """Commit (or drop) the current pending episode.

        Successful: intervention pending → inverted into intervention_buffer.
        Autonomous pending → either direct-insert (sampled_noise) or inverted
        (invert_action) into autonomous_buffer. Then bump the BC-trigger
        counter by the number of newly committed interventions.

        Failure: drop both pending lists entirely. No partial credit.
        """
        n_int = len(self._pending_intervened)
        n_auto = len(self._pending_autonomous)

        if not success:
            self._pending_intervened.clear()
            self._pending_autonomous.clear()
            self.total_episodes_dropped += 1
            logger.info(f"[FlowDAggerTrainer] dropped episode (intervened={n_int} autonomous={n_auto})")
            return

        # Interventions: always inverted.
        self._run_inversion(self._pending_intervened, self.intervention_buffer)
        self._interventions_since_last_update += n_int

        # Autonomous: depends on config.
        cfg = self.config
        if cfg.autonomous_target == "sampled_noise":
            for item in self._pending_autonomous:
                noise_arr = item.get("noise")
                if noise_arr is None:
                    continue  # robot didn't populate transition.noise for this step
                self.autonomous_buffer.add(item["images"], item["state"], noise_arr)
        elif cfg.autonomous_target == "invert_action":
            self._run_inversion(self._pending_autonomous, self.autonomous_buffer)
        else:
            raise ValueError(
                f"autonomous_target must be 'sampled_noise' or 'invert_action', got {cfg.autonomous_target!r}"
            )

        self._pending_intervened.clear()
        self._pending_autonomous.clear()
        self.total_episodes_committed += 1
        logger.info(
            f"[FlowDAggerTrainer] committed episode "
            f"(intervened={n_int} autonomous={n_auto}) "
            f"buf_int={self.intervention_buffer.size} buf_auto={self.autonomous_buffer.size}"
        )

    def _ingest(self, transition: Transition):
        self.total_transitions += 1

        # Eval block: state set instantly via ControlMessage handlers
        # (_enter_eval_block / _exit_eval_block). Here we just tally SR/steps
        # per episode and skip every downstream path — buffering, inversion,
        # update gating. BC updates pause as a side effect (no interventions
        # accrue).
        if self._in_eval_block:
            ep_id = getattr(transition, "episode_id", 0)
            if self._eval_current_episode_id != ep_id:
                self._eval_current_episode_id = ep_id
                self._eval_current_episode_steps = 0
            self._eval_current_episode_steps += 1

            if transition.done:
                # success is True → success; False/None → failure (None
                # buckets as failure, matching the normal commit path).
                success_bool = transition.success is True
                if success_bool:
                    steps = self._eval_current_episode_steps
                else:
                    # Failures are tallied at the rollout full length. Episodes
                    # only ever terminate naturally at max_steps, so a genuine
                    # failure already approx max_steps; this lets the operator
                    # discard an obviously-failing rollout early without dragging
                    # down eval avg_steps. Fall back to counted steps if the
                    # terminal did not carry a max_steps stamp.
                    max_steps = getattr(transition, "max_steps", 0) or 0
                    steps = max_steps if max_steps > 0 else self._eval_current_episode_steps
                self._eval_block_steps.append(steps)
                self._eval_block_episodes += 1
                if success_bool:
                    self._eval_block_successes += 1
                self._eval_current_episode_id = None
                self._eval_current_episode_steps = 0
            return

        # Episode-boundary detection. Robot stamps episode_id (time.time_ns)
        # on every transition. A change in episode_id without a prior
        # done=True,success=True commit means the previous episode was
        # interrupted (e-stop, process kill) — drop it as implicit failure.
        ep_id = getattr(transition, "episode_id", 0)
        if self._current_episode_id is None:
            self._current_episode_id = ep_id
            self._current_episode_saw_success = False
            self._autonomous_subsample_counter = 0
        elif ep_id != self._current_episode_id:
            # New episode arrived without us seeing the previous one's
            # terminal success. Drop the in-flight pending lists.
            if self._pending_intervened or self._pending_autonomous:
                self._commit_episode(success=False)
            self._current_episode_id = ep_id
            self._current_episode_saw_success = False
            self._autonomous_subsample_counter = 0

        # One-shot dump of the raw wire obs (pre any preprocessing).
        from rho.policies.dsrl._obs_dump import dump_obs_once

        dump_obs_once(
            "wire_transition",
            transition.obs,
            image_keys=list(transition.obs.keys()) if isinstance(transition.obs, dict) else None,
            extra_fields={
                "intervened": transition.intervened,
                "done": transition.done,
                "success": transition.success,
                "episode_id": ep_id,
                "action.shape": getattr(transition.action, "shape", None),
                "intervention_action.shape": (
                    getattr(transition.intervention_action, "shape", None)
                    if transition.intervention_action is not None
                    else None
                ),
                "config.image_keys": list(self.config.image_keys),
                "config.image_size": self.config.image_size,
                "config.state_dim": self.config.state_dim,
            },
        )

        # Partition into the right pending list. We do the obs preprocessing
        # once per kept transition here (not at commit time) — keeps the
        # pending list items small (extracted images at image_size, not the
        # raw wire obs) and avoids running env.process_input twice.
        cfg = self.config
        if transition.intervened:
            self.total_interventions += 1
            if transition.intervention_action is None:
                logger.warning("intervened transition has intervention_action=None; skipping")
            else:
                try:
                    processed = self._prepare_obs_for_inversion(transition, transition.intervention_action)
                    images = self._extract_images(processed)
                    state = self._extract_state(processed)
                    self._pending_intervened.append(
                        {
                            "images": images,
                            "state": state,
                            "processed_obs": processed,
                        }
                    )
                except (KeyError, ValueError) as e:
                    logger.warning(f"skipping intervention transition: {e}")
        else:
            # Autonomous transition. Subsample to bound memory + commit cost.
            self._autonomous_subsample_counter += 1
            if (
                cfg.autonomous_subsample_every <= 1
                or (self._autonomous_subsample_counter % cfg.autonomous_subsample_every) == 0
            ):
                try:
                    if cfg.autonomous_target == "sampled_noise":
                        # Cheap path: just extract images/state via the training
                        # preprocessing (no action injection, no inversion later).
                        # Skip silently if the robot didn't populate transition.noise.
                        if transition.noise is None:
                            pass
                        else:
                            processed_min = self._process_obs_for_training(transition.obs)
                            images = self._extract_images(processed_min)
                            state = self._extract_state(processed_min)
                            noise_arr = np.asarray(transition.noise, dtype=np.float32).reshape(-1)
                            noise_arr = noise_arr[: cfg.noise_dim]
                            if noise_arr.size < cfg.noise_dim:
                                # Robot's noise shape doesn't match config; pad with zeros.
                                padded = np.zeros(cfg.noise_dim, dtype=np.float32)
                                padded[: noise_arr.size] = noise_arr
                                noise_arr = padded
                            self._pending_autonomous.append(
                                {
                                    "images": images,
                                    "state": state,
                                    "noise": noise_arr,
                                }
                            )
                    else:
                        # Expensive path: invert transition.action at commit time.
                        processed = self._prepare_obs_for_inversion(transition, transition.action)
                        images = self._extract_images(processed)
                        state = self._extract_state(processed)
                        self._pending_autonomous.append(
                            {
                                "images": images,
                                "state": state,
                                "processed_obs": processed,
                            }
                        )
                except (KeyError, ValueError) as e:
                    logger.warning(f"skipping autonomous transition: {e}")

        # Terminal: commit or drop based on success.
        if transition.done:
            if transition.success is True:
                self._current_episode_saw_success = True
                self._commit_episode(success=True)
            else:
                if (
                    transition.success is None
                    and cfg.warn_on_missing_episode_signal
                    and not self._warned_missing_success
                ):
                    logger.warning(
                        "[FlowDAggerTrainer] terminal transition arrived with success=None — "
                        "this episode will be DROPPED. Verify the operator-side annotation flow "
                        "is wired up. (This warning fires once.)"
                    )
                    self._warned_missing_success = True
                self._commit_episode(success=False)
            # Either way, the next transition begins a fresh episode.
            self._current_episode_id = None

    # ── BC update ───────────────────────────────────────────────────────────

    def _sample_mixed_batch(self):
        """Draw a mixed batch from intervention + autonomous buffers.

        Composition is cfg.intervention_sample_ratio from intervention,
        the rest from autonomous. Falls back to interventions-only if the
        autonomous buffer is empty.
        Returns (images_dict, state_or_None, noise_target_arr).
        """
        cfg = self.config
        bsz = cfg.bc_batch_size
        if self.autonomous_buffer.size == 0:
            return self.intervention_buffer.sample(bsz)
        n_int = int(round(bsz * cfg.intervention_sample_ratio))
        n_int = (
            max(1, min(bsz - 1, n_int))
            if cfg.intervention_sample_ratio not in (0.0, 1.0)
            else (bsz if cfg.intervention_sample_ratio == 1.0 else 0)
        )
        n_auto = bsz - n_int

        if n_auto == 0:
            return self.intervention_buffer.sample(bsz)
        if n_int == 0:
            return self.autonomous_buffer.sample(bsz)

        i_images, i_state, i_w = self.intervention_buffer.sample(n_int)
        a_images, a_state, a_w = self.autonomous_buffer.sample(n_auto)

        images = {k: np.concatenate([i_images[k], a_images[k]], axis=0) for k in self.config.image_keys}
        if i_state is not None and a_state is not None:
            state = np.concatenate([i_state, a_state], axis=0)
        else:
            state = None
        w_target = np.concatenate([i_w, a_w], axis=0)
        return images, state, w_target

    def _bc_update_cycle(self):
        """One BC update cycle = cfg.bc_steps_per_update MSE steps + publish.

        Gated on the intervention buffer (autonomous-only training is useless —
        it would distill the policy onto its own stochastic noise samples
        without any corrective signal).
        """
        cfg = self.config
        if self.intervention_buffer.size < cfg.min_interventions_to_start:
            return

        if not cfg.freeze_encoder:
            self.encoder.train()
        self.actor.train()
        running_loss = 0.0
        for _ in range(cfg.bc_steps_per_update):
            images, state, w_target = self._sample_mixed_batch()

            cam_tensors = []
            for k in self.config.image_keys:
                imgs = torch.from_numpy(images[k]).to(self.device).float()  # (B, 3, H, W) already normalized
                cam_tensors.append(imgs)
            img_batch = torch.cat(cam_tensors, dim=1)  # (B, 3*num_cameras, H, W)
            state_batch = torch.from_numpy(state).to(self.device) if state is not None else None
            w_batch = torch.from_numpy(w_target).to(self.device)

            emb = self.encoder(img_batch, state=state_batch)
            pred = self.actor(emb)
            loss = F.mse_loss(pred, w_batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                clip_targets = list(self.actor.parameters())
                if not cfg.freeze_encoder:
                    clip_targets += list(self.encoder.parameters())
                nn.utils.clip_grad_norm_(clip_targets, cfg.max_grad_norm)
            self.optimizer.step()

            running_loss += loss.item()
            self.total_updates += 1
            if (self.total_updates % cfg.log_interval) == 0:
                logger.info(
                    f"[FlowDAggerTrainer] update={self.total_updates} "
                    f"loss={loss.item():.6f} "
                    f"buf_int={self.intervention_buffer.size} buf_auto={self.autonomous_buffer.size} "
                    f"interventions={self.total_interventions}"
                )
            if cfg.checkpoint_interval > 0 and (self.total_updates % cfg.checkpoint_interval) == 0:
                try:
                    self._save_checkpoint(f"step{self.total_updates}")
                except Exception as e:
                    logger.warning(f"[FlowDAggerTrainer] periodic checkpoint save failed: {e}")

        avg_loss = running_loss / cfg.bc_steps_per_update
        self.recent_bc_losses.append(avg_loss)
        if len(self.recent_bc_losses) > 50:
            self.recent_bc_losses.pop(0)

        if cfg.publish_after_each_update_cycle:
            self._publish_params()

    def _publish_params(self):
        params = {
            "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
            "actor": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            "update_step": self.total_updates,
        }
        self.param_publisher.publish(params)
        logger.debug(f"[FlowDAggerTrainer] published params @ update={self.total_updates}")

    # ── Control messages ───────────────────────────────────────────────────

    def _handle_control(self, msg: ControlMessage):
        """Dispatch operator commands received over the experience socket."""
        kind = msg.kind
        if kind == "enter_eval":
            self._enter_eval_block()
        elif kind == "exit_eval":
            self._exit_eval_block()
        elif kind == "save_checkpoint":
            tag = f"manual_step{self.total_updates}"
            try:
                self._save_checkpoint(tag)
            except Exception as e:
                logger.warning(f"[FlowDAggerTrainer] manual checkpoint save failed: {e}")
        else:
            logger.warning(f"[FlowDAggerTrainer] unknown ControlMessage kind: {kind!r}")

    def _enter_eval_block(self):
        if self._in_eval_block:
            return  # idempotent
        # In-flight non-eval episode (if any) is interrupted; drop it as a
        # failure so the drop counter stays honest and pending buffers don't
        # leak into the next training episode.
        if self._pending_intervened or self._pending_autonomous:
            self._commit_episode(success=False)
        self._current_episode_id = None
        self._in_eval_block = True
        self._eval_block_episodes = 0
        self._eval_block_successes = 0
        self._eval_block_steps = []
        self._eval_current_episode_id = None
        self._eval_current_episode_steps = 0
        logger.info("[FlowDAggerTrainer] entering eval block")

    def _exit_eval_block(self):
        if not self._in_eval_block:
            return  # idempotent
        if self._eval_current_episode_steps > 0:
            logger.warning(
                "[FlowDAggerTrainer] eval block exited mid-episode "
                f"({self._eval_current_episode_steps} steps abandoned); "
                "not counted in summary"
            )
        n = self._eval_block_episodes
        s = self._eval_block_successes
        if n == 0:
            logger.info("[FlowDAggerTrainer] eval block ended: no episodes completed")
        else:
            sr_pct = 100.0 * s / n
            avg_steps = sum(self._eval_block_steps) / len(self._eval_block_steps)
            logger.info(
                f"[FlowDAggerTrainer] eval block ended: SR={s}/{n} ({sr_pct:.1f}%) avg_steps={avg_steps:.1f}"
            )
        self._in_eval_block = False
        self._eval_block_episodes = 0
        self._eval_block_successes = 0
        self._eval_block_steps = []
        self._eval_current_episode_id = None
        self._eval_current_episode_steps = 0

    # ── Main loop ───────────────────────────────────────────────────────────

    def train(self):
        """Main loop: drain transitions, trigger BC update cycles.

        Episode-buffered: inversion + buffer commits now happen inside
        _commit_episode (driven by terminal transitions in _ingest), not on
        a per-transition timeout. The old timeout-flush logic is gone.
        """
        cfg = self.config
        last_log = time.monotonic()
        while self.running:
            batch = self.experience_receiver.receive_batch(max_batch=64, timeout_ms=100)
            for msg in batch:
                if isinstance(msg, ControlMessage):
                    self._handle_control(msg)
                else:
                    self._ingest(msg)

            if self._interventions_since_last_update >= cfg.update_every_n_interventions:
                self._bc_update_cycle()
                self._interventions_since_last_update = 0

            if cfg.max_training_steps > 0 and self.total_updates >= cfg.max_training_steps:
                logger.info(
                    f"[FlowDAggerTrainer] reached max_training_steps={cfg.max_training_steps}, stopping"
                )
                self.running = False
                break

            now = time.monotonic()
            if now - last_log > 10.0:
                last_log = now
                bc = f"{np.mean(self.recent_bc_losses):.5f}" if self.recent_bc_losses else "n/a"
                inv = f"{np.mean(self.recent_inversion_mse):.6f}" if self.recent_inversion_mse else "n/a"
                # Only surface the eval tally while a block is active.
                # Resets to nothing on exit (per-block, not lifetime).
                eval_str = (
                    f" eval={self._eval_block_successes}/{self._eval_block_episodes}"
                    if self._in_eval_block
                    else ""
                )
                logger.info(
                    f"[FlowDAggerTrainer] transitions={self.total_transitions} "
                    f"interventions={self.total_interventions} "
                    f"episodes={self.total_episodes_committed}/"
                    f"{self.total_episodes_committed + self.total_episodes_dropped} "
                    f"buf_int={self.intervention_buffer.size} buf_auto={self.autonomous_buffer.size} "
                    f"updates={self.total_updates} bc_loss~{bc} inv_mse~{inv}"
                    f"{eval_str}"
                )

    def train_in_background(self):
        self._thread = threading.Thread(target=self.train, daemon=True)
        self._thread.start()


def start_flowdagger_trainer(
    config: FlowDAggerConfig,
    base_policy: Any,
    prompt: str = "",
    experience_port: int = 5555,
    param_port: int = 5556,
    device: str = "cuda",
    policy_interface: Any | None = None,
    env: Any | None = None,
    blocking: bool = False,
) -> FlowDAggerTrainer:
    """Build, start, and (optionally) run a FlowDAggerTrainer."""
    trainer = FlowDAggerTrainer(
        config=config,
        base_policy=base_policy,
        prompt=prompt,
        experience_port=experience_port,
        param_port=param_port,
        device=device,
        policy_interface=policy_interface,
        env=env,
    )
    trainer.start()
    if blocking:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("FlowDAgger interrupted by user.")
        finally:
            trainer.stop()
    else:
        trainer.train_in_background()
    return trainer

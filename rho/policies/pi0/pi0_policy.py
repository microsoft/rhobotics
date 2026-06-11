import logging
from collections import deque
from pathlib import Path

import torch
from torch import Tensor

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES_PREFIX
from rho.common.constants import OBSERVATION_LANG as OBS_TASK
from rho.common.constants import OBSERVATION_STATE as OBS_STATE
from rho.policies.base import PreTrainedPolicy

from .configuration_pi0 import PI0Config
from .pi0_models import _PI0Pytorch
from .processing_pi0 import _SentencePieceTokenizer

logger = logging.getLogger(__name__)


class PI0Policy(PreTrainedPolicy):
    """PI0/PI05 wrapper compatible with `alku` training.

    Notes on compliance with `PolicyConfig`:
    - `feature_dict` provides the observation/action shapes; we pad to OpenPI's fixed 32D.
    - `dtype` controls model precision; OpenPI expects bf16/fp32.
    - `device` controls where tensors/models are placed.
    """

    config_class = PI0Config
    name = "pi0"

    def __init__(self, config: PI0Config, dataset_stats=None):
        super().__init__(config)
        import logging as _logging

        _log = _logging.getLogger(__name__)
        _log.info("[PI0Policy] __init__ started")
        config.validate_features()
        self.config = config
        self.device = torch.device(config.device)

        self._tokenizer = None
        if config.tokenizer_model_path is not None:
            _log.info(f"[PI0Policy] Loading tokenizer from {config.tokenizer_model_path}")
            self._tokenizer = _SentencePieceTokenizer(
                config.tokenizer_model_path,
                max_len=config.tokenizer_max_length,
            )
            _log.info("[PI0Policy] Tokenizer loaded.")

        _log.info("[PI0Policy] Creating _PI0Pytorch model...")
        self.model = _PI0Pytorch(config)
        _log.info("[PI0Policy] _PI0Pytorch created. Moving to device...")
        self.model.to(self.device)
        _log.info("[PI0Policy] Model on device.")

        if config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.n_action_steps = config.n_action_steps
        self._action_queue: deque[Tensor] = deque(maxlen=self.n_action_steps)
        self.reset()

    # ------------------------------------------------------------------
    # Checkpoint loading (supports openpi safetensors + standard .pt)
    # ------------------------------------------------------------------

    def load_from_pretrained(self, checkpoint_path):
        """Load pretrained weights into this PI0Policy.

        Supports:
        - Directory containing ``model.safetensors`` (openpi-converted checkpoint)
        - Single ``.safetensors`` file
        - Standard ``.pt`` / ``.pth`` checkpoint (delegates to base class)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Resolve directory → safetensors file inside it.
        if checkpoint_path.is_dir():
            safetensors_file = checkpoint_path / "model.safetensors"
            if safetensors_file.exists():
                checkpoint_path = safetensors_file
            else:
                raise FileNotFoundError(f"Directory {checkpoint_path} does not contain model.safetensors")

        # For .safetensors files, use safetensors.torch.load_model which
        # correctly restores tied weights (e.g. PaliGemma embed_tokens ↔ lm_head).
        if checkpoint_path.suffix == ".safetensors":
            import safetensors.torch

            logger.info(f"Loading safetensors checkpoint into inner model: {checkpoint_path}")
            safetensors.torch.load_model(self.model, str(checkpoint_path), device="cpu")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._post_checkpoint_load()

            if hasattr(self, "device"):
                self.to(self.device)
                logger.info(f"Pretrained weights loaded and moved to {self.device}")
            else:
                logger.info("Pretrained weights loaded successfully")
            return

        # Fall back to the base class for .pt / .pth files.
        super().load_from_pretrained(checkpoint_path)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.n_action_steps)

    def _pad_last_dim(self, x: torch.Tensor, new_dim: int) -> torch.Tensor:
        if x.shape[-1] == new_dim:
            return x
        if x.shape[-1] > new_dim:
            return x[..., :new_dim]
        pad_shape = list(x.shape)
        pad_shape[-1] = new_dim
        out = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        out[..., : x.shape[-1]] = x
        return out

    def _ensure_bchw(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 5 and img.shape[1] == 1:
            img = img[:, 0]
        return img

    def _to_minus1_1(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(torch.float32)
        mx = float(img.max()) if img.numel() else 0.0
        mn = float(img.min()) if img.numel() else 0.0
        if mx > 1.0:
            img = img / 255.0
            mx, mn = 1.0, 0.0
        if mn >= 0.0 and mx <= 1.0:
            img = img * 2.0 - 1.0
        return img

    def _build_openpi_image_dict(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Map phi image features -> OpenPI's fixed 3-view dict."""
        image_tensors: list[Tensor] = []
        image_masks: list[Tensor] = []

        img_keys = list(self.config.image_features.keys())
        img_keys.sort()
        if len(img_keys) == 0 or not any(k in batch for k in img_keys):
            # Fallback: environment rollouts may provide a single `observation.image` key
            # regardless of the training feature_dict.
            img_keys = sorted([k for k in batch if str(k).startswith(OBS_IMAGES_PREFIX)])

        for k in img_keys:
            if k not in batch:
                continue
            img = self._ensure_bchw(batch[k]).to(self.device)
            img = self._to_minus1_1(img)
            image_tensors.append(img)

            is_pad_key = f"{k}_is_pad"
            if is_pad_key in batch:
                mask = (~batch[is_pad_key].to(torch.bool)).to(self.device)
                if mask.ndim > 1:
                    mask = mask[:, -1]
            else:
                mask = torch.ones(img.shape[0], dtype=torch.bool, device=self.device)
            image_masks.append(mask)

        if len(image_tensors) == 0:
            raise ValueError(
                f"No image tensors found in batch. Expected keys starting with '{OBS_IMAGES_PREFIX}'."
            )

        while len(image_tensors) < 3:
            pad_img = torch.full_like(image_tensors[0], -1.0)
            pad_mask = torch.zeros_like(image_masks[0])
            image_tensors.append(pad_img)
            image_masks.append(pad_mask)
        if len(image_tensors) > 3:
            image_tensors = image_tensors[:3]
            image_masks = image_masks[:3]

        images = {
            "base_0_rgb": image_tensors[0],
            "left_wrist_0_rgb": image_tensors[1],
            "right_wrist_0_rgb": image_tensors[2],
        }
        masks = {
            "base_0_rgb": image_masks[0],
            "left_wrist_0_rgb": image_masks[1],
            "right_wrist_0_rgb": image_masks[2],
        }
        return images, masks

    def _get_tokenizer(self) -> _SentencePieceTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        raise ValueError(
            "PI0Policy requires tokenization but `PI0Config.tokenizer_model_path` is not set. "
            "Provide a path to the PaliGemma sentencepiece model (paligemma_tokenizer.model)."
        )

    def _tokenize_batch(
        self, prompts: list[str], *, state_for_pi05: torch.Tensor | None
    ) -> tuple[Tensor, Tensor]:
        tok = self._get_tokenizer()
        tokens = []
        masks = []
        for i, p in enumerate(prompts):
            st = state_for_pi05[i] if (state_for_pi05 is not None) else None
            t, m = tok.tokenize(p, state=st)
            tokens.append(t)
            masks.append(m)
        tokens_t = torch.stack(tokens, dim=0).to(self.device)
        masks_t = torch.stack(masks, dim=0).to(self.device)
        return tokens_t, masks_t

    def _make_observation(self, batch: dict[str, Tensor]):
        images, image_masks = self._build_openpi_image_dict(batch)

        state = batch[OBS_STATE]
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        state = state.to(self.device)
        state = self._pad_last_dim(state, self.config.max_state_dim)

        prompts = batch.get(OBS_TASK)
        if prompts is None:
            prompts = [""] * state.shape[0]
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.tolist()
        if not isinstance(prompts, (list, tuple)):
            prompts = [str(prompts)]

        state_for_pi05 = state if self.config.pi05 else None
        tokenized_prompt, tokenized_prompt_mask = self._tokenize_batch(
            [str(p) for p in prompts],
            state_for_pi05=state_for_pi05,
        )

        class _Obs:
            pass

        obs = _Obs()
        obs.images = images
        obs.image_masks = image_masks
        obs.state = state
        obs.tokenized_prompt = tokenized_prompt
        obs.tokenized_prompt_mask = tokenized_prompt_mask
        return obs

    # ------------------------------------------------------------------
    # FlowDAgger / DSRL trainer adapter methods
    #
    # rho.hil.trainers.flowdagger_trainer threads transitions through these
    # methods on the base policy to build the batched inputs for
    # inverse_noise_map. RhoAlphaPolicy exposes the same surface; we match it
    # here so the trainer is base-policy-agnostic.
    # ------------------------------------------------------------------

    def consolidate_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """No-op for pi0.

        The rhoalpha trainer pipeline stages a ``consolidate_images`` step that
        stacks per-camera image tensors under a single ``OBS_IMAGES`` key so the
        downstream ``prepare_image`` can index by batch x cam. Pi0 doesn't need
        that intermediate -- ``prepare_image`` reads the per-camera keys
        directly via ``_build_openpi_image_dict``. Return the batch unchanged.
        """
        return batch

    def prepare_image(self, batch: dict[str, Tensor]):
        """Returns ``((images_list, masks_list), None)`` for the inverter chain.

        The flowdagger trainer unpacks ``image, _ = model.prepare_image(batch)``
        (mirroring rhoalpha's ``(images, image_mask)`` two-tuple). Pi0 needs the
        per-camera masks alongside the images, so we bundle both into the first
        slot and leave the second as ``None``. ``_PI0Pytorch.get_image_text_hidden_state``
        unpacks ``image = (images_list, masks_list)``.

        ``images_list`` is a list of (B, 3, H, W) tensors (one per OpenPI camera
        slot, padded to 3 with -1 sentinels if fewer cameras). ``masks_list``
        mirrors with (B,) bool tensors.
        """
        images, masks = self._build_openpi_image_dict(batch)
        return (list(images.values()), list(masks.values())), None

    def prepare_prompt(self, batch: dict[str, Tensor]):
        """Returns (lang_tokens, lang_masks) ready for ``embed_prefix``.

        Tokenizes ``batch["task"]`` with the PaliGemma sentencepiece tokenizer.
        For pi05, the prompt prefix carries the state via tokenizer side input.
        """
        state = batch.get(OBS_STATE)
        if state is not None:
            if state.ndim == 3 and state.shape[1] == 1:
                state = state[:, 0]
            state = state.to(self.device)
            state = self._pad_last_dim(state, self.config.max_state_dim)

        prompts = batch.get(OBS_TASK)
        if prompts is None:
            bsize = state.shape[0] if state is not None else 1
            prompts = [""] * bsize
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.tolist()
        if not isinstance(prompts, (list, tuple)):
            prompts = [str(prompts)]

        state_for_pi05 = state if (self.config.pi05 and state is not None) else None
        return self._tokenize_batch([str(p) for p in prompts], state_for_pi05=state_for_pi05)

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns the robot state padded to ``max_state_dim`` and on device."""
        state = batch[OBS_STATE]
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        state = state.to(self.device)
        return self._pad_last_dim(state, self.config.max_state_dim)

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns the target action chunk padded to ``max_action_dim``.

        Trainer passes the operator-corrected action under ``ACTION``. Pi0 needs
        chunk shape (B, action_horizon, max_action_dim); upstream broadcast in
        flowdagger_trainer._process_obs_for_inversion already shaped it as
        (1, chunk_size, action_dim).
        """
        actions = batch[ACTION].to(self.device)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        return self._pad_last_dim(actions, self.config.max_action_dim)

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, float]]:
        self.train()
        batch = dict(batch)

        obs = self._make_observation(batch)

        actions = batch[ACTION]
        actions = actions.to(self.device)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        actions = self._pad_last_dim(actions, self.config.max_action_dim)
        if actions.shape[1] != self.config.action_horizon:
            raise ValueError(
                f"Expected action horizon {self.config.action_horizon}, got {actions.shape[1]}. "
                "Make sure your dataset is configured with delta_timestamps via "
                "PolicyConfig.action_delta_indices."
            )

        losses = self.model.forward(obs, actions, noise=noise, time=time)

        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            in_episode = (~actions_is_pad.to(torch.bool)).to(self.device)
            losses = losses * in_episode.unsqueeze(-1)

        original_action_dim = self.config.action_feature.shape[-1]
        losses = losses[:, :, :original_action_dim]

        loss = losses.mean()
        return loss, {"l2_loss": float(loss.item())}

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return self.forward(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        batch = dict(batch)

        if len(self._action_queue) == 0:
            obs = self._make_observation(batch)
            actions = self.model.sample_actions(
                self.device,
                obs,
                noise=noise,
                num_steps=self.config.num_inference_steps,
            )
            actions = actions[:, : self.config.n_action_steps, :]
            original_action_dim = self.config.action_feature.shape[-1]
            actions = actions[:, :, :original_action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> dict[str, Tensor]:
        """Sample an action chunk for environment rollouts.

        `GymEnvironment.evaluate_policy()` expects this method to return a dict with an
        `'actions'` key containing a sequence of actions (optionally batched).
        """
        self.eval()
        batch = dict(batch)

        obs = self._make_observation(batch)
        actions = self.model.sample_actions(
            self.device,
            obs,
            noise=noise,
            num_steps=self.config.num_inference_steps,
        )

        # Keep only the original (un-padded) action dims.
        original_action_dim = self.config.action_feature.shape[-1]
        actions = actions[:, :, :original_action_dim]

        # Environment uses chunked execution; prefer returning up to chunk_size if configured.
        chunk_size = getattr(self.config, "chunk_size", None)
        if isinstance(chunk_size, int) and chunk_size > 0:
            actions = actions[:, :chunk_size, :]

        return {"actions": actions}

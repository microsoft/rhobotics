"""
Rho flow-matching policy with shared rollout / preparation helpers.

Contains:
- ``_apply_action_padding_masks``: private action-padding utility
- ``RhoPolicy``: Rho flow-matching policy for public training/inference,
  with consolidate_images / prepare_image / prepare_state / prepare_action /
  prepare_prompt / select_action / sample_actions / sample_actions_rtc helpers.
"""

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rho.common.constants import ACTION
from rho.common.constants import OBSERVATION_ENVIRONMENT_STATE as OBS_ENV
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES
from rho.common.constants import OBSERVATION_LANG as OBS_TASK
from rho.common.constants import OBSERVATION_STATE as OBS_ROBOT
from rho.common.transforms import build_key_padding_transform, get_target_sequence_lengths
from rho.common.types import FeatureType
from rho.policies.base import PreTrainedPolicy, populate_queues
from rho.policies.rho.backbone import create_backbone_adapter
from rho.policies.rho.configuration_rho import RhoConfig
from rho.policies.rho.rho_model import RhoModel
from rho.policies.rho.rho_processor import RhoProcessor

logger = logging.getLogger(__name__)

OBS_IMAGES_IS_PAD = f"{OBS_IMAGES}_is_pad"
_SIGLIP2_LEGACY_CHECKPOINT_PREFIX = ".vision_tower.vision_tower.vision_model."
_SIGLIP2_CURRENT_CHECKPOINT_PREFIX = ".vision_tower.vision_tower."


def _remap_siglip2_checkpoint_keys(state_dict: dict, *, uses_nested_vision_model: bool) -> dict:
    """Adapt SigLIP2 checkpoint keys to the installed Transformers module layout."""
    if uses_nested_vision_model:
        return state_dict
    return {
        key.replace(_SIGLIP2_LEGACY_CHECKPOINT_PREFIX, _SIGLIP2_CURRENT_CHECKPOINT_PREFIX): value
        for key, value in state_dict.items()
    }


@dataclass(frozen=True)
class _PreparedModelInputs:
    image: Tensor
    image_mask: Tensor
    state: Tensor
    prompt: list[str]
    action: Tensor | None = None


RHOALPHA_EXTRA_PREFIXES = (
    "model.fast_model.",
    "model.knowledge_insulation_fast_model.",
    "model.knowledge_insulation_endstate_model.",
    "model.text_output_model.",
)
LEGACY_FLOW_COMPONENTS = (
    "vlm_backbone.",
    "vlm_projector.",
    "state_projector.",
    "action_in_proj.",
    "action_head.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "time_mlp_in.",
    "time_mlp_out.",
    "action_expert.",
)


# ---------------------------------------------------------------------------
# Private action-padding helper
# ---------------------------------------------------------------------------
def _apply_action_padding_masks(
    action_loss: Tensor,
    action_is_pad: Tensor | None = None,
    action_dim_is_pad: Tensor | None = None,
) -> Tensor:
    """Zero padded time steps and padded action dimensions without changing loss shape."""
    if action_is_pad is not None:
        action_is_pad = action_is_pad.to(device=action_loss.device, dtype=torch.bool)
        if action_is_pad.ndim == 1:
            action_is_pad = action_is_pad.unsqueeze(0)
        action_loss = action_loss * (~action_is_pad).unsqueeze(-1)
    if action_dim_is_pad is not None:
        action_dim_is_pad = action_dim_is_pad.to(device=action_loss.device, dtype=torch.bool)
        if action_dim_is_pad.ndim == 1:
            action_dim_is_pad = action_dim_is_pad.unsqueeze(0)
        loss_dim = action_loss.shape[-1]
        mask_dim = action_dim_is_pad.shape[-1]
        if mask_dim < loss_dim:
            action_dim_is_pad = F.pad(action_dim_is_pad, (0, loss_dim - mask_dim), value=True)
        elif mask_dim > loss_dim:
            action_dim_is_pad = action_dim_is_pad[..., :loss_dim]
        action_loss = action_loss * (~action_dim_is_pad).unsqueeze(1)
    return action_loss


# ---------------------------------------------------------------------------
# Checkpoint conversion
# ---------------------------------------------------------------------------
def convert_rhoalpha_state_dict(state_dict: dict) -> dict:
    """Extract Rho flow weights from current or legacy RhoAlpha layouts."""
    has_flow = any(key.startswith("model.flow_model.") for key in state_dict)
    ki_prefix = "model.knowledge_insulation_endstate_model.flow_model."
    has_ki_flow = any(key.startswith(ki_prefix) for key in state_dict)

    if has_flow:
        converted = {
            key: value for key, value in state_dict.items() if not key.startswith(RHOALPHA_EXTRA_PREFIXES)
        }
    elif has_ki_flow:
        converted = {
            f"model.flow_model.{key.removeprefix(ki_prefix)}": value
            for key, value in state_dict.items()
            if key.startswith(ki_prefix)
        }
    else:
        converted = {}
        for key, value in state_dict.items():
            if not key.startswith("model."):
                converted[key] = value
                continue
            component = key.removeprefix("model.")
            if component.startswith(LEGACY_FLOW_COMPONENTS):
                converted[f"model.flow_model.{component}"] = value

    dropped = len(state_dict) - len(converted)
    if dropped:
        logger.info("Dropped %d non-Rho keys while converting RhoAlpha weights", dropped)
    return converted


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class RhoPolicy(PreTrainedPolicy):
    """Rho flow-matching policy for public training and inference."""

    config_class = RhoConfig
    name = "rho"

    def __init__(
        self,
        config: RhoConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        del dataset_stats
        super().__init__(config)

        self.model = RhoModel(config)
        self.model.print_freezing_status()
        self._has_uninitialized_backbone = getattr(
            self.model.flow_model._backend,
            "_backbone_has_uninitialized_weights",
            False,
        )
        self._backend = create_backbone_adapter(config)
        self.n_action_steps = config.n_action_steps
        self._queues = None
        inference_padding_features = {
            key: feature
            for key, feature in config.feature_dict.items()
            if feature.type != FeatureType.VISUAL and key != ACTION
        }
        self.input_padding_transform = build_key_padding_transform(
            inference_padding_features,
            target_sequence_lengths=get_target_sequence_lengths(config.delta_indices_dict),
        )
        self.processor = RhoProcessor(config)
        self.reset()

    # ------------------------------------------------------------------
    # Reset / queues
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._queues = {
            OBS_ROBOT: deque(maxlen=self.config.n_obs_steps),
            OBS_TASK: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
            self._queues[OBS_IMAGES_IS_PAD] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV] = deque(maxlen=self.config.n_obs_steps)

    def get_optim_params(self):
        return self.parameters()

    def get_named_param_groups(self) -> list[dict] | None:
        action_expert_lr = self.config.optimizer_lr_action_expert
        if action_expert_lr is None:
            return None

        vlm_params = []
        expert_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "vlm_backbone" in name:
                vlm_params.append(parameter)
            else:
                expert_params.append(parameter)

        groups = []
        if vlm_params:
            groups.append(
                {
                    "name": "vlm",
                    "params": vlm_params,
                    "lr": float(self.config.optimizer_lr),
                }
            )
        if expert_params:
            groups.append(
                {
                    "name": "action_expert",
                    "params": expert_params,
                    "lr": float(action_expert_lr),
                }
            )
        return groups or None

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _remap_checkpoint_keys(self, state_dict: dict) -> dict:
        state_dict = convert_rhoalpha_state_dict(state_dict)
        flow_model = getattr(getattr(self, "model", None), "flow_model", None)
        vision_tower = getattr(
            getattr(getattr(flow_model, "vlm_backbone", None), "model", None), "vision_tower", None
        )
        inner_vision_model = getattr(vision_tower, "vision_tower", None)
        return _remap_siglip2_checkpoint_keys(
            state_dict,
            uses_nested_vision_model=hasattr(inner_vision_model, "vision_model"),
        )

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        state_dict = self._remap_checkpoint_keys(state_dict)
        # A checkpoint may carry a folded noise sampler (see rho.policies.rho.noise_policy). Loading it
        # with `noise_policy` unset is a legitimate request - it means "run this checkpoint the standard
        # Gaussian way" - so drop those tensors instead of failing the strict load on unexpected keys.
        if getattr(getattr(self, "model", None), "flow_model", None) is not None and (
            getattr(self.model.flow_model, "noise_policy", None) is None
        ):
            dropped = [k for k in state_dict if ".noise_policy." in k]
            if dropped:
                state_dict = {k: v for k, v in state_dict.items() if ".noise_policy." not in k}
                logger.info(
                    "Ignoring %d folded noise-policy tensors: config.noise_policy is unset, so this "
                    "checkpoint runs with Gaussian noise (standard evaluation).",
                    len(dropped),
                )
        # The mirror case: `noise_policy` is configured but the checkpoint predates it, which is
        # how online adaptation starts -- a freshly initialised noise policy on an existing
        # policy. Seed those entries from the live module so the strict load still holds
        # everything else to account.
        flow_model = getattr(getattr(self, "model", None), "flow_model", None)
        if getattr(flow_model, "noise_policy", None) is not None and not any(
            ".noise_policy." in k for k in state_dict
        ):
            own = {k: v for k, v in self.state_dict().items() if ".noise_policy." in k}
            state_dict = {**state_dict, **own}
            logger.info(
                "Checkpoint carries no noise policy; initialising %d tensors for the configured "
                "%r sampler.",
                len(own),
                getattr(flow_model.config, "noise_policy", None),
            )

        result = torch.nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
        self._has_uninitialized_backbone = False
        return result

    def _remap_backwards_compatible_keys(self, state_dict: dict) -> dict:
        return self._remap_checkpoint_keys(state_dict)

    # ------------------------------------------------------------------
    # Image consolidation
    # ------------------------------------------------------------------
    def _validate_required_observations(self, batch: dict) -> None:
        self.processor.validate_required_observations(batch)

    def consolidate_images(self, batch: dict) -> dict:
        return self.processor.consolidate_images(batch)

    # ------------------------------------------------------------------
    # prepare_image / prepare_state / prepare_action / prepare_prompt
    # ------------------------------------------------------------------
    def prepare_image(self, batch: dict) -> tuple[Tensor, Tensor]:
        return self.processor.prepare_image(batch)

    def prepare_prompt(self, batch: dict):
        if OBS_TASK in batch and isinstance(batch[OBS_TASK], torch.Tensor):
            from rho.common.task_encoding import decode_task_bytes

            batch = {**batch, OBS_TASK: decode_task_bytes(batch[OBS_TASK])}
        return self._backend.prepare_prompt(batch)

    def prepare_state(self, batch: dict):
        return self.processor.prepare_state(batch)

    def prepare_action(self, batch: dict):
        return self.processor.prepare_action(batch)

    def pad_vector(self, vector: Tensor, new_dim: int) -> Tensor:
        return self.processor.pad_vector(vector, new_dim)

    def _prepare_model_inputs(self, batch: dict, *, include_action: bool = False) -> _PreparedModelInputs:
        processed = self.processor.prepare(batch, include_action=include_action)
        return _PreparedModelInputs(
            image=processed.image,
            image_mask=processed.image_mask,
            state=processed.state,
            prompt=self.prepare_prompt(batch),
            action=processed.action,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._validate_required_observations(batch)
        batch_size = 1
        if isinstance(batch[OBS_TASK], (list, tuple, torch.Tensor, np.ndarray)):
            batch_size = len(batch[OBS_TASK])
        batch = self.input_padding_transform(batch, batch_size=batch_size, device=self.device)
        batch = dict(batch)
        batch = self.consolidate_images(batch)

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues["action"]) == 0:
            batch = {
                OBS_IMAGES: torch.cat(list(self._queues[OBS_IMAGES]), dim=1),
                OBS_IMAGES_IS_PAD: torch.cat(list(self._queues[OBS_IMAGES_IS_PAD]), dim=1),
                OBS_ROBOT: torch.stack(list(self._queues[OBS_ROBOT]), dim=1),
                OBS_TASK: self._queues[OBS_TASK][0],
            }
            inputs = self._prepare_model_inputs(batch)
            actions = self.model.sample_actions(
                inputs.image,
                inputs.prompt,
                inputs.state,
                noise=noise,
                image_mask=inputs.image_mask,
            )
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, : self.config.n_action_steps, :original_action_dim]
            self._queues["action"].extend(actions.transpose(0, 1))

        return self._queues["action"].popleft()

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        inputs = self._prepare_model_inputs(batch)
        actions = self.model.sample_actions(
            inputs.image,
            inputs.prompt,
            inputs.state,
            noise=noise,
            image_mask=inputs.image_mask,
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return {"actions": actions, "tactile_action": None}

    @torch.no_grad()
    def sample_actions_rtc(
        self,
        batch: dict[str, Tensor],
        inference_delay: int,
        execution_horizon: int,
        prev_actions: Tensor | None = None,
        noise: Tensor | None = None,
        beta: float = 40.0,
        guidance_schedule: str = "paper",
    ) -> Tensor:
        if prev_actions is None:
            return self.sample_actions(batch, noise=noise)

        self.eval()
        inputs = self._prepare_model_inputs(batch)

        actions = self.model.sample_actions_rtc(
            inputs.image,
            inputs.prompt,
            inputs.state,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            prev_actions=prev_actions,
            noise=noise,
            image_mask=inputs.image_mask,
            beta=beta,
            guidance_schedule=guidance_schedule,
        )
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return {"actions": actions, "tactile_action": None}

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        if self._has_uninitialized_backbone:
            raise RuntimeError(
                "Phi5 has uninitialized weights. Load a Rho/RhoAlpha checkpoint or "
                f"install model weights at {self.config.vlm_backbone_folder}."
            )

        batch = dict(batch)
        inputs = self._prepare_model_inputs(batch, include_action=True)
        if inputs.action is None:
            raise RuntimeError("Internal error: training inputs did not include actions.")
        actions_is_pad = batch.get("action_is_pad")
        action_dim_is_pad = batch.get("action_dim_is_pad")

        losses = self.model(
            inputs.image,
            inputs.prompt,
            inputs.state,
            inputs.action,
            noise=noise,
            time=time,
            image_mask=inputs.image_mask,
        )
        losses = _apply_action_padding_masks(losses, actions_is_pad, action_dim_is_pad)
        losses = losses[:, :, : self.config.max_action_dim]
        loss = losses.mean()
        loss_dict = {"l2_loss": loss.item()}
        loss_dict.update(self.model.flow_model._last_hidden_state_stats)
        return loss, loss_dict

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return self.forward(batch)

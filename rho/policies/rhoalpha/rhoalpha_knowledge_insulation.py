"""
Phi4MM Knowledge Insulation Model.

Implements dual-path training with:
- Flow matching path with stop gradient (no VLM updates)
- Autoregressive path with full gradients (VLM updates)

This allows the model to learn both continuous and discrete action prediction
while preventing the flow matching objective from affecting VLM parameters.
"""

import logging

from torch import Tensor

from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_fast import RhoAlphaFASTModel
from rho.policies.rhoalpha.rhoalpha_flow import RhoAlphaFlowMatchingModel
from rho.policies.rhoalpha.rhoalpha_model import RhoAlphaModel

logger = logging.getLogger(__name__)


class RhoAlphaKnowledgeInsulationModel(RhoAlphaModel):
    """
    Knowledge Insulation model combining flow matching and autoregressive.

    Runs both training paths:
    - Flow matching with detached VLM features (stop gradient)
    - Autoregressive with full gradients

    This allows the autoregressive path to update the VLM while the
    flow matching path benefits from VLM features without affecting them.

    Note: Uses weight tying to get FA2 efficiency for flow + SDPA compatibility for AR:
    - Flow model uses flash_attention_2 VLM (3x more memory efficient)
    - AR model uses SDPA VLM (required for prefix-LM mask)
    - Weights are tied so both VLMs share the same parameters
    """

    def __init__(self, config: RhoAlphaConfig):
        # Create parent with flash_attention_2 for flow model
        original_attn = config.attention_implementation
        config.attention_implementation = "flash_attention_2"
        super().__init__(config)

        # Flow model shares FA2 VLM
        flow_shared_components = {
            "vlm_backbone": self.vlm_backbone,
            "vlm_processor": self.vlm_processor,
            "generation_config": self.generation_config,
            "vlm_projector": self.vlm_projector,
            "state_projector": self.state_projector,
        }
        self.flow_model = RhoAlphaFlowMatchingModel(config, **flow_shared_components)

        # Create AR model with SDPA (shares processor/projectors, NOT vlm_backbone)
        config.attention_implementation = "sdpa"
        fast_shared_components = {
            "vlm_processor": self.vlm_processor,
            "generation_config": self.generation_config,
            "vlm_projector": self.vlm_projector,
            "state_projector": self.state_projector,
        }
        self.fast_model = RhoAlphaFASTModel(config, **fast_shared_components)

        # Tie weights: simple approach using load_state_dict with references
        logger.info("Tying VLM weights between flow (FA2) and AR (SDPA)...")
        self._tie_weights_simple()

        # Restore original attention setting
        config.attention_implementation = original_attn

        # Knowledge insulation alpha parameter (weight for flow loss)
        self.alpha = getattr(config, "knowledge_insulation_alpha", 1.0)

        self.print_freezing_status()

    def _tie_weights_simple(self):
        """Tie weights by making AR VLM parameters reference flow VLM parameters."""

        flow_vlm = self.vlm_backbone
        ar_vlm = self.fast_model.vlm_backbone

        # Load flow weights into AR VLM first (copies values)
        ar_vlm.load_state_dict(flow_vlm.state_dict())

        # Now tie parameters by replacing in _parameters dict
        tied = 0
        for name, flow_param in flow_vlm.named_parameters():
            parts = name.split(".")

            # Navigate to parent module
            module = ar_vlm
            for part in parts[:-1]:
                module = module[int(part)] if part.isdigit() else getattr(module, part)

            # Replace parameter in _parameters dict
            param_name = parts[-1]
            if hasattr(module, "_parameters") and param_name in module._parameters:
                module._parameters[param_name] = flow_param
                tied += 1

        # Also tie buffers (e.g. running stats)
        for name, flow_buffer in flow_vlm.named_buffers():
            parts = name.split(".")
            module = ar_vlm
            for part in parts[:-1]:
                module = module[int(part)] if part.isdigit() else getattr(module, part)

            buffer_name = parts[-1]
            if hasattr(module, "_buffers") and buffer_name in module._buffers:
                module._buffers[buffer_name] = flow_buffer

        # Verify tying
        flow_params = list(flow_vlm.parameters())
        ar_params = list(ar_vlm.parameters())
        shared = sum(1 for f, a in zip(flow_params, ar_params, strict=False) if f.data_ptr() == a.data_ptr())
        logger.info(f"Tied {shared}/{len(flow_params)} VLM parameters")

    def forward(
        self,
        image,
        prompt,
        state,
        actions,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=None,
    ):
        """
        Forward pass with knowledge insulation.

        Runs both paths:
        - Flow matching with detached VLM features (stop gradient)
        - Autoregressive with full gradients

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            actions: Ground truth actions
            noise: Optional noise for flow matching
            time: Optional time for flow matching
            image_mask: Optional image masking
            training_mode: Training mode (ignored, always runs both paths)

        Returns:
            Combined loss
        """
        # Compute VLM features once for flow path
        # (Fast model has different architecture and can't reuse these)
        vlm_features, vlm_mask = self.get_image_text_hidden_state(image, prompt, image_mask=image_mask)

        # Flow matching path with stop gradient (no VLM updates)
        # Detach features to prevent gradients from flowing back to VLM
        vlm_features_detached = vlm_features.detach()
        vlm_mask_detached = vlm_mask.detach() if vlm_mask is not None else None

        flow_losses = self.flow_model.forward(
            image,
            prompt,
            state,
            actions,
            noise=noise,
            time=time,
            image_mask=image_mask,
            precomputed_hidden_state=(vlm_features_detached, vlm_mask_detached),
        )
        flow_loss = flow_losses.mean()

        # Autoregressive path with full gradients (VLM updates)
        # Fast model uses its own VLM forward pass with custom masking
        ar_loss = self.fast_model.forward(
            image, prompt, state, actions, noise=None, time=None, image_mask=image_mask
        )

        # Combine losses (Eq. 4 from knowledge insulation paper)
        total_loss = ar_loss + self.alpha * flow_loss

        return total_loss

    def sample_actions(
        self, image, prompt, state, noise=None, image_mask=None, inference_mode="flow"
    ) -> Tensor:
        """
        Sample actions using either flow matching or autoregressive.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            noise: Optional noise for flow matching
            image_mask: Optional image masking
            inference_mode: Which model to use ("flow" or "fast")

        Returns:
            Sampled actions
        """
        if inference_mode == "flow":
            return self.flow_model.sample_actions(image, prompt, state, noise=noise, image_mask=image_mask)
        elif inference_mode == "fast":
            return self.fast_model.sample_actions(image, prompt, state, noise=None, image_mask=image_mask)
        else:
            raise ValueError(f"Unknown inference mode: {inference_mode}. Use 'flow' or 'fast'.")

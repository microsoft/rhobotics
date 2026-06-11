"""
Phi4MM Unified Robotics Model.

Implements a unified model that composes specialized models based on training modes:
- ROBOT_FLOWMATCH: RhoAlphaFlowMatchingModel
- ROBOT_AUTOREGRESSIVE: RhoAlphaFASTModel
- ROBOT_KNOWLEDGE_INSULATION: RhoAlphaKnowledgeInsulationModel
- VQA, BOUNDING_BOX, POINTING: RhoAlphaWebModel

The model instantiates the appropriate models based on config.training_modes and
routes forward/sample_actions calls based on the training_mode parameter.
"""

import logging

from torch import Tensor, nn

from rho.common.types import TrainingMode
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_fast import RhoAlphaFASTModel
from rho.policies.rhoalpha.rhoalpha_flow import RhoAlphaFlowMatchingModel
from rho.policies.rhoalpha.rhoalpha_knowledge_insulation import RhoAlphaKnowledgeInsulationModel
from rho.policies.rhoalpha.rhoalpha_web import (
    RhoAlphaBoundingBoxModel,
    RhoAlphaPointingModel,
    RhoAlphaVQAModel,
)

logger = logging.getLogger(__name__)


class RhoAlphaRoboticsModel(nn.Module):
    """
    Unified Phi4MM model that composes specialized models based on training modes.

    This is a lightweight router that instantiates specialized models and routes
    forward/sample_actions calls. It does NOT inherit from RhoAlphaModel to avoid
    creating duplicate VLM components - instead, each specialized model creates
    its own VLM, and knowledge insulation models share VLMs via kwargs.
    """

    def __init__(self, config: RhoAlphaConfig):
        super().__init__()
        self.config = config

        # Get training modes from config
        self.training_modes = getattr(config, "training_modes", [TrainingMode.ROBOT_FLOWMATCH])
        if not isinstance(self.training_modes, list):
            self.training_modes = [self.training_modes]

        # Initialize models based on training modes
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize specialized models based on configured training modes.

        Each model creates its own VLM backbone (no sharing at this level).
        Knowledge insulation model handles VLM sharing internally between its sub-models.
        """
        # Flow matching model
        if TrainingMode.ROBOT_FLOWMATCH in self.training_modes:
            self.flow_model = RhoAlphaFlowMatchingModel(self.config)
        else:
            self.flow_model = None

        # Autoregressive model (FAST)
        if TrainingMode.ROBOT_AUTOREGRESSIVE in self.training_modes:
            self.fast_model = RhoAlphaFASTModel(self.config)
        else:
            self.fast_model = None

        # Knowledge insulation model (combines flow + fast with stop gradient)
        # This model handles VLM sharing internally between its flow and fast sub-models
        if TrainingMode.ROBOT_KNOWLEDGE_INSULATION in self.training_modes:
            self.knowledge_insulation_model = RhoAlphaKnowledgeInsulationModel(self.config)
        else:
            self.knowledge_insulation_model = None

        # Web tasks models (separate model for each task)

        # VQA model
        if TrainingMode.VQA in self.training_modes:
            self.vqa_model = RhoAlphaVQAModel(self.config)
        else:
            self.vqa_model = None

        # Bounding box model
        if TrainingMode.BOUNDING_BOX in self.training_modes:
            self.bbox_model = RhoAlphaBoundingBoxModel(self.config)
        else:
            self.bbox_model = None

        # Pointing model
        if TrainingMode.POINTING in self.training_modes:
            self.pointing_model = RhoAlphaPointingModel(self.config)
        else:
            self.pointing_model = None

    def forward(
        self,
        image,
        prompt,
        state,
        actions,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=TrainingMode.ROBOT_FLOWMATCH,
        **kwargs,
    ):
        """
        Forward pass routing to appropriate model based on training_mode.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            actions: Ground truth actions
            noise: Optional noise (for flow matching)
            time: Optional time (for flow matching)
            image_mask: Optional image masking
            training_mode: Which training mode to use
            **kwargs: Additional task-specific arguments

        Returns:
            Loss or losses depending on training mode
        """
        # Route to appropriate model
        if training_mode == TrainingMode.ROBOT_FLOWMATCH:
            if self.flow_model is None:
                raise ValueError(f"ROBOT_FLOWMATCH not in config.training_modes: {self.training_modes}")
            return self.flow_model.forward(image, prompt, state, actions, noise, time, image_mask)

        elif training_mode == TrainingMode.ROBOT_AUTOREGRESSIVE:
            if self.fast_model is None:
                raise ValueError(f"ROBOT_AUTOREGRESSIVE not in config.training_modes: {self.training_modes}")
            return self.fast_model.forward(image, prompt, state, actions, noise, time, image_mask)

        elif training_mode == TrainingMode.ROBOT_KNOWLEDGE_INSULATION:
            if self.knowledge_insulation_model is None:
                raise ValueError(
                    f"ROBOT_KNOWLEDGE_INSULATION not in config.training_modes: {self.training_modes}"
                )
            return self.knowledge_insulation_model.forward(
                image, prompt, state, actions, noise, time, image_mask
            )

        elif training_mode == TrainingMode.VQA:
            if self.vqa_model is None:
                raise ValueError(f"VQA not in config.training_modes: {self.training_modes}")
            return self.vqa_model.forward(
                image, prompt, state, actions, noise, time, image_mask, training_mode, **kwargs
            )

        elif training_mode == TrainingMode.BOUNDING_BOX:
            if self.bbox_model is None:
                raise ValueError(f"BOUNDING_BOX not in config.training_modes: {self.training_modes}")
            return self.bbox_model.forward(
                image, prompt, state, actions, noise, time, image_mask, training_mode, **kwargs
            )

        elif training_mode == TrainingMode.POINTING:
            if self.pointing_model is None:
                raise ValueError(f"POINTING not in config.training_modes: {self.training_modes}")
            return self.pointing_model.forward(
                image, prompt, state, actions, noise, time, image_mask, training_mode, **kwargs
            )

        else:
            raise NotImplementedError(f"Training mode {training_mode} not implemented")

    def sample_actions(
        self, image, prompt, state, noise=None, image_mask=None, inference_mode=None
    ) -> Tensor:
        """
        Sample actions using specified inference mode.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            noise: Optional noise (for flow matching)
            image_mask: Optional image masking
            inference_mode: Which inference method to use (defaults to primary training mode)

        Returns:
            Sampled actions
        """
        # Default to primary training mode if not specified
        if inference_mode is None:
            inference_mode = self.training_modes[0]

        # Route to appropriate model
        if inference_mode == TrainingMode.ROBOT_FLOWMATCH:
            if self.flow_model is None:
                raise ValueError(f"ROBOT_FLOWMATCH not in config.training_modes: {self.training_modes}")
            return self.flow_model.sample_actions(image, prompt, state, noise, image_mask)

        elif inference_mode == TrainingMode.ROBOT_AUTOREGRESSIVE:
            if self.fast_model is None:
                raise ValueError(f"ROBOT_AUTOREGRESSIVE not in config.training_modes: {self.training_modes}")
            return self.fast_model.sample_actions(image, prompt, state, noise, image_mask)

        elif inference_mode == TrainingMode.ROBOT_KNOWLEDGE_INSULATION:
            # For knowledge insulation, we can use either flow or fast for inference
            # Default to flow matching
            if self.knowledge_insulation_model is None:
                raise ValueError(
                    f"ROBOT_KNOWLEDGE_INSULATION not in config.training_modes: {self.training_modes}"
                )
            return self.knowledge_insulation_model.sample_actions(
                image, prompt, state, noise, image_mask, inference_mode="flow"
            )

        else:
            raise NotImplementedError(
                f"Inference not implemented for mode {inference_mode}. "
                f"Use ROBOT_FLOWMATCH, ROBOT_AUTOREGRESSIVE, or ROBOT_KNOWLEDGE_INSULATION."
            )

    def sample_actions_rtc(
        self,
        image,
        prompt,
        state,
        inference_delay,
        execution_horizon,
        prev_actions=None,
        noise=None,
        beta=40.0,
    ) -> Tensor:
        """
        Sample actions in real-time with specified inference delay and execution horizon.

        This method is designed for real-time control where the model needs to generate
        actions within a certain time constraint. It uses the flow matching model for sampling
        due to its efficiency, but can be extended to use other models if needed.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            inference_delay: Time allowed for inference (in seconds)
            execution_horizon: Time horizon for executing sampled actions (in seconds)
            prev_actions: Optional previously executed actions for context
            noise: Optional noise for sampling
            beta: Temperature parameter for sampling (higher = more random)

        Returns:
            Sampled actions tensor
        """
        if self.flow_model is None:
            raise ValueError(f"ROBOT_FLOWMATCH not in config.training_modes: {self.training_modes}")

        return self.flow_model.sample_actions_rtc(
            image, prompt, state, inference_delay, execution_horizon, prev_actions, noise, beta
        )

    def print_freezing_status(self):
        """Print freezing status of the active model(s)."""
        # Print status for each initialized model
        if self.flow_model is not None:
            logger.info("=== Flow Matching Model ===")
            self.flow_model.print_freezing_status()

        if self.fast_model is not None:
            logger.info("=== FAST Autoregressive Model ===")
            self.fast_model.print_freezing_status()

        if self.knowledge_insulation_model is not None:
            logger.info("=== Knowledge Insulation Model ===")
            self.knowledge_insulation_model.print_freezing_status()

        if self.vqa_model is not None:
            logger.info("=== VQA Model ===")
            self.vqa_model.print_freezing_status()

        if self.bbox_model is not None:
            logger.info("=== Bounding Box Model ===")
            self.bbox_model.print_freezing_status()

        if self.pointing_model is not None:
            logger.info("=== Pointing Model ===")
            self.pointing_model.print_freezing_status()

"""
Phi4MM Web Tasks (VQA, Bounding Box, Pointing).

Implements web-related tasks like visual question answering,
bounding box prediction, and pointing.

Models:
- RhoAlphaWebModel: Base class for all web tasks
- RhoAlphaVQAModel: Visual question answering
- RhoAlphaBoundingBoxModel: Object localization with bounding boxes
- RhoAlphaPointingModel: 2D coordinate prediction

Note: These are currently stub implementations and need proper loss/tokenization.
"""

import torch
from torch import Tensor, nn

from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_model import RhoAlphaModel


class RhoAlphaWebModel(RhoAlphaModel):
    """
    Base model for web-related tasks.

    Currently a pass-through to RhoAlphaModel, but provides a common base
    for web-specific functionality that may be added later.
    """

    def __init__(self, config: RhoAlphaConfig, **kwargs):
        super().__init__(config, **kwargs)
        # No additional components yet, but this provides a place for
        # shared web-specific functionality in the future


class RhoAlphaVQAModel(RhoAlphaWebModel):
    """
    Visual Question Answering model.

    Projects VLM features to vocabulary for answer generation.
    """

    def __init__(self, config: RhoAlphaConfig, **kwargs):
        super().__init__(config, **kwargs)

        # VQA head - projects to vocabulary for answer generation
        self.vqa_head = nn.Linear(self.embed_dim, self.vlm_backbone.config.vocab_size).to(
            device=self.device, dtype=self.dtype
        )

        self.print_freezing_status()

    def forward(
        self,
        image,
        prompt,
        state=None,
        actions=None,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=None,
        target_text=None,
        **kwargs,
    ):
        """
        Forward pass for VQA task.

        TODO: Implement proper VQA loss with target_text
        This should tokenize target_text and compute cross-entropy.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state (may be None for web tasks)
            actions: Ground truth actions (may be None for web tasks)
            noise: Ignored for VQA
            time: Ignored for VQA
            image_mask: Optional image masking
            training_mode: Ignored (always VQA)
            target_text: Ground truth answer text
            **kwargs: Additional arguments

        Returns:
            VQA loss
        """
        # Extract shared VLM features
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        # Pool features and project to vocabulary
        pooled_features = image_text_embed.mean(dim=1)  # Simple mean pooling
        logits = self.vqa_head(pooled_features)  # noqa: F841

        # TODO: Implement proper VQA loss
        # Should tokenize target_text and compute cross-entropy loss
        # For now, return zero loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss

    def sample_actions(self, image, prompt, state=None, noise=None, image_mask=None) -> Tensor:
        """
        Inference for VQA.

        TODO: Implement proper answer decoding.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state (may be None)
            noise: Ignored
            image_mask: Optional image masking

        Returns:
            VQA logits (would need decoding for actual answer)
        """
        # Extract VLM features
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        pooled_features = image_text_embed.mean(dim=1)
        logits = self.vqa_head(pooled_features)
        return logits


class RhoAlphaBoundingBoxModel(RhoAlphaWebModel):
    """
    Bounding Box prediction model.

    Predicts 4 coordinates (x1, y1, x2, y2) for object localization.
    """

    def __init__(self, config: RhoAlphaConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Bounding box head - predicts 4 coordinates (x1, y1, x2, y2)
        self.bbox_head = nn.Linear(self.embed_dim, 4).to(device=self.device, dtype=self.dtype)

        self.print_freezing_status()

    def forward(
        self,
        image,
        prompt,
        state=None,
        actions=None,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=None,
        target_bbox=None,
        **kwargs,
    ):
        """
        Forward pass for bounding box prediction.

        TODO: Implement proper bbox loss (e.g., smooth L1 or IoU loss)

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state (may be None for web tasks)
            actions: Ground truth actions (may be None for web tasks)
            noise: Ignored for bbox
            time: Ignored for bbox
            image_mask: Optional image masking
            training_mode: Ignored (always BOUNDING_BOX)
            target_bbox: Ground truth bounding box (x1, y1, x2, y2)
            **kwargs: Additional arguments

        Returns:
            Bounding box loss
        """
        # Extract shared VLM features
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        # Pool features and predict bbox
        pooled_features = image_text_embed.mean(dim=1)
        bbox_pred = self.bbox_head(pooled_features)  # noqa: F841

        # TODO: Implement proper bbox loss
        # if target_bbox is not None:
        #     loss = F.smooth_l1_loss(bbox_pred, target_bbox)
        # or use IoU loss for better performance
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss

    def sample_actions(self, image, prompt, state=None, noise=None, image_mask=None) -> Tensor:
        """
        Inference for bounding box prediction.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state (may be None)
            noise: Ignored
            image_mask: Optional image masking

        Returns:
            Predicted bbox coordinates (batch_size, 4)
        """
        # Extract VLM features
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        pooled_features = image_text_embed.mean(dim=1)
        bbox = self.bbox_head(pooled_features)
        return bbox


class RhoAlphaPointingModel(RhoAlphaWebModel):
    """
    Pointing model for 2D coordinate prediction.

    Predicts a single 2D point (x, y) for pointing tasks.
    """

    def __init__(self, config: RhoAlphaConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Pointing head - predicts 2D coordinates (x, y)
        self.pointing_head = nn.Linear(self.embed_dim, 2).to(device=self.device, dtype=self.dtype)

        self.print_freezing_status()

    def forward(
        self,
        image,
        prompt,
        state=None,
        actions=None,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=None,
        target_point=None,
        **kwargs,
    ):
        """
        Forward pass for pointing task.

        TODO: Implement proper pointing loss (e.g., MSE loss)

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state (may be None for web tasks)
            actions: Ground truth actions (may be None for web tasks)
            noise: Ignored for pointing
            time: Ignored for pointing
            image_mask: Optional image masking
            training_mode: Ignored (always POINTING)
            target_point: Ground truth 2D point (x, y)
            **kwargs: Additional arguments

        Returns:
            Pointing loss
        """
        # Extract shared VLM features
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        # Pool features and predict point
        pooled_features = image_text_embed.mean(dim=1)
        point_pred = self.pointing_head(pooled_features)  # noqa: F841

        # TODO: Implement proper pointing loss
        # if target_point is not None:
        #     loss = F.mse_loss(point_pred, target_point)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss

    def sample_actions(self, image, prompt, state=None, noise=None, image_mask=None) -> Tensor:
        """
        Inference for pointing task.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state (may be None)
            noise: Ignored
            image_mask: Optional image masking

        Returns:
            Predicted point coordinates (batch_size, 2)
        """
        # Extract VLM features
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        pooled_features = image_text_embed.mean(dim=1)
        point = self.pointing_head(pooled_features)
        return point

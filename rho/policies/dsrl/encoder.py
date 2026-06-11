"""Observation encoders for DSRL-SAC.

Ported from dsrl_pi0 JAX implementation to PyTorch.
Provides SmallEncoder (4-layer CNN), ResNet34Encoder, VLMObsEncoder,
and the unified ObsEncoder wrapper.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Spatial softmax layer: converts feature maps to (x, y) keypoint coordinates.

    Input: (B, C, H, W) feature maps.
    Output: (B, 2*C) expected (x, y) coordinates per channel.
    """

    def __init__(self, height: int, width: int, num_channels: int, temperature: float = -1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Create coordinate grids
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        # (H*W,)
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

        if temperature == -1.0:
            # Learnable temperature
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature: (B, C, H, W) feature maps

        Returns:
            (B, 2*C) expected x,y coordinates per channel
        """
        B, C, H, W = feature.shape
        # Reshape to (B, C, H*W)
        feature_flat = feature.reshape(B, C, H * W)

        # Softmax attention over spatial dims
        attention = F.softmax(feature_flat / self.temperature, dim=2)  # (B, C, H*W)

        # Weighted sum of coordinates
        expected_x = (self.pos_x * attention).sum(dim=2)  # (B, C)
        expected_y = (self.pos_y * attention).sum(dim=2)  # (B, C)

        return torch.cat([expected_x, expected_y], dim=1)  # (B, 2*C)


class SmallEncoder(nn.Module):
    """Small 4-layer CNN encoder matching dsrl_pi0's 'small' encoder.

    Architecture: 4 conv layers with GroupNorm + ReLU.
    Default: features=(32,32,32,32), kernels=3x3, strides=(2,1,1,1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: Sequence[int] = (32, 32, 32, 32),
        strides: Sequence[int] = (2, 1, 1, 1),
        kernel_size: int = 3,
        norm_type: str = "group",
        num_groups: int = 4,
    ):
        super().__init__()
        assert len(features) == len(strides)

        layers = []
        ch_in = in_channels
        for feat, stride in zip(features, strides, strict=False):
            # Conv with VALID padding (no padding)
            layers.append(nn.Conv2d(ch_in, feat, kernel_size, stride=stride, padding=0))
            if norm_type == "group":
                layers.append(nn.GroupNorm(min(num_groups, feat), feat))
            elif norm_type == "batch":
                layers.append(nn.BatchNorm2d(feat))
            elif norm_type == "layer":
                layers.append(nn.GroupNorm(1, feat))  # GroupNorm with 1 group = LayerNorm
            layers.append(nn.ReLU(inplace=True))
            ch_in = feat

        self.conv = nn.Sequential(*layers)
        self.out_channels = features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) images in [0, 1]

        Returns:
            (B, C_out, H', W') feature maps
        """
        return self.conv(x)


class ResNetBlock(nn.Module):
    """Standard ResNet residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_type: str = "group",
        num_groups: int = 4,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = self._make_norm(out_channels, norm_type, num_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = self._make_norm(out_channels, norm_type, num_groups)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                self._make_norm(out_channels, norm_type, num_groups),
            )

    @staticmethod
    def _make_norm(channels: int, norm_type: str, num_groups: int) -> nn.Module:
        if norm_type == "group":
            return nn.GroupNorm(min(num_groups, channels), channels)
        elif norm_type == "batch":
            return nn.BatchNorm2d(channels)
        elif norm_type == "layer":
            return nn.GroupNorm(1, channels)
        raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder with GroupNorm and optional SpatialSoftmax.

    Matches dsrl_pi0's ResNet34 encoder architecture.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        norm_type: str = "group",
        num_groups: int = 4,
        use_spatial_softmax: bool = True,
        softmax_temperature: float = -1.0,
    ):
        super().__init__()
        self.use_spatial_softmax = use_spatial_softmax

        # Initial conv + pool
        self.conv1 = nn.Conv2d(in_channels, num_filters, 7, stride=2, padding=3, bias=False)
        self.norm1 = ResNetBlock._make_norm(num_filters, norm_type, num_groups)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        # ResNet-34 stage sizes: (3, 4, 6, 3)
        stage_sizes = (3, 4, 6, 3)
        strides = (1, 2, 2, 1)  # stride for first block of each stage
        self.stages = nn.ModuleList()
        ch_in = num_filters
        for stage_idx, (num_blocks, stride) in enumerate(zip(stage_sizes, strides, strict=False)):
            ch_out = num_filters * (2**stage_idx)
            blocks = []
            for block_idx in range(num_blocks):
                s = (
                    stride
                    if block_idx == 0 and stage_idx > 0
                    else (stride if block_idx == 0 and stage_idx == 0 else 1)
                )
                if stage_idx == 0:
                    s = 1
                elif block_idx == 0:
                    s = stride
                else:
                    s = 1
                blocks.append(
                    ResNetBlock(ch_in, ch_out, stride=s, norm_type=norm_type, num_groups=num_groups)
                )
                ch_in = ch_out
            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = ch_in
        self.softmax_temperature = softmax_temperature
        # SpatialSoftmax is created dynamically based on feature map size
        self._spatial_softmax: SpatialSoftmax | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) images in [0, 1]

        Returns:
            (B, feature_dim) — if spatial_softmax: 2*C_out, else C_out
        """
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)

        if self.use_spatial_softmax:
            B, C, H, W = x.shape
            if self._spatial_softmax is None or self._spatial_softmax.height != H:
                self._spatial_softmax = SpatialSoftmax(H, W, C, self.softmax_temperature).to(x.device)
            return self._spatial_softmax(x)
        else:
            return x.mean(dim=(2, 3))  # Global average pool


def _compute_encoder_output_dim(
    encoder: nn.Module,
    in_channels: int,
    image_size: int,
    use_spatial_softmax: bool,
) -> int:
    """Compute the output dimension of an encoder by doing a forward pass."""
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, image_size, image_size)
        if isinstance(encoder, SmallEncoder):
            feat = encoder(dummy)
            B, C, H, W = feat.shape
            if use_spatial_softmax:
                return 2 * C
            return C * H * W
        elif isinstance(encoder, ResNet34Encoder):
            out = encoder(dummy)
            return out.shape[1]
    return 0


class ObsEncoder(nn.Module):
    """Unified observation encoder.

    Takes raw images (+ optional state), processes through CNN + optional
    SpatialSoftmax + bottleneck to produce a fixed-size embedding.

    Args:
        encoder_type: "small" or "resnet34"
        in_channels: input channels (3 * num_cameras)
        image_size: expected image height/width
        latent_dim: bottleneck output dimension
        state_dim: robot state dimension (0 to disable)
        norm_type: normalization type for CNN
        use_spatial_softmax: whether to use spatial softmax
        softmax_temperature: spatial softmax temperature (-1 = learnable)
    """

    def __init__(
        self,
        encoder_type: str = "small",
        in_channels: int = 3,
        image_size: int = 128,
        latent_dim: int = 50,
        state_dim: int = 0,
        norm_type: str = "group",
        use_spatial_softmax: bool = True,
        softmax_temperature: float = -1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.use_spatial_softmax = use_spatial_softmax

        # Build CNN encoder
        if encoder_type == "small":
            self.cnn = SmallEncoder(in_channels=in_channels, norm_type=norm_type)
            # Compute output size
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, image_size, image_size)
                feat = self.cnn(dummy)
                _, C, H, W = feat.shape
            if use_spatial_softmax:
                self.spatial_softmax = SpatialSoftmax(H, W, C, softmax_temperature)
                encoder_out_dim = 2 * C
            else:
                self.spatial_softmax = None
                encoder_out_dim = C * H * W
        elif encoder_type == "resnet34":
            self.cnn = ResNet34Encoder(
                in_channels=in_channels,
                norm_type=norm_type,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, image_size, image_size)
                out = self.cnn(dummy)
                encoder_out_dim = out.shape[1]
            self.spatial_softmax = None  # Handled internally by ResNet34Encoder
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Bottleneck: Linear -> LayerNorm -> Tanh
        self.bottleneck = nn.Sequential(
            nn.Linear(encoder_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        # Optional state projection
        if state_dim > 0:
            self.state_proj = nn.Linear(state_dim, state_dim)
        else:
            self.state_proj = None

    @property
    def output_dim(self) -> int:
        """Total output dimension (latent_dim + state_dim if applicable)."""
        return self.latent_dim + (self.state_dim if self.state_proj is not None else 0)

    def forward(self, images: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) float tensor in [0, 1]
            state: (B, state_dim) optional robot state

        Returns:
            (B, output_dim) observation embedding
        """
        feat = self.cnn(images)

        if isinstance(self.cnn, SmallEncoder) and self.spatial_softmax is not None:
            feat = self.spatial_softmax(feat)
        elif isinstance(self.cnn, SmallEncoder):
            feat = feat.reshape(feat.shape[0], -1)
        # ResNet34Encoder already outputs (B, dim)

        emb = self.bottleneck(feat)

        if self.state_proj is not None and state is not None:
            state_emb = self.state_proj(state)
            emb = torch.cat([emb, state_emb], dim=1)

        return emb


class VLMObsEncoder(nn.Module):
    """Observation encoder that reuses VLM hidden states from Phi4MM.

    Instead of processing raw images, takes pre-computed VLM hidden states
    and projects them to a fixed embedding.

    Args:
        hidden_dim: dimension of VLM hidden states
        latent_dim: bottleneck output dimension
        state_dim: robot state dimension (0 to disable)
    """

    def __init__(self, hidden_dim: int = 1024, latent_dim: int = 50, state_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        if state_dim > 0:
            self.state_proj = nn.Linear(state_dim, state_dim)
        else:
            self.state_proj = None

    @property
    def output_dim(self) -> int:
        return self.latent_dim + (self.state_dim if self.state_proj is not None else 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_dim) VLM hidden states
            mask: (B, seq_len) attention mask (1 = valid, 0 = padding)
            state: (B, state_dim) optional robot state

        Returns:
            (B, output_dim) observation embedding
        """
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden_states.mean(dim=1)

        emb = self.proj(pooled)

        if self.state_proj is not None and state is not None:
            state_emb = self.state_proj(state)
            emb = torch.cat([emb, state_emb], dim=1)

        return emb

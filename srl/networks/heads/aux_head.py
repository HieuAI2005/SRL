"""Auxiliary heads for self-supervised representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection used for CURL / contrastive InfoNCE loss."""

    type_name = "projection"

    def __init__(self, input_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvDecoderHead(nn.Module):
    """Convolutional decoder for pixel reconstruction (AE auxiliary loss).

    Decodes a flat latent vector back to an image using transposed convolutions.
    """

    type_name = "decoder"

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int, int] = (3, 84, 84),
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        C, H, W = output_shape
        # Compute stem size: assume 4× upsampling → stem_h = H // 4
        sh, sw = H // 8, W // 8
        self.stem_h, self.stem_w = sh, sw
        self.stem_channels = base_channels * 4

        self.fc = nn.Linear(latent_dim, self.stem_channels * sh * sw)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.stem_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, C, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.fc(z).view(B, self.stem_channels, self.stem_h, self.stem_w)
        return self.deconv(x)

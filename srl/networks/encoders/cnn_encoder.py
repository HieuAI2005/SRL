"""CNN encoder — for image/pixel observations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from srl.networks.layers.cnn_builder import build_cnn


class CNNEncoder(nn.Module):
    """Visual encoder: convolutional backbone + flatten + linear projection.

    Parameters
    ----------
    input_shape:
        ``(C, H, W)`` of the input image.
    layer_configs:
        List of layer configs forwarded to :func:`~srl.networks.layers.cnn_builder.build_cnn`.
    latent_dim:
        Output embedding size.
    kwargs:
        Forwarded to ``build_cnn`` (conv_type, default_activation, …).
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        layer_configs: list,
        latent_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.cnn, flat_dim = build_cnn(layer_configs, input_shape, **kwargs)
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(flat_dim, latent_dim)
        self._apply_layer_norm = kwargs.get("layer_norm_out", True)
        if self._apply_layer_norm:
            self.out_norm = nn.LayerNorm(latent_dim)
        else:
            self.out_norm = nn.Identity()

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [..., C, H, W] — handle batch dims
        x = obs
        if obs.dtype == torch.uint8:
            x = x.float() / 255.0
        z = self.out_norm(self.proj(self.flatten(self.cnn(x))))
        return z

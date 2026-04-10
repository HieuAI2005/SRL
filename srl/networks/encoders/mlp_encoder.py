"""MLP encoder — for state/vector observations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from srl.networks.layers.mlp_builder import build_mlp


class MLPEncoder(nn.Module):
    """Fully-connected encoder for 1-D state observations.

    Accepts either a flat ``Tensor[..., input_dim]`` or the first element of a
    dict obs (``obs['state']``).

    Parameters
    ----------
    input_dim:
        Dimensionality of the input vector.
    layer_configs:
        List of ints or dicts (see :func:`~srl.networks.layers.mlp_builder.build_mlp`).
    latent_dim:
        Optional projection to a fixed-size output.  ``None`` keeps the final
        hidden size as the latent dimension.
    kwargs:
        Forwarded to ``build_mlp`` (default_activation, default_norm, …).
    """

    def __init__(
        self,
        input_dim: int,
        layer_configs: list[int | dict],
        latent_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net, hidden_dim = build_mlp(layer_configs, input_dim, **kwargs)

        if latent_dim is not None and latent_dim != hidden_dim:
            self.projection: nn.Module = nn.Linear(hidden_dim, latent_dim)
            self._latent_dim = latent_dim
        else:
            self.projection = nn.Identity()
            self._latent_dim = hidden_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.net(obs))

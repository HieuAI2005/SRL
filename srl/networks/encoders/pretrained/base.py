"""Base class for pre-trained encoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PretrainedEncoderBase(nn.Module):
    """Shared base for all pre-trained encoders.

    Handles three common concerns so subclasses stay minimal:

    1. **Projection** — backbone output dim → ``latent_dim`` via ``Linear + LayerNorm``.
    2. **Freeze** — freeze all or the first N child blocks of the backbone.
    3. **Normalization** — optional ImageNet mean/std normalisation; registered as
       buffers so they move with ``.to(device)``.

    Subclass contract:

    * Call ``super().__init__(cfg)`` first.
    * Build ``self.backbone``, compute ``backbone_dim``.
    * Call ``self._build_proj(backbone_dim)`` to wire up the projection head.
    * Optionally call ``self._maybe_freeze(self.backbone, cfg.extra)`` after building.
    * Implement ``forward(self, obs) -> Tensor``, call ``self._preprocess(obs)``
      on the input then ``self._project(features)`` on the backbone output.
    """

    def __init__(self, cfg: Any) -> None:  # cfg: EncoderConfig
        super().__init__()
        self._latent_dim: int = cfg.latent_dim
        self._normalize = cfg.extra.get("normalize_input", True)
        # Always register buffers so state_dict shape is identical regardless of
        # normalize_input — this prevents strict load_state_dict mismatches when
        # a checkpoint is reloaded with a different config value.
        self.register_buffer(
            "_norm_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_norm_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _build_proj(self, backbone_dim: int) -> None:
        """Wire up Linear projection + LayerNorm (backbone_dim → latent_dim)."""
        self.proj = nn.Linear(backbone_dim, self._latent_dim)
        self.out_norm = nn.LayerNorm(self._latent_dim)

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        """Apply projection head: flatten → proj → norm."""
        return self.out_norm(self.proj(features.flatten(1)))

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """uint8 → float32 / 255, then optional ImageNet normalisation."""
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
        if self._normalize:
            x = (x - self._norm_mean) / self._norm_std
        return x

    def _maybe_freeze(self, backbone: nn.Module, extra: dict) -> None:
        """Freeze backbone based on ``extra`` flags.

        Parameters (read from extra dict):
            freeze_backbone (bool): Freeze if True. Default False.
            freeze_layers (int | None): If set, only freeze the first N direct
                children. If None, freezes everything.
        """
        if not extra.get("freeze_backbone", False):
            return
        freeze_n = extra.get("freeze_layers", None)
        if freeze_n is None:
            for p in backbone.parameters():
                p.requires_grad_(False)
        else:
            for child in list(backbone.children())[:freeze_n]:
                for p in child.parameters():
                    p.requires_grad_(False)

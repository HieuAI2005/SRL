"""Vision pre-trained encoders backed by torchvision.

Registered types (use as ``type:`` in YAML):
    - ``resnet``       — ResNet-18 / 34 / 50 / 101 / 152
    - ``efficientnet`` — EfficientNet-B0 … B7
    - ``vit``          — ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32

Common ``extra`` YAML fields (all optional):
    model_variant (str):      backbone variant name, e.g. ``"resnet50"``.
    pretrained (bool|str):    ``true`` / ``"DEFAULT"`` / ``"IMAGENET1K_V1"``
                              / ``false``.  Default ``true``.
    freeze_backbone (bool):   freeze all backbone weights.  Default ``false``.
    freeze_layers (int):      freeze only the first N child modules.
    normalize_input (bool):   apply ImageNet mean/std.  Default ``true``.

Installation::

    pip install torchvision
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from srl.registry import register_encoder
from srl.registry.config_schema import EncoderConfig

from .base import PretrainedEncoderBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tv_weights(variant: str, pretrained: bool | str) -> Any:
    """Resolve the ``weights=`` argument for torchvision model constructors.

    torchvision >= 0.13 accepts ``None``, ``"DEFAULT"``, or a specific tag
    string (e.g. ``"IMAGENET1K_V1"``).
    """
    if pretrained is False or pretrained == "none":
        return None
    if isinstance(pretrained, str) and pretrained.upper() not in ("TRUE", "DEFAULT"):
        return pretrained  # specific tag passed verbatim
    return "DEFAULT"


def _require_torchvision() -> Any:
    try:
        import torchvision.models as M  # noqa: PLC0415
        return M
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for vision pre-trained encoders.\n"
            "Install with: pip install torchvision"
        ) from exc


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

# Feature dim of the global-average-pooled representation (before FC).
_RESNET_DIMS: dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


@register_encoder("resnet")
class ResNetEncoder(PretrainedEncoderBase):
    """ResNet backbone encoder.

    Example YAML::

        encoders:
          - name: visual_enc
            type: resnet
            input_shape: [3, 224, 224]
            latent_dim: 256
            model_variant: resnet50
            pretrained: true
            freeze_backbone: true
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__(cfg)
        M = _require_torchvision()

        variant: str = cfg.extra.get("model_variant", "resnet18").lower()
        if variant not in _RESNET_DIMS:
            raise ValueError(
                f"Unknown ResNet variant '{variant}'. "
                f"Choose from: {list(_RESNET_DIMS)}"
            )

        weights = _tv_weights(variant, cfg.extra.get("pretrained", True))
        backbone_full = getattr(M, variant)(weights=weights)

        # Strip the final FC layer → output is [B, C, 1, 1] after AvgPool
        self.backbone = nn.Sequential(*list(backbone_full.children())[:-1])

        self._build_proj(_RESNET_DIMS[variant])
        self._maybe_freeze(self.backbone, cfg.extra)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(obs)
        return self._project(self.backbone(x))


# ---------------------------------------------------------------------------
# EfficientNet
# ---------------------------------------------------------------------------

_EFFICIENTNET_DIMS: dict[str, int] = {
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b2": 1408,
    "efficientnet_b3": 1536,
    "efficientnet_b4": 1792,
    "efficientnet_b5": 2048,
    "efficientnet_b6": 2304,
    "efficientnet_b7": 2560,
}


@register_encoder("efficientnet")
class EfficientNetEncoder(PretrainedEncoderBase):
    """EfficientNet backbone encoder.

    Example YAML::

        encoders:
          - name: visual_enc
            type: efficientnet
            input_shape: [3, 224, 224]
            latent_dim: 256
            model_variant: efficientnet_b3
            pretrained: true
            freeze_backbone: false
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__(cfg)
        M = _require_torchvision()

        variant: str = cfg.extra.get("model_variant", "efficientnet_b0").lower()
        if variant not in _EFFICIENTNET_DIMS:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}'. "
                f"Choose from: {list(_EFFICIENTNET_DIMS)}"
            )

        weights = _tv_weights(variant, cfg.extra.get("pretrained", True))
        net = getattr(M, variant)(weights=weights)

        # features (conv + BN blocks) + AdaptiveAvgPool2d → [B, C, 1, 1]
        self.backbone = nn.Sequential(net.features, net.avgpool)

        self._build_proj(_EFFICIENTNET_DIMS[variant])
        self._maybe_freeze(self.backbone, cfg.extra)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(obs)
        return self._project(self.backbone(x))


# ---------------------------------------------------------------------------
# Vision Transformer (ViT)
# ---------------------------------------------------------------------------

_VIT_DIMS: dict[str, int] = {
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
    "vit_h_14": 1280,
}


@register_encoder("vit")
class ViTEncoder(PretrainedEncoderBase):
    """Vision Transformer (ViT) backbone encoder.

    Uses the CLS-token representation from torchvision's ViT implementation.
    The classification head is replaced with ``nn.Identity`` so the raw CLS
    token embedding flows into the projection.

    Example YAML::

        encoders:
          - name: visual_enc
            type: vit
            input_shape: [3, 224, 224]
            latent_dim: 256
            model_variant: vit_b_16
            pretrained: true
            freeze_backbone: true
            freeze_layers: 6        # freeze first 6 transformer blocks
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__(cfg)
        M = _require_torchvision()

        variant: str = cfg.extra.get("model_variant", "vit_b_16").lower()
        if variant not in _VIT_DIMS:
            raise ValueError(
                f"Unknown ViT variant '{variant}'. "
                f"Choose from: {list(_VIT_DIMS)}"
            )

        weights = _tv_weights(variant, cfg.extra.get("pretrained", True))
        self._vit = getattr(M, variant)(weights=weights)
        # Hijack the classification head → raw CLS token [B, hidden_dim]
        self._vit.heads = nn.Identity()

        self._build_proj(_VIT_DIMS[variant])
        # Freeze the transformer encoder blocks, not the patch embedding
        self._maybe_freeze(self._vit.encoder, cfg.extra)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(obs)
        if x.shape[-2:] != (224, 224):
            raise ValueError(
                f"ViTEncoder expects spatial input 224×224, got {tuple(x.shape[-2:])}. "
                "Set input_shape: [3, 224, 224] in your YAML config, or add a resize "
                "wrapper to your environment."
            )
        cls_token = self._vit(x)  # [B, hidden_dim]
        return self._project(cls_token)

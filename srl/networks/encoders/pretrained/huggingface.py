"""HuggingFace Transformers pre-trained encoders.

Registered types (use as ``type:`` in YAML):
    - ``huggingface``    — any encoder-only text model (BERT, DistilBERT,
                           RoBERTa, ALBERT, DeBERTa, …)
    - ``hf_vision``      — any HuggingFace vision model (ViT, Swin, ConvNeXt,
                           ResNet from HF Hub, DeiT, BEiT, …)

Text encoder (``huggingface``) ``extra`` fields:
    model_name (str):        HuggingFace model ID, e.g. ``"bert-base-uncased"``.
                             **Required.**
    pooling (str):           ``"cls"`` (default) or ``"mean"``.
    freeze_backbone (bool):  freeze all transformer weights.  Default ``true``
                             (recommended — language models are large).
    max_length (int):        truncation length for tokeniser.  Default 128.
    normalize_input (bool):  not used for text; kept for API consistency.
    pad_token_id (int):      token ID treated as padding.  Default ``0``.

Vision encoder (``hf_vision``) ``extra`` fields:
    model_name (str):        HuggingFace model ID, e.g. ``"google/vit-base-patch16-224"``.
                             **Required.**
    freeze_backbone (bool):  freeze all backbone weights.  Default ``true``.
    normalize_input (bool):  apply ImageNet mean/std normalisation.  Default ``true``.

Text input format:
    Integer tensor of token IDs, shape ``[B, seq_len]``.
    Attention mask derived automatically (non-zero positions).

Vision input format:
    Float/uint8 pixel tensor, shape ``[B, C, H, W]`` (CHW layout).

Installation::

    pip install transformers

Example YAML (text)::

    encoders:
      - name: text_enc
        type: huggingface
        input_dim: 128
        latent_dim: 128
        model_name: distilbert-base-uncased
        pooling: cls
        freeze_backbone: true

Example YAML (vision)::

    encoders:
      - name: visual_enc
        type: hf_vision
        input_shape: [3, 224, 224]
        latent_dim: 256
        model_name: google/vit-base-patch16-224
        freeze_backbone: true
"""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.registry import register_encoder
from srl.registry.config_schema import EncoderConfig

from .base import PretrainedEncoderBase


def _require_transformers():
    try:
        from transformers import AutoConfig, AutoModel  # noqa: PLC0415
        return AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for HuggingFace encoders.\n"
            "Install with: pip install transformers"
        ) from exc


def _probe_backbone_dim(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Run a single dummy forward to infer the flat backbone output dimension.

    Handles all HuggingFace vision output conventions:
      - ``pooler_output``: [B, D] or [B, D, 1, 1]
      - ``last_hidden_state``: [B, N, D] (transformer) or [B, D, H, W] (CNN)
    """
    device = next(model.parameters()).device
    dummy = torch.zeros(1, *input_shape, device=device)
    with torch.no_grad():
        out = model(pixel_values=dummy)

    if out.pooler_output is not None:
        po = out.pooler_output.flatten(1)
        return po.shape[-1]

    lhs = out.last_hidden_state
    # Transformer: [B, N, D] → CLS token dim
    if lhs.ndim == 3:
        return lhs.shape[-1]
    # CNN: [B, D, H, W] → flatten
    return lhs.flatten(1).shape[-1]


def _pool_vision_output(out, last_hidden_dim: int) -> torch.Tensor:
    """Extract a flat feature vector from a HuggingFace vision model output."""
    if out.pooler_output is not None:
        return out.pooler_output.flatten(1)          # [B, D] or [B, D, 1, 1]→[B, D]

    lhs = out.last_hidden_state
    if lhs.ndim == 3:
        return lhs[:, 0]                              # CLS token: [B, D]
    return lhs.flatten(1)                             # CNN spatial: [B, D*H*W]


@register_encoder("huggingface")
class HuggingFaceTextEncoder(nn.Module):
    """Encoder-only HuggingFace Transformers backbone.

    Wraps any model from the HuggingFace Hub that follows the
    ``BaseModelOutput`` convention (``last_hidden_state``).

    The attention mask is derived from ``input_ids != pad_token_id``, so no
    separate mask tensor needs to be passed through the obs dict.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        AutoConfig, AutoModel = _require_transformers()

        model_name: str | None = cfg.extra.get("model_name")
        if not model_name:
            raise ValueError(
                "HuggingFaceTextEncoder requires 'model_name' in the encoder "
                "extra fields, e.g.:\n"
                "  model_name: bert-base-uncased"
            )

        hf_config = AutoConfig.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

        backbone_dim: int = hf_config.hidden_size
        self._latent_dim: int = cfg.latent_dim
        self._pooling: str = cfg.extra.get("pooling", "cls")
        self._pad_id: int = cfg.extra.get("pad_token_id", 0)

        self.proj = nn.Linear(backbone_dim, cfg.latent_dim)
        self.out_norm = nn.LayerNorm(cfg.latent_dim)

        # Default: freeze — language models are expensive to fine-tune in RL
        if cfg.extra.get("freeze_backbone", True):
            for p in self._model.parameters():
                p.requires_grad_(False)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        obs:
            Integer tensor of token IDs, shape ``[B, seq_len]``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, latent_dim]``.
        """
        input_ids = obs.long()
        attention_mask = (input_ids != self._pad_id).long()

        out = self._model(input_ids=input_ids, attention_mask=attention_mask)

        if self._pooling == "cls":
            # CLS token is always position 0 for BERT-style models
            z = out.last_hidden_state[:, 0]
        else:
            # Mean pooling over non-padding tokens
            mask_f = attention_mask.unsqueeze(-1).float()
            z = (out.last_hidden_state * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)

        return self.out_norm(self.proj(z))


# ---------------------------------------------------------------------------
# HuggingFace Vision Encoder
# ---------------------------------------------------------------------------

@register_encoder("hf_vision")
class HuggingFaceVisionEncoder(PretrainedEncoderBase):
    """HuggingFace vision backbone encoder.

    Loads any vision model from the HuggingFace Hub via ``AutoModel``.
    Compatible architectures include:

    * **ViT** (``google/vit-base-patch16-224``, ``google/vit-large-patch16-224``)
    * **Swin Transformer** (``microsoft/swin-tiny-patch4-window7-224``)
    * **ConvNeXt** (``facebook/convnext-tiny-224``)
    * **DeiT** (``facebook/deit-small-patch16-224``)
    * **BEiT** (``microsoft/beit-base-patch16-224``)
    * **HF ResNet** (``microsoft/resnet-50``)

    The backbone output dimension is inferred automatically via a probe forward
    pass so no manual ``backbone_dim`` is needed.

    Example YAML::

        encoders:
          - name: visual_enc
            type: hf_vision
            input_shape: [3, 224, 224]
            latent_dim: 256
            model_name: google/vit-base-patch16-224
            freeze_backbone: true
            normalize_input: true
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__(cfg)
        AutoConfig, AutoModel = _require_transformers()

        model_name: str | None = cfg.extra.get("model_name")
        if not model_name:
            raise ValueError(
                "HuggingFaceVisionEncoder requires 'model_name' in the encoder "
                "extra fields, e.g.:\n"
                "  model_name: google/vit-base-patch16-224"
            )

        if cfg.input_shape is None:
            raise ValueError(
                "HuggingFaceVisionEncoder requires 'input_shape' in the encoder "
                "config, e.g.:\n"
                "  input_shape: [3, 224, 224]"
            )

        self._input_shape: tuple[int, ...] = tuple(cfg.input_shape)
        self._model = AutoModel.from_pretrained(model_name)

        # Auto-detect backbone output dimension
        backbone_dim = _probe_backbone_dim(self._model, self._input_shape)

        self._build_proj(backbone_dim)

        if cfg.extra.get("freeze_backbone", True):
            for p in self._model.parameters():
                p.requires_grad_(False)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        obs:
            Pixel tensor of shape ``[B, C, H, W]`` (CHW layout).
            uint8 inputs are scaled to ``[0, 1]`` automatically.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, latent_dim]``.
        """
        x = self._preprocess(obs)
        out = self._model(pixel_values=x)
        z = _pool_vision_output(out, self._latent_dim)
        return self.out_norm(self.proj(z))

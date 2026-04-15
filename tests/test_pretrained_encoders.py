"""Smoke tests for pre-trained encoder registration and config parsing.

These tests do NOT require torchvision or transformers — they only verify that:
  1. The four new encoder types are correctly registered in EncoderRegistry.
  2. EncoderConfig.from_dict() correctly routes extra fields.
  3. ModelBuilder wires up a pretrained encoder entry without instantiating it
     (via a mock registry entry that doesn't need the optional dep).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from srl.networks.encoders import pretrained  # noqa: F401 — triggers registrations
from srl.registry.registry import EncoderRegistry
from srl.registry.config_schema import EncoderConfig
from srl.registry.builder import ModelBuilder


# ---------------------------------------------------------------------------
# Registry smoke tests (no torchvision / transformers needed)
# ---------------------------------------------------------------------------

def test_pretrained_encoder_types_registered() -> None:
    """All pretrained encoder types must appear in the registry after import."""
    for name in ("resnet", "efficientnet", "vit", "huggingface", "hf_vision"):
        assert name in EncoderRegistry, (
            f"'{name}' not found in EncoderRegistry. "
            f"Registered: {EncoderRegistry.available()}"
        )


def test_encoder_config_extra_passthrough() -> None:
    """model_variant and freeze_backbone must land in cfg.extra."""
    cfg = EncoderConfig.from_dict({
        "name": "visual_enc",
        "type": "resnet",
        "input_shape": [3, 224, 224],
        "latent_dim": 256,
        "model_variant": "resnet50",
        "pretrained": False,
        "freeze_backbone": True,
        "freeze_layers": 6,
    })
    assert cfg.extra["model_variant"] == "resnet50"
    assert cfg.extra["pretrained"] is False
    assert cfg.extra["freeze_backbone"] is True
    assert cfg.extra["freeze_layers"] == 6
    assert cfg.latent_dim == 256


def test_encoder_config_huggingface_extra() -> None:
    """model_name and pooling must land in cfg.extra for huggingface type."""
    cfg = EncoderConfig.from_dict({
        "name": "text_enc",
        "type": "huggingface",
        "input_dim": 64,
        "latent_dim": 128,
        "model_name": "distilbert-base-uncased",
        "pooling": "cls",
        "freeze_backbone": True,
    })
    assert cfg.type == "huggingface"
    assert cfg.extra["model_name"] == "distilbert-base-uncased"
    assert cfg.extra["pooling"] == "cls"
    assert cfg.extra["freeze_backbone"] is True


def test_encoder_config_hf_vision_extra() -> None:
    """model_name must land in cfg.extra for hf_vision type."""
    cfg = EncoderConfig.from_dict({
        "name": "visual_enc",
        "type": "hf_vision",
        "input_shape": [3, 224, 224],
        "latent_dim": 256,
        "model_name": "google/vit-base-patch16-224",
        "freeze_backbone": True,
        "normalize_input": True,
    })
    assert cfg.type == "hf_vision"
    assert cfg.extra["model_name"] == "google/vit-base-patch16-224"
    assert cfg.extra["freeze_backbone"] is True
    assert cfg.input_shape == [3, 224, 224]


# ---------------------------------------------------------------------------
# ModelBuilder with a stub pretrained encoder (no optional deps)
# ---------------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """Lightweight stand-in for a pretrained encoder used in builder tests."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self._latent_dim = cfg.latent_dim
        self.linear = nn.Linear(10, cfg.latent_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs[..., :10])


def test_model_builder_with_registered_pretrained_encoder(monkeypatch) -> None:
    """ModelBuilder must instantiate a registry encoder and wire up flows."""
    # Temporarily register stub under a test-only key to avoid touching real registry
    EncoderRegistry._store["_stub_enc"] = _StubEncoder

    try:
        model = ModelBuilder.from_dict({
            "encoders": [
                {
                    "name": "enc",
                    "type": "_stub_enc",
                    "input_dim": 10,
                    "latent_dim": 32,
                }
            ],
            "flows": ["enc -> actor"],
            "actor": {"name": "actor", "type": "gaussian", "action_dim": 2},
        })
        assert "enc" in model.encoders
        assert model.encoders["enc"].latent_dim == 32
    finally:
        EncoderRegistry._store.pop("_stub_enc", None)


# ---------------------------------------------------------------------------
# base.py — norm buffers always present regardless of normalize_input
# ---------------------------------------------------------------------------

def test_norm_buffers_always_registered() -> None:
    """_norm_mean and _norm_std must be in state_dict even when normalize_input=False."""
    EncoderRegistry._store["_stub_enc"] = _StubEncoder

    try:
        cfg_no_norm = EncoderConfig.from_dict({
            "name": "enc",
            "type": "_stub_enc",
            "input_dim": 10,
            "latent_dim": 16,
            "normalize_input": False,
        })

        # Import base to verify buffer presence directly
        from srl.networks.encoders.pretrained.base import PretrainedEncoderBase

        class _BareEncoder(PretrainedEncoderBase):
            def __init__(self, cfg):
                super().__init__(cfg)
                self._build_proj(10)

            def forward(self, obs):
                return self._project(obs)

        enc = _BareEncoder(cfg_no_norm)
        sd = enc.state_dict()
        assert "_norm_mean" in sd, "_norm_mean buffer missing from state_dict"
        assert "_norm_std" in sd, "_norm_std buffer missing from state_dict"
        assert sd["_norm_mean"].shape == (1, 3, 1, 1)
        assert sd["_norm_std"].shape == (1, 3, 1, 1)
    finally:
        EncoderRegistry._store.pop("_stub_enc", None)


# ---------------------------------------------------------------------------
# sac._unique_encoder_params — frozen params excluded
# ---------------------------------------------------------------------------

def test_unique_encoder_params_excludes_frozen() -> None:
    """_unique_encoder_params must not include requires_grad=False parameters."""
    from srl.algorithms.sac import _unique_encoder_params

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleDict({
                "enc": nn.Sequential(
                    nn.Linear(4, 8),   # will be frozen
                    nn.Linear(8, 4),   # trainable
                )
            })

    model = _FakeModel()
    # Freeze only the first linear
    for p in model.encoders["enc"][0].parameters():
        p.requires_grad_(False)

    params = _unique_encoder_params(model)
    for p in params:
        assert p.requires_grad, "Frozen param leaked into encoder optimizer param list"

    # Should have only the second linear's params (weight + bias = 2)
    assert len(params) == 2


# ---------------------------------------------------------------------------
# ViT input size validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torchvision"),
    reason="torchvision not installed",
)
def test_vit_rejects_wrong_input_size() -> None:
    """ViTEncoder must raise ValueError for non-224×224 inputs."""
    from srl.networks.encoders.pretrained.vision import ViTEncoder

    cfg = EncoderConfig.from_dict({
        "name": "v",
        "type": "vit",
        "input_shape": [3, 84, 84],
        "latent_dim": 64,
        "model_variant": "vit_b_16",
        "pretrained": False,
    })
    enc = ViTEncoder(cfg)
    obs = torch.zeros(1, 3, 84, 84)
    with pytest.raises(ValueError, match="224"):
        enc(obs)

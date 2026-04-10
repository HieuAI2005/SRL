"""Config schema dataclasses for YAML-driven model building."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Layer-level config (shared by MLP / CNN layer lists)
# ---------------------------------------------------------------------------

@dataclass
class LayerConfig:
    """A single layer description inside an encoder or head."""
    # For shorthand int entries in YAML the builder converts them automatically
    out_features: int | None = None          # MLP
    out_channels: int | None = None          # CNN
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    activation: str = "relu"
    norm: str = "none"
    dropout: float = 0.0
    dropout_type: str = "standard"
    pooling: str = "none"
    pooling_kernel: int = 2
    residual: bool = False
    depthwise: bool = False
    norm_order: str = "post"                 # "pre" | "post"
    weight_init: str = "none"


# ---------------------------------------------------------------------------
# Encoder config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    name: str
    type: str                                # "mlp" | "cnn" | "lstm" | "text" | custom
    # mlp-specific
    input_dim: int | None = None
    # cnn-specific
    input_shape: list[int] | None = None     # [C, H, W]
    # shared
    latent_dim: int = 128
    layers: list[Any] = field(default_factory=list)
    # Auxiliary representation
    aux_type: str | None = None              # "autoencoder" | "contrastive" | "byol"
    aux_latent_dim: int = 64
    # Momentum / EMA encoder
    use_momentum: bool = False
    momentum_tau: float = 0.99
    # Frame stacking
    frame_stack: int = 1
    # Recurrent wrapper
    recurrent: bool = False
    lstm_hidden: int = 256
    # Raw extra kwargs passed through to the encoder class
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EncoderConfig":
        allowed = cls.__dataclass_fields__.keys()
        known = {k: v for k, v in d.items() if k in allowed}
        extra = {k: v for k, v in d.items() if k not in allowed}
        obj = cls(**known)
        obj.extra = extra
        return obj


# ---------------------------------------------------------------------------
# Head config (actor / critic / value)
# ---------------------------------------------------------------------------

@dataclass
class HeadConfig:
    name: str
    type: str                                # "gaussian" | "squashed_gaussian" | "deterministic" | "value" | "twin_q" | "q"
    action_dim: int | None = None
    layers: list[Any] = field(default_factory=list)
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    state_dependent_std: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeadConfig":
        allowed = cls.__dataclass_fields__.keys()
        known = {k: v for k, v in d.items() if k in allowed}
        extra = {k: v for k, v in d.items() if k not in allowed}
        obj = cls(**known)
        obj.extra = extra
        return obj


# ---------------------------------------------------------------------------
# Loss config
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    name: str
    weight: float = 1.0
    # optional schedule: "linear_decay" | "cosine" | "constant"
    schedule: str = "constant"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LossConfig":
        allowed = cls.__dataclass_fields__.keys()
        known = {k: v for k, v in d.items() if k in allowed}
        extra = {k: v for k, v in d.items() if k not in allowed}
        obj = cls(**known)
        obj.extra = extra
        return obj


# ---------------------------------------------------------------------------
# Top-level agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentModelConfig:
    """Full model config parsed from a YAML file.

    YAML structure example::

        encoders:
          - name: visual_enc
            type: cnn
            input_shape: [3, 64, 64]
            latent_dim: 256
            aux_type: autoencoder
            layers: [[32,3,0], [64,3,0], [64,3,0]]
          - name: state_enc
            type: mlp
            input_dim: 12
            latent_dim: 64
            layers: [128, 64]

        flows:
          - "visual_enc -> actor"
          - "state_enc  -> actor"
          - "visual_enc -> critic"
          - "state_enc  -> critic"

        actor:
          name: actor
          type: squashed_gaussian
          action_dim: 6

        critic:
          name: critic
          type: twin_q
          action_dim: 6

        losses:
          - name: ppo_clip
            weight: 1.0
          - name: reconstruction
            weight: 0.1
    """

    encoders: list[EncoderConfig] = field(default_factory=list)
    flows: list[str] = field(default_factory=list)
    actor: HeadConfig | None = None
    critic: HeadConfig | None = None
    losses: list[LossConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentModelConfig":
        encoders = [EncoderConfig.from_dict(e) for e in d.get("encoders", [])]
        flows = d.get("flows", [])

        actor_d = d.get("actor")
        actor = HeadConfig.from_dict(actor_d) if actor_d else None

        critic_d = d.get("critic")
        critic = HeadConfig.from_dict(critic_d) if critic_d else None

        losses = [LossConfig.from_dict(l) for l in d.get("losses", [])]

        return cls(
            encoders=encoders,
            flows=flows,
            actor=actor,
            critic=critic,
            losses=losses,
        )

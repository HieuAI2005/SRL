"""Pooling layer factory."""

from __future__ import annotations

import torch.nn as nn


def get_pooling(name: str, kernel_size: int = 2, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("none", "identity", ""):
        return nn.Identity()
    if name == "maxpool":
        return nn.MaxPool2d(kernel_size=kernel_size, **kwargs)
    if name == "avgpool":
        return nn.AvgPool2d(kernel_size=kernel_size, **kwargs)
    if name == "adaptiveavgpool":
        output_size = kwargs.pop("output_size", (1, 1))
        return nn.AdaptiveAvgPool2d(output_size)
    raise ValueError(f"Unknown pooling '{name}'. Options: maxpool, avgpool, adaptiveavgpool, none")

"""Contrastive / CURL representation learning module."""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.losses.aux_losses import info_nce_loss
from srl.networks.encoders.augmentations import augment


class ContrastiveModule(nn.Module):
    """Applies two augmentations to the same obs and computes InfoNCE.

    Wraps an encoder and an optional projection head.

    Parameters
    ----------
    encoder:
        Shared online encoder.
    projection:
        Small MLP that maps latent → projection space.
    aug_mode:
        Augmentation mode passed to :func:`~srl.networks.encoders.augmentations.augment`.
    temperature:
        InfoNCE temperature.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        aug_mode: str = "drq",
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.aug_mode = aug_mode
        self.temperature = temperature

    def compute_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss from two augmented views of *obs*.

        Parameters
        ----------
        obs:
            Pixel observations, uint8 or float, shape ``(B, C, H, W)``.
        """
        obs_f = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs

        aug1 = augment(obs_f, mode=self.aug_mode)
        aug2 = augment(obs_f, mode=self.aug_mode)

        z1 = self.projection(self.encoder(aug1))
        z2 = self.projection(self.encoder(aug2))
        return info_nce_loss(z1, z2, temperature=self.temperature)

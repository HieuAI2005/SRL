"""Auxiliary self-supervised losses: InfoNCE, reconstruction, BYOL."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE / SimCLR contrastive loss.

    Parameters
    ----------
    z_anchor, z_positive:
        L2-normalised projection embeddings, shape ``(B, D)``.
    """
    B = z_anchor.size(0)
    z_a = F.normalize(z_anchor, dim=-1)
    z_p = F.normalize(z_positive, dim=-1)

    # Similarity matrix (B, B)
    logits = torch.mm(z_a, z_p.T) / temperature
    labels = torch.arange(B, device=z_anchor.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    return loss / 2.0


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE pixel reconstruction loss for autoencoder."""
    return F.mse_loss(recon, target.float(), reduction=reduction)


def byol_loss(
    online: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """BYOL regression loss (stop-gradient on target side, already handled
    externally by using a momentum encoder)."""
    online = F.normalize(online, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (online * target).sum(dim=-1).mean()

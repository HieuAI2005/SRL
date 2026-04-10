"""CheckpointManager — save and load model weights using safetensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Save and load agent checkpoints.

    Uses ``safetensors`` when available, falls back to ``torch.save``.

    Parameters
    ----------
    save_dir:
        Directory where checkpoints are stored.
    max_keep:
        Keep only the last *max_keep* checkpoints.
    """

    def __init__(self, save_dir: str | Path, max_keep: int = 5) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._saved: list[Path] = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        metrics: dict[str, Any] | None = None,
        tag: str = "ckpt",
    ) -> Path:
        ckpt_path = self.save_dir / f"{tag}_{step:010d}.pt"
        payload: dict[str, Any] = {
            "step": step,
            "model_state": model.state_dict(),
            "metrics": metrics or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()

        # Try safetensors for model weights
        try:
            from safetensors.torch import save_file as st_save

            st_path = ckpt_path.with_suffix(".safetensors")
            st_save(model.state_dict(), str(st_path))
            # Save non-tensor metadata separately
            meta_path = ckpt_path.with_suffix(".meta.pt")
            torch.save(
                {"step": step, "metrics": metrics or {},
                 "optimizer_state": payload.get("optimizer_state")},
                meta_path,
            )
            self._record(st_path)
            return st_path
        except (ImportError, Exception):
            pass

        torch.save(payload, ckpt_path)
        self._record(ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        model: nn.Module,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file as st_load
            state = st_load(str(path), device=str(device))
            model.load_state_dict(state)

            meta_path = path.with_suffix(".meta.pt")
            if meta_path.exists():
                meta = torch.load(meta_path, map_location=device, weights_only=False)
                if optimizer is not None and meta.get("optimizer_state"):
                    optimizer.load_state_dict(meta["optimizer_state"])
                return meta
            return {}
        else:
            payload = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            if optimizer is not None and "optimizer_state" in payload:
                optimizer.load_state_dict(payload["optimizer_state"])
            return payload

    def latest(self) -> Path | None:
        """Return the most recently saved checkpoint path."""
        if self._saved:
            return self._saved[-1]
        # Scan directory
        candidates = sorted(self.save_dir.glob("ckpt_*.pt")) + sorted(
            self.save_dir.glob("ckpt_*.safetensors")
        )
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, path: Path) -> None:
        self._saved.append(path)
        while len(self._saved) > self.max_keep:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink(missing_ok=True)
            # Remove companion meta file if present
            meta = old.with_suffix(".meta.pt")
            if meta.exists():
                meta.unlink(missing_ok=True)

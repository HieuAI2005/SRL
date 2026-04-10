"""TensorBoard + stdout logger for SRL training runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class Logger:
    """Lightweight logger that writes to TensorBoard and optionally stdout.

    Parameters
    ----------
    log_dir:
        Directory for TensorBoard event files.
    verbose:
        Print scalars to stdout.
    """

    def __init__(self, log_dir: str | Path = "runs", verbose: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._writer = None
        self._step = 0

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                pass
        return self._writer

    def log(self, tag: str, value: float, step: int | None = None) -> None:
        step = step if step is not None else self._step
        writer = self._get_writer()
        if writer is not None:
            writer.add_scalar(tag, value, global_step=step)
        if self.verbose:
            print(f"[step {step:>8d}] {tag}: {value:.4f}", flush=True)

    def log_dict(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            self.log(k, v, step=step)

    def set_step(self, step: int) -> None:
        self._step = step

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

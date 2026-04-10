"""GAE (Generalised Advantage Estimation) computation utilities."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute returns and advantages using GAE-λ.

    Parameters
    ----------
    rewards, values, dones:
        Shape ``(T,)`` arrays (single environment).
    last_value:
        Bootstrap value for the final step.

    Returns
    -------
    (returns, advantages) — both shape ``(T,)``.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages

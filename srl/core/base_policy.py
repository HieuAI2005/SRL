"""BasePolicy — thin wrapper that combines encoder(s) + actor/critic heads."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BasePolicy(nn.Module, abc.ABC):
    """Abstract base for all SRL policies.

    A *policy* owns the :class:`~srl.networks.agent_model.AgentModel` (or a
    traditional actor-critic pair) and exposes the two canonical methods used
    by every algorithm:

    * :meth:`forward` — full stochastic forward pass → ``(action, log_prob, value)``
    * :meth:`evaluate_actions` — re-evaluate stored actions for PPO-style updates
    """

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        deterministic: bool = False,
        recurrent_state: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, tuple | None]:
        """Full forward pass.

        Returns
        -------
        action : Tensor
        log_prob : Tensor | None
            ``None`` for deterministic policies (DDPG).
        value : Tensor | None
            ``None`` for pure-actor policies (DDPG actor).
        next_recurrent_state : tuple | None
        """

    @abc.abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        recurrent_state: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Evaluate stored *actions* under the current policy.

        Returns
        -------
        log_prob : Tensor
        entropy : Tensor
        value : Tensor | None
        """

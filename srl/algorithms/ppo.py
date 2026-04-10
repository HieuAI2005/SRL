"""PPO (Proximal Policy Optimisation) — on-policy, continuous actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from srl.core.base_agent import BaseAgent
from srl.core.config import PPOConfig
from srl.core.rollout_buffer import RolloutBuffer
from srl.losses.loss_composer import LossComposer
from srl.losses.rl_losses import (
    entropy_loss,
    ppo_clip_loss,
    ppo_value_loss,
)
from srl.utils.checkpoint import CheckpointManager
from srl.utils.normalizer import RunningNormalizer


class PPO(BaseAgent):
    """Proximal Policy Optimisation.

    Parameters
    ----------
    model:
        :class:`~srl.networks.agent_model.AgentModel` with a Gaussian /
        SquashedGaussian actor and a Value critic.
    config:
        :class:`~srl.core.config.PPOConfig`.
    device:
        PyTorch device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.cfg = config or PPOConfig()
        self._device = torch.device(device)
        self.model.to(self._device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, eps=1e-5
        )

        self.buffer = RolloutBuffer(
            capacity=self.cfg.n_steps,
            num_envs=getattr(self.cfg, "num_envs", 1),
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
            device=self.device,
        )

        self.checkpoint_manager: CheckpointManager | None = None
        self._global_step = 0

        # Optional observation/reward normalisation
        self._obs_normalizer: RunningNormalizer | None = None
        self._ret_normalizer: RunningNormalizer | None = None

        self.composer = LossComposer()
        self.composer.add("policy", weight=1.0)
        self.composer.add("value", weight=self.cfg.vf_coef)
        self.composer.add("entropy", weight=self.cfg.entropy_coef)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: dict[str, torch.Tensor],
        hidden: dict | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict]:
        self.model.eval()
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
        actor_out = result["actor_out"]

        # Actor out: dict from GaussianActorHead/SquashedGaussianActorHead,
        # or (action, log_prob) tuple from older heads
        if isinstance(actor_out, dict):
            if deterministic:
                action = actor_out.get("mean", actor_out.get("action"))
            else:
                action = actor_out.get("action")
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            action, log_prob = actor_out
        else:
            action, log_prob = actor_out, None

        value = result["value"]
        return action, log_prob, value, result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError(
            "Call PPO via a TrainingLoop or pass pre-collected batches to update()."
        )

    def update(self) -> dict[str, float]:
        """Run *n_epochs × n_batches* gradient updates on the filled buffer."""
        self.model.train()
        batch = self.buffer.get_batch()
        metrics_accum: dict[str, list[float]] = {}

        for _ in range(self.cfg.n_epochs):
            for mini in self.buffer.get_batches(self.cfg.batch_size):
                obs = {k: v.to(self.device) for k, v in mini.obs.items()}
                result = self.model(obs)
                actor_out = result["actor_out"]

                if isinstance(actor_out, dict):
                    log_prob_eval = actor_out.get("log_prob")
                    ent_raw = torch.zeros(1, device=self.device)
                elif isinstance(actor_out, tuple):
                    _, log_prob_eval = actor_out
                    ent_raw = torch.zeros(1, device=self.device)
                else:
                    log_prob_eval = actor_out
                    ent_raw = torch.zeros(1, device=self.device)

                # Try to get entropy from distribution if available
                if hasattr(self.model, "actor") and hasattr(self.model.actor, "get_distribution"):
                    dist = self.model.actor.get_distribution(result.get("actor_latent", obs))
                    if dist is not None:
                        ent_raw = dist.entropy().mean()
                ent = ent_raw

                adv = mini.advantages.to(self.device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pol_loss = ppo_clip_loss(
                    log_prob_eval,
                    mini.log_probs.to(self.device),
                    adv,
                    clip_eps=self.cfg.clip_range,
                )
                val_loss = ppo_value_loss(
                    result["value"].squeeze(-1),
                    mini.returns.to(self.device),
                    clip_eps=self.cfg.clip_range,
                )
                ent_loss = entropy_loss(ent)

                total, info = self.composer.compute(
                    step=self._global_step,
                    policy=pol_loss,
                    value=val_loss,
                    entropy=ent_loss,
                )

                self.optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                for k, v in info.items():
                    metrics_accum.setdefault(k, []).append(v)
                self._global_step += 1

        self.buffer.reset()
        return {k: sum(v) / len(v) for k, v in metrics_accum.items()}

    def save(self, path: str) -> None:
        import torch
        torch.save({"model": self.model.state_dict(), "step": self._global_step}, path)

    def load(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self._global_step = ckpt.get("step", 0)

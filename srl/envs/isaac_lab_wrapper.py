"""Isaac Lab environment wrapper for SRL."""

from __future__ import annotations

from typing import Any

import numpy as np


class IsaacLabWrapper:
    """Wrap an Isaac Lab ``ManagerBasedRLEnv`` to match the SRL interface.

    Isaac Lab envs return torch tensors on GPU; this wrapper converts them
    to numpy arrays (CPU) for the SRL buffers.

    Parameters
    ----------
    env:
        A ``ManagerBasedRLEnv`` or ``DirectRLEnv`` instance from Isaac Lab.
    obs_key:
        Key used when packaging the observation dict.
    """

    def __init__(self, env: Any, obs_key: str = "state") -> None:
        self.env = env
        self.obs_key = obs_key
        self.num_envs: int = getattr(env, "num_envs", 1)
        self.obs_space = getattr(env, "observation_space", None)
        self.act_space = getattr(env, "action_space", None)

    @property
    def device(self):
        if hasattr(self.env, "device"):
            return self.env.device
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "device"):
            return self.env.unwrapped.device
        raise AttributeError("Isaac Lab environment does not expose a device attribute")

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Isaac Lab returns (obs_dict, extras) or obs_dict
        if isinstance(out, tuple):
            obs, info = out[0], out[1]
        else:
            obs, info = out, {}
        return self._wrap_obs(obs), info

    def step(self, actions):
        import torch  # local import — optional dep

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)

        out = self.env.step(actions)
        # Returns (obs, reward, terminated, truncated, info) in newer Isaac Lab
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = done, done

        return (
            self._wrap_obs(obs),
            _to_np(reward),
            _to_np(terminated).astype(bool),
            _to_np(truncated).astype(bool),
            info,
        )

    def close(self) -> None:
        self.env.close()

    def _wrap_obs(self, obs: Any) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            return {k: _to_np(v) for k, v in obs.items()}
        return {self.obs_key: _to_np(obs)}


def _to_np(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)

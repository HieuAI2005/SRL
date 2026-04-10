"""Synchronous vectorised environment (single-process, low-overhead)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class SyncVectorEnv:
    """Run *n* copies of an environment sequentially in one process.

    Compatible with the SRL wrapper interface:
    ``reset() → (dict_obs, infos)``
    ``step(actions) → (dict_obs, rewards, dones, truncateds, infos)``

    Parameters
    ----------
    env_fns:
        List of callables that each return a (wrapped) gymnasium environment.
    """

    def __init__(self, env_fns: list[Callable]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.obs_space = self.envs[0].obs_space
        self.act_space = self.envs[0].act_space

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], list[dict]]:
        obs_list, info_list = zip(*[e.reset(**kwargs) for e in self.envs])
        return _stack_obs(obs_list), list(info_list)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        obs_list, rews, dones, truncs, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, d, t, info = env.step(actions[i])
            if d or t:
                o, _ = env.reset()
            obs_list.append(o)
            rews.append(r)
            dones.append(d)
            truncs.append(t)
            infos.append(info)
        return (
            _stack_obs(obs_list),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _stack_obs(obs_list: list[dict]) -> dict[str, np.ndarray]:
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}

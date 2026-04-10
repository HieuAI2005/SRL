"""Gymnasium single-env wrapper that normalises the SRL obs/action interface."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class GymnasiumWrapper(gym.Wrapper):
    """Thin wrapper around any Gymnasium environment.

    * Converts ``obs`` to ``dict`` form (key = ``"state"`` or ``"pixels"``).
    * Exposes ``obs_space``, ``act_space`` helpers.
    * Stores ``num_envs = 1`` for API uniformity with vector envs.
    """

    def __init__(self, env: gym.Env, obs_key: str = "state") -> None:
        super().__init__(env)
        self.obs_key = obs_key
        self.num_envs = 1

    @property
    def obs_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def act_space(self) -> gym.Space:
        return self.env.action_space

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.env.reset(**kwargs)
        return {self.obs_key: np.asarray(obs)}, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return {self.obs_key: np.asarray(obs)}, float(reward), done, truncated, info

"""Headless Isaac Lab integration test for SRL.

Run this file through tests/IsaacLab/isaaclab.sh so the correct Isaac Sim runtime
and Python environment are used.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = ROOT / "tests" / "IsaacLab"
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
import omni.usd
import carb

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from srl.algorithms.a2c import A2C
from srl.algorithms.ddpg import DDPG
from srl.algorithms.ppo import PPO
from srl.algorithms.sac import SAC
from srl.core.config import A2CConfig, DDPGConfig
from srl.core.config import PPOConfig
from srl.core.config import SACConfig
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper
from srl.registry.builder import ModelBuilder


TASKS = [
    "Isaac-Cartpole-v0",
    "Isaac-Ant-v0",
    "Isaac-Humanoid-v0",
]
ON_POLICY_ALGOS = ("ppo", "a2c")
OFF_POLICY_ALGOS = ("sac", "ddpg")


def _build_model(obs_dim: int, action_dim: int, algo: str):
    cfg = {
        "encoders": [
            {
                "name": "state_enc",
                "type": "mlp",
                "input_dim": obs_dim,
                "latent_dim": 64,
                "layers": [{"out_features": 64, "activation": "elu", "norm": "none"}],
            }
        ],
        "flows": ["state_enc -> actor", "state_enc -> critic"],
        "losses": [],
    }
    if algo in {"ppo", "a2c"}:
        cfg["actor"] = {"name": "actor", "type": "gaussian", "action_dim": action_dim}
        cfg["critic"] = {"name": "critic", "type": "value", "layers": []}
    elif algo == "sac":
        cfg["actor"] = {"name": "actor", "type": "squashed_gaussian", "action_dim": action_dim}
        cfg["critic"] = {"name": "critic", "type": "twin_q", "action_dim": action_dim, "layers": []}
    elif algo == "ddpg":
        cfg["actor"] = {"name": "actor", "type": "deterministic", "action_dim": action_dim}
        cfg["critic"] = {"name": "critic", "type": "q_function", "action_dim": action_dim, "layers": []}
    else:
        raise ValueError(f"Unsupported Isaac Lab algo: {algo}")
    return ModelBuilder.from_dict(cfg)


def _normalize_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = obs.get("policy") if "policy" in obs else obs.get("state")
    if state is None:
        first_key = next(iter(obs))
        state = obs[first_key]
    return {"state": np.asarray(state)}


def _action_dim_from(env) -> int:
    shape = getattr(env.act_space, "shape", None) or ()
    return int(np.prod(shape)) if shape else 1


def _to_tensor_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {"state_enc": torch.from_numpy(np.asarray(obs["state"])).float()}


def _run_on_policy_task(task_name: str, algo: str, num_envs: int = 1, n_steps: int = 6) -> None:
    omni.usd.get_context().new_stage()
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    env_cfg = parse_env_cfg(task_name, device="cpu", num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None
    wrapped = IsaacLabWrapper(env)

    obs, _ = wrapped.reset(seed=0)
    obs = _normalize_obs(obs)
    obs_dim = int(np.asarray(obs["state"]).shape[-1])
    action_dim = _action_dim_from(wrapped)
    model = _build_model(obs_dim, action_dim, algo)
    if algo == "ppo":
        agent = PPO(model, PPOConfig(n_steps=n_steps, num_envs=num_envs, batch_size=max(4, num_envs), n_epochs=1), device="cpu")
    else:
        agent = A2C(model, A2CConfig(n_steps=n_steps, num_envs=num_envs, batch_size=max(4, num_envs)), device="cpu")

    for _ in range(n_steps):
        action, log_prob, value, _ = agent.predict(_to_tensor_obs(obs))
        next_obs, reward, done, truncated, _ = wrapped.step(action.detach().cpu().numpy())
        next_obs = _normalize_obs(next_obs)
        agent.buffer.add(
            obs=obs,
            action=action.detach().cpu().numpy(),
            reward=np.asarray(reward),
            done=np.asarray(done),
            log_prob=log_prob.detach().cpu().numpy() if log_prob is not None else None,
            value=value.detach().cpu().numpy() if value is not None else None,
        )
        obs = next_obs

    _, _, last_value, _ = agent.predict(_to_tensor_obs(obs))
    agent.buffer.compute_returns_and_advantages(
        last_value=last_value.detach().cpu().numpy() if last_value is not None else 0.0
    )
    metrics = agent.update()
    assert metrics, f"No {algo.upper()} metrics returned for {task_name}"
    assert all(np.isfinite(v) for v in metrics.values()), f"Non-finite {algo.upper()} metric for {task_name}: {metrics}"
    wrapped.close()
    print(f"[PASS] {task_name} headless {algo.upper()} update ok")


def _run_off_policy_task(task_name: str, algo: str, steps: int = 16, warmup: int = 8) -> None:
    omni.usd.get_context().new_stage()
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    env_cfg = parse_env_cfg(task_name, device="cpu", num_envs=1)
    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None
    wrapped = IsaacLabWrapper(env)

    if not isinstance(wrapped.act_space, gym.spaces.Box):
        wrapped.close()
        print(f"[SKIP] {task_name} headless {algo.upper()} requires continuous actions")
        return

    obs, _ = wrapped.reset(seed=0)
    obs = _normalize_obs(obs)
    obs_dim = int(np.asarray(obs["state"]).shape[-1])
    action_dim = _action_dim_from(wrapped)
    model = _build_model(obs_dim, action_dim, algo)
    target = copy.deepcopy(model)

    if algo == "sac":
        agent = SAC(model, target, SACConfig(buffer_size=128, batch_size=8, action_dim=action_dim, learning_starts=0), device="cpu")
    else:
        agent = DDPG(model, target, DDPGConfig(buffer_size=128, batch_size=8, action_dim=action_dim, learning_starts=0), device="cpu")

    for step in range(steps):
        if step < warmup:
            action_np = np.asarray(wrapped.act_space.sample(), dtype=np.float32)
            if action_np.ndim == 1:
                action_np = np.expand_dims(action_np, axis=0)
        else:
            action, _, _, _ = agent.predict(_to_tensor_obs(obs))
            action_np = action.detach().cpu().numpy()

        next_obs, reward, done, truncated, _ = wrapped.step(action_np)
        next_obs = _normalize_obs(next_obs)
        reward_f = float(np.asarray(reward).reshape(-1)[0])
        done_f = bool(np.asarray(done).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])
        action_store = np.asarray(action_np[0] if np.asarray(action_np).ndim > 1 else action_np, dtype=np.float32)
        agent.buffer.add(
            obs={"state": np.asarray(obs["state"])[0]},
            action=action_store,
            reward=np.array([reward_f], dtype=np.float32),
            next_obs={"state": np.asarray(next_obs["state"])[0]},
            done=np.array([done_f], dtype=bool),
        )
        obs = next_obs

    metrics = agent.update()
    assert metrics, f"No {algo.upper()} metrics returned for {task_name}"
    assert all(np.isfinite(v) for v in metrics.values()), f"Non-finite {algo.upper()} metric for {task_name}: {metrics}"
    wrapped.close()
    print(f"[PASS] {task_name} headless {algo.upper()} update ok")


def main() -> int:
    for task_name in TASKS:
        for algo in ON_POLICY_ALGOS:
            _run_on_policy_task(task_name, algo)
        for algo in OFF_POLICY_ALGOS:
            _run_off_policy_task(task_name, algo)
    print("All Isaac Lab headless multi-algorithm tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
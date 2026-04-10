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

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper
from srl.registry.builder import ModelBuilder


TASKS = [
    ("Isaac-Cartpole-v0", 8, 4, 1),
    ("Isaac-Ant-v0", 8, 4, 8),
    ("Isaac-Humanoid-v0", 8, 4, 21),
]


def _build_model(obs_dim: int, action_dim: int):
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
        "actor": {"name": "actor", "type": "gaussian", "action_dim": action_dim},
        "critic": {"name": "critic", "type": "value", "layers": []},
        "losses": [],
    }
    return ModelBuilder.from_dict(cfg)


def _run_task(task_name: str, num_envs: int, n_steps: int, action_dim: int) -> None:
    omni.usd.get_context().new_stage()
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)

    env_cfg = parse_env_cfg(task_name, device="cpu", num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None
    wrapped = IsaacLabWrapper(env)

    obs, _ = wrapped.reset(seed=0)
    state = obs.get("policy") if "policy" in obs else obs.get("state")
    if state is None:
        first_key = next(iter(obs))
        state = obs[first_key]
        obs = {"state": state}
    elif "state" not in obs:
        obs = {"state": state}

    obs_dim = int(np.asarray(obs["state"]).shape[-1])
    model = _build_model(obs_dim, action_dim)
    agent = PPO(model, PPOConfig(n_steps=n_steps, num_envs=num_envs, batch_size=max(4, num_envs), n_epochs=1), device="cpu")

    for _ in range(n_steps):
        obs_t = {"state": torch.from_numpy(np.asarray(obs["state"])).float()}
        action, log_prob, value, _ = agent.predict({"state_enc": obs_t["state"]})
        next_obs, reward, done, truncated, _ = wrapped.step(action.detach().cpu().numpy())
        state = next_obs.get("policy") if "policy" in next_obs else next_obs.get("state")
        if state is None:
            state = next_obs[next(iter(next_obs))]
        next_obs = {"state": state}
        agent.buffer.add(
            obs=obs,
            action=action.detach().cpu().numpy(),
            reward=np.asarray(reward),
            done=np.asarray(done),
            log_prob=log_prob.detach().cpu().numpy() if log_prob is not None else None,
            value=value.detach().cpu().numpy() if value is not None else None,
        )
        obs = next_obs

    last_obs = {"state_enc": torch.from_numpy(np.asarray(obs["state"])).float()}
    _, _, last_value, _ = agent.predict(last_obs)
    agent.buffer.compute_returns_and_advantages(
        last_value=last_value.detach().cpu().numpy() if last_value is not None else 0.0
    )
    metrics = agent.update()
    assert metrics, f"No PPO metrics returned for {task_name}"
    assert all(np.isfinite(v) for v in metrics.values()), f"Non-finite metric for {task_name}: {metrics}"
    wrapped.close()
    print(f"[PASS] {task_name} headless PPO update ok")


def main() -> int:
    for task_name, num_envs, n_steps, action_dim in TASKS:
        _run_task(task_name, num_envs, n_steps, action_dim)
    print("All Isaac Lab headless tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Deep training tests across runnable environments and algorithms.

These tests do real rollouts and optimizer updates rather than simple reset/step
smoke checks. They validate that the model builder, wrappers, buffers, and
algorithms work together for each runnable environment config.
"""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from srl.algorithms.a2c import A2C
from srl.algorithms.a3c import A3C
from srl.algorithms.ddpg import DDPG
from srl.algorithms.ppo import PPO
from srl.algorithms.sac import SAC
from srl.core.config import A2CConfig, A3CConfig, DDPGConfig, PPOConfig, SACConfig
from srl.envs.goal_env_wrapper import GoalEnvWrapper
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.racecar_wrapper import RacecarWrapper
from srl.registry.builder import ModelBuilder

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "envs"

try:
    import gymnasium_robotics

    gymnasium_robotics.register_robotics_envs()
    HAS_ROBOTICS = True
except Exception:
    HAS_ROBOTICS = False

try:
    import racecar_gym  # noqa: F401

    HAS_RACECAR = True
except Exception:
    HAS_RACECAR = False


def _env_params() -> list:
    params = []
    for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        env_id = cfg.get("env_id") or cfg.get("train", {}).get("env_id")
        env_type = cfg.get("env_type") or cfg.get("train", {}).get("env_type", "flat")
        if env_id:
            params.append(pytest.param(env_id, env_type, id=yaml_path.stem))
    return params


def _make_wrapped_env(env_id: str, env_type: str):
    if env_type == "goal":
        if not HAS_ROBOTICS:
            pytest.skip("gymnasium_robotics not installed")
        return GoalEnvWrapper(gym.make(env_id))

    if env_type == "racecar":
        if not HAS_RACECAR:
            pytest.skip("racecar_gym unavailable on this interpreter")
        return RacecarWrapper(gym.make(env_id))

    if env_type == "isaaclab":
        pytest.skip("Isaac Lab requires Isaac Sim runtime")

    base_env = gym.make(env_id)
    if isinstance(base_env.observation_space, gym.spaces.Box):
        shape = base_env.observation_space.shape or ()
        if len(shape) > 1:
            base_env = gym.wrappers.FlattenObservation(base_env)
    return GymnasiumWrapper(base_env)


def _obs_dim_from(obs: dict[str, np.ndarray]) -> int:
    return int(np.asarray(obs["state"]).size)


def _action_dim_from(env) -> int:
    return int(np.prod(env.act_space.shape))


def _build_model(obs_dim: int, action_dim: int, algo: str):
    cfg = {
        "encoders": [
            {
                "name": "state_enc",
                "type": "mlp",
                "input_dim": obs_dim,
                "latent_dim": 64,
                "layers": [
                    {"out_features": 64, "activation": "relu", "norm": "none"},
                ],
            }
        ],
        "flows": ["state_enc -> actor", "state_enc -> critic"],
        "losses": [],
    }
    if algo in {"ppo", "a2c", "a3c"}:
        cfg["actor"] = {"name": "actor", "type": "gaussian", "action_dim": action_dim}
        cfg["critic"] = {"name": "critic", "type": "value", "layers": []}
    elif algo == "sac":
        cfg["actor"] = {"name": "actor", "type": "squashed_gaussian", "action_dim": action_dim}
        cfg["critic"] = {"name": "critic", "type": "twin_q", "action_dim": action_dim, "layers": []}
    elif algo == "ddpg":
        cfg["actor"] = {"name": "actor", "type": "deterministic", "action_dim": action_dim}
        cfg["critic"] = {"name": "critic", "type": "q_function", "action_dim": action_dim, "layers": []}
    else:
        raise ValueError(f"Unsupported algo {algo}")
    return ModelBuilder.from_dict(cfg)


def _to_tensor_obs(obs: dict[str, np.ndarray], batch: bool) -> dict[str, torch.Tensor]:
    tensor_obs: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        arr = np.asarray(value, dtype=np.float32)
        if batch:
            arr = np.expand_dims(arr, axis=0)
        tensor_obs[key] = torch.from_numpy(arr).float()
    return tensor_obs


def _run_on_policy_update(env, algo: str) -> None:
    obs, _ = env.reset(seed=0)
    obs_dim = _obs_dim_from(obs)
    action_dim = _action_dim_from(env)
    model = _build_model(obs_dim, action_dim, algo)

    if algo == "ppo":
        agent = PPO(model, PPOConfig(n_steps=8, num_envs=1, batch_size=4, n_epochs=1), device="cpu")
    else:
        agent = A2C(model, A2CConfig(n_steps=8, num_envs=1, batch_size=4), device="cpu")

    for _ in range(agent.cfg.n_steps):
        obs_t = _to_tensor_obs(obs, batch=False)
        action, log_prob, value, _ = agent.predict(obs_t)
        action_np = action.detach().cpu().numpy()
        next_obs, reward, done, truncated, _ = env.step(action_np)
        agent.buffer.add(
            obs=obs,
            action=action_np,
            reward=np.asarray(reward),
            done=np.asarray(done),
            log_prob=log_prob.detach().cpu().numpy() if log_prob is not None else None,
            value=value.detach().cpu().numpy() if value is not None else None,
        )
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset(seed=0)

    last_obs = _to_tensor_obs(obs, batch=False)
    _, _, last_value, _ = agent.predict(last_obs)
    agent.buffer.compute_returns_and_advantages(
        last_value=last_value.detach().cpu().numpy() if last_value is not None else 0.0
    )
    metrics = agent.update()
    assert metrics
    assert all(np.isfinite(v) for v in metrics.values())


def _run_off_policy_update(env, algo: str) -> None:
    obs, _ = env.reset(seed=0)
    obs_dim = _obs_dim_from(obs)
    action_dim = _action_dim_from(env)
    model = _build_model(obs_dim, action_dim, algo)
    target = copy.deepcopy(model)

    if algo == "sac":
        agent = SAC(
            model,
            target,
            SACConfig(buffer_size=64, batch_size=8, action_dim=action_dim, learning_starts=0),
            device="cpu",
        )
    else:
        agent = DDPG(
            model,
            target,
            DDPGConfig(buffer_size=64, batch_size=8, action_dim=action_dim, learning_starts=0),
            device="cpu",
        )

    for step in range(16):
        if step < 8:
            action_np = env.act_space.sample()
        else:
            obs_t = _to_tensor_obs(obs, batch=True)
            action, _, _, _ = agent.predict(obs_t)
            action_np = action.squeeze(0).detach().cpu().numpy()

        next_obs, reward, done, truncated, _ = env.step(action_np)
        agent.buffer.add(
            obs=obs,
            action=action_np,
            reward=np.array([reward], dtype=np.float32),
            next_obs=next_obs,
            done=np.array([done or truncated], dtype=bool),
        )
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset(seed=0)

    metrics = agent.update()
    assert metrics
    assert all(np.isfinite(v) for v in metrics.values())


def _run_a3c_train(env_id: str, env_type: str) -> None:
    probe_env = _make_wrapped_env(env_id, env_type)
    obs, _ = probe_env.reset(seed=0)
    model = _build_model(_obs_dim_from(obs), _action_dim_from(probe_env), "a3c")
    probe_env.close()

    agent = A3C(model, A3CConfig(n_steps=4, n_workers=1, batch_size=4), device="cpu")
    agent.train(total_timesteps=4, env_fn=partial(_make_wrapped_env, env_id, env_type))
    assert agent._global_step >= 4


@pytest.mark.parametrize("env_id,env_type", _env_params())
def test_deep_env_training_update_matrix(env_id: str, env_type: str) -> None:
    env = _make_wrapped_env(env_id, env_type)
    try:
        _run_on_policy_update(env, "ppo")
        _run_on_policy_update(env, "a2c")
        _run_off_policy_update(env, "sac")
        _run_off_policy_update(env, "ddpg")
    finally:
        env.close()


@pytest.mark.parametrize("env_id,env_type", _env_params())
def test_deep_env_a3c_matrix(env_id: str, env_type: str) -> None:
    _run_a3c_train(env_id, env_type)
"""Stable-Baselines3 helpers for training and evaluating the PPO attacker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO


def create_ppo_model(env: Any, seed: int = 42) -> PPO:
    """Create a minimally configured PPO model for the attack environment."""
    return PPO("MlpPolicy", env, verbose=1, seed=seed)


def train_ppo(model: PPO, total_timesteps: int = 5000) -> PPO:
    """Train a PPO model for a positive number of timesteps and return it."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    model.learn(total_timesteps=int(total_timesteps))
    return model


def save_ppo(model: PPO, path: str | Path) -> None:
    """Save a PPO checkpoint, creating its parent directory when necessary."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_path))


def load_ppo(path: str | Path, env: Any) -> PPO:
    """Load a PPO checkpoint and attach it to ``env``."""
    return PPO.load(str(path), env=env)


def run_episode(model: PPO, env: Any, deterministic: bool = True) -> list[dict[str, Any]]:
    """Run one attack episode and return a tabular-friendly step history."""
    observation, _ = env.reset()
    history: list[dict[str, Any]] = []
    while True:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(int(action))
        record = dict(info)
        record["reward"] = float(reward)
        record["terminated"] = bool(terminated)
        record["truncated"] = bool(truncated)
        history.append(record)
        if terminated or truncated:
            return history

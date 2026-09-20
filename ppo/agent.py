"""Stable-Baselines3 PPO construction and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO


def create_ppo_agent(env: Any, seed: int = 42, device: str = "auto", **kwargs: Any) -> PPO:
    """Create an MLP PPO agent with conservative defaults."""
    settings = {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
    }
    settings.update(kwargs)
    return PPO("MlpPolicy", env, seed=seed, device=device, **settings)


def load_ppo_agent(model_path: str | Path, env: Any = None, device: str = "auto") -> PPO:
    """Load a PPO checkpoint and optionally attach an environment."""
    path = Path(model_path)
    candidate = path if path.is_file() else path.with_suffix(".zip")
    if not candidate.is_file():
        raise FileNotFoundError(f"PPO model checkpoint not found: {path}")
    return PPO.load(str(candidate), env=env, device=device)

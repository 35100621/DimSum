from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.ppo.environment import DeepfakeAttackEnv
from src.ppo.mock_detector import MockDetector
from src.ppo.train import create_ppo_model, load_ppo, run_episode, save_ppo, train_ppo


@pytest.fixture
def env():
    x = np.linspace(64, 224, 16, dtype=np.uint8)
    pixels = np.tile(x, (16, 1))
    image = Image.fromarray(np.stack([pixels] * 3, axis=2))
    return DeepfakeAttackEnv(image, MockDetector(), max_steps=2, seed=42)


def test_training_helpers_validate_and_save(tmp_path: Path, env):
    model = create_ppo_model(env, seed=42)
    with pytest.raises(ValueError):
        train_ppo(model, total_timesteps=0)
    checkpoint = tmp_path / "nested" / "ppo_test"
    save_ppo(model, checkpoint)
    assert checkpoint.with_suffix(".zip").exists()
    loaded = load_ppo(checkpoint, env)
    history = run_episode(loaded, env)
    assert history
    assert {"action_name", "reward", "terminated", "truncated"} <= history[0].keys()

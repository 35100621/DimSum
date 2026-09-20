from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces
from PIL import Image
from stable_baselines3.common.env_checker import check_env

from ppo.environment import DeepfakeAttackEnv


class SequenceVictim:
    def __init__(self, fake_probabilities):
        self.probabilities = iter(fake_probabilities)

    def predict_image(self, image):
        fake = float(next(self.probabilities))
        return {"label": "FAKE" if fake >= 0.5 else "REAL", "real_probability": 1.0 - fake, "fake_probability": fake}


class ImageVictim:
    def predict_image(self, image):
        fake = float(np.asarray(image, dtype=float).mean() / 255.0)
        return {"label": "FAKE" if fake >= 0.5 else "REAL", "real_probability": 1.0 - fake, "fake_probability": fake}


@pytest.fixture
def image_path(tmp_path: Path):
    path = tmp_path / "fake.png"
    Image.fromarray(np.full((32, 32, 3), 220, dtype=np.uint8)).save(path)
    return path


def test_reset_and_step_contract(image_path):
    env = DeepfakeAttackEnv([image_path], ImageVictim(), seed=42)
    observation, reset_info = env.reset()
    transition = env.step(np.array([0, 0]))
    assert observation.shape == (6,) and observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert len(transition) == 5 and np.isfinite(transition[1])
    assert isinstance(reset_info, dict)
    required = {"image_path", "action_name", "strength", "step", "initial_fake_probability", "previous_fake_probability", "current_fake_probability", "current_real_probability", "ssim", "perturbation", "attack_success"}
    assert required <= transition[4].keys()


def test_success_terminates(image_path):
    env = DeepfakeAttackEnv([image_path], SequenceVictim([0.9, 0.4]))
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated and not truncated and info["attack_success"] and reward > 1.0


def test_max_steps_truncates(image_path):
    env = DeepfakeAttackEnv([image_path], SequenceVictim([0.9, 0.8, 0.7]), max_steps=2)
    env.reset()
    assert not env.step(np.array([0, 0]))[3]
    _, _, terminated, truncated, _ = env.step(np.array([0, 0]))
    assert not terminated and truncated


def test_environment_passes_sb3_checker(image_path):
    check_env(DeepfakeAttackEnv([image_path], ImageVictim(), seed=42), warn=True)

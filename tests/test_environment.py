import numpy as np
import pytest
from gymnasium import spaces
from PIL import Image
from stable_baselines3.common.env_checker import check_env

from src.ppo.environment import DeepfakeAttackEnv
from src.ppo.mock_detector import MockDetector


class SequenceDetector:
    def __init__(self, scores):
        self.scores = iter(scores)

    def predict(self, image):
        return next(self.scores)


@pytest.fixture
def sample_image():
    x = np.linspace(64, 224, 32, dtype=np.uint8)
    pixels = np.tile(x, (24, 1))
    return Image.fromarray(np.stack([pixels] * 3, axis=2), mode="RGB")


def test_reset_and_step_follow_gymnasium_api(sample_image):
    env = DeepfakeAttackEnv(sample_image, MockDetector(), seed=42)
    obs, info = env.reset()
    transition = env.step(0)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 7
    assert len(transition) == 5


def test_successful_attack_terminates(sample_image):
    env = DeepfakeAttackEnv(sample_image, SequenceDetector([0.9, 0.4]))
    env.reset()
    _, _, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert truncated is False
    assert info["success"] is True


def test_max_steps_truncates_unsuccessful_episode(sample_image):
    env = DeepfakeAttackEnv(sample_image, SequenceDetector([0.9, 0.8, 0.7]), max_steps=2)
    env.reset()
    assert env.step(0)[3] is False
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is True


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), "invalid"])
def test_detector_output_is_validated(sample_image, score):
    env = DeepfakeAttackEnv(sample_image, SequenceDetector([score]))
    with pytest.raises(ValueError):
        env.reset()


def test_environment_passes_sb3_checker(sample_image):
    env = DeepfakeAttackEnv(sample_image, MockDetector(), seed=42)
    check_env(env, warn=True)

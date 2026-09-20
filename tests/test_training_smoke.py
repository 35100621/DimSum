from pathlib import Path

import numpy as np
from PIL import Image

from ppo.agent import create_ppo_agent, load_ppo_agent
from ppo.environment import DeepfakeAttackEnv


class MockVictim:
    def predict_image(self, image):
        fake = float(np.clip(np.asarray(image).mean() / 255.0, 0.51, 0.99))
        return {"label": "FAKE", "real_probability": 1 - fake, "fake_probability": fake}


def test_tiny_ppo_training(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((24, 24, 3), 200, dtype=np.uint8)).save(image_path)
    env = DeepfakeAttackEnv([image_path], MockVictim(), max_steps=2, seed=42)
    model = create_ppo_agent(env, seed=42, device="cpu", n_steps=8, batch_size=4, verbose=0)
    model.learn(total_timesteps=8)
    checkpoint = tmp_path / "ppo_smoke"
    model.save(checkpoint)
    loaded = load_ppo_agent(checkpoint, env=env, device="cpu")
    observation, _ = env.reset(seed=42)
    action, _ = loaded.predict(observation, deterministic=True)
    assert env.action_space.contains(action)

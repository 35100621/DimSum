import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ppo.environment import DeepfakeAttackEnv
from victim.detector import VictimDetector


ROOT = Path("/Users/hgtan/Desktop/DeepfakeBench_Handover")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPFAKEBENCH_INTEGRATION") != "1",
    reason="set RUN_DEEPFAKEBENCH_INTEGRATION=1 to load the external Xception checkpoint",
)
def test_real_xception_to_environment_transition():
    detector = VictimDetector(
        ROOT,
        weights_path=ROOT / "training/weights/xception_best.pth",
        device="cpu",
    )
    assert detector.model.training is False
    assert not any(parameter.requires_grad for parameter in detector.model.parameters())
    fake_path = ROOT / "datasets/rgb/FF_small/fake/Deepfakes_0000.png"
    prediction = detector.predict_path(fake_path)
    assert prediction["label"] in {"REAL", "FAKE"}
    assert prediction["real_probability"] + prediction["fake_probability"] == pytest.approx(1.0)

    image = torch.rand(1, 3, 224, 224, requires_grad=True)
    detector.predict_tensor(image, differentiable=True)[0, 1].backward()
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())

    env = DeepfakeAttackEnv([fake_path], detector, max_steps=1, seed=42)
    observation, _ = env.reset(seed=42)
    transition = env.step(np.array([1, 2]))
    assert observation.shape == (6,)
    assert np.isfinite(transition[1])
    assert 0.0 <= transition[4]["current_fake_probability"] <= 1.0

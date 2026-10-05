import numpy as np
import pytest
import torch
from PIL import Image

from ppo.actions import fgsm_attack, frequency_lowpass_attack, pgd_attack


class DifferentiableVictim:
    device = torch.device("cpu")
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
    def predict_tensor(self, image, differentiable=False):
        fake = torch.sigmoid(self.weight * (image.mean(dim=(1, 2, 3)) - 0.5))
        return torch.stack((1 - fake, fake), dim=1)


@pytest.fixture
def image():
    return Image.fromarray(np.full((24, 32, 3), 190, dtype=np.uint8), "RGB")


@pytest.mark.parametrize("attack", [fgsm_attack, pgd_attack])
def test_gradient_attack_is_bounded_valid_and_freezes_model(image, attack):
    victim = DifferentiableVictim()
    before = np.asarray(image, dtype=np.int16)
    result = attack(image, victim, epsilon=8 / 255)
    delta = np.abs(np.asarray(result, dtype=np.int16) - before)
    assert result.mode == "RGB" and result.size == image.size
    assert delta.max() <= 8
    assert victim.weight.grad is None and not victim.weight.requires_grad


def test_fgsm_targets_real_direction(image):
    victim = DifferentiableVictim()
    before = victim.predict_tensor(torch.tensor(np.asarray(image) / 255).permute(2, 0, 1).float().unsqueeze(0))[0, 0]
    attacked = fgsm_attack(image, victim, 8 / 255)
    tensor = torch.tensor(np.asarray(attacked) / 255).permute(2, 0, 1).float().unsqueeze(0)
    assert victim.predict_tensor(tensor)[0, 0] > before


def test_pgd_is_deterministic(image):
    victim = DifferentiableVictim()
    np.testing.assert_array_equal(pgd_attack(image, victim), pgd_attack(image, victim))


def test_frequency_attack_is_bounded_and_deterministic(image):
    first = frequency_lowpass_attack(image, 8 / 255)
    second = frequency_lowpass_attack(image, 8 / 255)
    np.testing.assert_array_equal(first, second)
    assert np.abs(np.asarray(first, dtype=np.int16) - np.asarray(image, dtype=np.int16)).max() <= 8

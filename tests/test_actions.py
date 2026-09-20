import numpy as np
import pytest
from PIL import Image

from ppo.actions import ACTION_NAMES, apply_action


@pytest.fixture
def sample_image():
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    return Image.fromarray(pixels, mode="RGB")


@pytest.mark.parametrize("action_id", ACTION_NAMES)
def test_actions_preserve_rgb_dimensions_and_range(sample_image, action_id):
    result = apply_action(sample_image, action_id, 0.75, np.random.default_rng(7))
    assert result.mode == "RGB"
    assert result.size == sample_image.size
    pixels = np.asarray(result)
    assert pixels.dtype == np.uint8 and pixels.min() >= 0 and pixels.max() <= 255


def test_noop_is_pixel_identical(sample_image):
    np.testing.assert_array_equal(np.asarray(apply_action(sample_image, 0, 1.0)), sample_image)


def test_seeded_noise_is_reproducible(sample_image):
    first = apply_action(sample_image, 1, 0.5, np.random.default_rng(42))
    second = apply_action(sample_image, 1, 0.5, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("strength", [-0.1, 1.1, np.nan])
def test_invalid_strength_raises(sample_image, strength):
    with pytest.raises(ValueError):
        apply_action(sample_image, 1, strength)


def test_invalid_action_raises(sample_image):
    with pytest.raises(ValueError):
        apply_action(sample_image, 7, 0.5)

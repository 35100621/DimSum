import numpy as np
import pytest
from PIL import Image

from src.attacks.perturbations import (
    ACTIONS,
    apply_action,
    gaussian_blur,
    jpeg_compression,
    resize_attack,
)


@pytest.fixture
def sample_image():
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    return Image.fromarray(pixels, mode="RGB")


@pytest.mark.parametrize("action", ACTIONS)
def test_all_actions_preserve_rgb_size_and_original(sample_image, action):
    before = np.asarray(sample_image).copy()
    result = apply_action(sample_image, np.int64(action), rng=np.random.default_rng(7))
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == sample_image.size
    np.testing.assert_array_equal(np.asarray(sample_image), before)
    assert result is not sample_image


def test_invalid_action_raises(sample_image):
    with pytest.raises(ValueError):
        apply_action(sample_image, 7)


@pytest.mark.parametrize("kernel", [0, -1, 2, 1.5])
def test_invalid_blur_kernel_raises(sample_image, kernel):
    with pytest.raises(ValueError):
        gaussian_blur(sample_image, kernel)


@pytest.mark.parametrize("quality", [0, 101, 50.5])
def test_invalid_jpeg_quality_raises(sample_image, quality):
    with pytest.raises(ValueError):
        jpeg_compression(sample_image, quality)


@pytest.mark.parametrize("scale", [0, -0.1, 1.1])
def test_invalid_resize_scale_raises(sample_image, scale):
    with pytest.raises(ValueError):
        resize_attack(sample_image, scale)

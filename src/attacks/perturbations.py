"""Controlled image perturbations used by the PPO attacker."""

from __future__ import annotations

from io import BytesIO
from numbers import Integral

import cv2
import numpy as np
from PIL import Image, ImageEnhance


ACTIONS = {
    0: "none",
    1: "noise",
    2: "blur",
    3: "brightness_up",
    4: "brightness_down",
    5: "jpeg",
    6: "resize",
}


def _as_rgb(image: Image.Image) -> Image.Image:
    """Return an independent RGB copy of a PIL image."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    return image.convert("RGB").copy()


def add_noise(
    image: Image.Image,
    std: float = 5,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Add zero-mean Gaussian noise with standard deviation ``std``."""
    if std < 0:
        raise ValueError("std must be non-negative")
    rgb = _as_rgb(image)
    generator = rng if rng is not None else np.random.default_rng()
    pixels = np.asarray(rgb, dtype=np.float32)
    noise = generator.normal(0.0, std, size=pixels.shape)
    noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def gaussian_blur(image: Image.Image, kernel_size: int = 3) -> Image.Image:
    """Apply an OpenCV Gaussian blur with an odd, positive kernel size."""
    if not isinstance(kernel_size, Integral) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    pixels = np.asarray(_as_rgb(image))
    blurred = cv2.GaussianBlur(pixels, (int(kernel_size), int(kernel_size)), 0)
    return Image.fromarray(blurred, mode="RGB")


def change_brightness(image: Image.Image, factor: float = 1.05) -> Image.Image:
    """Scale image brightness by a strictly positive factor."""
    if factor <= 0:
        raise ValueError("factor must be greater than zero")
    return ImageEnhance.Brightness(_as_rgb(image)).enhance(float(factor)).convert("RGB")


def jpeg_compression(image: Image.Image, quality: int = 80) -> Image.Image:
    """Round-trip an image through an in-memory JPEG encoding."""
    if not isinstance(quality, Integral) or not 1 <= quality <= 100:
        raise ValueError("quality must be an integer between 1 and 100")
    rgb = _as_rgb(image)
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def resize_attack(image: Image.Image, scale: float = 0.9) -> Image.Image:
    """Resize down by ``scale`` and restore the original dimensions."""
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be greater than zero and at most one")
    rgb = _as_rgb(image)
    width, height = rgb.size
    scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced = rgb.resize(scaled_size, Image.Resampling.LANCZOS)
    return reduced.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")


def apply_action(
    image: Image.Image,
    action: int,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Apply one discrete attack action without mutating ``image``."""
    if not isinstance(action, Integral) or int(action) not in ACTIONS:
        raise ValueError(f"action must be one of {tuple(ACTIONS)}")

    action_id = int(action)
    if action_id == 0:
        return _as_rgb(image)
    if action_id == 1:
        return add_noise(image, std=5, rng=rng)
    if action_id == 2:
        return gaussian_blur(image, kernel_size=3)
    if action_id == 3:
        return change_brightness(image, factor=1.05)
    if action_id == 4:
        return change_brightness(image, factor=0.95)
    if action_id == 5:
        return jpeg_compression(image, quality=80)
    return resize_attack(image, scale=0.9)

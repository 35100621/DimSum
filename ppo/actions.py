"""Bounded image transformations available to the PPO policy."""

from __future__ import annotations

from io import BytesIO
from numbers import Integral

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ACTION_NAMES = {
    0: "noop",
    1: "gaussian_noise",
    2: "gaussian_blur",
    3: "brightness_down",
    4: "brightness_up",
    5: "jpeg_compression",
    6: "resize_restore",
}


def _to_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
            raise ValueError("NumPy image must have shape (H, W, C)")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("NumPy image contains non-finite pixels")
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array[..., :3], mode="RGB")
    raise TypeError("image must be a PIL image or NumPy array")


def apply_action(
    image: Image.Image | np.ndarray,
    action_id: int,
    strength: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Apply one bounded action while preserving size and RGB channels."""
    if not isinstance(action_id, Integral) or int(action_id) not in ACTION_NAMES:
        raise ValueError(f"action_id must be one of {tuple(ACTION_NAMES)}")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    rgb = _to_rgb(image)
    original_size = rgb.size
    action_id = int(action_id)

    if action_id == 0 or strength == 0.0:
        result = rgb.copy()
    elif action_id == 1:
        generator = rng if rng is not None else np.random.default_rng()
        pixels = np.asarray(rgb, dtype=np.float32)
        noise = generator.normal(0.0, 12.0 * strength, pixels.shape)
        result = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), "RGB")
    elif action_id == 2:
        result = rgb.filter(ImageFilter.GaussianBlur(radius=2.0 * strength))
    elif action_id == 3:
        result = ImageEnhance.Brightness(rgb).enhance(1.0 - 0.25 * strength)
    elif action_id == 4:
        result = ImageEnhance.Brightness(rgb).enhance(1.0 + 0.25 * strength)
    elif action_id == 5:
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=max(35, round(100 - 65 * strength)))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            result = decoded.convert("RGB").copy()
    else:
        width, height = original_size
        scale = 1.0 - 0.65 * strength
        reduced_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = rgb.resize(reduced_size, Image.Resampling.LANCZOS)
        result = reduced.resize(original_size, Image.Resampling.LANCZOS)

    result = result.convert("RGB")
    if result.size != original_size:
        raise RuntimeError("Image action failed to preserve dimensions")
    return result

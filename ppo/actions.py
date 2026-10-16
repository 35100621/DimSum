"""Bounded image transformations available to the PPO policy."""

from __future__ import annotations

from io import BytesIO
from numbers import Integral

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch


ACTION_NAMES = {
    0: "noop",
    1: "gaussian_noise",
    2: "gaussian_blur",
    3: "brightness_down",
    4: "brightness_up",
    5: "jpeg_compression",
    6: "resize_restore",
}
EXPANDED_ACTION_NAMES = {
    **ACTION_NAMES,
    7: "fgsm",
    8: "pgd",
    9: "frequency_lowpass",
}


def _pixels(image: Image.Image | np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return one RGB image as a differentiable BCHW tensor in [0, 1]."""
    array = np.asarray(_to_rgb(image), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), "RGB")


def fgsm_attack(
    image: Image.Image | np.ndarray,
    detector: object,
    epsilon: float = 8 / 255,
) -> Image.Image:
    """Target class 0 (REAL) with a bounded single gradient step."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    device = getattr(detector, "device", "cpu")
    original = _pixels(image, device)
    attacked = original.clone().detach().requires_grad_(True)
    real_probability = detector.predict_tensor(attacked, differentiable=True)[0, 0]
    loss = -torch.log(real_probability.clamp_min(1e-12))
    gradient = torch.autograd.grad(loss, attacked)[0]
    result = attacked - epsilon * gradient.sign()
    return _pil(result)


def pgd_attack(
    image: Image.Image | np.ndarray,
    detector: object,
    epsilon: float = 8 / 255,
    step_size: float = 2 / 255,
    iterations: int = 5,
) -> Image.Image:
    """Target class 0 (REAL) with deterministic projected gradient descent."""
    if not 0.0 <= epsilon <= 1.0 or not 0.0 < step_size <= 1.0:
        raise ValueError("epsilon and step_size must be valid pixel-space bounds")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    device = getattr(detector, "device", "cpu")
    original = _pixels(image, device)
    attacked = original.clone().detach()
    for _ in range(iterations):
        attacked.requires_grad_(True)
        real_probability = detector.predict_tensor(attacked, differentiable=True)[0, 0]
        loss = -torch.log(real_probability.clamp_min(1e-12))
        gradient = torch.autograd.grad(loss, attacked)[0]
        attacked = attacked.detach() - step_size * gradient.sign()
        attacked = torch.maximum(torch.minimum(attacked, original + epsilon), original - epsilon)
        attacked = attacked.clamp(0.0, 1.0).detach()
    return _pil(attacked)


def frequency_lowpass_attack(
    image: Image.Image | np.ndarray,
    epsilon: float = 8 / 255,
    cutoff_fraction: float = 0.35,
) -> Image.Image:
    """Suppress high spatial frequencies and bound the per-pixel change by epsilon."""
    if not 0.0 <= epsilon <= 1.0 or not 0.0 < cutoff_fraction <= 1.0:
        raise ValueError("invalid frequency attack parameters")
    rgb = np.asarray(_to_rgb(image), dtype=np.float32) / 255.0
    height, width, _ = rgb.shape
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = cutoff_fraction * min(height, width) / 2.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2)[..., None]
    spectrum = np.fft.fftshift(np.fft.fft2(rgb, axes=(0, 1)), axes=(0, 1))
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask, axes=(0, 1)), axes=(0, 1)).real
    result = np.clip(rgb + np.clip(filtered - rgb, -epsilon, epsilon), 0.0, 1.0)
    return Image.fromarray(np.rint(result * 255.0).astype(np.uint8), "RGB")


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
    detector: object | None = None,
    expanded: bool = False,
) -> Image.Image:
    """Apply one bounded action while preserving size and RGB channels."""
    names = EXPANDED_ACTION_NAMES if expanded else ACTION_NAMES
    if not isinstance(action_id, Integral) or int(action_id) not in names:
        raise ValueError(f"action_id must be one of {tuple(names)}")
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
    elif action_id == 6:
        width, height = original_size
        scale = 1.0 - 0.65 * strength
        reduced_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = rgb.resize(reduced_size, Image.Resampling.LANCZOS)
        result = reduced.resize(original_size, Image.Resampling.LANCZOS)
    elif action_id in {7, 8}:
        if detector is None or not callable(getattr(detector, "predict_tensor", None)):
            raise TypeError("gradient actions require a differentiable detector")
        epsilon = (2.0 + 6.0 * strength) / 255.0
        result = (fgsm_attack(rgb, detector, epsilon) if action_id == 7 else
                  pgd_attack(rgb, detector, epsilon, max(epsilon / 4.0, 1 / 255), 5))
    else:
        result = frequency_lowpass_attack(
            rgb, epsilon=(2.0 + 10.0 * strength) / 255.0,
            cutoff_fraction=0.45 - 0.20 * strength,
        )

    result = result.convert("RGB")
    if result.size != original_size:
        raise RuntimeError("Image action failed to preserve dimensions")
    return result

"""A deterministic image-responsive detector used only for integration tests."""

from __future__ import annotations

import numpy as np
from PIL import Image


class MockDetector:
    """Estimate P(fake) from stable brightness and edge statistics.

    This class is an integration aid, not a real deepfake detector. Its score
    changes under several available perturbations so the environment provides a
    meaningful reward signal during smoke tests.
    """

    def predict(self, image: Image.Image) -> float:
        """Return a deterministic fake probability in the inclusive range [0, 1]."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        luminance = pixels.mean(axis=2)
        brightness = float(luminance.mean())
        horizontal_edges = float(np.abs(np.diff(luminance, axis=1)).mean())
        vertical_edges = float(np.abs(np.diff(luminance, axis=0)).mean())
        edge_strength = min(1.0, 4.0 * (horizontal_edges + vertical_edges))
        score = 0.15 + 0.70 * brightness + 0.15 * edge_strength
        return float(np.clip(score, 0.0, 1.0))

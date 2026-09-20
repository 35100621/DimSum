"""Gymnasium environment connecting image actions to a victim detector."""

from __future__ import annotations

import logging
from numbers import Integral
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image
from skimage.metrics import structural_similarity

from .actions import ACTION_NAMES, apply_action


LOGGER = logging.getLogger(__name__)
N_STRENGTH_LEVELS = 5


class DeepfakeAttackEnv(gym.Env[np.ndarray, np.ndarray]):
    """Compose image transformations to reduce a detector's fake confidence."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_paths: list[str | Path],
        victim_detector: Any,
        image_size: tuple[int, int] = (224, 224),
        max_steps: int = 5,
        attack_target: str = "fake_to_real",
        quality_weight: float = 0.25,
        perturbation_weight: float = 0.10,
        seed: int | None = None,
        strength_levels: int = N_STRENGTH_LEVELS,
    ) -> None:
        super().__init__()
        self.image_paths = [Path(path) for path in image_paths]
        if not self.image_paths:
            raise ValueError("image_paths must contain at least one image")
        missing = next((path for path in self.image_paths if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Input image not found: {missing}")
        if not callable(getattr(victim_detector, "predict_image", None)):
            raise TypeError("victim_detector must provide predict_image(image)")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain positive width and height")
        if not isinstance(max_steps, Integral) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if attack_target != "fake_to_real":
            raise ValueError("Only attack_target='fake_to_real' is currently supported")
        if quality_weight < 0 or perturbation_weight < 0:
            raise ValueError("reward penalty weights must be non-negative")
        if not isinstance(strength_levels, Integral) or strength_levels < 2:
            raise ValueError("strength_levels must be an integer of at least 2")

        self.victim_detector = victim_detector
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_steps = int(max_steps)
        self.attack_target = attack_target
        self.quality_weight = float(quality_weight)
        self.perturbation_weight = float(perturbation_weight)
        self.strength_levels = int(strength_levels)
        self.action_space = spaces.MultiDiscrete([len(ACTION_NAMES), self.strength_levels])
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self._initial_seed = seed

        self.original_image: Image.Image | None = None
        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.initial_prediction: dict[str, Any] | None = None
        self.current_prediction: dict[str, Any] | None = None
        self.step_count = 0
        self.cumulative_distortion = 0.0

    @staticmethod
    def _validate_prediction(prediction: Any) -> dict[str, Any]:
        required = {"label", "real_probability", "fake_probability"}
        if not isinstance(prediction, dict) or not required <= prediction.keys():
            raise ValueError(f"Victim prediction must contain keys: {sorted(required)}")
        label = prediction["label"]
        real = float(prediction["real_probability"])
        fake = float(prediction["fake_probability"])
        if label not in {"REAL", "FAKE"} or not np.isfinite([real, fake]).all():
            raise ValueError("Victim prediction contains invalid values")
        if not 0 <= real <= 1 or not 0 <= fake <= 1 or abs(real + fake - 1.0) >= 1e-5:
            raise ValueError("Victim probabilities must be normalized into [0, 1]")
        return {"label": label, "real_probability": real, "fake_probability": fake}

    def _observation(self) -> np.ndarray:
        assert self.initial_prediction is not None and self.current_prediction is not None
        return np.array(
            [
                self.current_prediction["real_probability"],
                self.current_prediction["fake_probability"],
                self.initial_prediction["real_probability"],
                self.initial_prediction["fake_probability"],
                self.step_count / self.max_steps,
                np.clip(self.cumulative_distortion, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quality(original: Image.Image, attacked: Image.Image) -> tuple[float, float]:
        before = np.asarray(original, dtype=np.float32) / 255.0
        after = np.asarray(attacked, dtype=np.float32) / 255.0
        ssim = structural_similarity(before, after, channel_axis=2, data_range=1.0)
        perturbation = float(np.mean(np.abs(before - after)))
        return float(np.clip(ssim, 0.0, 1.0)), float(np.clip(perturbation, 0.0, 1.0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample and initialize one image episode."""
        effective_seed = self._initial_seed if seed is None else seed
        super().reset(seed=effective_seed)
        requested_index = None if options is None else options.get("image_index")
        index = int(requested_index) if requested_index is not None else int(
            self.np_random.integers(len(self.image_paths))
        )
        if not 0 <= index < len(self.image_paths):
            raise ValueError("options['image_index'] is outside image_paths")
        self.current_image_path = self.image_paths[index]
        try:
            with Image.open(self.current_image_path) as image:
                resized = image.convert("RGB").resize(self.image_size, Image.Resampling.LANCZOS)
        except (OSError, ValueError) as error:
            raise ValueError(f"Unable to load training image: {self.current_image_path}") from error
        self.original_image = resized.copy()
        self.current_image = resized.copy()
        self.step_count = 0
        self.cumulative_distortion = 0.0
        prediction = self._validate_prediction(self.victim_detector.predict_image(self.current_image))
        self.initial_prediction = prediction.copy()
        self.current_prediction = prediction.copy()
        return self._observation(), {
            "image_path": str(self.current_image_path),
            "initial_prediction": self.initial_prediction.copy(),
        }

    def step(
        self, action: np.ndarray | list[int] | tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action and compute confidence, quality, and distortion reward."""
        if self.current_image is None or self.original_image is None or self.current_prediction is None:
            raise RuntimeError("reset() must be called before step()")
        action_array = np.asarray(action)
        if action_array.shape != (2,) or not np.issubdtype(action_array.dtype, np.integer):
            raise ValueError("action must contain integer [action_type, strength_level]")
        if not self.action_space.contains(action_array):
            raise ValueError(f"action is outside {self.action_space}")
        action_id, strength_level = (int(value) for value in action_array)
        strength = strength_level / (self.strength_levels - 1)
        previous_fake = float(self.current_prediction["fake_probability"])
        candidate = apply_action(self.current_image, action_id, strength, rng=self.np_random)
        prediction = self._validate_prediction(self.victim_detector.predict_image(candidate))
        ssim_score, perturbation = self._quality(self.original_image, candidate)
        confidence_gain = previous_fake - prediction["fake_probability"]
        attack_success = bool(
            self.initial_prediction["label"] == "FAKE"
            and self.current_prediction["label"] == "FAKE"
            and prediction["label"] == "REAL"
        )
        reward = confidence_gain - self.quality_weight * (1.0 - ssim_score)
        reward -= self.perturbation_weight * perturbation
        if attack_success:
            reward += 1.0
        reward = float(np.clip(reward, -2.0, 2.0))

        self.current_image = candidate
        self.current_prediction = prediction
        self.step_count += 1
        self.cumulative_distortion = float(
            np.clip(self.cumulative_distortion + perturbation, 0.0, 1.0)
        )
        terminated = bool(attack_success)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        info = {
            "image_path": str(self.current_image_path),
            "action_name": ACTION_NAMES[action_id],
            "strength": float(strength),
            "step": self.step_count,
            "initial_fake_probability": float(self.initial_prediction["fake_probability"]),
            "previous_fake_probability": previous_fake,
            "current_fake_probability": float(prediction["fake_probability"]),
            "current_real_probability": float(prediction["real_probability"]),
            "ssim": ssim_score,
            "perturbation": perturbation,
            "attack_success": attack_success,
        }
        return self._observation(), reward, terminated, truncated, info

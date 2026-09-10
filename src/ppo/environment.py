"""Gymnasium environment for detector-agnostic image perturbation attacks."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from src.attacks.perturbations import ACTIONS, apply_action
from src.ppo.reward import confidence_reduction_reward

RewardFunction = Callable[[float, float], float]


class DeepfakeAttackEnv(gym.Env[np.ndarray, int]):
    """Apply discrete perturbations to reduce a detector's P(fake) score."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        image: Image.Image,
        detector: Any,
        max_steps: int = 5,
        success_threshold: float = 0.5,
        reward_fn: RewardFunction | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not isinstance(max_steps, Integral) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0 and 1")
        if not callable(getattr(detector, "predict", None)):
            raise TypeError("detector must provide a callable predict(image) method")

        self.original_image = image.convert("RGB").copy()
        self.detector = detector
        self.max_steps = int(max_steps)
        self.success_threshold = float(success_threshold)
        self.reward_fn = reward_fn or confidence_reduction_reward
        self.action_space = spaces.Discrete(len(ACTIONS), seed=seed)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_image = self.original_image.copy()
        self.current_confidence = 0.0
        self.step_count = 0
        self._initial_seed = seed

    @staticmethod
    def _validate_prediction(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("detector.predict(image) must return a numeric value") from error
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("detector.predict(image) must return P(fake) in [0, 1]")
        return score

    def _observation(self) -> np.ndarray:
        return np.array(
            [self.current_confidence, self.step_count / self.max_steps],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore the original image and return the initial compact observation."""
        del options
        effective_seed = self._initial_seed if seed is None else seed
        super().reset(seed=effective_seed)
        self.current_image = self.original_image.copy()
        self.step_count = 0
        self.current_confidence = self._validate_prediction(
            self.detector.predict(self.current_image)
        )
        return self._observation(), {
            "fake_confidence": self.current_confidence,
            "step": self.step_count,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action, score the result, and return a Gymnasium transition."""
        if not isinstance(action, Integral) or int(action) not in ACTIONS:
            raise ValueError(f"action must be one of {tuple(ACTIONS)}")
        action_id = int(action)
        before = self.current_confidence
        self.current_image = apply_action(self.current_image, action_id, rng=self.np_random)
        after = self._validate_prediction(self.detector.predict(self.current_image))
        reward = float(self.reward_fn(before, after))
        self.current_confidence = after
        self.step_count += 1
        terminated = bool(after < self.success_threshold)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        info = {
            "step": self.step_count,
            "action_id": action_id,
            "action_name": ACTIONS[action_id],
            "before_confidence": before,
            "after_confidence": after,
            "confidence_reduction": float(before - after),
            "success": terminated,
        }
        return self._observation(), reward, terminated, truncated, info

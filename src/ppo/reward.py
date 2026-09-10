"""Reward functions for detector-confidence reduction attacks."""


def _validate_confidence(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def confidence_reduction_reward(
    before_confidence: float,
    after_confidence: float,
) -> float:
    """Reward a reduction in the detector's fake probability."""
    before = _validate_confidence(before_confidence, "before_confidence")
    after = _validate_confidence(after_confidence, "after_confidence")
    return float(before - after)


def quality_aware_reward(
    before_confidence: float,
    after_confidence: float,
    distortion: float,
    lambda_quality: float = 0.1,
) -> float:
    """Reward confidence reduction while penalizing an external distortion value.

    A later image-quality metric such as SSIM or LPIPS can supply ``distortion``.
    """
    before = _validate_confidence(before_confidence, "before_confidence")
    after = _validate_confidence(after_confidence, "after_confidence")
    distortion = float(distortion)
    lambda_quality = float(lambda_quality)
    if distortion < 0:
        raise ValueError("distortion must be non-negative")
    if lambda_quality < 0:
        raise ValueError("lambda_quality must be non-negative")
    return float((before - after) - lambda_quality * distortion)

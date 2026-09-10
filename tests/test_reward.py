import pytest

from src.ppo.reward import confidence_reduction_reward, quality_aware_reward


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [(0.9, 0.5, 0.4), (0.5, 0.9, -0.4), (0.7, 0.7, 0.0)],
)
def test_confidence_reduction_reward(before, after, expected):
    assert confidence_reduction_reward(before, after) == pytest.approx(expected)


@pytest.mark.parametrize(("before", "after"), [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_invalid_confidence_raises(before, after):
    with pytest.raises(ValueError):
        confidence_reduction_reward(before, after)


def test_quality_aware_reward():
    assert quality_aware_reward(0.9, 0.5, 0.2, 0.1) == pytest.approx(0.38)


@pytest.mark.parametrize(("distortion", "weight"), [(-0.1, 0.1), (0.1, -0.1)])
def test_invalid_quality_inputs_raise(distortion, weight):
    with pytest.raises(ValueError):
        quality_aware_reward(0.9, 0.5, distortion, weight)

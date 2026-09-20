import numpy as np
import pytest
import torch
from PIL import Image

from victim.detector import VictimDetector


class FakeModel(torch.nn.Module):
    def forward(self, data, inference=False):
        assert inference is True
        return {"cls": torch.tensor([[1.0, 2.0]], device=data["image"].device)}


def test_predict_image_contract_without_deepfakebench():
    detector = VictimDetector.__new__(VictimDetector)
    detector.device = torch.device("cpu")
    detector.config = {"resolution": 32, "mean": [0.5] * 3, "std": [0.5] * 3}
    detector.model = FakeModel().eval()
    result = detector.predict_image(Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)))
    assert set(result) == {"label", "real_probability", "fake_probability"}
    assert result["label"] in {"REAL", "FAKE"}
    assert 0 <= result["real_probability"] <= 1
    assert 0 <= result["fake_probability"] <= 1
    assert result["real_probability"] + result["fake_probability"] == pytest.approx(1.0)


@pytest.mark.integration
def test_real_deepfakebench_integration_is_configuration_dependent():
    pytest.skip("Requires local DeepfakeBench checkout and Xception weights")

"""DeepfakeBench detector adapter used by the DimSum attack pipeline."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda`` or ``mps`` to an available device."""
    choice = str(device).lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is not available")
    if choice not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(choice)


class VictimDetector:
    """Load DeepfakeBench Xception once and expose normalized predictions.

    The wrapper owns all detector-specific imports and preprocessing. DeepfakeBench
    labels class 0 as REAL and class 1 as FAKE.
    """

    def __init__(
        self,
        deepfakebench_root: str | Path,
        detector_name: str = "xception",
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        if detector_name != "xception":
            raise ValueError("This integration currently supports detector_name='xception'")
        self.root = Path(deepfakebench_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"DeepfakeBench directory not found: {self.root}")
        self.training_dir = self.root / "training"
        if not self.training_dir.is_dir():
            raise FileNotFoundError(
                f"DeepfakeBench training directory not found: {self.training_dir}"
            )
        self.config_path = Path(config_path).expanduser().resolve() if config_path else (
            self.training_dir / "config" / "detector" / "xception.yaml"
        )
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Victim detector config not found: {self.config_path}")
        if weights_path is None:
            raise FileNotFoundError(
                "Victim detector weights not configured. Expected Xception checkpoint "
                "configured by victim.weights_path."
            )
        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Victim detector weights not found: {self.weights_path}")
        self.device = resolve_device(device)
        self.detector_name = detector_name
        self.config = self._load_config()
        self.model = self._load_model()

    def _load_config(self) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("PyYAML is required to load DeepfakeBench configuration") from error
        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"Invalid detector configuration: {self.config_path}")
        config["model_name"] = self.detector_name
        pretrained = config.get("pretrained")
        if isinstance(pretrained, str) and pretrained.lower() != "none":
            pretrained_path = Path(pretrained).expanduser()
            if not pretrained_path.is_absolute():
                pretrained_path = self.root / pretrained_path
            config["pretrained"] = str(pretrained_path.resolve())
        return config

    def _import_xception_class(self) -> type[torch.nn.Module]:
        """Import only Xception, avoiding optional dependencies of other detectors."""
        registry_module = importlib.import_module("metrics.registry")
        registry = getattr(registry_module, "DETECTOR")
        package = types.ModuleType("detectors")
        package.__path__ = [str(self.training_dir / "detectors")]
        package.__package__ = "detectors"
        package.DETECTOR = registry
        sys.modules["detectors"] = package
        importlib.import_module("detectors.xception_detector")
        return registry[self.detector_name]

    def _load_model(self) -> torch.nn.Module:
        training_path = str(self.training_dir)
        if training_path not in sys.path:
            sys.path.insert(0, training_path)
        try:
            model_class = self._import_xception_class()
            model = model_class(self.config)
        except Exception as error:
            raise RuntimeError(
                "Failed to construct DeepfakeBench Xception. Check its dependencies "
                f"and detector config at: {self.config_path}"
            ) from error

        try:
            checkpoint = torch.load(
                self.weights_path, map_location=self.device, weights_only=False
            )
            if isinstance(checkpoint, dict):
                for key in ("state_dict", "model", "model_state_dict"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        checkpoint = checkpoint[key]
                        break
            if not isinstance(checkpoint, dict):
                raise TypeError("checkpoint does not contain a state dictionary")
            state_dict = {
                key.removeprefix("module."): value for key, value in checkpoint.items()
            }
            model.load_state_dict(state_dict, strict=True)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load victim detector weights: {self.weights_path}"
            ) from error
        model.to(self.device)
        model.eval()
        return model

    def _to_tensor(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
                raise ValueError("NumPy image must have shape (H, W, C)")
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            tensor = torch.from_numpy(array[..., :3].copy()).permute(2, 0, 1).float()
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().clone().float()
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3:
                raise ValueError("Torch image must have shape (C, H, W) or (1, C, H, W)")
            if tensor.shape[0] not in {1, 3, 4}:
                raise ValueError("Torch image must have 1, 3, or 4 channels")
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            tensor = tensor[:3]
        else:
            raise TypeError("image must be a PIL image, NumPy array, or Torch tensor")

        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0).unsqueeze(0)
        resolution = int(self.config.get("resolution", 256))
        tensor = torch.nn.functional.interpolate(
            tensor, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        mean = torch.tensor(self.config.get("mean", [0.5] * 3)).view(1, 3, 1, 1)
        std = torch.tensor(self.config.get("std", [0.5] * 3)).view(1, 3, 1, 1)
        return ((tensor - mean) / std).to(self.device)

    def predict_image(
        self, image: Image.Image | np.ndarray | torch.Tensor
    ) -> dict[str, str | float]:
        """Predict one image and return REAL/FAKE probabilities."""
        tensor = self._to_tensor(image)
        with torch.no_grad():
            output = self.model({"image": tensor}, inference=True)
            logits = output["cls"] if isinstance(output, dict) and "cls" in output else output
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("Victim detector did not return tensor logits")
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape[-1] == 2:
                probabilities = torch.softmax(logits, dim=-1)[0]
                real_probability, fake_probability = probabilities.tolist()
            elif logits.shape[-1] == 1:
                fake_probability = torch.sigmoid(logits.flatten()[0]).item()
                real_probability = 1.0 - fake_probability
            else:
                raise RuntimeError("Victim detector output must contain one or two logits")
        total = float(real_probability + fake_probability)
        real_probability = float(np.clip(real_probability / total, 0.0, 1.0))
        fake_probability = float(np.clip(fake_probability / total, 0.0, 1.0))
        return {
            "label": "FAKE" if fake_probability >= real_probability else "REAL",
            "real_probability": real_probability,
            "fake_probability": fake_probability,
        }

    def predict_path(self, image_path: str | Path) -> dict[str, str | float]:
        """Load an image path and pass it to :meth:`predict_image`."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        try:
            with Image.open(path) as image:
                return self.predict_image(image.convert("RGB"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Unable to load input image: {path}") from error

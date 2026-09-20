"""Small shared utilities for training and evaluation entry points."""

from __future__ import annotations

import random
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch, and available accelerators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_image_paths(directory: str | Path, dataset_name: str) -> list[Path]:
    """Recursively collect supported images with a descriptive empty-set error."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{dataset_name.capitalize()} directory not found: {root}")
    candidates = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    paths: list[Path] = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                image.verify()
            paths.append(path)
        except (OSError, ValueError) as error:
            LOGGER.warning("Skipping invalid image %s: %s", path, error)
    if not paths:
        raise ValueError(f"No supported image files found in {dataset_name} directory: {root}")
    return paths


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required for configuration files") from error
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return config

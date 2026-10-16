"""Small shared utilities for training and evaluation entry points."""

from __future__ import annotations

import random
import logging
import json
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


def load_image_manifest(
    manifest_path: str | Path,
    dataset_name: str,
    label: str | None = None,
) -> list[Path]:
    """Load validated image paths from a split manifest, optionally filtering labels."""
    records = load_image_manifest_records(manifest_path, dataset_name)
    selected = [record["path"] for record in records if label is None or record["label"] == label]
    if not selected:
        path = Path(manifest_path).expanduser().resolve()
        suffix = f" with label {label}" if label else ""
        raise ValueError(f"No images found in {dataset_name} manifest{suffix}: {path}")
    return selected


def load_image_manifest_records(
    manifest_path: str | Path,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Load validated paths and labels from a split manifest."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{dataset_name.capitalize()} manifest not found: {path}")
    with path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError(f"Invalid image manifest: {path}")
    root_value = manifest.get("dataset_root")
    if not isinstance(root_value, str):
        raise ValueError(f"Manifest dataset_root must be a string: {path}")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    records: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        if not isinstance(sample, dict) or not isinstance(sample.get("path"), str):
            raise ValueError(f"Manifest contains an invalid sample: {path}")
        sample_label = sample.get("label")
        if sample_label not in {"REAL", "FAKE"}:
            raise ValueError(f"Manifest sample label must be REAL or FAKE: {path}")
        image_path = (root / sample["path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image not found: {image_path}")
        records.append({**sample, "path": image_path})
    if not records:
        raise ValueError(f"No images found in {dataset_name} manifest: {path}")
    return records


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

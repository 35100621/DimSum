import json
from pathlib import Path

import pytest

from ppo.common import load_image_manifest
from scripts.create_ff_split import build_split


def test_stratified_split_is_deterministic_and_disjoint(tmp_path: Path):
    dataset = tmp_path / "FF_small"
    for folder, prefixes in (("fake", ("Deepfakes", "FaceSwap")), ("real", ("actor_real", "youtube_real"))):
        class_dir = dataset / folder
        class_dir.mkdir(parents=True)
        for prefix in prefixes:
            for index in range(10):
                (class_dir / f"{prefix}_{index:04d}.png").write_bytes(b"placeholder")
    first_train, first_test = build_split(dataset, 0.2, 42)
    second_train, second_test = build_split(dataset, 0.2, 42)
    assert (first_train, first_test) == (second_train, second_test)
    train_paths = {row["path"] for row in first_train}
    test_paths = {row["path"] for row in first_test}
    assert len(first_train) == 32 and len(first_test) == 8
    assert train_paths.isdisjoint(test_paths)


def test_manifest_loader_filters_fake_images(tmp_path: Path):
    dataset = tmp_path / "images"
    dataset.mkdir()
    fake = dataset / "fake.png"
    real = dataset / "real.png"
    from PIL import Image
    Image.new("RGB", (4, 4)).save(fake)
    Image.new("RGB", (4, 4)).save(real)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "dataset_root": "images",
        "samples": [
            {"path": "fake.png", "label": "FAKE"},
            {"path": "real.png", "label": "REAL"},
        ],
    }))
    assert load_image_manifest(manifest, "training", label="FAKE") == [fake.resolve()]


def test_manifest_loader_rejects_missing_images(tmp_path: Path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "dataset_root": ".",
        "samples": [{"path": "missing.png", "label": "FAKE"}],
    }))
    with pytest.raises(FileNotFoundError, match="Manifest image not found"):
        load_image_manifest(manifest, "training")

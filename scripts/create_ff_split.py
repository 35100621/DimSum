"""Create deterministic, stratified FF_small train/test manifests."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def sample_group(path: Path) -> str:
    """Return the manipulation or real-source group encoded in an FF_small name."""
    return re.sub(r"_\d+$", "", path.stem)


def build_split(dataset_root: Path, test_fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split every class/source group independently using a fixed seed."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for folder, label in (("real", "REAL"), ("fake", "FAKE")):
        class_dir = dataset_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"FF_small class directory not found: {class_dir}")
        for path in sorted(class_dir.glob("*.png")):
            groups[(label, sample_group(path))].append(path)
    if not groups:
        raise ValueError(f"No FF_small PNG images found at: {dataset_root}")

    generator = random.Random(seed)
    training: list[dict[str, str]] = []
    testing: list[dict[str, str]] = []
    for (label, group), paths in sorted(groups.items()):
        shuffled = list(paths)
        generator.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_fraction))
        test_paths = set(shuffled[:test_count])
        for path in sorted(shuffled):
            record = {
                "path": str(path.relative_to(dataset_root)),
                "label": label,
                "group": group,
            }
            (testing if path in test_paths else training).append(record)
    training.sort(key=lambda item: item["path"])
    testing.sort(key=lambda item: item["path"])
    return training, testing


def write_manifest(
    path: Path,
    dataset_root: Path,
    split: str,
    samples: list[dict[str, str]],
    seed: int,
    test_fraction: float,
) -> None:
    """Write one portable manifest with a dataset path relative to the manifest."""
    relative_root = Path("../../../DeepfakeBench_Handover/datasets/rgb/FF_small")
    payload: dict[str, Any] = {
        "dataset": "FF_small",
        "split": split,
        "seed": seed,
        "test_fraction": test_fraction,
        "dataset_root": str(relative_root),
        "source_dataset_root": str(dataset_root.resolve()),
        "samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Create `ff_ppo_train.json` and `ff_test.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", default="data/splits", type=Path)
    parser.add_argument("--test-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    training, testing = build_split(dataset_root, args.test_fraction, args.seed)
    write_manifest(args.output_dir / "ff_ppo_train.json", dataset_root, "ppo_train", training, args.seed, args.test_fraction)
    write_manifest(args.output_dir / "ff_test.json", dataset_root, "test", testing, args.seed, args.test_fraction)
    print(f"training records: {len(training)}")
    print(f"testing records: {len(testing)}")
    print(f"PPO-labelled FAKE training records: {sum(row['label'] == 'FAKE' for row in training)}")


if __name__ == "__main__":
    main()

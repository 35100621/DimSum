"""Reproduce the handed-over Xception baseline on FF_small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from victim.detector import VictimDetector


def evaluate_victim_baseline(
    dataset_dir: str | Path,
    detector: VictimDetector,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate REAL/FAKE folders using the handover's label mapping and threshold."""
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    labels: list[int] = []
    scores: list[float] = []
    counts: dict[str, int] = {}
    for label, folder_name in ((0, "real"), (1, "fake")):
        folder = dataset_path / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Baseline class directory not found: {folder}")
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise ValueError(f"No PNG images found in baseline class directory: {folder}")
        counts[folder_name] = len(paths)
        for path in paths:
            prediction = detector.predict_path(path)
            labels.append(label)
            scores.append(float(prediction["fake_probability"]))
    binary = [int(score >= threshold) for score in scores]
    true_positive = sum(expected == predicted == 1 for expected, predicted in zip(labels, binary))
    true_negative = sum(expected == predicted == 0 for expected, predicted in zip(labels, binary))
    false_positive = sum(expected == 0 and predicted == 1 for expected, predicted in zip(labels, binary))
    false_negative = sum(expected == 1 and predicted == 0 for expected, predicted in zip(labels, binary))
    return {
        "total_images": len(labels),
        "real_images": counts["real"],
        "fake_images": counts["fake"],
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, binary)),
        "auc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, binary)),
        "recall": float(recall_score(labels, binary)),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone baseline CLI parser."""
    parser = argparse.ArgumentParser(description="Reproduce the FF_small Xception baseline.")
    parser.add_argument("--deepfakebench-root", required=True, help="Extracted handover root.")
    parser.add_argument("--dataset", help="FF_small directory; defaults inside the handover.")
    parser.add_argument("--weights", help="Xception checkpoint; defaults inside the handover.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output", help="Optional path for the JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run baseline evaluation and optionally save its summary."""
    args = build_parser().parse_args(argv)
    root = Path(args.deepfakebench_root).expanduser().resolve()
    dataset = Path(args.dataset) if args.dataset else root / "datasets/rgb/FF_small"
    weights = Path(args.weights) if args.weights else root / "training/weights/xception_best.pth"
    detector = VictimDetector(root, weights_path=weights, device=args.device)
    summary = evaluate_victim_baseline(dataset, detector)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

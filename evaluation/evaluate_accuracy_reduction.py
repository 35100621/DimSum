"""Measure frozen-detector accuracy before and after FAKE-only PPO attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ppo.common import load_image_manifest_records


LABELS = ("REAL", "FAKE")


def _confusion(truth: pd.Series, prediction: pd.Series) -> list[list[int]]:
    """Return rows=true and columns=predicted in REAL, FAKE order."""
    return [
        [int(((truth == true) & (prediction == predicted)).sum()) for predicted in LABELS]
        for true in LABELS
    ]


def evaluate_accuracy_reduction(
    test_manifest: str | Path,
    attack_results: str | Path,
    output_dir: str | Path,
    min_ssim: float = 0.90,
) -> dict[str, object]:
    """Use original REALs and PPO-processed FAKEs to measure accuracy reduction."""
    records = load_image_manifest_records(test_manifest, "test")
    manifest = pd.DataFrame({
        "image_path": [str(record["path"]) for record in records],
        "manifest_label": [record["label"] for record in records],
    })
    results = pd.read_csv(attack_results)
    required = {
        "image_path", "ground_truth_label", "initial_label", "final_label",
        "attack_success", "ssim",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Attack results missing columns: {sorted(missing)}")
    if results.image_path.duplicated().any() or manifest.image_path.duplicated().any():
        raise ValueError("Evaluation paths must be unique")
    if set(results.image_path) != set(manifest.image_path):
        raise ValueError("Attack results and test manifest do not contain the same paths")
    frame = manifest.merge(results, on="image_path", how="inner", validate="one_to_one")
    if not (frame.manifest_label == frame.ground_truth_label).all():
        raise ValueError("Ground-truth labels disagree with the test manifest")
    if len(frame) != 40 or (frame.ground_truth_label == "REAL").sum() != 20 or (frame.ground_truth_label == "FAKE").sum() != 20:
        raise ValueError("Expected the fixed held-out split: 20 REAL and 20 FAKE")

    frame["after_label"] = frame["initial_label"]
    fake_mask = frame.ground_truth_label == "FAKE"
    frame.loc[fake_mask, "after_label"] = frame.loc[fake_mask, "final_label"]
    frame["after_source"] = "unchanged_original_real"
    frame.loc[fake_mask, "after_source"] = "quality_ppo_processed_fake"
    frame["clean_correct"] = frame.initial_label == frame.ground_truth_label
    frame["after_correct"] = frame.after_label == frame.ground_truth_label

    eligible = fake_mask & (frame.initial_label == "FAKE")
    successful = eligible & (frame.final_label == "REAL")
    quality_successful = successful & (frame.ssim >= min_ssim)
    clean_correct = int(frame.clean_correct.sum())
    after_correct = int(frame.after_correct.sum())
    summary: dict[str, object] = {
        "evaluation_set": {"total": 40, "real": 20, "fake": 20},
        "label_order": list(LABELS),
        "clean": {
            "correct": clean_correct,
            "incorrect": 40 - clean_correct,
            "accuracy": clean_correct / 40,
            "confusion_matrix": _confusion(frame.ground_truth_label, frame.initial_label),
        },
        "after_quality_ppo": {
            "correct": after_correct,
            "incorrect": 40 - after_correct,
            "accuracy": after_correct / 40,
            "confusion_matrix": _confusion(frame.ground_truth_label, frame.after_label),
            "real_images_unchanged": int((~fake_mask).sum()),
            "fake_images_processed": int(fake_mask.sum()),
        },
        "accuracy_drop": clean_correct / 40 - after_correct / 40,
        "accuracy_drop_percentage_points": 100 * (clean_correct - after_correct) / 40,
        "attack_performance": {
            "eligible_fake_images": int(eligible.sum()),
            "successful_attacks": int(successful.sum()),
            "asr": float(successful.sum() / eligible.sum()),
            "quality_threshold_ssim": min_ssim,
            "quality_successful_attacks": int(quality_successful.sum()),
            "quality_constrained_asr": float(quality_successful.sum() / eligible.sum()),
            "mean_ssim_eligible": float(frame.loc[eligible, "ssim"].mean()),
        },
        "test_manifest": str(Path(test_manifest).expanduser().resolve()),
        "attack_results": str(Path(attack_results).expanduser().resolve()),
    }
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "results.csv", index=False)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector accuracy before and after FAKE-only PPO attack.")
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--attack-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-ssim", type=float, default=0.90)
    args = parser.parse_args()
    print(json.dumps(evaluate_accuracy_reduction(
        args.test_manifest, args.attack_results, args.output_dir, args.min_ssim
    ), indent=2))


if __name__ == "__main__":
    main()

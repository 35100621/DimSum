"""Evaluate no-attack and standalone adversarial baselines on one held-out manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ppo.actions import fgsm_attack, frequency_lowpass_attack, pgd_attack
from ppo.common import load_image_manifest_records, seed_everything
from victim.detector import VictimDetector


def evaluate_baseline(manifest: str | Path, deepfakebench_root: str | Path,
                      weights: str | Path, attack: str, output_dir: str | Path,
                      seed: int = 42, device: str = "cpu") -> dict[str, object]:
    """Run a fixed attack once per image and save metrics and attacked images."""
    if attack not in {"none", "fgsm", "pgd", "frequency"}:
        raise ValueError("attack must be none, fgsm, pgd, or frequency")
    seed_everything(seed)
    detector = VictimDetector(deepfakebench_root, weights_path=weights, device=device)
    records = load_image_manifest_records(manifest, "test")
    destination = Path(output_dir).expanduser().resolve()
    samples = destination / "adversarial_samples"
    samples.mkdir(parents=True, exist_ok=True)
    rows, reductions, ssims, psnrs = [], [], [], []
    successes = eligible = 0
    started = time.perf_counter()
    for record in records:
        path, label = record["path"], record["label"]
        with Image.open(path) as source:
            before_image = source.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        initial = detector.predict_image(before_image)
        if attack == "fgsm":
            after_image = fgsm_attack(before_image, detector, 8 / 255)
        elif attack == "pgd":
            after_image = pgd_attack(before_image, detector, 8 / 255, 2 / 255, 10)
        elif attack == "frequency":
            after_image = frequency_lowpass_attack(before_image, 8 / 255, 0.35)
        else:
            after_image = before_image.copy()
        final = detector.predict_image(after_image)
        before = np.asarray(before_image, dtype=np.float32) / 255.0
        after = np.asarray(after_image, dtype=np.float32) / 255.0
        ssim = float(structural_similarity(before, after, channel_axis=2, data_range=1.0))
        psnr = float(peak_signal_noise_ratio(before, after, data_range=1.0))
        is_eligible = label == "FAKE" and initial["label"] == "FAKE"
        success = is_eligible and final["label"] == "REAL"
        if is_eligible:
            eligible += 1
            successes += int(success)
            reductions.append(initial["fake_probability"] - final["fake_probability"])
            ssims.append(ssim)
            psnrs.append(psnr)
        rows.append({"image_path": str(path), "ground_truth_label": label,
                     "initial_label": initial["label"], "initial_fake_probability": initial["fake_probability"],
                     "final_label": final["label"], "final_fake_probability": final["fake_probability"],
                     "eligible": is_eligible, "attack_success": success, "ssim": ssim, "psnr": psnr})
        after_image.save(samples / f"{path.stem}_adv.png")
    runtime = time.perf_counter() - started
    def stats(values: list[float]) -> dict[str, float | None]:
        a = np.asarray(values, dtype=float)
        finite = a[np.isfinite(a)]
        if len(a) and not len(finite):
            return {"mean": None, "median": None, "std": None}
        return ({"mean": float(finite.mean()), "median": float(np.median(finite)), "std": float(finite.std())}
                if len(finite) else {"mean": None, "median": None, "std": None})
    summary = {"attack": attack, "total_images": len(rows), "eligible_images": eligible,
               "successful_attacks": successes, "attack_success_rate": successes / eligible if eligible else None,
               "fake_confidence_reduction": stats(reductions), "ssim": stats(ssims), "psnr": stats(psnrs),
               "runtime_seconds": runtime, "seconds_per_image": runtime / len(rows),
               "epsilon": 8 / 255 if attack != "none" else 0.0, "seed": seed,
               "weights": str(Path(weights).resolve()), "manifest": str(Path(manifest).resolve())}
    pd.DataFrame(rows).to_csv(destination / "results.csv", index=False)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--deepfakebench-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--attack", choices=["none", "fgsm", "pgd", "frequency"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    args = parser.parse_args()
    print(json.dumps(evaluate_baseline(args.test_manifest, args.deepfakebench_root, args.weights,
                                       args.attack, args.output_dir, args.seed, args.device), indent=2))


if __name__ == "__main__":
    main()

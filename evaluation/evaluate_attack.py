"""Evaluate a trained PPO policy on a separate image directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ppo.agent import load_ppo_agent
from ppo.common import (
    collect_image_paths,
    load_image_manifest_records,
    load_yaml,
    seed_everything,
)
from ppo.environment import DeepfakeAttackEnv
from victim.detector import VictimDetector


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
QUALITY_THRESHOLDS = (0.90, 0.85, 0.80, 0.70, 0.50)


def _aggregate(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"mean_{prefix}": None, f"median_{prefix}": None, f"std_{prefix}": None}
    array = np.asarray(values, dtype=float)
    statistics = (np.mean(array), np.median(array), np.std(array))
    return {
        f"mean_{prefix}": float(statistics[0]) if np.isfinite(statistics[0]) else None,
        f"median_{prefix}": float(statistics[1]) if np.isfinite(statistics[1]) else None,
        f"std_{prefix}": float(statistics[2]) if np.isfinite(statistics[2]) else None,
    }


def quality_constrained_asr(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Count successful eligible attacks meeting fixed SSIM thresholds."""
    eligible_rows = [
        row for row in rows
        if row["ground_truth_label"] == "FAKE" and row["initial_label"] == "FAKE"
    ]
    denominator = len(eligible_rows)
    return {
        f"ssim_gte_{threshold:.2f}": {
            "successes": (count := sum(
                bool(row["attack_success"]) and float(row["ssim"]) >= threshold
                for row in eligible_rows
            )),
            "eligible": denominator,
            "asr": count / denominator if denominator else None,
        }
        for threshold in QUALITY_THRESHOLDS
    }


def evaluate_attack(
    test_dir: str | Path | None,
    model_path: str | Path,
    deepfakebench_root: str | Path,
    weights_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = "outputs/evaluation",
    max_steps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    save_images: bool | None = None,
    test_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic attacks and write per-image CSV plus JSON summary."""
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("pandas is required for evaluation output") from error
    config = load_yaml(config_path)
    project = config.get("project", {})
    env_config = config.get("environment", {})
    victim_config = config.get("victim", {})
    evaluation_config = config.get("evaluation", {})
    effective_seed = int(project.get("seed", 42) if seed is None else seed)
    effective_device = str(project.get("device", "auto") if device is None else device)
    effective_steps = int(env_config.get("max_steps", 5) if max_steps is None else max_steps)
    should_save = bool(evaluation_config.get("save_images", False) if save_images is None else save_images)
    seed_everything(effective_seed)
    if (test_dir is None) == (test_manifest is None):
        raise ValueError("Provide exactly one of test_dir or test_manifest")
    if test_manifest is not None:
        manifest_records = load_image_manifest_records(test_manifest, "test")
        image_paths = [record["path"] for record in manifest_records]
        ground_truth = {record["path"]: record["label"] for record in manifest_records}
    else:
        image_paths = collect_image_paths(test_dir, "test")
        ground_truth = {
            path: path.parent.name.upper() if path.parent.name.lower() in {"real", "fake"} else None
            for path in image_paths
        }
    detector = VictimDetector(
        deepfakebench_root=deepfakebench_root,
        detector_name=victim_config.get("detector_name", "xception"),
        config_path=victim_config.get("config_path"),
        weights_path=weights_path,
        device=effective_device,
    )
    env = DeepfakeAttackEnv(
        image_paths=image_paths,
        victim_detector=detector,
        image_size=(int(env_config.get("image_width", 224)), int(env_config.get("image_height", 224))),
        max_steps=effective_steps,
        quality_weight=float(env_config.get("quality_weight", 0.25)),
        perturbation_weight=float(env_config.get("perturbation_weight", 0.10)),
        seed=effective_seed,
        strength_levels=int(env_config.get("strength_levels", 5)),
        action_set=str(env_config.get("action_set", "basic")),
        min_ssim=env_config.get("min_ssim"),
        quality_violation_penalty=float(env_config.get("quality_violation_penalty", 0.5)),
    )
    model = load_ppo_agent(model_path, env=env, device=effective_device)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    samples_path = output_path / "adversarial_samples"
    if should_save:
        samples_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    reductions: list[float] = []
    eligible_ssim: list[float] = []
    eligible_psnr: list[float] = []
    eligible_l1: list[float] = []
    successful = 0
    eligible = 0
    action_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    rejected_action_counts: Counter[str] = Counter()
    started = time.perf_counter()
    for index, path in enumerate(image_paths):
        try:
            observation, _ = env.reset(seed=effective_seed + index, options={"image_index": index})
            assert env.initial_prediction is not None and env.original_image is not None
            initial = env.initial_prediction.copy()
            terminated = truncated = False
            trace: list[str] = []
            confidence_trace = [float(initial["fake_probability"])]
            candidate_ssim_trace: list[float] = []
            rejected_trace: list[bool] = []
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, step_info = env.step(action)
                action_counts[step_info["action_name"]] += 1
                strength_counts[f'{step_info["strength"]:.2f}'] += 1
                trace.append(f'{step_info["action_name"]}:{step_info["strength"]:.2f}')
                confidence_trace.append(float(step_info["current_fake_probability"]))
                candidate_ssim_trace.append(float(step_info["candidate_ssim"]))
                rejected = bool(step_info["quality_violation"])
                rejected_trace.append(rejected)
                if rejected:
                    rejected_action_counts[step_info["action_name"]] += 1
            assert env.current_prediction is not None and env.current_image is not None
            final = env.current_prediction
            before = np.asarray(env.original_image, dtype=np.float32) / 255.0
            after = np.asarray(env.current_image, dtype=np.float32) / 255.0
            ssim = float(structural_similarity(before, after, channel_axis=2, data_range=1.0))
            psnr = float(peak_signal_noise_ratio(before, after, data_range=1.0))
            l1 = float(np.mean(np.abs(before - after)))
            true_label = ground_truth[path]
            is_eligible = true_label == "FAKE" and initial["label"] == "FAKE"
            initially_misclassified = true_label == "FAKE" and initial["label"] == "REAL"
            attack_success = bool(is_eligible and final["label"] == "REAL")
            if is_eligible:
                eligible += 1
                successful += int(attack_success)
                reductions.append(float(initial["fake_probability"] - final["fake_probability"]))
                eligible_ssim.append(ssim)
                eligible_psnr.append(psnr)
                eligible_l1.append(l1)
            rows.append({
                "image_path": str(path),
                "ground_truth_label": true_label,
                "initial_label": initial["label"],
                "initial_real_probability": initial["real_probability"],
                "initial_fake_probability": initial["fake_probability"],
                "final_label": final["label"],
                "final_real_probability": final["real_probability"],
                "final_fake_probability": final["fake_probability"],
                "initially_misclassified": initially_misclassified,
                "attack_success": attack_success,
                "steps_used": env.step_count,
                "ssim": ssim,
                "psnr": psnr,
                "l1_distortion": l1,
                "action_trace": "|".join(trace),
                "fake_probability_trace": "|".join(f"{value:.8f}" for value in confidence_trace),
                "candidate_ssim_trace": "|".join(f"{value:.8f}" for value in candidate_ssim_trace),
                "rejected_action_trace": "|".join("1" if value else "0" for value in rejected_trace),
                "rejected_actions": sum(rejected_trace),
            })
            if should_save:
                digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
                env.original_image.save(samples_path / f"{path.stem}_{digest}_before.png")
                env.current_image.save(samples_path / f"{path.stem}_{digest}_adv.png")
        except (OSError, ValueError, RuntimeError) as error:
            LOGGER.warning("Skipping invalid image %s: %s", path, error)

    pd.DataFrame(rows).to_csv(output_path / "results.csv", index=False)
    runtime_seconds = time.perf_counter() - started
    gradient_evaluations = action_counts["fgsm"] + 5 * action_counts["pgd"]
    environment_steps = sum(action_counts.values())
    summary: dict[str, Any] = {
        "total_images": len(rows),
        "eligible_images": eligible,
        "ground_truth_fake_images": sum(row["ground_truth_label"] == "FAKE" for row in rows),
        "real_reference_images": sum(row["ground_truth_label"] == "REAL" for row in rows),
        "initially_misclassified_fake_images": sum(row["initially_misclassified"] for row in rows),
        "successful_attacks": successful,
        "attack_success_rate": float(successful / eligible) if eligible else None,
        **_aggregate(reductions, "fake_confidence_reduction"),
        **_aggregate(eligible_ssim, "ssim"),
        **_aggregate(eligible_psnr, "psnr"),
        **_aggregate(eligible_l1, "l1_distortion"),
        "action_selection_frequency": dict(sorted(action_counts.items())),
        "strength_distribution": dict(sorted(strength_counts.items())),
        "rejected_action_frequency": dict(sorted(rejected_action_counts.items())),
        "rejected_actions": sum(rejected_action_counts.values()),
        "applied_actions": environment_steps - sum(rejected_action_counts.values()),
        "model_checkpoint": str(Path(model_path).expanduser().resolve()),
        "configuration": str(Path(config_path).expanduser().resolve()),
        "evaluation_runtime_seconds": runtime_seconds,
        "seconds_per_image": runtime_seconds / len(rows) if rows else None,
        "environment_steps": environment_steps,
        "gradient_evaluations": gradient_evaluations,
        "estimated_detector_forward_calls": len(rows) + environment_steps + gradient_evaluations,
        "quality_constrained_asr": quality_constrained_asr(rows),
        "configured_min_ssim": env.min_ssim,
        "minimum_final_ssim": min((float(row["ssim"]) for row in rows), default=None),
        "final_quality_violations": (
            sum(float(row["ssim"]) < env.min_ssim - 1e-9 for row in rows)
            if env.min_ssim is not None else None
        ),
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=False)
    LOGGER.info(
        "images=%d eligible=%d successful=%d ASR=%s mean_reduction=%s mean_ssim=%s mean_psnr=%s",
        len(rows), eligible, successful, summary["attack_success_rate"],
        summary["mean_fake_confidence_reduction"], summary["mean_ssim"], summary["mean_psnr"],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate a PPO attack on held-out images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-dir", help="Directory containing held-out test images.")
    source.add_argument("--test-manifest", help="JSON manifest containing held-out test images.")
    parser.add_argument("--model", required=True, help="Trained PPO checkpoint.")
    parser.add_argument("--deepfakebench-root", required=True, help="DeepfakeBench repository root.")
    parser.add_argument("--weights", required=True, help="Trained Xception checkpoint path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="DimSum YAML configuration path.")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Evaluation artifact directory.")
    parser.add_argument("--max-steps", type=int, help="Maximum transformations per image.")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--save-images", action="store_true", default=None, help="Save adversarial PNG files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run held-out evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    evaluate_attack(
        test_dir=args.test_dir, model_path=args.model,
        deepfakebench_root=args.deepfakebench_root, weights_path=args.weights,
        config_path=args.config, output_dir=args.output_dir, max_steps=args.max_steps,
        seed=args.seed, device=args.device, save_images=args.save_images,
        test_manifest=args.test_manifest,
    )


if __name__ == "__main__":
    main()

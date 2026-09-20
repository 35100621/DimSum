"""Command-line and notebook helpers for training the DimSum PPO attacker."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from stable_baselines3.common.env_checker import check_env

from victim.detector import VictimDetector

from .agent import create_ppo_agent, load_ppo_agent
from .common import collect_image_paths, load_yaml, seed_everything
from .environment import DeepfakeAttackEnv


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def train_attack(
    train_dir: str | Path,
    deepfakebench_root: str | Path,
    weights_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    timesteps: int | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    output_dir: str | Path = "outputs",
    resume: str | Path | None = None,
) -> Path:
    """Train PPO from a dedicated training directory and save its checkpoint."""
    config = load_yaml(config_path)
    project = config.get("project", {})
    environment_config = config.get("environment", {})
    ppo_config = config.get("ppo", {})
    victim_config = config.get("victim", {})
    effective_seed = int(project.get("seed", 42) if seed is None else seed)
    effective_device = str(project.get("device", "auto") if device is None else device)
    effective_timesteps = int(
        config.get("training", {}).get("total_timesteps", 10000)
        if timesteps is None else timesteps
    )
    effective_max_steps = int(environment_config.get("max_steps", 5) if max_steps is None else max_steps)
    if effective_timesteps <= 0:
        raise ValueError("timesteps must be positive")
    seed_everything(effective_seed)
    image_paths = collect_image_paths(train_dir, "training")
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
        image_size=(
            int(environment_config.get("image_width", 224)),
            int(environment_config.get("image_height", 224)),
        ),
        max_steps=effective_max_steps,
        attack_target=environment_config.get("attack_target", "fake_to_real"),
        quality_weight=float(environment_config.get("quality_weight", 0.25)),
        perturbation_weight=float(environment_config.get("perturbation_weight", 0.10)),
        seed=effective_seed,
        strength_levels=int(environment_config.get("strength_levels", 5)),
    )
    check_env(env, warn=True)
    model = load_ppo_agent(resume, env=env, device=effective_device) if resume else create_ppo_agent(
        env, seed=effective_seed, device=effective_device, **ppo_config
    )
    output_path = Path(output_dir).expanduser().resolve()
    checkpoint = output_path / "checkpoints" / "ppo_final"
    (output_path / "logs").mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=effective_timesteps)
    model.save(str(checkpoint))

    used_config: dict[str, Any] = dict(config)
    used_config["project"] = {**project, "seed": effective_seed, "device": effective_device}
    used_config["training"] = {"total_timesteps": effective_timesteps, "train_dir": str(Path(train_dir).resolve())}
    used_config["environment"] = {**environment_config, "max_steps": effective_max_steps}
    used_config["victim"] = {
        **victim_config,
        "deepfakebench_root": str(Path(deepfakebench_root).resolve()),
        "weights_path": str(Path(weights_path).resolve()),
    }
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to save training configuration") from error
    with (output_path / "training_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(used_config, output_file, sort_keys=False)
    LOGGER.info(
        "device=%s images=%d detector=%s seed=%d max_steps=%d timesteps=%d checkpoint=%s.zip",
        effective_device, len(image_paths), detector.detector_name, effective_seed,
        effective_max_steps, effective_timesteps, checkpoint,
    )
    return checkpoint.with_suffix(".zip")


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""
    parser = argparse.ArgumentParser(description="Train a PPO deepfake attack policy.")
    parser.add_argument("--train-dir", required=True, help="Directory containing training images.")
    parser.add_argument("--deepfakebench-root", required=True, help="DeepfakeBench repository root.")
    parser.add_argument("--weights", required=True, help="Trained Xception checkpoint path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="DimSum YAML configuration path.")
    parser.add_argument("--timesteps", type=int, help="PPO training timesteps (overrides config).")
    parser.add_argument("--max-steps", type=int, help="Maximum attack steps per episode.")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--output-dir", default="outputs", help="Training artifact directory.")
    parser.add_argument("--resume", help="Optional PPO checkpoint to resume.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the training command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    train_attack(
        train_dir=args.train_dir,
        deepfakebench_root=args.deepfakebench_root,
        weights_path=args.weights,
        config_path=args.config,
        timesteps=args.timesteps,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

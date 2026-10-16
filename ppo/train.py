"""Command-line and notebook helpers for training the DimSum PPO attacker."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from victim.detector import VictimDetector

from .agent import create_ppo_agent, load_ppo_agent
from .common import collect_image_paths, load_image_manifest, load_yaml, seed_everything
from .environment import DeepfakeAttackEnv


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


class ActionFrequencyCallback(BaseCallback):
    """Persist action/strength counts after every rollout, including partial runs."""

    def __init__(self, output_path: Path, load_existing: bool = False) -> None:
        super().__init__()
        self.output_path = output_path
        self.actions: Counter[str] = Counter()
        self.strengths: Counter[str] = Counter()
        if load_existing and output_path.is_file():
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            self.actions.update(previous.get("action_selection_frequency", {}))
            self.strengths.update(previous.get("strength_distribution", {}))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "action_name" in info:
                self.actions[str(info["action_name"])] += 1
            if "strength" in info:
                self.strengths[f'{float(info["strength"]):.2f}'] += 1
        return True

    def _on_rollout_end(self) -> None:
        payload = {
            "completed_timesteps": int(self.num_timesteps),
            "action_selection_frequency": dict(sorted(self.actions.items())),
            "strength_distribution": dict(sorted(self.strengths.items())),
        }
        self.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_attack(
    train_dir: str | Path | None,
    deepfakebench_root: str | Path,
    weights_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    timesteps: int | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    output_dir: str | Path = "outputs",
    resume: str | Path | None = None,
    train_manifest: str | Path | None = None,
    checkpoint_freq: int = 0,
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
    if (train_dir is None) == (train_manifest is None):
        raise ValueError("Provide exactly one of train_dir or train_manifest")
    image_paths = (
        load_image_manifest(train_manifest, "training", label="FAKE")
        if train_manifest is not None
        else collect_image_paths(train_dir, "training")
    )
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
        action_set=str(environment_config.get("action_set", "basic")),
        min_ssim=environment_config.get("min_ssim"),
        quality_violation_penalty=float(environment_config.get("quality_violation_penalty", 0.5)),
    )
    check_env(env, warn=True)
    model = load_ppo_agent(resume, env=env, device=effective_device) if resume else create_ppo_agent(
        env, seed=effective_seed, device=effective_device, **ppo_config
    )
    output_path = Path(output_dir).expanduser().resolve()
    checkpoint = output_path / "checkpoints" / "ppo_final"
    (output_path / "logs").mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    callbacks: list[BaseCallback] = [ActionFrequencyCallback(
        output_path / "training_actions.json", load_existing=resume is not None
    )]
    if checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=int(checkpoint_freq),
            save_path=str(output_path / "checkpoints" / "intermediate"),
            name_prefix="ppo",
        ))
    model.learn(
        total_timesteps=effective_timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=resume is None,
    )
    completed_timesteps = int(model.num_timesteps)
    model.save(str(checkpoint))

    used_config: dict[str, Any] = dict(config)
    used_config["project"] = {**project, "seed": effective_seed, "device": effective_device}
    used_config["training"] = {
        "requested_timesteps": effective_timesteps,
        "completed_timesteps": completed_timesteps,
        "train_dir": str(Path(train_dir).resolve()) if train_dir is not None else None,
        "train_manifest": str(Path(train_manifest).resolve()) if train_manifest is not None else None,
        "training_images": len(image_paths),
    }
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
        "device=%s images=%d detector=%s seed=%d max_steps=%d requested_timesteps=%d completed_timesteps=%d checkpoint=%s.zip",
        effective_device, len(image_paths), detector.detector_name, effective_seed,
        effective_max_steps, effective_timesteps, completed_timesteps, checkpoint,
    )
    return checkpoint.with_suffix(".zip")


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""
    parser = argparse.ArgumentParser(description="Train a PPO deepfake attack policy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-dir", help="Directory containing training images.")
    source.add_argument("--train-manifest", help="JSON split manifest; only labelled FAKE rows train PPO.")
    parser.add_argument("--deepfakebench-root", required=True, help="DeepfakeBench repository root.")
    parser.add_argument("--weights", required=True, help="Trained Xception checkpoint path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="DimSum YAML configuration path.")
    parser.add_argument("--timesteps", type=int, help="PPO training timesteps (overrides config).")
    parser.add_argument("--max-steps", type=int, help="Maximum attack steps per episode.")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--output-dir", default="outputs", help="Training artifact directory.")
    parser.add_argument("--resume", help="Optional PPO checkpoint to resume.")
    parser.add_argument(
        "--checkpoint-freq", type=int, default=0,
        help="Save an intermediate model every N steps. Continuation restores the model and optimizer, not exact RNG/environment state.",
    )
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
        train_manifest=args.train_manifest,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()

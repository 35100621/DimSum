"""Build the checked-in DimSum notebooks using only the Python standard library."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def write(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "notebooks" / name).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


SETUP = """from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root: {PROJECT_ROOT}")"""


write("01_detector_setup.ipynb", [
    markdown("# 01 — DeepfakeBench Xception setup\n\nThis notebook uses the extracted handover. The model is loaded once by `VictimDetector`, which owns the teammate's exact preprocessing and returns normalized REAL/FAKE probabilities."),
    code(SETUP),
    code("""from victim.detector import VictimDetector

DEEPFAKEBENCH_ROOT = PROJECT_ROOT.parent / "DeepfakeBench_Handover"
XCEPTION_CONFIG = DEEPFAKEBENCH_ROOT / "training/config/detector/xception.yaml"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_ROOT / "training/weights/xception_best.pth"
SAMPLE_IMAGE = DEEPFAKEBENCH_ROOT / "datasets/rgb/FF_small/fake/Deepfakes_0000.png"
DEVICE = "cpu"
RUN_DETECTOR = True"""),
    code("""for label, path in {
    "DeepfakeBench": DEEPFAKEBENCH_ROOT,
    "Xception config": XCEPTION_CONFIG,
    "Xception weights": XCEPTION_WEIGHTS,
    "sample image": SAMPLE_IMAGE,
}.items():
    print(f"{label:18s} {path}  exists={path.exists()}")"""),
    code("""if RUN_DETECTOR:
    detector = VictimDetector(
        DEEPFAKEBENCH_ROOT,
        config_path=XCEPTION_CONFIG,
        weights_path=XCEPTION_WEIGHTS,
        device=DEVICE,
    )
    prediction = detector.predict_path(SAMPLE_IMAGE)
    print(prediction)
    assert prediction["label"] in {"REAL", "FAKE"}
    assert abs(prediction["real_probability"] + prediction["fake_probability"] - 1) < 1e-5
else:
    print("Set RUN_DETECTOR=True after configuring the paths above.")"""),
])


write("02_attack_baselines.ipynb", [
    markdown("# 02 — Controlled attack baselines\n\nEvery action preserves image size, RGB channels, and the valid pixel range. Strength is normalized to `[0, 1]`."),
    code(SETUP),
    code("""import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ppo.actions import ACTION_NAMES, apply_action

height, width = 160, 224
x = np.linspace(32, 240, width, dtype=np.uint8)
gradient = np.tile(x, (height, 1))
checker = ((np.indices((height, width)).sum(axis=0) // 10) % 2 * 20).astype(np.uint8)
image = Image.fromarray(np.stack([gradient, np.clip(gradient + checker, 0, 255), gradient], axis=2)).convert("RGB")"""),
    code("""strength = 0.75
outputs = {
    name: apply_action(image, action_id, strength, np.random.default_rng(42))
    for action_id, name in ACTION_NAMES.items()
}
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for axis, (name, output) in zip(axes.flat, outputs.items()):
    axis.imshow(output)
    axis.set_title(name.replace("_", " ").title())
    axis.axis("off")
axes.flat[-1].axis("off")
plt.tight_layout()"""),
    code("""for name, output in outputs.items():
    pixels = np.asarray(output)
    assert output.mode == "RGB" and output.size == image.size
    assert pixels.dtype == np.uint8 and 0 <= pixels.min() <= pixels.max() <= 255
    difference = np.abs(pixels.astype(float) - np.asarray(image).astype(float)).mean()
    print(f"{name:20s} mean absolute pixel change={difference:.3f}")"""),
])


write("03_ppo_environment.ipynb", [
    markdown("# 03 — PPO attack environment\n\nThis notebook uses a synthetic image and deterministic mock victim, so it runs without DeepfakeBench. The real detector uses the same `predict_image` contract."),
    code(SETUP),
    code("""import numpy as np
from PIL import Image
from stable_baselines3.common.env_checker import check_env

from ppo.environment import DeepfakeAttackEnv
from src.ppo.mock_detector import MockDetector

demo_dir = PROJECT_ROOT / "outputs/notebook_demo"
demo_dir.mkdir(parents=True, exist_ok=True)
image_path = demo_dir / "synthetic_fake.png"
x = np.linspace(96, 240, 224, dtype=np.uint8)
pixels = np.tile(x, (160, 1))
Image.fromarray(np.stack([pixels] * 3, axis=2)).save(image_path)

env = DeepfakeAttackEnv([image_path], MockDetector(), max_steps=5, seed=42)
check_env(env, warn=True)
print("Environment check passed", env.action_space, env.observation_space)"""),
    code("""observation, reset_info = env.reset(seed=42)
print("reset", observation, reset_info)
for action in ([5, 3], [3, 2], [1, 1]):
    observation, reward, terminated, truncated, info = env.step(np.array(action))
    print(f"action={action} reward={reward:.4f} obs={observation}")
    print(info)
    if terminated or truncated:
        break"""),
    code("""assert observation.shape == (6,) and observation.dtype == np.float32
assert env.original_image is not env.current_image
assert 0 <= info["ssim"] <= 1 and 0 <= info["perturbation"] <= 1
print("Reward direction: positive confidence gain means P(FAKE) decreased.")"""),
])


write("04_ppo_training.ipynb", [
    markdown("# 04 — PPO training\n\nThe default path runs a short end-to-end PPO integration against the frozen Xception victim using only labelled FAKE images from the training manifest. It writes an integration checkpoint, not a final experimental model. Set `FULL_TRAINING=True` for the configured 10,000-timestep run."),
    code(SETUP),
    code("""FULL_TRAINING = False
TRAIN_MANIFEST = PROJECT_ROOT / "data/splits/ff_ppo_train.json"
DEEPFAKEBENCH_ROOT = PROJECT_ROOT.parent / "DeepfakeBench_Handover"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_ROOT / "training/weights/xception_best.pth"
TIMESTEPS = 10_000 if FULL_TRAINING else 64
OUTPUT_DIR = PROJECT_ROOT / ("outputs" if FULL_TRAINING else "outputs/integration_smoke")
SEED = 42
DEVICE = "cpu"
"""),
    code("""from ppo.train import train_attack

checkpoint = train_attack(
    train_dir=None,
    train_manifest=TRAIN_MANIFEST,
    deepfakebench_root=DEEPFAKEBENCH_ROOT,
    weights_path=XCEPTION_WEIGHTS,
    timesteps=TIMESTEPS,
    seed=SEED,
    device=DEVICE,
    output_dir=OUTPUT_DIR,
)
print(checkpoint)"""),
    code("""assert checkpoint.is_file()
print("Integration training complete." if not FULL_TRAINING else "Full PPO training complete.")
print("Training and held-out test manifests are disjoint.")"""),
])


write("05_evaluation.ipynb", [
    markdown("# 05 — Held-out evaluation\n\nEvaluation never updates PPO. It uses deterministic actions on the disjoint test manifest. Attack success rate counts only ground-truth FAKE images that Xception initially detects as FAKE. The default checkpoint is the short integration model and its metrics are a pipeline check, not final experimental results."),
    code(SETUP),
    code("""from evaluation.evaluate_attack import evaluate_attack

TEST_MANIFEST = PROJECT_ROOT / "data/splits/ff_test.json"
MODEL_PATH = PROJECT_ROOT / "outputs/integration_smoke/checkpoints/ppo_final.zip"
DEEPFAKEBENCH_ROOT = PROJECT_ROOT.parent / "DeepfakeBench_Handover"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_ROOT / "training/weights/xception_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs/integration_smoke_evaluation"
RUN_EVALUATION = True"""),
    code("""for label, path in {
    "held-out manifest": TEST_MANIFEST,
    "PPO checkpoint": MODEL_PATH,
    "DeepfakeBench": DEEPFAKEBENCH_ROOT,
    "Xception weights": XCEPTION_WEIGHTS,
}.items():
    print(f"{label:18s} {path}  exists={path.exists()}")"""),
    code("""if RUN_EVALUATION:
    summary = evaluate_attack(
        test_dir=None,
        test_manifest=TEST_MANIFEST,
        model_path=MODEL_PATH,
        deepfakebench_root=DEEPFAKEBENCH_ROOT,
        weights_path=XCEPTION_WEIGHTS,
        output_dir=OUTPUT_DIR,
        seed=42,
        device="cpu",
        save_images=True,
    )
    display(summary)
    print(OUTPUT_DIR / "results.csv")
    print(OUTPUT_DIR / "summary.json")
else:
    print("Set RUN_EVALUATION=True to evaluate the configured checkpoint.")"""),
])

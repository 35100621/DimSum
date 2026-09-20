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
    markdown("# 01 — DeepfakeBench Xception setup\n\nConfigure the three paths below. The model is loaded once by `VictimDetector`, which also owns resizing and normalization. Leave `RUN_DETECTOR = False` to validate imports without external weights."),
    code(SETUP),
    code("""from victim.detector import VictimDetector

DEEPFAKEBENCH_ROOT = PROJECT_ROOT.parent / "DeepfakeBench"
XCEPTION_CONFIG = DEEPFAKEBENCH_ROOT / "training/config/detector/xception.yaml"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_ROOT / "training/weights/xception_best.pth"
SAMPLE_IMAGE = PROJECT_ROOT / "data/test/sample.jpg"
DEVICE = "auto"
RUN_DETECTOR = False"""),
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
    markdown("# 04 — PPO training\n\nThe default smoke path trains against a mock victim and writes `outputs/checkpoints/ppo_notebook_smoke.zip`. Switch `USE_MOCK` off and configure the paths to train against DeepfakeBench Xception."),
    code(SETUP),
    code("""USE_MOCK = True
TRAIN_DIR = PROJECT_ROOT / "data/train"
DEEPFAKEBENCH_ROOT = PROJECT_ROOT.parent / "DeepfakeBench"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_ROOT / "training/weights/xception_best.pth"
TIMESTEPS = 32 if USE_MOCK else 10_000
SEED = 42
DEVICE = "auto"
"""),
    code("""if USE_MOCK:
    import numpy as np
    from PIL import Image
    from ppo.agent import create_ppo_agent
    from ppo.environment import DeepfakeAttackEnv
    from src.ppo.mock_detector import MockDetector

    demo_dir = PROJECT_ROOT / "outputs/notebook_demo/train"
    demo_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, level in enumerate((160, 190, 220)):
        path = demo_dir / f"synthetic_{index}.png"
        Image.fromarray(np.full((64, 64, 3), level, dtype=np.uint8)).save(path)
        paths.append(path)
    env = DeepfakeAttackEnv(paths, MockDetector(), max_steps=3, seed=SEED)
    model = create_ppo_agent(env, seed=SEED, device=DEVICE, n_steps=16, batch_size=8, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)
    checkpoint = PROJECT_ROOT / "outputs/checkpoints/ppo_notebook_smoke"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    print(checkpoint.with_suffix(".zip"))
else:
    from ppo.train import train_attack
    checkpoint = train_attack(
        train_dir=TRAIN_DIR,
        deepfakebench_root=DEEPFAKEBENCH_ROOT,
        weights_path=XCEPTION_WEIGHTS,
        timesteps=TIMESTEPS,
        seed=SEED,
        device=DEVICE,
        output_dir=PROJECT_ROOT / "outputs",
    )
    print(checkpoint)"""),
    code("""observation, _ = env.reset(seed=SEED) if USE_MOCK else (None, None)
if USE_MOCK:
    action, _ = model.predict(observation, deterministic=True)
    print("deterministic action [type, strength]:", action)
    assert env.action_space.contains(action)"""),
])


write("05_evaluation.ipynb", [
    markdown("# 05 — Held-out evaluation\n\nEvaluation never updates PPO. It uses deterministic actions and reports attack success only for images initially classified FAKE. Configure the paths, keep the test directory separate from training, and set `RUN_EVALUATION=True`."),
    code(SETUP),
    code("""from evaluation.evaluate_attack import evaluate_attack

TEST_DIR = PROJECT_ROOT / "data/test"
MODEL_PATH = PROJECT_ROOT / "outputs/checkpoints/ppo_final.zip"
DEEPFAKEBENCH_ROOT = PROJECT_ROOT.parent / "DeepfakeBench"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_ROOT / "training/weights/xception_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs/evaluation"
RUN_EVALUATION = False"""),
    code("""for label, path in {
    "held-out images": TEST_DIR,
    "PPO checkpoint": MODEL_PATH,
    "DeepfakeBench": DEEPFAKEBENCH_ROOT,
    "Xception weights": XCEPTION_WEIGHTS,
}.items():
    print(f"{label:18s} {path}  exists={path.exists()}")"""),
    code("""if RUN_EVALUATION:
    summary = evaluate_attack(
        test_dir=TEST_DIR,
        model_path=MODEL_PATH,
        deepfakebench_root=DEEPFAKEBENCH_ROOT,
        weights_path=XCEPTION_WEIGHTS,
        output_dir=OUTPUT_DIR,
        seed=42,
        device="auto",
        save_images=True,
    )
    display(summary)
    print(OUTPUT_DIR / "results.csv")
    print(OUTPUT_DIR / "summary.json")
else:
    print("Set RUN_EVALUATION=True after configuring the four paths above.")"""),
])

import json
from pathlib import Path

import pandas as pd

from evaluation.evaluate_accuracy_reduction import evaluate_accuracy_reduction


def test_accuracy_uses_all_images_while_asr_uses_eligible_fake_only(tmp_path: Path):
    root = tmp_path / "images"; root.mkdir()
    samples, rows = [], []
    for index in range(40):
        label = "REAL" if index < 20 else "FAKE"
        path = root / f"{index}.png"; path.write_bytes(b"x")
        initial = "REAL" if index < 20 or index in {20, 21, 22, 23} else "FAKE"
        final = "REAL" if label == "FAKE" else "FAKE"
        samples.append({"path": path.name, "label": label})
        rows.append({"image_path": str(path.resolve()), "ground_truth_label": label,
                     "initial_label": initial, "final_label": final,
                     "attack_success": label == "FAKE" and initial == "FAKE",
                     "ssim": 0.95})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_root": str(root), "samples": samples}))
    results = tmp_path / "attack.csv"; pd.DataFrame(rows).to_csv(results, index=False)
    summary = evaluate_accuracy_reduction(manifest, results, tmp_path / "out")
    assert summary["clean"]["correct"] == 36
    assert summary["after_quality_ppo"]["correct"] == 20
    assert summary["attack_performance"]["eligible_fake_images"] == 16
    assert summary["attack_performance"]["asr"] == 1.0
    output = pd.read_csv(tmp_path / "out/results.csv")
    real = output[output.ground_truth_label == "REAL"]
    assert (real.after_label == real.initial_label).all()
    assert (real.after_source == "unchanged_original_real").all()

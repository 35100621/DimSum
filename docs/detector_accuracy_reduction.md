# Xception Detector Accuracy Reduction

## Evaluation set

- Total: 40 held-out images
- REAL: 20
- FAKE: 20
- Manifest: `data/splits/ff_test.json`
- Detector: frozen pretrained DeepfakeBench Xception
- Attack: completed Quality-Constrained PPO, deterministic policy

The before and after evaluations use the same fixed paths and original labels. In
the after-attack set, all 20 REAL images remain unchanged and all 20 FAKE images use
their saved Quality-PPO outputs. Accuracy uses all 40 images. ASR remains separate
and uses only the 16 FAKE images that Xception originally classified correctly.

## Before attack

- Correct: **29/40**
- Incorrect: 11/40
- Clean accuracy: **72.5%**

Confusion matrix, rows = true `[REAL, FAKE]`, columns = predicted `[REAL, FAKE]`:

```text
[[13,  7],
 [ 4, 16]]
```

## After Quality-Constrained PPO attack

- Correct: **15/40**
- Incorrect: 25/40
- Attacked accuracy: **37.5%**

```text
[[13,  7],
 [18,  2]]
```

The REAL row is identical because those 20 samples were unchanged. Among the 20
FAKE images, 18 are classified REAL after processing: four were pre-existing false
negatives and 14 were successful attacks. Two remain correctly classified FAKE.

## Accuracy reduction

```text
72.5% − 37.5% = 35.0 percentage points
```

## Attack performance, kept separate

- Eligible FAKE images: 16
- Successful attacks: 14
- Raw ASR: **87.5%**
- ASR at SSIM ≥ 0.90: **87.5%**
- Mean SSIM over eligible FAKE images: **0.9541**

## Command

```bash
.venv/bin/python -m evaluation.evaluate_accuracy_reduction \
  --test-manifest data/splits/ff_test.json \
  --attack-results outputs/expanded_quality_seed42_evaluation_completed/results.csv \
  --output-dir outputs/detector_accuracy_reduction_quality_ppo
```

No PPO or Xception training was performed. The script validates unique paths, exact
manifest/result path equality, ground-truth agreement, and the expected 20/20 class
split. It explicitly assigns every REAL sample its original prediction after attack
and every FAKE sample its saved final Quality-PPO prediction.

## Artifacts

- Summary: `outputs/detector_accuracy_reduction_quality_ppo/summary.json`
- Per-image before/after correctness:
  `outputs/detector_accuracy_reduction_quality_ppo/results.csv`
- Source attack evaluation:
  `outputs/expanded_quality_seed42_evaluation_completed/results.csv`

## Limitations

- The held-out set contains only 40 images.
- Seven of 20 clean REAL images are already false positives, and four of 20 clean
  FAKE images are already false negatives.
- The result uses one approximately resumed seed-42 policy and one Xception detector.
- Source-video and identity independence remain unverifiable from the flattened
  manifests.

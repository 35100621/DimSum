# DimSum Project Status

**Project:** PPO-Based Adversarial Attacks Against Deepfake Detectors
**Current stage:** Expanded and quality-constrained PPO completed and evaluated
**Last updated:** 4 October 2026 (post next-steps execution)

## 1. Current system

The working pipeline is:

```text
Held-out or training image
        ↓
DeepfakeAttackEnv
        ↓
PPO selects [transformation, strength]
        ↓
Adversarial image
        ↓
Frozen DeepfakeBench Xception detector
        ↓
REAL/FAKE probabilities
        ↓
Confidence gain − quality penalties + success bonus
```

The projects remain separate:

- DimSum: `/Users/hgtan/Desktop/DimSum`
- DeepfakeBench handover: `/Users/hgtan/Desktop/DeepfakeBench_Handover`
- Xception checkpoint: `DeepfakeBench_Handover/training/weights/xception_best.pth`
- Dataset: `DeepfakeBench_Handover/datasets/rgb/FF_small`

## 2. Completed work

### DeepfakeBench handover

- Inspected and selectively extracted the supplied handover archive.
- Retained only the Xception checkpoint and required ImageNet backbone weights.
- Excluded approximately 1.4 GB of unrelated detector checkpoints.
- Confirmed the standalone `FF_small.zip` is byte-identical to the dataset already
  included in the handover, so it was not duplicated.
- Restored the archive's missing `networks`, `loss`, and `metrics` modules from
  official DeepfakeBench commit `f188b1c105465e2e5377eb536a95022ae0e4522d`.

### Xception detector interface

`victim/detector.py` now:

- loads Xception once;
- loads `xception_best.pth` successfully;
- keeps the model in evaluation mode;
- freezes every detector parameter;
- maps class 0 to REAL and class 1 to FAKE;
- returns normalized REAL and FAKE probabilities;
- accepts PIL, NumPy, and Torch images;
- isolates DeepfakeBench imports from PPO and evaluation code;
- provides no-gradient inference for ordinary evaluation;
- provides an opt-in differentiable tensor path for future FGSM/PGD attacks.

The wrapper reproduces the handover preprocessing exactly:

- RGB conversion;
- Pillow default resize to 224 × 224;
- ImageNet mean `[0.485, 0.456, 0.406]`;
- ImageNet standard deviation `[0.229, 0.224, 0.225]`;
- FAKE decision threshold `0.5`.

### Baseline reproduction

The full 200-image baseline was reproduced with 100 REAL and 100 FAKE images.

| Metric | Reported | Reproduced |
|---|---:|---:|
| Accuracy | 0.745 | 0.745 |
| AUC | 0.854 | 0.854 |
| Precision | 0.7130434782608696 | 0.7130434782608696 |
| Recall | 0.82 | 0.82 |

Confusion matrix:

```text
[[67, 33],
 [18, 82]]
```

The baseline can be rerun with:

```bash
python -m evaluation.evaluate_victim \
  --deepfakebench-root ../DeepfakeBench_Handover \
  --device cpu
```

### Dataset split

A deterministic seed-42, stratified 80/20 split was created without copying the
images.

| Partition | Total | FAKE | REAL |
|---|---:|---:|---:|
| PPO training manifest | 160 | 80 | 80 |
| Held-out test manifest | 40 | 20 | 20 |

PPO training selects only the 80 labelled FAKE training records because the
current objective is FAKE → REAL. REAL records remain in the manifest for split
accounting and future objectives.

Each fake manipulation contributes 16 training and 4 test images:

- Deepfakes
- Face2Face
- FaceShifter
- FaceSwap
- NeuralTextures

Each real source contributes 40 training and 10 test images:

- actor_real
- youtube_real

The partitions have no path overlap and no duplicate file-content hashes across
the split.

Split files:

- `data/splits/ff_ppo_train.json`
- `data/splits/ff_test.json`
- Generator: `scripts/create_ff_split.py`

### PPO environment and actions

The Gymnasium environment supports sequential transformations with five strength
levels. Implemented actions are:

1. No-op
2. Gaussian noise
3. Gaussian blur
4. Brightness decrease
5. Brightness increase
6. JPEG compression
7. Downscale and restore

The observation contains current and initial detector probabilities, normalized
episode progress, and accumulated distortion.

The FAKE → REAL reward is:

```text
previous FAKE probability − current FAKE probability
− quality weight × (1 − SSIM)
− perturbation weight × normalized L1 difference
+ 1.0 only for a valid FAKE → REAL flip
```

Successful attacks terminate an episode. Reaching the maximum number of steps
truncates it.

### Training and evaluation integration

Training now accepts either a directory or a split manifest:

```bash
python -m ppo.train \
  --train-manifest data/splits/ff_ppo_train.json \
  --deepfakebench-root ../DeepfakeBench_Handover \
  --weights ../DeepfakeBench_Handover/training/weights/xception_best.pth \
  --timesteps 10000 \
  --device cpu
```

Evaluation accepts the held-out manifest and uses deterministic PPO actions:

```bash
python -m evaluation.evaluate_attack \
  --test-manifest data/splits/ff_test.json \
  --model outputs/checkpoints/ppo_final.zip \
  --deepfakebench-root ../DeepfakeBench_Handover \
  --weights ../DeepfakeBench_Handover/training/weights/xception_best.pth \
  --device cpu
```

Attack success rate counts only ground-truth FAKE test images that Xception
initially detects as FAKE. Ground-truth REAL images and existing false negatives
remain in `results.csv` but cannot inflate ASR.

Evaluation reports:

- attack success rate;
- mean, median, and standard deviation of FAKE-confidence reduction;
- mean and median SSIM;
- mean and median PSNR;
- one row per test image;
- optional adversarial PNG files.

### Notebooks

All five notebooks compile and run through their configured default paths:

1. `01_detector_setup.ipynb` — real Xception loading and inference
2. `02_attack_baselines.ipynb` — transformation visualization
3. `03_ppo_environment.ipynb` — environment and reward validation
4. `04_ppo_training.ipynb` — real-Xception integration rollout or full training
5. `05_evaluation.ipynb` — deterministic held-out evaluation

Notebook 04 defaults to a short integration run. Set `FULL_TRAINING = True` to
launch the configured full run.

## 3. Integration-smoke results

A real 256-step PPO rollout completed on CPU. Stable-Baselines3 completes a full
256-step rollout even though the smoke request is 64 timesteps.

Artifacts:

- Checkpoint: `outputs/integration_smoke/checkpoints/ppo_final.zip`
- Training config: `outputs/integration_smoke/training_config.yaml`
- Per-image results: `outputs/integration_smoke_evaluation/results.csv`
- Summary: `outputs/integration_smoke_evaluation/summary.json`

Held-out smoke evaluation:

| Metric | Result |
|---|---:|
| Test images | 40 |
| Ground-truth FAKE images | 20 |
| Ground-truth REAL references | 20 |
| Eligible attacks | 16 |
| Existing FAKE false negatives | 4 |
| Successful FAKE → REAL flips | 3 |
| Attack success rate | 0.1875 |
| Mean FAKE-confidence reduction | 0.2909 |
| Mean SSIM | 0.9471 |
| Mean PSNR | 30.98 |

These are software-integration results, not final experimental findings. The PPO
policy received only one rollout and is not considered fully trained.

## 4. Verification status

- Unit test suite: **32 passed, 2 optional integration tests skipped**
- Real checkpoint integration test: **passed**
- Detector probabilities: normalized and valid
- Detector weights: frozen
- Model mode: evaluation
- Differentiable input gradients: verified finite
- Training/test path overlap: none
- Training/test content-hash overlap: none
- Training and evaluation CLI help: working
- All notebook code cells: syntactically valid

The only recurring warning is an invalid escape sequence inside the original
DeepfakeBench Xception source docstring. It does not affect execution or results.

## 5. Next-steps execution results

### Verified: full basic-action PPO

The untouched seven-action policy was trained from scratch with seed 42 on the 80
FAKE training records. The request was 10,000 timesteps; Stable-Baselines3 completed
the enclosing 256-step rollout, so the saved model contains 10,240 timesteps.

- Hardware: Apple arm64 CPU, four Torch threads; CUDA and MPS unavailable
- Wall time: 482.28 seconds
- Checkpoint: `outputs/basic_actions_seed42/checkpoints/ppo_final.zip`
- Configuration: `outputs/basic_actions_seed42/training_config.yaml`
- Full console log: `outputs/basic_actions_seed42_training.log`

Deterministic evaluation on the 40 held-out records produced:

| Metric | Basic PPO |
|---|---:|
| Eligible FAKE samples | 16 |
| Successful attacks | 11 |
| ASR | 0.6875 |
| Mean / median / std confidence reduction | 0.4102 / 0.4843 / 0.1334 |
| Mean / median / std SSIM | 0.3751 / 0.2378 / 0.2081 |
| Mean / median / std PSNR | 26.37 / 25.71 / 1.87 dB |

The deterministic policy selected Gaussian noise 132 times and resize/restore 33
times. Every selected action used strength 0.75. This is policy collapse and the
very low SSIM shows excessive degradation; the ASR must be interpreted with that
quality failure. Results, action traces, before/after images, and exact artifact
paths are in `outputs/basic_actions_seed42_evaluation/`.

Direct inspection of the lowest-SSIM pairs shows that `FF_small` contains
high-contrast, mask-like images with large uniform black and white regions. The
policy applies resize/restore and then stacks three or four 0.75-strength Gaussian
noise actions, or applies Gaussian noise five times. Repeated noise over uniform
regions causes the SSIM collapse, and cumulative perturbation is not constrained
against the original image. The inspection montage is
`outputs/basic_actions_seed42_evaluation/lowest_ssim_montage.jpg`.

Checkpoint hashes are recorded in `outputs/checkpoint_sha256.txt`; the completed
basic checkpoint SHA-256 is
`cea01be41af9c5a9137e28296b72463671f4806bb46d97b84447f02393ca6e7e`.

### Implemented and independently evaluated attacks

`ppo/actions.py` now provides bounded targeted FGSM, deterministic targeted PGD,
and a deterministic FFT low-pass modification. FGSM and PGD optimize class 0
(REAL), clamp pixels into `[0, 1]`, and never enable gradients on Xception weights.
Standalone defaults use epsilon `8/255`; PGD uses step size `2/255` for 10
iterations. The frequency baseline uses epsilon `8/255` and cutoff 0.35.

| Method | Successes / eligible | ASR | Mean confidence reduction | Mean SSIM | Mean PSNR | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| No attack | 0 / 16 | 0.0000 | 0.0000 | 1.0000 | infinite | 2.59 s |
| FGSM | 4 / 16 | 0.2500 | 0.1894 | 0.8728 | 39.60 dB | 66.07 s |
| PGD | 7 / 16 | 0.4375 | 0.4409 | 0.9363 | 42.04 dB | 555.94 s |
| Frequency low-pass | 2 / 16 | 0.1250 | 0.1470 | 0.9163 | 39.31 dB | 5.15 s |
| Basic PPO | 11 / 16 | 0.6875 | 0.4102 | 0.3751 | 26.37 dB | evaluation 9.32 s |

These are preliminary single-seed results. Basic PPO has higher ASR but far worse
image quality, and PGD has a much higher computational cost. The results do not
support an unqualified PPO superiority claim. Standalone CSV, JSON, and images are
under `outputs/baselines/`.

Quality-constrained ASR uses the original 16 eligible samples as the denominator
and counts a success only when it also meets the SSIM threshold:

| Method | SSIM ≥ 0.90 | SSIM ≥ 0.85 | SSIM ≥ 0.80 | SSIM ≥ 0.70 | SSIM ≥ 0.50 |
|---|---:|---:|---:|---:|---:|
| No attack | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FGSM | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.2500 |
| PGD | 0.0000 | 0.3125 | 0.4375 | 0.4375 | 0.4375 |
| Frequency low-pass | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 |
| Basic PPO | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3125 |

This comparison is saved in `outputs/comparison_seed42.json`. At SSIM ≥ 0.80,
PGD is the only evaluated attack with nonzero constrained ASR. Frequency low-pass
was previously the only evaluated method with successes at SSIM ≥ 0.90. The now
completed expanded PPO evaluation supersedes that earlier incomplete comparison;
see `docs/expanded_ppo_evaluation.md`.

### Verified: completed expanded PPO

The environment accepts `action_set: expanded`, adding FGSM, five-iteration PGD,
and frequency low-pass to the existing seven actions. The basic action set remains
the default, preserving its checkpoint compatibility. Expanded strengths map to
predefined perturbation budgets and sequential actions retain the existing reward,
quality penalties, success termination, and step-limit truncation.

A new-from-scratch seed-42 reference run was started with `config/expanded.yaml`.
The approximately resumed seed-42 reference run completed 10,240 cumulative steps.
The final checkpoint is `outputs/expanded_actions_seed42/checkpoints/ppo_final.zip`
with SHA-256 `9607610a04a6fcb3d48727b23dcc7272179d7f0db3aa8af2da2b0ccdcd5e2f4c`.
Exact RNG, rollout-buffer, and environment state were not restored at continuation,
so this is not described as a bit-identical uninterrupted run.

Held-out deterministic evaluation achieved 16/16 raw ASR (100%), mean SSIM 0.9468,
mean PSNR 31.42 dB, and mean L1 distortion 0.0090. Quality-constrained ASR is 12/16
(75%) at SSIM ≥ 0.90 and 16/16 at SSIM ≥ 0.85. The policy selected only
strength-0.50 PGD and Gaussian blur on held-out episodes. Full evidence is in
`docs/expanded_ppo_evaluation.md` and
`outputs/expanded_actions_seed42_evaluation_completed/`.

Training now supports `--checkpoint-freq` and records action/strength counts after
each rollout. The exact continuation command and state limitations are stored in
`outputs/expanded_actions_seed42/reference_run_status.json`. Continuation restores
the PPO model and optimizer, but is explicitly approximate because exact RNG,
rollout-buffer, and environment state are not serialized.

### Verification after extension

- Full suite with real checkpoint enabled: **41 passed, 1 skipped**
- Only remaining skip: optional training integration coverage
- Detector mode, frozen parameters, valid probabilities, and differentiable input
  gradients passed against the real checkpoint
- New tests cover perturbation bounds, RGB/range validity, targeted direction,
  deterministic PGD/frequency behavior, and frozen model parameters
- Path and content-hash overlap remain absent
- Source-video and identity overlap remain unresolved because the flattened manifest
  records contain only broad group labels, not source video or identity metadata

### Files added or changed in this execution

- `ppo/actions.py`: FGSM, PGD, frequency operator, and expanded action registry
- `ppo/environment.py`: selectable basic/expanded action spaces
- `ppo/train.py`: reads the configured action set
  and supports periodic checkpoints/action-frequency artifacts for long runs
- `evaluation/evaluate_attack.py`: action/strength counts, action traces, exact
  artifact references, and before/after images
- `evaluation/evaluate_baselines.py`: standalone baseline CLI and reproducible reports
- `config/default.yaml`: explicit basic action set
- `config/expanded.yaml`: isolated expanded-action experiment configuration
- `tests/test_gradient_attacks.py`: attack correctness and safety tests
- `docs/project_status.md`: actual configurations, outcomes, limitations, and next work

## 6. Quality-constrained result and remaining work

### Quality-constrained expanded PPO

`config/expanded_quality.yaml` implements a separate hard original-relative
SSIM ≥ 0.90 constraint with rejected-action penalty 0.5. Its 10,240-step policy is
complete and achieved 14/16 raw and SSIM ≥ 0.90 ASR (87.5%), mean SSIM 0.9541,
mean PSNR 31.66 dB, and mean L1 distortion 0.00822. All final images met the hard
constraint; 59 attempted PGD actions were rejected across 40 episodes. See
`docs/expanded_quality_ppo_evaluation.md`.

### Larger-scale validation

- Only `FF_small` has been used.
- No full FaceForensics++ dataset was downloaded or evaluated.
- No transfer evaluation against another detector has been performed.
- Hyperparameters have not been tuned.
- Multiple training seeds have not been compared.

## 7. Recommended next steps

1. **Run stronger experiments**
   - Repeat training with multiple seeds.
   - Report mean and variation across runs.
   - Evaluate transfer to another frozen detector when available.

## 8. Next development task

Repeat the final comparison with additional seeds and, when an approved pretrained
checkpoint is available, a second frozen detector. Preserve the current seed-42
models as completed reference artifacts.

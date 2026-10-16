# Completed Expanded PPO Evaluation

## Result

The approximately resumed seed-42 expanded PPO policy achieved **16/16 attacks
(100% raw ASR)** on the originally eligible held-out FAKE images. Mean SSIM was
**0.9468**, mean PSNR was **31.42 dB**, and mean original-relative L1 distortion
was **0.0090**.

At the predeclared primary quality threshold, SSIM ≥ 0.90, expanded PPO retained
**12/16 successes (75%)**. Standalone ten-iteration PGD retained 0/16 at this
threshold. On this small held-out set, expanded PPO therefore outperformed
standalone PGD in attack success at comparable measured image quality. This is a
single approximately resumed run on 16 eligible images, not evidence of broad
generalisation.

## Audit and method

- Model: `outputs/expanded_actions_seed42/checkpoints/ppo_final.zip`
- Model SHA-256:
  `9607610a04a6fcb3d48727b23dcc7272179d7f0db3aa8af2da2b0ccdcd5e2f4c`
- Stored timesteps: 10,240; action space: `MultiDiscrete([10, 5])`
- Config: `config/expanded.yaml`, SHA-256
  `81b119330c9e8212f99f9cc7e0caa2fa28178b244e125672f620fb36a8f04ebd`
- Test manifest SHA-256:
  `1a8e0f9d825e66e15239a61d4fb08ea3e9894b7dc9e5fe968e4d6717d0236177`
- Xception SHA-256:
  `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47`
- Hardware: Apple arm64 CPU, four Torch threads; CUDA and MPS unavailable
- Software: Python 3.13.7, Torch 2.14.0, Stable-Baselines3 2.9.0,
  Gymnasium 1.3.0, NumPy 2.5.3
- Preprocessing: RGB, 224×224, ImageNet normalization; REAL=0, FAKE=1,
  threshold 0.5; frozen Xception
- Split: 40 images, 20 FAKE and 20 REAL; 16 initially correct FAKE images are
  eligible and four existing FAKE false negatives are excluded from ASR
- Train/test path overlap: zero; file-content hash overlap: zero

The training is labelled **approximately resumed seed-42 expanded PPO**. The model
and optimizer were restored at continuation points, but exact RNG, rollout-buffer,
and environment state were not.

## Evaluation command

```bash
.venv/bin/python -m evaluation.evaluate_attack \
  --test-manifest data/splits/ff_test.json \
  --model outputs/expanded_actions_seed42/checkpoints/ppo_final.zip \
  --deepfakebench-root ../DeepfakeBench_Handover \
  --weights ../DeepfakeBench_Handover/training/weights/xception_best.pth \
  --config config/expanded.yaml \
  --seed 42 --device cpu \
  --output-dir outputs/expanded_actions_seed42_evaluation_completed \
  --save-images
```

The evaluation took 480.09 seconds internally and 482.49 seconds wall time for all
40 images. It used 152 environment actions, 555 gradient evaluations, and an
estimated 747 detector forward calls. Training cost is separate; the final
continuation log was not preserved, while checkpoint timestamps show the sequence
continued from 1,536 at 17:46 to 10,240 at 23:33 on 4 October 2026.

## Expanded PPO metrics

| Metric | Value |
|---|---:|
| Successful / eligible | 16 / 16 |
| Raw ASR | 100.00% |
| FAKE-confidence reduction, mean / median / std | 0.6151 / 0.4968 / 0.2012 |
| SSIM, mean / median / std | 0.9468 / 0.9735 / 0.0360 |
| PSNR, mean / median / std | 31.42 / 30.05 / 2.86 dB |
| L1 distortion, mean / median / std | 0.0090 / 0.0061 / 0.0045 |
| Evaluation time | 480.09 s; 12.00 s/image |

### Quality-constrained ASR

The denominator remains the same 16 originally eligible images at every threshold.

| Minimum SSIM | Successes | Constrained ASR |
|---:|---:|---:|
| 0.90 (primary) | 12 / 16 | 75.00% |
| 0.85 | 16 / 16 | 100.00% |
| 0.80 | 16 / 16 | 100.00% |
| 0.70 | 16 / 16 | 100.00% |
| 0.50 | 16 / 16 | 100.00% |

## Fair comparison

| Method | Successes | Raw ASR | Mean SSIM | ASR at SSIM ≥ 0.90 | ASR at SSIM ≥ 0.80 |
|---|---:|---:|---:|---:|---:|
| No attack | 0 / 16 | 0.00% | 1.0000 | 0.00% | 0.00% |
| FGSM | 4 / 16 | 25.00% | 0.8728 | 0.00% | 0.00% |
| PGD, 10 iterations | 7 / 16 | 43.75% | 0.9363 | 0.00% | 43.75% |
| Frequency low-pass | 2 / 16 | 12.50% | 0.9163 | 12.50% | 12.50% |
| Basic PPO | 11 / 16 | 68.75% | 0.3751 | 0.00% | 0.00% |
| Expanded PPO | 16 / 16 | 100.00% | 0.9468 | 75.00% | 100.00% |

The 95% Wilson interval for expanded raw ASR is approximately 80.6%–100%; for
standalone PGD it is 23.1%–66.8%. One held-out success changes ASR by 6.25
percentage points.

The compute budgets differ. Standalone PGD applies ten gradient iterations once.
Embedded PPO PGD applies five iterations each time selected and may select it more
than once in an episode, with blur or other actions in between. Expanded evaluation
took 480 seconds versus 556 seconds for standalone PGD on this machine, but this
does not make their query or perturbation budgets identical.

Paired results show that both methods succeeded on seven images, expanded PPO
succeeded where standalone PGD failed on nine, and there were no cases where PGD
succeeded but expanded PPO failed.

## Learned deterministic behaviour

Across all 40 held-out episodes, the policy selected only:

- PGD: 111 actions
- Gaussian blur: 41 actions
- Strength: 0.50 for all 152 actions

On the 16 eligible images it selected 19 PGD and 28 blur actions. Episode lengths
were one step for one image, two for seven, three for four, and five for four.
The most common sequence was `gaussian_blur:0.50 | pgd:0.50` (seven images).
Four five-step episodes used four repeated blurs followed by PGD; these produced
the four eligible results below SSIM 0.90. The policy therefore collapsed to a
two-action medium-strength strategy rather than choosing diversely, but it avoided
the basic policy's repeated high-strength noise collapse.

## Artifacts

- Per-image metrics and trajectories:
  `outputs/expanded_actions_seed42_evaluation_completed/results.csv`
- Summary: `outputs/expanded_actions_seed42_evaluation_completed/summary.json`
- Policy analysis: `outputs/expanded_actions_seed42_evaluation_completed/policy_analysis.json`
- Before/after PNGs and montage:
  `outputs/expanded_actions_seed42_evaluation_completed/adversarial_samples/` and
  `representative_pairs.png`
- Paired comparison: `outputs/comparison_completed_expanded/paired_eligible_results.csv`
- Aggregate comparison: `outputs/comparison_completed_expanded/comparison.csv`
- Charts: `outputs/comparison_completed_expanded/asr_vs_ssim.svg` and
  `constrained_asr.svg`

## Separate quality-constrained variant

Because four expanded successes missed the primary SSIM threshold and showed
visible repeated blur, `config/expanded_quality.yaml` defines a separate hard
constraint. Candidate actions with original-relative SSIM below 0.90 are rejected,
the previous valid image is retained, and the agent receives a −0.5 reward. The
reference config and checkpoint are unchanged.

A 256-step training-only smoke run completed in 93.20 seconds with finite metrics.
This projects to roughly one hour for 10,240 steps on the same CPU. It is an
integration test, not a trained quality-constrained result, and it has not been
evaluated on the held-out set.

## Limitations

- Only 16 eligible held-out images and one approximately resumed seed-42 policy
- `FF_small`, not full FaceForensics++
- Source-video and identity overlap remain unresolved because flattened manifests
  lack those identifiers
- One Xception detector; no cross-detector transfer evaluation
- Different gradient/query budgets across attacks
- Deterministic expanded policy collapsed to blur and PGD rather than a diverse
  observation-dependent action repertoire

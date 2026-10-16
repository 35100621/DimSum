# Quality-Constrained Expanded PPO Evaluation

## Outcome

The completed hard-quality PPO policy achieved **14/16 attacks (87.5% raw ASR)**.
All final attacked images met the configured original-relative SSIM ≥ 0.90 bound,
so quality-constrained ASR was also **14/16 (87.5%)** at both SSIM ≥ 0.90 and
SSIM ≥ 0.85.

Compared with unconstrained expanded PPO, the hard constraint improved mean SSIM
from 0.9468 to 0.9541 and primary quality-constrained ASR from 75% to 87.5%, while
raw ASR decreased from 100% to 87.5%. The constraint therefore improved compliance
and the predeclared primary metric, but prevented two attacks that the reference
policy completed below or near its intended quality budget.

## Reproducibility

- Checkpoint: `outputs/expanded_quality_seed42/checkpoints/ppo_final.zip`
- Checkpoint SHA-256:
  `b9b09911e6afad9d2771fbb2dc5a95756dca3b6bc60521950e0b9bc3cc6a5cc3`
- Timesteps: 10,240; seed 42; CPU; `MultiDiscrete([10, 5])`
- Config: `config/expanded_quality.yaml`, SHA-256
  `ecfe21f1e47d7d52ea0153fd576e26a4da7d36b98d3e1a1e058e05ada326c8f9`
- Test manifest: `data/splits/ff_test.json`, SHA-256
  `1a8e0f9d825e66e15239a61d4fb08ea3e9894b7dc9e5fe968e4d6717d0236177`
- Xception SHA-256:
  `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47`
- Hard rule: reject any candidate whose SSIM against the episode's original image
  is below 0.90; retain the previous valid image; apply reward −0.5

```bash
.venv/bin/python -m evaluation.evaluate_attack \
  --test-manifest data/splits/ff_test.json \
  --model outputs/expanded_quality_seed42/checkpoints/ppo_final.zip \
  --deepfakebench-root ../DeepfakeBench_Handover \
  --weights ../DeepfakeBench_Handover/training/weights/xception_best.pth \
  --config config/expanded_quality.yaml \
  --seed 42 --device cpu \
  --output-dir outputs/expanded_quality_seed42_evaluation_completed \
  --save-images
```

## Metrics

| Metric | Result |
|---|---:|
| Successful / eligible | 14 / 16 |
| Raw ASR | 87.50% |
| ASR at SSIM ≥ 0.90 | 14 / 16 (87.50%) |
| ASR at SSIM ≥ 0.85 | 14 / 16 (87.50%) |
| Mean / median / std confidence reduction | 0.5050 / 0.4928 / 0.2315 |
| Mean / median / std SSIM | 0.9541 / 0.9735 / 0.0272 |
| Mean / median / std PSNR | 31.66 / 30.27 / 2.92 dB |
| Mean / median / std L1 distortion | 0.00822 / 0.00614 / 0.00346 |
| Evaluation runtime | 487.45 s; 12.19 s/image |

The minimum final SSIM across all 40 images was **0.9021**. There were **zero final
quality violations**, confirming that accepted cumulative modifications remained
within the hard constraint relative to each original image.

## Action and rejection behaviour

Across all 40 deterministic episodes:

- PGD attempts: 114
- Gaussian blur attempts: 31
- Strength: 0.50 for all 145 actions
- Rejected PGD attempts: 59
- Applied actions: 86
- Gradient evaluations: 570

On the 16 eligible images, the policy attempted 21 PGD and 25 blur actions; four
PGD attempts were rejected. Both failed images ended after the rejected PGD action
could not cross the detector boundary without falling below SSIM 0.90. Candidate
SSIM, rejection flags, confidence trajectories, and final original-relative metrics
are stored per image in `results.csv`.

## Comparison

| Method | Raw ASR | Mean SSIM | Mean PSNR | ASR at SSIM ≥ 0.90 | ASR at SSIM ≥ 0.85 |
|---|---:|---:|---:|---:|---:|
| FGSM | 25.00% | 0.8728 | 39.60 | 0.00% | 0.00% |
| PGD, 10 iterations | 43.75% | 0.9363 | 42.04 | 0.00% | 31.25% |
| Frequency low-pass | 12.50% | 0.9163 | 39.31 | 12.50% | 12.50% |
| Basic PPO | 68.75% | 0.3751 | 26.37 | 0.00% | 0.00% |
| Expanded PPO | 100.00% | 0.9468 | 31.42 | 75.00% | 100.00% |
| Quality PPO | 87.50% | 0.9541 | 31.66 | **87.50%** | 87.50% |

The quality PPO has a 95% Wilson interval of approximately 64.0%–96.5% for 14/16.
One success changes ASR by 6.25 percentage points, so differences should not be
generalized beyond this small test set.

## Did hard quality constraints improve the result?

**Yes for constraint compliance and the primary SSIM ≥ 0.90 metric; no for raw
attack success.** The hard constraint eliminated all final SSIM violations,
increased mean SSIM by 0.0073, reduced mean L1 distortion, and raised primary
quality-constrained ASR from 75% to 87.5%. It reduced raw ASR from 100% to 87.5%
because two attacks could not complete within the quality budget.

## Artifacts

- Summary: `outputs/expanded_quality_seed42_evaluation_completed/summary.json`
- Per-image results: `outputs/expanded_quality_seed42_evaluation_completed/results.csv`
- Before/after images: `outputs/expanded_quality_seed42_evaluation_completed/adversarial_samples/`
- Representative successes/failures: `representative_pairs.png`
- Policy analysis: `policy_analysis.json`
- Seven-method comparison: `outputs/comparison_completed_expanded/comparison.csv`
- Paired eligible results: `outputs/comparison_completed_expanded/paired_eligible_results.csv`
- Charts: `outputs/comparison_completed_expanded/asr_vs_ssim.svg` and
  `constrained_asr.svg`

## Limitations

- Only 16 eligible FAKE images; four original FAKE images were already false negatives
- One seed and one frozen Xception detector
- `FF_small`, not full FaceForensics++
- Source-video and identity overlap remain unverifiable from flattened manifests
- PPO sequences and standalone attacks have different gradient/query budgets
- The deterministic policy still collapsed to medium-strength blur and PGD

# Expanded PPO vs Standalone PGD

## Conclusion: yes on this held-out set, with major limitations

The completed approximately resumed expanded PPO policy outperformed standalone
PGD in both raw ASR and quality-constrained ASR on the same 16 eligible images.
This is a single small-set result and does not establish generalisation.

## Verified evidence available now

| Method | Successes / eligible | Raw ASR | Mean SSIM | ASR at SSIM ≥ 0.90 | ASR at SSIM ≥ 0.80 |
|---|---:|---:|---:|---:|---:|
| Standalone PGD | 7 / 16 | 43.75% | 0.9363 | 0 / 16 (0%) | 7 / 16 (43.75%) |
| Basic PPO | 11 / 16 | 68.75% | 0.3751 | 0 / 16 (0%) | 0 / 16 (0%) |
| Expanded PPO | 16 / 16 | 100.00% | 0.9468 | 12 / 16 (75.00%) | 16 / 16 (100.00%) |
| Quality PPO | 14 / 16 | 87.50% | 0.9541 | 14 / 16 (87.50%) | 14 / 16 (87.50%) |

The basic PPO result shows why raw ASR alone is insufficient. Expanded PPO avoids
that noise collapse and exceeds standalone PGD at the primary SSIM ≥ 0.90 threshold
(12/16 versus 0/16). Their attack budgets differ: standalone PGD uses ten gradient
iterations once, while PPO may sequence multiple five-iteration PGD actions.

## Reference-run status

- Configuration: `config/expanded.yaml`, unchanged
- Seed: 42
- Device: CPU; CUDA and MPS unavailable
- Completed: 10,240 cumulative steps
- Latest checkpoint:
  `outputs/expanded_actions_seed42/checkpoints/ppo_final.zip`
- Action and strength counts: `outputs/expanded_actions_seed42/training_actions.json`
- Status and continuation command:
  `outputs/expanded_actions_seed42/reference_run_status.json`

Continuation is approximate rather than bit-identical: Stable-Baselines3 restores
the policy and optimizer, but the current workflow does not serialize exact RNG,
rollout-buffer, or environment state.

## Quality-constraint conclusion

The completed hard-quality policy improved primary SSIM ≥ 0.90 ASR from 75% to
87.5%, increased mean SSIM, and eliminated final quality violations. Raw ASR fell
from 100% to 87.5%. The constraint therefore improved the quality-controlled result
while enforcing the intended tradeoff.

## Limitations

- `FF_small` contains only 200 images and the ASR denominator contains 16 samples.
- Results use one seed and one frozen Xception detector.
- The flattened manifests cannot rule out source-video or identity overlap.
- PPO sequences and ten-iteration PGD have different compute and perturbation
  budgets; both must be reported before any superiority claim.
- No cross-detector transfer or Kitt0uz paired-image evaluation has been run.

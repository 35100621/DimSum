# DimSum — Milestone 2 (Show of Force): Weekly Update 2

## Main Results So Far / Changes to the Reference Implementation

### 1. Completed detector-to-PPO integration

Since our first update, we have connected the pretrained DeepfakeBench Xception
detector to our PPO environment. Xception stays in evaluation mode with its weights
frozen, and the environment receives its REAL/FAKE probabilities after every attack
action.

We reproduced the detector baseline on 200 `FF_small` images: **74.5% accuracy and
0.854 AUC**. The main change from the original DeepfakeBench workflow is that we
are no longer using Xception only for classification. Its confidence now provides
feedback to an agent that learns a sequence of image modifications.

### 2. Completed the basic PPO experiment

We created a deterministic train/test split with 160 training-manifest records and
40 held-out records. The training manifest contains 80 FAKE and 80 REAL images;
the current FAKE-to-REAL PPO objective uses the 80 FAKE images. The held-out set
contains 20 FAKE and 20 REAL images.

Our basic PPO run requested 10,000 steps and completed **10,240 actual steps**
because Stable-Baselines3 finished the enclosing rollout. On the held-out set,
16 FAKE images were eligible: they were ground-truth FAKE and initially predicted
FAKE by Xception. PPO flipped **11/16**, giving **68.75% ASR**.

However, mean SSIM was only **0.3751**, so raw ASR does not represent a good
quality-preserving result. The policy collapsed to repeated strength-0.75 Gaussian
noise and resize/restore actions. In several episodes it stacked noise three to
five times, causing severe cumulative degradation.

### 3. Implemented and evaluated additional attacks

We implemented bounded FGSM, iterative PGD, and a reproducible frequency-domain
low-pass operator. They work as standalone baselines and are also available in the
expanded PPO action space. Our attack tests cover perturbation bounds, output
validity, deterministic behaviour where applicable, targeted direction, gradient
flow to the input, and frozen detector weights.

| Method | ASR | Mean SSIM |
|---|---:|---:|
| FGSM | 25.00% | 0.8728 |
| PGD | 43.75% | 0.9363 |
| Frequency low-pass | 12.50% | 0.9163 |
| Basic PPO | 68.75% | 0.3751 |

These are preliminary single-seed results based on only **16 eligible FAKE
images**. At a common **SSIM ≥ 0.80** requirement, PGD retains **43.75%
quality-constrained ASR**, while basic PPO falls to **0%**. This shows why we need
to compare attack success and image quality together rather than rank methods by
unconstrained ASR.

### 4. Expanded PPO status

The expanded environment now offers FGSM, PGD, and frequency-domain actions along
with the original transformations. A 256-step rollout completed successfully, but
it ran at approximately two steps per second on our CPU. This projects to about
85 minutes for the rollout-aligned 10,240-step experiment. We stopped after the
benchmark boundary, so **expanded PPO training and held-out results are not yet
complete**.

## Current Analysis

Basic PPO can reduce detector confidence and flip predictions, but its current
policy often sacrifices image quality. The next question is whether expanded,
quality-constrained PPO can outperform standalone PGD while preserving comparable
image quality. This remains an open question.

## Next Steps

1. Complete expanded PPO training and evaluate it on the same held-out set.
2. Add stronger cumulative image-quality constraints or reward penalties and
   evaluate that revised policy as a separate experiment.
3. Compare PGD and PPO using matched SSIM thresholds and clearly reported compute
   budgets.
4. Inspect learned attack sequences, run additional seeds if feasible, and document
   the unresolved dataset source-overlap risk.

## Challenge Target: Kitt0uz

Our proposed challenge remains a comparison between ordinary SimSwap outputs and
modified Weak Feature Matching (wFM) SimSwap outputs using detector predictions,
confidence, and attack success rates. We have not found challenge images or results
in our repository yet, so this comparison is **planned and awaiting images**.

---

## Evidence Checklist

- **Xception baseline — 74.5% accuracy, 0.854 AUC, 200 images:**
  `outputs/baseline_summary.json`
- **160/40 split and class counts:** `data/splits/ff_ppo_train.json` and
  `data/splits/ff_test.json`
- **10,000 requested / 10,240 completed PPO steps and 80 training images:**
  `outputs/basic_actions_seed42/training_config.yaml`
- **Basic PPO — 11/16, 68.75% ASR, mean SSIM 0.3751, action frequencies and
  strength distribution:** `outputs/basic_actions_seed42_evaluation/summary.json`
- **Basic PPO training progress:** `outputs/basic_actions_seed42_training.log`
- **FGSM — 4/16, 25.00% ASR, mean SSIM 0.8728:**
  `outputs/baselines/fgsm/summary.json`
- **PGD — 7/16, 43.75% ASR, mean SSIM 0.9363:**
  `outputs/baselines/pgd/summary.json`
- **Frequency low-pass — 2/16, 12.50% ASR, mean SSIM 0.9163:**
  `outputs/baselines/frequency/summary.json`
- **SSIM-threshold comparison:** `outputs/comparison_seed42.json`
- **Expanded PPO — 256 steps, 102 seconds, projected 85.33 minutes, no saved
  checkpoint:** `outputs/expanded_actions_seed42_benchmark.json` and
  `outputs/expanded_actions_seed42_training.log`
- **Latest verification recorded during implementation:** 37 tests passed with two
  optional integration tests skipped; with the real-checkpoint integration enabled,
  38 passed and one optional training integration test was skipped.

## Outdated or Unverifiable Claims

- The earlier statement that full PPO training had not run is outdated for the
  **basic** action space: that experiment is now complete. It remains true for the
  expanded action space.
- The 256-step expanded rollout is a speed/integration benchmark, not a trained
  expanded PPO result. There is no expanded checkpoint to evaluate.
- No repository evidence confirms receipt or evaluation of Kitt0uz SimSwap/wFM
  images.
- The manifests prove no path or content-hash overlap, but they do not contain
  source-video or identity metadata, so those overlap risks remain unverifiable.
- Results use one small dataset, one seed, one detector, and only 16 eligible held-out
  FAKE images. They do not establish multi-seed or cross-detector generalisation.

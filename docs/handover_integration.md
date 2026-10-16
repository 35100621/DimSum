# DeepfakeBench handover integration

The supplied handover was extracted to the sibling directory
`/Users/hgtan/Desktop/DeepfakeBench_Handover`. The standalone `FF_small.zip` was
not extracted because the handover already contains the same 100 REAL and 100
FAKE images.

Only the Xception checkpoint and Xception ImageNet backbone weights were retained;
unrelated detector checkpoints were excluded. The archive omitted DeepfakeBench's
`training/networks`, `training/loss`, and `training/metrics` source directories.
Those directories were restored from official DeepfakeBench commit
`f188b1c105465e2e5377eb536a95022ae0e4522d`.

The supplied baseline procedure uses:

- labels REAL = 0 and FAKE = 1;
- Pillow RGB resize to 224 × 224;
- ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
  `[0.229, 0.224, 0.225]`;
- a FAKE threshold of 0.5;
- evaluation mode and no gradient tracking.

Reproduced on all 200 images:

| Metric | Reported | Reproduced |
|---|---:|---:|
| Accuracy | 0.745 | 0.745 |
| AUC | 0.854 | 0.854 |
| Precision | 0.7130434782608696 | 0.7130434782608696 |
| Recall | 0.82 | 0.82 |

The reproduced confusion matrix is `[[67, 33], [18, 82]]`. No Xception training
was performed. The DimSum wrapper loads the model once, freezes detector weights,
and provides both ordinary no-gradient prediction and an opt-in differentiable
tensor path for future FGSM/PGD input gradients.

The supplied requirements pin NumPy 1.23.5, which does not support this project's
Python 3.13 runtime. DimSum retains its compatible NumPy version and adds only the
missing Xception runtime dependencies (`scikit-learn` and `tensorboard`).

## PPO data split and integration smoke run

`scripts/create_ff_split.py` creates a seed-42, 80/20 split independently within
each label/source group. The resulting manifests are:

- `data/splits/ff_ppo_train.json`: 160 records (80 FAKE, 80 REAL); PPO selects
  only its 80 labelled FAKE records.
- `data/splits/ff_test.json`: 40 records (20 FAKE, 20 REAL).

The partitions have no path or file-content overlap. Every fake manipulation has
16 training and 4 test images. The two real sources each have 40 training and 10
test images.

A 256-step real-Xception integration rollout completed successfully and produced
`outputs/integration_smoke/checkpoints/ppo_final.zip`. Deterministic evaluation on
the 40 held-out records also completed. Sixteen of the twenty labelled FAKE images
were initially detected as FAKE and therefore eligible; three were flipped to REAL
(smoke ASR 0.1875). These values validate the complete software path and are not
final training results.

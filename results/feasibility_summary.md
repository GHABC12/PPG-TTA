# QM-TTA feasibility pilot — 2026-10-10

## Scope

- Used only the pre-existing VitalDB data; no additional dataset was downloaded.
- Audited 24 deterministic, evenly spaced case files out of 6,388. Both PPG and monitor SpO₂ tracks were readable in all 24; 23 cases yielded at least one usable window.
- Labels require at least five valid 1 Hz values in `[70,100]`; each label is the median for its non-overlapping window.
- All windows from a case stay in one GroupKFold fold. Each fold uses the next fold for validation and the remaining three for training.
- All five held-out folds were evaluated, covering the 23 cases with usable windows. This is one random seed.

## Data and preprocessing

| Backbone | Protocol | Usable windows | SpO₂ label distribution |
| --- | --- | ---: | --- |
| PaPaGei-S | 10 s, 125 Hz; Chebyshev 0.5–12 Hz; per-window z-score | 25,638 | median 100; min 74; mean 99.425 |
| SIGMA-PPG-Pro | 20 s, 50 Hz; Butterworth 0.5–8 Hz; `[-1,1]` min-max | 12,913 | median 100; min 74; mean 99.390 |

The highly right-skewed label distribution makes a median constant baseline important. These values are bedside monitor SpO₂ labels, not SaO₂ measurements.

## Fold 0 outcomes

Errors are in SpO₂ percentage points.

| Backbone | Method | MAE | RMSE | Bias | Prediction SD |
| --- | --- | ---: | ---: | ---: | ---: |
| PaPaGei-S | Source-only LP | 0.918 | 1.718 | -0.084 | 0.373 |
| PaPaGei-S | Train-median constant | 0.473 | 1.712 | 0.473 | 0.000 |
| PaPaGei-S | Episodic QM-TTA | 0.917 | 1.717 | -0.083 | 0.373 |
| SIGMA-PPG-Pro | Source-only LP | 1.011 | 1.719 | -0.183 | 0.922 |
| SIGMA-PPG-Pro | Train-median constant | 0.502 | 1.849 | 0.502 | 0.000 |
| SIGMA-PPG-Pro | Episodic QM-TTA | 1.016 | 1.725 | -0.191 | 0.930 |

`ΔMAE = MAE(source) - MAE(TTA)` is +0.00074 for PaPaGei-S and -0.00463 for SIGMA. Both learned source models are worse than the constant median on MAE. The current pilot does not establish a useful SpO₂ predictor or a QM-TTA benefit.

## Five-fold out-of-fold outcomes

Each window is predicted once by a model trained without its case. OOF aggregates use all windows; case-level means treat each of the 23 cases equally.

| Backbone | Method | OOF MAE | OOF RMSE | Mean case MAE | SD of case MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| PaPaGei-S | Source-only LP | 0.835 | 1.671 | 0.929 | 0.503 |
| PaPaGei-S | Train-fold median constant | 0.575 | 1.717 | 0.683 | 0.631 |
| PaPaGei-S | Episodic QM-TTA | 0.835 | 1.671 | 0.929 | 0.503 |
| SIGMA-PPG-Pro | Source-only LP | 0.916 | 1.792 | 1.026 | 0.543 |
| SIGMA-PPG-Pro | Train-fold median constant | 0.610 | 1.895 | 0.765 | 0.692 |
| SIGMA-PPG-Pro | Episodic QM-TTA | 0.927 | 1.797 | 1.039 | 0.542 |

OOF `ΔMAE = MAE(source) - MAE(TTA)` is -0.00005 for PaPaGei-S and -0.01038 for SIGMA. Neither source model nor TTA beats the train-fold median baseline on MAE.

## Integrity and limitations

- SIGMA checkpoint: 363,228,975 bytes; SHA-256 `b185072e04a2c7e0ce99dbbe96a241d0150a99f7b1c49414749ba3b1b404b68b`.
- SIGMA upstream commit: `b1cc8d90da3878958b890c6e352cd3ae2a644005`; local compatibility changes are documented in `third_party/SigmaPPG_PATCH.md`.
- Adapted SIGMA smoke input `(2,1,1000)` produced finite `(2,360)` features. All 253 backbone tensors loaded; only the downstream task head was random and is trained from source labels.
- TTA is episodic with reset per case; the update function has no SpO₂ target argument. All five folds but only one seed were run.
- Twenty-three cases and the near-100 label concentration are inadequate for broad generalization claims. PaPaGei/SIGMA pretraining exposure may include VitalDB, so this is not a clean unseen-domain test.

## Follow-up stage: source-tail fine-tuning and TTA tuning

Completed the next implementation stage using only VitalDB and the same five case-grouped folds. For each backbone and outer fold, source-tail fine-tuning trained only the last backbone block and prediction/auxiliary heads; the outer test fold was excluded from training. TTA then compared six update candidates (`lr` 1e-6 or 1e-5 crossed with morphology weight 0, 0.1, or 0.5) and a no-update control on that fold's validation set. Quality threshold was 0.5 for update candidates, regularization weight 1e-4, and batch size 4. The selected candidate was evaluated once on the outer test fold. TTA scored each batch before updating from that batch, and reset its state at case boundaries.

OOF metrics, in SpO₂ percentage points:

| Backbone | Method | Window MAE | Mean case MAE |
| --- | --- | ---: | ---: |
| PaPaGei-S | Linear probe | 0.8454 | 0.9339 |
| PaPaGei-S | Linear probe + validation-selected TTA | 0.8455 | 0.9340 |
| PaPaGei-S | Source-tail fine-tuning | 0.7975 | 0.9011 |
| PaPaGei-S | Fine-tuning + validation-selected TTA | 0.7975 | 0.9010 |
| PaPaGei-S | Train-fold median constant | 0.5749 | 0.6827 |
| SIGMA-PPG-Pro | Linear probe | 0.9244 | 1.0331 |
| SIGMA-PPG-Pro | Linear probe + validation-selected TTA | 0.9250 | 1.0337 |
| SIGMA-PPG-Pro | Source-tail fine-tuning | 0.8872 | 1.0160 |
| SIGMA-PPG-Pro | Fine-tuning + validation-selected TTA | 0.8893 | 1.0181 |
| SIGMA-PPG-Pro | Train-fold median constant | 0.6104 | 0.7651 |

Fine-tuning improves on the corresponding linear probe, but all learned methods remain worse than the train-fold median constant in both reported MAE summaries. TTA's aggregate changes are negligible and inconsistent in direction: for PaPaGei, mean case MAE changes by -0.00008 after fine-tuning and +0.00004 for the linear probe; for SIGMA it increases by 0.00210 and 0.00061, respectively. This small validation-selected grid does not establish a robust TTA benefit or globally optimal hyperparameters.

Per-fold/window and case-level outputs are in `results/tuned_all_metrics.csv`, `results/tuned_all_case_metrics.csv`, and `results/tuned_all_summary.json`; selected configurations are retained under each backbone's `tta_tuning/{lp,finetune}/` directory. Each validation fold was reused for source early stopping and TTA selection, so validation scores are optimistic; outer test labels were not used for selection. The experiment still has only 23 cases and one random seed, with severe label concentration near 100 and possible pretraining exposure to VitalDB. The next useful step is more independent cases/seeds and low-SpO₂ coverage, followed by case-level uncertainty estimates, rather than a larger search on this pilot.

### Fixed-parameter component ablation

To separate quality and morphology effects, added a true uniform-weight consistency control. `consistency_only` uses every sample uniformly; `quality_gated_consistency` adds the fixed quality gate/weights; `morphology_without_quality` adds morphology without quality control; `full_qm` combines quality and morphology. All used `lr=1e-5`, `lambda_reg=1e-4`, `lambda_morph=0.1` when enabled, batch size 4, and the same LP source folds. Each variant was scored once on each outer test fold; test scores did not drive parameter choice.

| Backbone | Method | OOF window MAE | Mean case MAE |
| --- | --- | ---: | ---: |
| PaPaGei-S | Source-only | 0.8454 | 0.9339 |
| PaPaGei-S | Consistency only | 0.8461 | 0.9349 |
| PaPaGei-S | Quality-gated consistency | 0.8456 | 0.9341 |
| PaPaGei-S | Morphology, no quality | 0.8456 | 0.9339 |
| PaPaGei-S | Full QM-TTA | 0.8454 | 0.9337 |
| PaPaGei-S | Train-fold median constant | 0.5749 | 0.6827 |
| SIGMA-PPG-Pro | Source-only | 0.9244 | 1.0331 |
| SIGMA-PPG-Pro | Consistency only | 0.9432 | 1.0535 |
| SIGMA-PPG-Pro | Quality-gated consistency | 0.9374 | 1.0467 |
| SIGMA-PPG-Pro | Morphology, no quality | 0.9340 | 1.0425 |
| SIGMA-PPG-Pro | Full QM-TTA | 0.9305 | 1.0429 |
| SIGMA-PPG-Pro | Train-fold median constant | 0.6104 | 0.7651 |

Case-bootstrap estimate of average-case-MAE improvement over source-only (positive is better), 95% CI, 23 cases:

| Backbone | Variant | Improvement | 95% case-bootstrap CI |
| --- | --- | ---: | ---: |
| PaPaGei-S | Consistency only | -0.0010 | [-0.0016, -0.0003] |
| PaPaGei-S | Quality-gated consistency | -0.0002 | [-0.0008, 0.0004] |
| PaPaGei-S | Morphology, no quality | -0.0000 | [-0.0005, 0.0005] |
| PaPaGei-S | Full QM-TTA | +0.0002 | [-0.0003, 0.0007] |
| SIGMA-PPG-Pro | Consistency only | -0.0204 | [-0.0285, -0.0132] |
| SIGMA-PPG-Pro | Quality-gated consistency | -0.0136 | [-0.0214, -0.0065] |
| SIGMA-PPG-Pro | Morphology, no quality | -0.0094 | [-0.0134, -0.0055] |
| SIGMA-PPG-Pro | Full QM-TTA | -0.0098 | [-0.0158, -0.0048] |

PaPaGei full QM-TTA's tiny positive point estimate is not distinguishable from zero in this case bootstrap, while consistency-only degrades its case MAE. All four SIGMA variants degrade mean case MAE relative to source-only. Every learned method remains substantially worse than its train-fold median constant. No nonfinite-loss updates occurred. Taken together with the parameter search, these results do not establish an effective configuration; broader hyperparameter search on the current sample is unlikely to answer the generalization question.

Summary files: `results/tta_ablation_metrics.csv`, `results/tta_ablation_case_metrics.csv`, and `results/tta_ablation_summary.json`. The bootstrap is exploratory, uses only 23 cases, and does not correct for comparing multiple variants.

## Next implementation steps

1. Expand the deterministic VitalDB case sample and inspect low-SpO₂ coverage before spending time on more training; keep this phase VitalDB-only.
2. If coverage supports it, repeat the case-grouped study with additional random seeds and case-level paired bootstrap intervals.
3. Treat the current component ablation as exploratory; only revisit its settings if a larger VitalDB sample changes the baseline or label coverage materially.

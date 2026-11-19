# Hyperparameter tuning: findings

Tuning script: [`src/tune_hyperparameters.py`](../../src/tune_hyperparameters.py) · Outputs: this folder · Run date: 2026-11-19

## Summary

- **The models were tuned without leakage.** Hyperparameters were chosen only on the 2016–2017 warm-up data. The 2018–2025 evaluation stream was used once, after selection, to score the chosen configurations under the unchanged notebook protocol.
- **Mean macro-F1 over all 28 configurations rose from 0.8994 to 0.9203** (+0.0209). Of the 28 configurations, 18 improved, 6 got slightly worse (worst: −0.0057) and 4 stayed the same (Gaussian Naive Bayes, which has no hyperparameters).
- **Tuned Online Bagging on representation B is the new best model in both cities:**
  - Covilhã: **0.9818** (was 0.9763)
  - Portimão: **0.9751** (was 0.9647)
- **UnimSafe gained the most (+0.112 mean).** On Covilhã A it went from a model that almost never predicted an event (recall 0.10, macro-F1 0.55) to a usable one (recall 0.62, macro-F1 0.79).
- **Adaptive Random Forest (+0.017) and Online Logistic Regression (+0.010) improved in all four city × representation runs.**
- **Hoeffding Tree and UnimPA barely changed on 2018–2025**, even though the Hoeffding Tree looked much better on the 2017 validation year. That is a sign of selection noise (see [Caveats](#caveats)).

## Protocol

| Step | Data | Purpose |
|---|---|---|
| Inner fit | 2016 work hours (learn only) | Initial training of each candidate |
| Inner validation | 2017 work hours (predict → score → learn) | **Selection score** = 2017 macro-F1, averaged over both cities |
| Official replay | Learn 2016–2017, then predict → evaluate → learn on 2018–2025 | Final reported scores; never used for selection |

- **Preprocessing for the inner loop:** median imputation and standard scaling were fitted on 2016 only, so the 2017 validation features are also out-of-sample. The official replay uses the notebook's preprocessing, fitted on the pooled 2016–2017 warm-up.
- **One winner per (model, representation):** each candidate was scored on both cities and both representations. The selected configuration for each model × representation is the one with the highest mean validation macro-F1 across the two cities, and the same parameters are used for both cities.
- **The script reproduces the notebook exactly:** run with the original notebook parameters, the script's official replay gives the **same confusion matrices as `results/metrics.csv` for all 28 configurations**. So the baseline and tuned numbers below are directly comparable. Input data is `data/processed/city_*_model_stream.csv`, read with `float_precision='round_trip'` so the values are bit-identical to the notebook's.
- **Seeds:** `random_state` / `seed` = 42 for all stochastic models. Candidate sampling is seeded per model.

## Search space and budget

Random search was used. Each model's candidate list includes the notebook baseline as candidate 0.

| Model | Candidates | Parameters searched |
|---|---|---|
| Gaussian Naive Bayes | 1 | none (River's GaussianNB has no hyperparameters) |
| Online Logistic Regression | 120 | optimizer {SGD, Adam, AdaGrad}, lr ∈ [1e-3, 0.5] (log), l2 ∈ {0, 1e-5 … 1e-2}, intercept_lr ∈ {0.001, 0.01, 0.1} |
| Hoeffding Tree | 180 | grace_period {10 … 400}, delta ∈ [1e-9, 1e-1] (log), tau {0.01 … 0.2}, leaf_prediction {mc, nb, nba}, split_criterion {info_gain, gini, hellinger} |
| Online Bagging | 120 | n_models {5, 10, 15, 20, 30} + Hoeffding Tree space for the base learner |
| Adaptive Random Forest | 120 | n_models {5 … 30}, max_features {sqrt, log2, 0.3, 0.5, 0.8}, lambda_value {1, 3, 6, 10} + tree space |
| UnimPA | 180 | max_rules {2 … 20}, alpha_add, tau_merge, lambda_sim, e, act_min_update, dist_safe_factor, C ∈ [0.01, 10] (log), buffer_size_separability |
| UnimSafe | 180 | max_rules {2 … 20}, alpha_add, tau_merge, e, act_min_update, dist_safe_factor, forget_counts {0.9 … 1.0}, min_rules_for_merge, min_support_for_merge, buffer_size_separability |

In total, 901 candidates × 2 cities × 2 representations = **3,604 validation runs**, plus 52 official replays. The search took about 5 minutes on 22 worker processes: 261 s for validation and 43 s for the official replays.

Ensemble size and `max_rules` were allowed to differ from the notebook's 5, as agreed for this run.

## Results on the official 2018–2025 stream

### Best model per city × representation

| City | Rep | Best model (tuned) | Macro-F1 baseline | Macro-F1 tuned |
|---|---|---|---|---|
| Portimão | A | Online Logistic Regression | 0.9483 | **0.9595** |
| Portimão | B | Online Bagging | 0.9647 | **0.9751** |
| Covilhã | A | Online Logistic Regression | 0.9569 | **0.9675** |
| Covilhã | B | Online Bagging | 0.9763 | **0.9818** |

With baseline parameters, the Hoeffding Tree was the best model on Portimão A and Covilhã A. After tuning, Online Logistic Regression takes that place.

### Mean per model (4 city × representation runs)

| Model | Macro-F1 baseline | Macro-F1 tuned | Δ | Improved / worse |
|---|---|---|---|---|
| Gaussian Naive Bayes | 0.8661 | 0.8661 | +0.0000 | 0 / 0 |
| Online Logistic Regression | 0.9549 | 0.9652 | +0.0103 | 4 / 0 |
| Hoeffding Tree | 0.9615 | 0.9627 | +0.0012 | 1 / 3 |
| Online Bagging | 0.9635 | 0.9686 | +0.0051 | 4 / 0 |
| Adaptive Random Forest | 0.9452 | 0.9624 | +0.0172 | 4 / 0 |
| UnimPA | 0.9130 | 0.9133 | +0.0004 | 1 / 3 |
| UnimSafe | 0.6914 | 0.8035 | +0.1121 | 4 / 0 |

### All 28 configurations

| City | Rep | Model | Macro-F1 baseline | Macro-F1 tuned | Δ | Recall baseline | Recall tuned | Precision baseline | Precision tuned |
|---|---|---|---|---|---|---|---|---|---|
| Portimão | A | Gaussian Naive Bayes | 0.8664 | 0.8664 | +0.0000 | 0.9794 | 0.9794 | 0.6878 | 0.6878 |
| Portimão | A | Online Logistic Regression | 0.9483 | 0.9595 | +0.0113 | 0.9151 | 0.9306 | 0.9258 | 0.9450 |
| Portimão | A | Hoeffding Tree | 0.9500 | 0.9499 | -0.0001 | 0.9032 | 0.9009 | 0.9432 | 0.9451 |
| Portimão | A | Online Bagging | 0.9500 | 0.9530 | +0.0030 | 0.8950 | 0.9053 | 0.9520 | 0.9507 |
| Portimão | A | Adaptive Random Forest | 0.9437 | 0.9521 | +0.0084 | 0.9009 | 0.9098 | 0.9259 | 0.9430 |
| Portimão | A | UnimPA | 0.9235 | 0.9177 | -0.0057 | 0.8649 | 0.8579 | 0.8995 | 0.8888 |
| Portimão | A | UnimSafe | 0.7582 | 0.8156 | +0.0575 | 0.6187 | 0.7732 | 0.6360 | 0.6788 |
| Portimão | B | Gaussian Naive Bayes | 0.8593 | 0.8593 | +0.0000 | 0.9590 | 0.9590 | 0.6822 | 0.6822 |
| Portimão | B | Online Logistic Regression | 0.9561 | 0.9643 | +0.0083 | 0.9331 | 0.9409 | 0.9321 | 0.9495 |
| Portimão | B | Hoeffding Tree | 0.9619 | 0.9693 | +0.0074 | 0.9598 | 0.9607 | 0.9246 | 0.9455 |
| Portimão | B | Online Bagging | 0.9647 | 0.9751 | +0.0104 | 0.9366 | 0.9541 | 0.9551 | 0.9694 |
| Portimão | B | Adaptive Random Forest | 0.9425 | 0.9655 | +0.0230 | 0.8954 | 0.9364 | 0.9279 | 0.9577 |
| Portimão | B | UnimPA | 0.8983 | 0.8961 | -0.0022 | 0.8439 | 0.8381 | 0.8439 | 0.8426 |
| Portimão | B | UnimSafe | 0.7618 | 0.8169 | +0.0551 | 0.6379 | 0.7813 | 0.6319 | 0.6771 |
| Covilhã | A | Gaussian Naive Bayes | 0.8828 | 0.8828 | +0.0000 | 0.9782 | 0.9782 | 0.6804 | 0.6804 |
| Covilhã | A | Online Logistic Regression | 0.9569 | 0.9675 | +0.0106 | 0.9290 | 0.9360 | 0.9211 | 0.9508 |
| Covilhã | A | Hoeffding Tree | 0.9635 | 0.9629 | -0.0006 | 0.9382 | 0.9267 | 0.9348 | 0.9442 |
| Covilhã | A | Online Bagging | 0.9630 | 0.9644 | +0.0014 | 0.9227 | 0.9242 | 0.9486 | 0.9520 |
| Covilhã | A | Adaptive Random Forest | 0.9486 | 0.9618 | +0.0132 | 0.8886 | 0.9238 | 0.9328 | 0.9433 |
| Covilhã | A | UnimPA | 0.9320 | 0.9429 | +0.0108 | 0.8528 | 0.8794 | 0.9114 | 0.9220 |
| Covilhã | A | UnimSafe | 0.5538 | 0.7890 | +0.2352 | 0.0999 | 0.6182 | 0.6733 | 0.6454 |
| Covilhã | B | Gaussian Naive Bayes | 0.8558 | 0.8558 | +0.0000 | 0.9715 | 0.9715 | 0.6233 | 0.6233 |
| Covilhã | B | Online Logistic Regression | 0.9583 | 0.9696 | +0.0112 | 0.9356 | 0.9423 | 0.9196 | 0.9518 |
| Covilhã | B | Hoeffding Tree | 0.9705 | 0.9687 | -0.0018 | 0.9489 | 0.9112 | 0.9482 | 0.9817 |
| Covilhã | B | Online Bagging | 0.9763 | 0.9818 | +0.0055 | 0.9626 | 0.9645 | 0.9549 | 0.9720 |
| Covilhã | B | Adaptive Random Forest | 0.9460 | 0.9702 | +0.0242 | 0.8827 | 0.9267 | 0.9299 | 0.9702 |
| Covilhã | B | UnimPA | 0.8980 | 0.8965 | -0.0015 | 0.8224 | 0.8158 | 0.8224 | 0.8237 |
| Covilhã | B | UnimSafe | 0.6919 | 0.7925 | +0.1006 | 0.3640 | 0.6385 | 0.5840 | 0.6388 |

Recall and precision are for the heat-event class.

## Selected hyperparameters

Full-precision values are in [`best_parameters.json`](best_parameters.json). Values shown here are rounded.

| Model | Rep | Selected parameters | Notebook baseline |
|---|---|---|---|
| Online Logistic Regression | A | SGD lr=0.319, l2=1e-5, intercept_lr=0.1 | SGD lr=0.05, l2=0, intercept_lr=0.01 |
| Online Logistic Regression | B | SGD lr=0.105, l2=1e-4, intercept_lr=0.1 | 〃 |
| Hoeffding Tree | A | grace=100, delta=1.2e-8, tau=0.2, leaf=nba, split=gini | grace=50, delta=1e-3, tau=0.1, leaf=mc, split=info_gain |
| Hoeffding Tree | B | grace=10, delta=1.8e-4, tau=0.05, leaf=nba, split=info_gain | 〃 |
| Online Bagging | A | n_models=15; tree: grace=50, delta=6.8e-7, tau=0.2, leaf=nba, split=gini | n_models=5 + baseline tree |
| Online Bagging | B | n_models=20; tree: grace=10, delta=7.1e-5, tau=0.1, leaf=nba, split=info_gain | 〃 |
| Adaptive Random Forest | A | n_models=30, max_features=log2, λ=1; tree: grace=100, delta=8.4e-3, tau=0.01, leaf=nb, split=info_gain | n_models=5, sqrt, λ=6 + baseline tree |
| Adaptive Random Forest | B | n_models=30, max_features=log2, λ=3; tree: grace=10, delta=2.3e-2, tau=0.1, leaf=nb, split=gini | 〃 |
| UnimPA | A | max_rules=5, alpha_add=0.27, tau_merge=0.74, lambda_sim=0.61, e=0.44, act_min_update=0.27, dist_safe_factor=5.14, C=7.63, buffer=200 | max_rules=5, source defaults (0.3, 0.7, 0.5, 0.5, 0.3, 3.0, C=1.0, 200) |
| UnimPA | B | max_rules=3, alpha_add=0.39, tau_merge=0.91, lambda_sim=0.27, e=0.78, act_min_update=0.10, dist_safe_factor=4.91, C=0.42, buffer=500 | 〃 |
| UnimSafe | A | max_rules=12, alpha_add=0.15, tau_merge=0.78, e=0.73, act_min_update=0.27, dist_safe_factor=4.76, forget_counts=0.9, min_rules_for_merge=3, min_support_for_merge=5, buffer=500 | max_rules=5, source defaults (0.3, 0.7, 0.5, 0.3, 3.0, 0.99, 5, 10, 200) |
| UnimSafe | B | max_rules=12, alpha_add=0.35, tau_merge=0.94, e=0.58, act_min_update=0.24, dist_safe_factor=4.26, forget_counts=0.9, min_rules_for_merge=3, min_support_for_merge=5, buffer=500 | 〃 |

## Observations

1. **Leaf prediction matters most for the tree models.** Every selected tree-based configuration replaced majority-class leaves (`mc`) with Naive Bayes leaves (`nb` or `nba`). With only 4 or 11 continuous features, a Gaussian model inside each leaf gives smoother decision boundaries than a hard majority vote. Larger ensembles (15–30 members instead of 5) add a further gain.
2. **Adaptive Random Forest was under-configured at 5 trees.** With 5 trees and `max_features='sqrt'`, each tree sees very few features. Moving to 30 trees with Naive Bayes leaves lifts it from the weakest tree model (0.945) to on par with the others (0.962).
3. **Logistic regression wanted a higher learning rate.** A higher SGD learning rate (0.1–0.3, versus 0.05) and a faster intercept rate (0.1, versus 0.01) let the linear model track seasonal shifts faster. The gain is consistent: +0.008 to +0.011 in every run.
4. **UnimSafe was limited by stale class counts.** The decisive change is `forget_counts=0.9`, which forgets old class evidence much faster than 0.99. Together with more rules (12) and easier merging (`min_rules_for_merge=3`, `min_support_for_merge=5`), the rule consequents adapt to the minority heat class instead of collapsing to "no event". UnimSafe is still clearly the weakest learner apart from Gaussian Naive Bayes.
5. **UnimPA is close to its ceiling with this model structure.** Validation gains were tiny (+0.006 on A, +0.0003 on B), and the 2018–2025 changes are within noise (−0.006 to +0.011).
6. **Representation B still beats A for the strong learners.** The extra rolling and calendar features help the tuned tree ensembles most. For UnimPA, A remains better than B.

## Caveats

- **The validation year is small and selection is noisy.** 2017 has about 2,600 paired work hours per city (Covilhã has only about 380 heat events). The gain measured on 2017 overstates the real gain:

  | Model | Rep | 2017 val baseline | 2017 val tuned | Val Δ | 2018–25 Δ (mean of cities) |
  |---|---|---|---|---|---|
  | Online Logistic Regression | A | 0.9413 | 0.9571 | +0.0158 | +0.0109 |
  | Online Logistic Regression | B | 0.9405 | 0.9573 | +0.0168 | +0.0098 |
  | Hoeffding Tree | A | 0.9373 | 0.9566 | +0.0193 | -0.0004 |
  | Hoeffding Tree | B | 0.9372 | 0.9653 | +0.0281 | +0.0028 |
  | Online Bagging | A | 0.9465 | 0.9581 | +0.0117 | +0.0022 |
  | Online Bagging | B | 0.9449 | 0.9738 | +0.0288 | +0.0079 |
  | Adaptive Random Forest | A | 0.9422 | 0.9541 | +0.0119 | +0.0108 |
  | Adaptive Random Forest | B | 0.9477 | 0.9601 | +0.0123 | +0.0236 |
  | UnimPA | A | 0.9003 | 0.9062 | +0.0059 | +0.0025 |
  | UnimPA | B | 0.8919 | 0.8922 | +0.0003 | -0.0018 |
  | UnimSafe | A | 0.6115 | 0.7869 | +0.1755 | +0.1463 |
  | UnimSafe | B | 0.6623 | 0.7871 | +0.1248 | +0.0778 |

  The Hoeffding Tree is the clearest case: +0.02 to +0.03 on 2017 but about zero on 2018–2025. The search found configurations that fit 2017 well, not ones that generalise better. Gains of a few thousandths should be read as "no meaningful change".
- **Single seed.** Each configuration was run with seed 42 only. For the ensembles, differences below about 0.003 may be within seed-to-seed variation.
- **Recall trade-off for the tuned Hoeffding Tree on Covilhã B.** Recall drops from 0.949 to 0.911 while precision rises from 0.948 to 0.982. For a heat-safety use case, missed events (FN) may matter more than macro-F1 suggests.
- **Higher cost.** Tuned Online Bagging and Adaptive Random Forest use 3–6× more trees and took about 6× longer per replay in this run (timed under parallel load, so the figures are indicative only). Tuned UnimSafe uses up to 12 rules, which is more than the notebook's assertion `history.rules.between(1, 5)` allows. That assertion needs relaxing before the tuned UnimSafe can go into the notebook.
- **Same parameters for both cities.** Selecting per city might gain a little more, but it would halve the validation data behind each choice and make selection noisier.

## Reproducing

```powershell
.\venv\Scripts\Activate.ps1
python src\tune_hyperparameters.py --trials 120
```

- **Run time:** about 5 minutes on 22 workers.
- **Options:** `--workers N` sets the number of processes. `--trials N` sets the base candidate count per model, scaled ×1.0 / ×1.5 per model.
- **Inputs:** the processed streams written by `src/Final_Project_Heat_Stress.ipynb`. Run the notebook first if `data/processed/` is missing.

## Files in this folder

| File | Content |
|---|---|
| `validation_trials.csv` | All 3,604 validation runs (2017 scores, parameters, runtime) |
| `best_parameters.json` | Selected configuration per model × representation, with validation scores |
| `official_runs.csv` | 2018–2025 replays of baseline and selected configurations (full confusion matrices) |
| `official_comparison.csv` | Baseline vs tuned per city × representation × model |
| `FINDINGS.md` | This report |

The notebook (`src/Final_Project_Heat_Stress.ipynb`) and the main `results/` files still use the original parameters.

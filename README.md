# Finance Timing Project

This project studies a simple index timing task. The workflow is organized as a few layers:

1. Data layer

   Raw daily and intraday files are stored in `data/raw/`. The main experiments use the processed daily data in `data/processed/`.

2. Feature layer

   The selected model features are defined directly in `src/model_features.py`:

   - `logistic_regression`: 12 selected high-importance technical and trend features
   - `random_forest`: 16 selected momentum and trend-strength features
   - `gradient_boosting`: all 20 selected feature-engineering features
   - `hist_gradient_boosting`: all 20 selected feature-engineering features
   - `hist_gradient_boosting_alignment_confirmation`: all 20 selected feature-engineering features
   - `lightgbm`: all 20 selected feature-engineering features

3. Model layer

   The implemented models are:

   - logistic regression
   - random forest
   - gradient boosting
   - histogram gradient boosting
   - locked histogram gradient boosting with alignment-confirmation position policy
   - LightGBM

4. Position policy layer

   Model probabilities are converted into positions with searched mappings:

   - `linear_clipped`
   - `rank_linear`
   - `sigmoid`
   - `power`
   - `threshold`
   - `rank_confirmation`

   The search also includes `smoothing_window = 1, 2, 3, 5, 7, 10`, where
   `1` means no smoothing, and `smoothing_method = sma` or `ewma`.

5. Evaluation layer

   The final comparison uses `data/processed/ml_dataset.csv` for both
   baselines and ML models, so every reported strategy shares the same
   supervised-learning test period:

   ```text
   train: 2017-06-01 to 2023-12-29
   valid: 2024-01-02 to 2024-12-31
   test:  2025-01-02 to 2026-04-24
   ```

   Parameters are selected on the validation period using:

   ```text
   valid_selection_score = 0.5 * valid_return_score + 0.5 * valid_sharpe_score
   ```

   The final comparison is reported only on the test period. The same-period
   buy-and-hold return is 99.28%.

6. Feature importance layer

   `scripts/analyze_feature_importance.py` fits a Random Forest on the training
   split and ranks the current feature columns. This is a diagnostic step,
   not an additional trading model.

## Current Results

Current locked ML strategy:

```text
hist_gradient_boosting_alignment_confirmation + explicit 20 feature-engineering features
mapping = rank_confirmation
entry_rank = 0.50
exit_rank = 0.4875
confirm_days = 2
ma_alignment floor = 1.0

test_return = 163.27%
max_drawdown = -9.27%
sharpe = 3.54
excess_vs_buy_hold = +64.00%
```

Same-period buy-and-hold:

```text
test_return = 99.28%
max_drawdown = -17.26%
sharpe = 2.18
```

Best test Sharpe:

```text
hist_gradient_boosting + explicit 20 feature-engineering features
mapping = rank_linear
smoothing_window = 1

test_return = 110.16%
max_drawdown = -5.85%
sharpe = 3.83
```

The locked strategy has a slightly lower Sharpe than the best searched single
validation-period HGB row, but a much higher test return under the same test
period.

Best LightGBM combinations:

```text
current explicit features: lightgbm, test_return = 96.01%, sharpe = 2.61
historical feature-set leader: lightgbm + momentum, test_return = 105.13%, sharpe = 2.77
```

## Locked Strategy

The final locked ML strategy is reported as
`hist_gradient_boosting_alignment_confirmation`. It keeps the same 20 model
features as `hist_gradient_boosting`, changes only the model hyperparameters
and position policy, then reruns on the same train, validation, and test split.

Locked model:

```text
HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_leaf_nodes=8,
    min_samples_leaf=38,
    random_state=42,
)
```

Locked position policy:

```text
mapping = rank_confirmation
entry_rank = 0.50
exit_rank = 0.4875
confirm_days = 2
min_position = 0.00
max_position = 1.00
smoothing_window = 1
smoothing_method = sma
regime_floor_feature = ma_alignment
regime_floor_threshold = 0.5
regime_floor_position = 1.0
```

`ma_alignment` is used only by the position policy. It is not added to the
model's 20 input features.

## Reproduce

For classroom or teammate reproduction, run only the locked strategy checker:

```bash
python scripts/reproduce_locked_strategy.py
```

This script does not rerun the old baselines, does not search policy grids, and
does not write output files. It only verifies the final locked strategy from
the fixed dataset and fails loudly if the data or result differs.

Expected data contract:

```text
data file = data/processed/ml_dataset.csv
sha256 = a00bf3d3918e5230e0a4f7906a6b0233d4af93a19837e53ae9973434348fb51e
rows = 2162
train rows = 1604
valid rows = 242
test rows = 316
model features = original 20 FEATURE_COLUMNS
ma_alignment = position-policy feature only, not a model input
transaction cost = 0 bps
random_state = 42
```

Expected output:

```text
reproduction passed
cumulative_return: 1.632727292589
max_drawdown: -0.092726987482
sharpe: 3.535132344229
test_auc: 0.653581133540
buy_hold_cumulative_return: 0.992770066097
excess_return_vs_buy_hold: 0.639957226492
```

For maintainers who need to rerun the entire workflow, use the `finance` conda
environment, then run:

```bash
python scripts/run_all.py
```

The full runner uses aggressive CPU settings by default. Set `FINANCE_WORKERS`
to override the worker count on smaller machines.

To rerun only the feature importance diagnostic:

```bash
python scripts/analyze_feature_importance.py
```

To refresh only the locked ML strategy without rerunning the older ML rows:

```bash
python scripts/run_ml_baselines.py --models hist_gradient_boosting_alignment_confirmation --merge-existing
python scripts/plot_ml_baselines.py
```

To save an experiment record after a full run, opt in explicitly:

```bash
python scripts/run_all.py --record --name "experiment name" --purpose "why this run matters"
```

Main outputs are written to:

- `outputs/metrics/`
- `outputs/equity/`
- `outputs/plots/`

The main readable result tables are summarized in `result.md`.

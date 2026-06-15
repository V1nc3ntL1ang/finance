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
   - `lightgbm`: all 20 selected feature-engineering features

3. Model layer

   The implemented models are:

   - logistic regression
   - random forest
   - gradient boosting
   - histogram gradient boosting
   - LightGBM

4. Position policy layer

   Model probabilities are converted into positions with searched mappings:

   - `linear_clipped`
   - `rank_linear`
   - `sigmoid`
   - `power`
   - `threshold`

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

Best test return among the searched ML combinations:

```text
hist_gradient_boosting + explicit 20 feature-engineering features
mapping = rank_linear
smoothing_window = 1

test_return = 110.16%
max_drawdown = -5.85%
sharpe = 3.83
excess_vs_buy_hold = +10.88%
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

Best LightGBM combinations:

```text
current explicit features: lightgbm, test_return = 96.01%, sharpe = 2.61
historical feature-set leader: lightgbm + momentum, test_return = 105.13%, sharpe = 2.77
```

## Reproduce

Use the `finance` conda environment, then run:

```bash
python scripts/run_all.py
```

The runner uses aggressive CPU settings by default. Set `FINANCE_WORKERS` to
override the worker count on smaller machines.

To rerun only the feature importance diagnostic:

```bash
python scripts/analyze_feature_importance.py
```

Main outputs are written to:

- `outputs/metrics/`
- `outputs/equity/`
- `outputs/plots/`

The main readable result tables are summarized in `result.md`.

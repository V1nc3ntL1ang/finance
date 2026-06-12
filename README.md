# Finance Timing Project

This project studies a simple index timing task. The workflow is organized as a few layers:

1. Data layer

   Raw daily and intraday files are stored in `data/raw/`. The main experiments use the processed daily data in `data/processed/`.

2. Feature layer

   The feature groups are defined in `src/feature_groups.py`:

   - `market`: basic price and volume features
   - `regime`: market features plus trend and risk state
   - `momentum`: market features plus normalized momentum state
   - `composite`: all available features

3. Model layer

   The implemented models are:

   - logistic regression
   - random forest
   - gradient boosting
   - histogram gradient boosting
   - LightGBM

4. Position policy layer

   Model probabilities are converted into positions with continuous mappings:

   - `linear_clipped`
   - `rank_linear`
   - `sigmoid`

   The search also includes `smoothing_window = 1, 3, 5`, where `1` means no smoothing.

5. Evaluation layer

   Parameters are selected on the validation period using:

   ```text
   valid_selection_score = 0.5 * valid_return_score + 0.5 * valid_sharpe_score
   ```

   The final comparison is reported on the test period.

6. Feature importance layer

   `scripts/analyze_feature_importance.py` fits a Random Forest on the training
   split and ranks the current 28 feature columns. This is a diagnostic step,
   not an additional trading model.

## Current Results

Best test return among the searched ML combinations:

```text
logistic_regression + composite
mapping = linear_clipped
smoothing_window = 3

test_return = 107.35%
max_drawdown = -14.62%
sharpe = 2.69
excess_vs_buy_hold = +8.07%
```

Best test Sharpe:

```text
random_forest + momentum
mapping = linear_clipped
smoothing_window = 5

test_return = 87.07%
max_drawdown = -7.94%
sharpe = 2.97
```

Best LightGBM combination:

```text
lightgbm + momentum
mapping = linear_clipped
smoothing_window = 3

test_return = 105.13%
max_drawdown = -11.95%
sharpe = 2.77
```

## Reproduce

Use the `finance` conda environment, then run:

```bash
python scripts/run_all.py
```

To rerun only the model and feature-group search:

```bash
python scripts/run_feature_group_ablation.py
```

To rerun only the feature importance diagnostic:

```bash
python scripts/analyze_feature_importance.py
```

The number of workers can be controlled with:

```bash
FINANCE_WORKERS=4 python scripts/run_feature_group_ablation.py
```

Main outputs are written to:

- `outputs/metrics/`
- `outputs/equity/`
- `outputs/plots/`

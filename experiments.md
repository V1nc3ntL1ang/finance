# Experiment Record

This file records the reproducible experiments used for the course report.

## Evaluation Protocol

| Item | Setting |
|---|---|
| Initial capital | 100,000 RMB |
| Investment period | 2025-01-01 to 2026-05-06 |
| Trading calendar used in backtest | 2025-01-02 to 2026-05-06, 321 trading days |
| Data source for formal baselines | `data/processed/daily_features.csv` |
| Data source for formal machine-learning panels | `data/processed/daily.csv` |
| Baseline output metrics | `outputs/metrics/baselines.csv` |
| Baseline output equity curves | `outputs/equity/baselines.csv` |
| Machine-learning output metrics | `outputs/metrics/ml_baselines.csv` |
| Machine-learning output equity curves | `outputs/equity/ml_baselines.csv` |
| StableHGB output metrics | `outputs/metrics/stable_hgb_metrics.csv` |
| StableHGB validation folds | `outputs/metrics/stable_hgb_validation_folds.csv` |
| StableHGB output equity curve | `outputs/equity/stable_hgb_equity.csv` |
| Environment file | `environment.yml` |
| Label requirement | Baseline strategies do not use labels; machine-learning panels use binary five-trading-day forward direction labels for training, validation selection, and ex-post test evaluation |
| Formal validation selection rule | Compute `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)`, keep candidates within 0.02 of the best score, then prefer the least severe worst-fold drawdown |

The formal baseline evaluation uses the full available trading feature data through 2026-05-06. It does not use `data/processed/ml_dataset.csv`, because that labeled dataset drops the final days that do not have a future five-trading-day label.

The `split` column in `data/processed/ml_dataset.csv` is a label-window split based on `target_end_date`. It is not the formal investment-period filter; the experiment scripts use explicit `date` and `target_end_date` conditions.

For the formal machine-learning panels, the final model-fitting set uses labels whose five-trading-day target window ends before 2024-01-01. The 2024 labels are kept as the final validation holdout, and the investment-period backtest starts on 2025-01-01.

Panel A is a direct tradable baseline panel and has no validation-based model or policy selection. Panel B and Panel C use the formal validation rule above. After the 0.02 score screen and least-severe worst-drawdown choice, remaining ties are resolved by valid score, mean validation return, mean validation selection score, and mean validation Sharpe.

Environment setup:

```bash
conda env create -f environment.yml
conda activate finance-StableHGB
```

Full formal workflow:

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_all_experiments.py --workers 10
```

StableHGB reproduction check without rewriting output files:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/reproduce_stable_hgb.py
```

## Panel A: Tradable Baseline Strategies

| Strategy | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold (percentage points) |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 109.21% | 78.52% | -17.26% | 2.30 | 0.00 pp |
| MA20 timing | 77.64% | 57.00% | -10.90% | 2.76 | -31.58 pp |
| 20-day momentum timing | 55.73% | 41.59% | -17.91% | 1.77 | -53.48 pp |
| MA20 and momentum combo | 66.94% | 49.53% | -11.69% | 2.38 | -42.27 pp |

Command:

```bash
python -u scripts/run_baselines.py
```

## Panel B: Competitive Machine-Learning Methods

Panel B uses the same 20 technical features for every model. It evaluates five fixed model specifications; for each model, the probability-to-position policy is selected using rolling validation folds for 2022, 2023, and 2024. For each fold, training labels must end before the validation year begins. The final model is trained with labels whose target window ends before 2024-01-01 and is backtested from 2025-01-01 to 2026-05-06 using daily features through 2026-05-06.

Within each model, candidate policies are first ranked by `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)` across the three validation folds. Candidates within 0.02 of the best score are retained, and the final policy is the retained candidate with the least severe worst-fold drawdown.

Panel B only uses standard probability-to-position mappings: `linear_clipped`, `rank_linear`, `sigmoid`, `power`, and `threshold`. It does not use the StableHGB-specific `relative_signal_stabilizer` mapping or the Trend Position Guard.

The documented Panel B command is configured with 10 workers and a fixed discrete validation grid. The common position parameters are `min_position={0.00,0.10,0.20,0.30}`, `max_position={0.70,0.80,0.90,1.00}`, `smoothing_window={1,3,5}`, and `smoothing_method={sma}`. Combined with the mapping-specific grids, this produces 10,656 candidates per model per validation fold and 159,840 validation rows across all five models and three folds.

| Mapping | Candidates per Fold |
|---|---:|
| `linear_clipped` | 1,296 |
| `rank_linear` | 3,072 |
| `sigmoid` | 1,440 |
| `power` | 4,608 |
| `threshold` | 240 |
| Total | 10,656 |

| Model | Policy Mapping | Valid Score | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold (percentage points) | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | power | 0.2297 | 98.57% | 71.35% | -14.62% | 2.31 | -10.64 pp | 0.62 |
| Random forest | rank_linear | 0.2077 | 85.39% | 62.36% | -9.43% | 2.82 | -23.82 pp | 0.64 |
| Gradient boosting | linear_clipped | 0.2575 | 105.29% | 75.88% | -12.21% | 2.72 | -3.92 pp | 0.63 |
| Histogram gradient boosting | threshold | 0.1528 | 90.38% | 65.78% | -9.29% | 2.81 | -18.83 pp | 0.64 |
| LightGBM | rank_linear | 0.2010 | 88.72% | 64.64% | -10.48% | 2.81 | -20.49 pp | 0.65 |

Command:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/run_ml_baselines.py
```

## Panel C: StableHGB

Panel C evaluates StableHGB. It uses a histogram gradient boosting classifier with the same 20 model features as Panel B. The moving-average alignment signal `ma_alignment` is not used as a model feature; it is used only by the Trend Position Guard inside the trading policy.

The trading policy has two components. The Relative Signal Stabilizer converts predicted probabilities into prior-only expanding percentile ranks and changes position only after the rank signal is confirmed for consecutive days. The Trend Position Guard keeps the strategy fully invested when `ma_alignment > 0.5`. StableHGB model hyperparameters are fixed before the reported run; the trading-policy candidate is selected by the same formal validation rule used in Panel B from a 108-candidate local grid. Rolling folds for 2022, 2023, and 2024 are reported as validation evidence, while the final model is trained with labels whose target window ends before 2024-01-01 and is tested on the 2025-01-01 to 2026-05-06 investment period.

| Strategy | Policy | Valid Score | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold (percentage points) | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.3164 | 153.44% | 107.52% | -9.84% | 3.38 | 44.22 pp | 0.63 |

Validation fold details. The Return/Sharpe tie-break score is an auxiliary fold-level score used for tie-breaking, not the final cross-fold `valid_score`.

| Fold | Cumulative Return | Return/Sharpe Tie-Break Score | Max Drawdown | Sharpe | Buy-and-Hold Return |
|---|---:|---:|---:|---:|---:|
| valid_2022 | 30.99% | 1.36 | -22.29% | 1.18 | 22.20% |
| valid_2023 | 26.80% | 0.70 | -7.39% | 1.99 | 47.60% |
| valid_2024 | 43.76% | 3.56 | -40.84% | 1.06 | 9.33% |

Headline results use 0 bps transaction cost. Transaction-cost sensitivity is reported for StableHGB only and is not used for selecting the strategy; costs are one-way and proportional to absolute position changes. Cash earns 0%, bid-ask spread and slippage are not modeled separately, and the investment period inherits the prior close-derived target position without charging an additional initial establishment cost.

| Transaction Cost | Cumulative Return | Max Drawdown |
|---|---:|---:|
| 0 bps | 153.44% | -9.84% |
| 5 bps | 151.93% | -9.93% |
| 10 bps | 150.43% | -10.02% |
| 20 bps | 147.46% | -10.20% |

Command:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/run_stable_hgb.py
```

## Component Ablation

This ablation fixes the StableHGB histogram gradient boosting model and the same formal train/validation/test protocol. It changes only the position-construction components, so the comparison provides component-level diagnostic evidence for the Relative Signal Stabilizer and the Trend Position Guard.

| Experiment | Position rule | Valid score | Cumulative return | Excess vs buy-hold (percentage points) | Max drawdown | Sharpe | Test AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Buy-and-hold reference | constant_full_position |  | 109.21% | 0.00 pp | -17.26% | 2.30 |  |
| Validation-selected Gradient Boosting | Panel B linear_clipped | 0.2575 | 105.29% | -3.92 pp | -12.21% | 2.72 | 0.625 |
| StableHGB classifier + standard policy | rank_linear | 0.1993 | 82.20% | -27.01 pp | -10.48% | 2.65 | 0.635 |
| StableHGB classifier + standard policy + Trend Position Guard | rank_linear + Trend Position Guard | 0.2237 | 106.66% | -2.55 pp | -12.58% | 2.67 | 0.635 |
| StableHGB classifier + Relative Signal Stabilizer | Relative Signal Stabilizer | 0.2072 | 121.51% | 12.30 pp | -7.52% | 3.59 | 0.635 |
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.3164 | 153.44% | 44.22 pp | -9.84% | 3.38 | 0.635 |

All ablation headline results use 0 bps transaction cost.

Command:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/run_stable_hgb_ablation.py
```

Outputs:

| Output | Path |
|---|---|
| Component ablation table | `outputs/metrics/stable_hgb_ablation.csv` |
| Markdown table | `outputs/metrics/stable_hgb_ablation.md` |

## Figures

Experiment figures are generated from the formal output CSV files only.

| Figure | Output |
|---|---|
| All tradable baselines | `outputs/plots/all_baselines.png` |
| All competitive machine-learning baselines with buy-and-hold reference | `outputs/plots/all_ml_baselines.png` |
| StableHGB vs best baseline and best machine-learning baseline | `outputs/plots/stable_hgb_vs_references.png` |
| Drawdown comparison against best references | `outputs/plots/drawdown_stable_hgb_references.png` |
| Risk-return scatter | `outputs/plots/risk_return_scatter.png` |
| StableHGB position and equity | `outputs/plots/stable_hgb_position.png` |

Command:

```bash
PYTHONUNBUFFERED=1 python -u scripts/plot_results.py
```

## Notes

- `theoretical_optimal` is excluded from the formal baseline panel because it uses future returns and is not tradable.
- Baselines are reference strategies. StableHGB is evaluated separately against the same investment period.
- The environment is pinned in `environment.yml`.

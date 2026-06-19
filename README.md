# Finance Timing Project

This repository implements an index-enhancement strategy for the course project. It keeps one reproducible experiment workflow so teammates can rerun the reported results cleanly.

## Formal Protocol

| Item | Setting |
|---|---|
| Initial capital | 100,000 RMB |
| Investment period | 2025-01-01 to 2026-05-06 |
| Backtest trading days | 2025-01-02 to 2026-05-06, 321 trading days |
| Label target | Binary five-trading-day forward direction label `future_up_5d`, derived from `future_ret_5d` |
| Final model-fitting labels | Labels whose target window ends before 2024-01-01 |
| Validation evidence | Rolling folds for 2022, 2023, and 2024 |
| Formal selection rule | First compute `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)`, keep candidates within 0.02 of the best score, then prefer the least severe worst-fold drawdown |
| Experiment record | `experiments.md` |

Daily labels naturally end at 2026-04-24 because `future_ret_5d` needs five future trading days. Trading features do not need future labels, so the formal investment backtest still runs through 2026-05-06.

Panel A contains direct tradable baselines and does not require validation-based selection. Panel B and Panel C use the formal validation rule above. After the 0.02 score screen and least-severe worst-drawdown choice, remaining ties are resolved by valid score, mean validation return, mean validation selection score, and mean validation Sharpe.

## Environment

Create the project environment from the checked-in file:

```bash
conda env create -f environment.yml
conda activate finance-StableHGB
```

The environment pins the packages used for the formal run, including Python, pandas, scikit-learn, LightGBM, matplotlib, joblib, tqdm, and Pillow.

## Data Processing

The reproducible preprocessing chain starts from the checked-in raw daily text file:

| Step | Script | Output |
|---|---|---|
| Parse raw daily text | `scripts/build_daily_from_raw.py` | `data/processed/daily.csv` |
| Build tradable daily features | `scripts/build_daily_features.py` | `data/processed/daily_features.csv` |
| Build machine-learning features and labels | `scripts/build_ml_dataset.py` | `data/processed/ml_dataset.csv` |

Feature construction is implemented in `src/features.py` and `src/ml_dataset.py`. The machine-learning label is the binary direction target `future_up_5d`; `future_ret_5d` is retained as the underlying five-trading-day return. The position-only trend signal `ma_alignment` is generated with the daily features but is not used as a model input.

The `split` column in `data/processed/ml_dataset.csv` is assigned by the end date of the five-trading-day label window. Formal experiments still use explicit `date` and `target_end_date` filters, because the labeled test split is not identical to the final investment-period trading calendar.

## Experiments

The experiment results are organized into three panels.

| Panel | Script | Main outputs |
|---|---|---|
| Panel A: tradable baselines | `scripts/run_baselines.py` | `outputs/metrics/baselines.csv`, `outputs/equity/baselines.csv` |
| Panel B: competitive ML baselines | `scripts/run_ml_baselines.py` | `outputs/metrics/ml_baselines.csv`, `outputs/equity/ml_baselines.csv` |
| Panel C: StableHGB | `scripts/run_stable_hgb.py` | `outputs/metrics/stable_hgb_metrics.csv`, `outputs/metrics/stable_hgb_validation_folds.csv`, `outputs/equity/stable_hgb_equity.csv` |

Run the full workflow:

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_all_experiments.py --workers 10
```

Adjust `--workers` only if the machine has fewer available cores.

To verify only StableHGB without rewriting output files:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/reproduce_stable_hgb.py
```

## Reported Results

The formal comparison uses the same investment period for all strategies.

| Group | Best strategy | Cumulative return | Max drawdown | Sharpe | Excess vs buy-and-hold (percentage points) |
|---|---|---:|---:|---:|---:|
| Panel A | Buy and hold | 109.21% | -17.26% | 2.30 | 0.00 pp |
| Panel B | Gradient boosting | 105.29% | -12.21% | 2.72 | -3.92 pp |
| Panel C | StableHGB | 153.44% | -9.84% | 3.38 | 44.22 pp |

The full tables and validation-fold details are in `experiments.md`.

## StableHGB

StableHGB combines a histogram gradient boosting classifier with two trading-policy components:

- `Relative Signal Stabilizer`: converts each predicted probability into a prior-only expanding percentile rank and changes position only after the rank signal is confirmed for consecutive days.
- `Trend Position Guard`: uses the moving-average alignment signal `ma_alignment` to keep the strategy fully invested when the recent trend is strong.

Model:

```text
HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=70,
    random_state=42,
)
```

Trading policy:

```text
mapping_type = relative_signal_stabilizer
entry_rank = 0.550
exit_rank = 0.450
confirm_days = 2
min_position = 0.00
max_position = 1.00
smoothing_window = 1
smoothing_method = sma
trend_guard_feature = ma_alignment
trend_guard_threshold = 0.5
trend_guard_min_position = 1.0
```

`ma_alignment` is not a model input feature. It is used only by the Trend Position Guard inside the trading policy.

Backtests use a close-to-close accounting convention with `execution_lag=1`: the previous row's target position is applied to the next close-to-close return. In the equity CSV files, `position` is the target position generated by the policy, while `executed_position` is the position actually applied to daily returns. Headline results use 0 bps transaction cost; the StableHGB table also reports one-way cost sensitivity based on absolute position changes.

## Figures

Experiment figures are generated by `scripts/plot_results.py`:

| Figure | Output |
|---|---|
| All tradable baselines | `outputs/plots/all_baselines.png` |
| All competitive ML baselines with buy-and-hold reference | `outputs/plots/all_ml_baselines.png` |
| StableHGB vs best baseline and best ML baseline | `outputs/plots/stable_hgb_vs_references.png` |
| Drawdown comparison against best references | `outputs/plots/drawdown_stable_hgb_references.png` |
| Risk-return scatter | `outputs/plots/risk_return_scatter.png` |
| StableHGB position and equity | `outputs/plots/stable_hgb_position.png` |

## Reproduction Check

For the StableHGB contract based on `data/processed/daily.csv`, run:

```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/reproduce_stable_hgb.py
```

This checker recomputes StableHGB, validates the input data hash, confirms that `ma_alignment` is position-only, and verifies the final 2025-01-01 to 2026-05-06 investment-period metrics.

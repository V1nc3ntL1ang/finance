# StableHGB: Stable Index Timing with Histogram Gradient Boosting

StableHGB is a machine-learning enhanced index timing strategy for micro-cap stocks (SH#880823). It combines a histogram gradient boosting classifier with two proprietary trading-policy components to deliver superior risk-adjusted returns while maintaining stable portfolio positioning.

## 📊 Key Results

StableHGB significantly outperforms both simple baselines and competitive machine-learning approaches:

| Strategy | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess vs Buy-and-Hold |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 109.21% | 78.52% | -17.26% | 2.30 | 0.00 pp |
| Best ML baseline (Gradient boosting) | 105.29% | 75.88% | -12.21% | 2.72 | -3.92 pp |
| **StableHGB** | **153.44%** | **107.52%** | **-9.84%** | **3.38** | **+44.22 pp** |

## 🏗️ Strategy Architecture

StableHGB integrates three core components:

### 1. Prediction Engine
```text
HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=70,
    random_state=42,
)
```

### 2. Relative Signal Stabilizer
Converts predicted probabilities into expanding percentile ranks, requiring consecutive-day confirmation before position changes to filter noise.

### 3. Trend Position Guard
Uses a moving-average alignment signal (`ma_alignment`) to maintain full investment during strong trends, preventing premature exits.

## 🔧 Trading Policy Configuration

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

## 🛠️ Environment Setup

```bash
conda env create -f environment.yml
conda activate finance-StableHGB
```

## 📁 Project Structure

```
finance/
├── data/                    # Data storage
│   ├── raw/                 # Raw input data
│   └── processed/           # Processed features and datasets
├── scripts/                 # Experiment scripts
│   ├── run_all_experiments.py    # Run full pipeline
│   ├── run_stable_hgb.py         # StableHGB experiment
│   ├── run_baselines.py          # Panel A: tradable baselines
│   ├── run_ml_baselines.py       # Panel B: competitive ML baselines
│   └── plot_results.py           # Generate figures
├── src/                     # Core source code
│   ├── backtest.py          # Backtest engine
│   ├── features.py          # Feature construction
│   ├── ml_models.py         # ML model definitions
│   ├── position_policy.py   # Position mapping strategies
│   └── stable_hgb.py        # StableHGB implementation
└── outputs/                 # Experiment outputs
    ├── metrics/             # Performance metrics
    ├── equity/              # Equity curves
    └── plots/               # Visualizations
```

## 🚀 Quick Start

### Run Full Experiments
```bash
PYTHONUNBUFFERED=1 python -u scripts/run_all_experiments.py --workers 10
```

### Reproduce StableHGB Only
```bash
PYTHONUNBUFFERED=1 FINANCE_WORKERS=10 python -u scripts/reproduce_stable_hgb.py
```

## 📈 Experimental Protocol

| Item | Setting |
|---|---|
| Initial capital | 100,000 RMB |
| Investment period | 2025-01-01 to 2026-05-06 |
| Trading days | 321 trading days |
| Label target | Binary five-trading-day direction `future_up_5d` |
| Validation | Rolling folds: 2022, 2023, 2024 |
| Selection rule | `valid_score = mean(return) - 0.25 * std(return)` |
| Backtest convention | Close-to-close with `execution_lag=1` |

## 📋 Experiment Panels

| Panel | Focus | Script |
|---|---|---|
| **Panel A** | Traditional timing baselines (MA20, momentum) | `scripts/run_baselines.py` |
| **Panel B** | Competitive ML baselines (LR, RF, GBDT, LGBM) | `scripts/run_ml_baselines.py` |
| **Panel C** | **StableHGB strategy** | `scripts/run_stable_hgb.py` |

## 📊 Figures

| Figure | Description | File |
|---|---|---|
| All baselines | Equity curves of traditional strategies | `outputs/plots/all_baselines.png` |
| ML baselines | Equity curves of ML strategies | `outputs/plots/all_ml_baselines.png` |
| Risk-return | Scatter plot of strategy candidates | `outputs/plots/risk_return_scatter.png` |
| StableHGB vs references | StableHGB performance comparison | `outputs/plots/stable_hgb_vs_references.png` |
| Drawdown comparison | Drawdown profiles | `outputs/plots/drawdown_stable_hgb_references.png` |
| StableHGB position | Position and equity over time | `outputs/plots/stable_hgb_position.png` |

## 📝 Full Results

Complete experiment records and validation details are available in:
- `result.md` - Final summary for course report
- `experiments.md` - Detailed experiment history

---

*StableHGB achieves +44.22 percentage points excess return over buy-and-hold with a Sharpe ratio of 3.38 and maximum drawdown limited to -9.84%.*
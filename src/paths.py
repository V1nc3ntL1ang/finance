from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
EQUITY_DIR = OUTPUT_DIR / "equity"
PLOTS_DIR = OUTPUT_DIR / "plots"
BASELINE_PLOTS_DIR = PLOTS_DIR / "baseline"
ML_PLOTS_DIR = PLOTS_DIR / "ml"
ABLATION_PLOTS_DIR = PLOTS_DIR / "ablation"

DAILY_CSV = PROCESSED_DATA_DIR / "daily.csv"
DAILY_FEATURES_CSV = PROCESSED_DATA_DIR / "daily_features.csv"
ML_DATASET_CSV = PROCESSED_DATA_DIR / "ml_dataset.csv"

BASELINE_METRICS_CSV = METRICS_DIR / "baseline_metrics.csv"
ML_METRICS_CSV = METRICS_DIR / "ml_metrics.csv"
ML_ABLATION_METRICS_CSV = METRICS_DIR / "ml_ablation_metrics.csv"
FEATURE_GROUP_ABLATION_METRICS_CSV = METRICS_DIR / "feature_group_ablation_metrics.csv"

BASELINE_EQUITY_CSV = EQUITY_DIR / "baseline_equity.csv"
ML_EQUITY_CSV = EQUITY_DIR / "ml_equity.csv"


def ensure_output_dirs() -> None:
    for path in [
        METRICS_DIR,
        EQUITY_DIR,
        BASELINE_PLOTS_DIR,
        ML_PLOTS_DIR,
        ABLATION_PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
EQUITY_DIR = OUTPUT_DIR / "equity"
PLOTS_DIR = OUTPUT_DIR / "plots"

DAILY_CSV = PROCESSED_DATA_DIR / "daily.csv"
RAW_DAILY_TXT = RAW_DATA_DIR / "SH#880823.txt"
DAILY_FEATURES_CSV = PROCESSED_DATA_DIR / "daily_features.csv"
ML_DATASET_CSV = PROCESSED_DATA_DIR / "ml_dataset.csv"

BASELINE_METRICS_CSV = METRICS_DIR / "baselines.csv"
ML_BASELINE_METRICS_CSV = METRICS_DIR / "ml_baselines.csv"
ML_BASELINE_VALIDATION_CSV = METRICS_DIR / "ml_validation_folds.csv"
STABLE_HGB_METRICS_CSV = METRICS_DIR / "stable_hgb_metrics.csv"
STABLE_HGB_VALIDATION_CSV = METRICS_DIR / "stable_hgb_validation_folds.csv"
STABLE_HGB_ABLATION_CSV = METRICS_DIR / "stable_hgb_ablation.csv"
STABLE_HGB_ABLATION_MD = METRICS_DIR / "stable_hgb_ablation.md"
STABLE_HGB_ABLATION_VALIDATION_CSV = METRICS_DIR / "stable_hgb_ablation_validation.csv"

BASELINE_EQUITY_CSV = EQUITY_DIR / "baselines.csv"
ML_BASELINE_EQUITY_CSV = EQUITY_DIR / "ml_baselines.csv"
STABLE_HGB_EQUITY_CSV = EQUITY_DIR / "stable_hgb_equity.csv"

BASELINES_PLOT = PLOTS_DIR / "all_baselines.png"
ML_BASELINES_PLOT = PLOTS_DIR / "all_ml_baselines.png"
STABLE_HGB_REFERENCE_PLOT = PLOTS_DIR / "stable_hgb_vs_references.png"
DRAWDOWN_REFERENCE_PLOT = PLOTS_DIR / "drawdown_stable_hgb_references.png"
RISK_RETURN_PLOT = PLOTS_DIR / "risk_return_scatter.png"
STABLE_HGB_POSITION_PLOT = PLOTS_DIR / "stable_hgb_position.png"


def ensure_output_dirs() -> None:
    for path in [
        METRICS_DIR,
        EQUITY_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

from src.ml_dataset import FEATURE_COLUMNS


BASIC_FEATURES = [
    "ret_1d",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "momentum5",
    "momentum10",
    "momentum20",
    "volatility5",
    "volatility20",
    "range_pct",
    "volume_change20",
]

LEVEL_FEATURES = [
    "close_vs_ma20",
    "close_vs_ma60",
    "ma20_slope",
    "ma60_slope",
    "drawdown_from_20d_high",
    "drawdown_from_60d_high",
    "volume_zscore20",
    "volatility20_rank",
]

STRENGTH_FEATURES = [
    "ma_alignment",
    "trend_strength20",
    "trend_strength60",
    "ret5_over_vol20",
    "ret20_over_vol20",
    "rebound_from_20d_low",
    "rebound_from_60d_low",
    "range_zscore20",
]

FEATURE_GROUPS = {
    "market": BASIC_FEATURES,
    "regime": BASIC_FEATURES + LEVEL_FEATURES,
    "momentum": BASIC_FEATURES + STRENGTH_FEATURES,
    "composite": FEATURE_COLUMNS,
}

MODEL_FEATURE_GROUPS = {
    "logistic_regression": "composite",
    "random_forest": "momentum",
    "gradient_boosting": "composite",
    "hist_gradient_boosting": "composite",
    "lightgbm": "composite",
}

from __future__ import annotations

from collections.abc import Collection

from src.ml_dataset import FEATURE_COLUMNS


LOGISTIC_REGRESSION_FEATURES = [
    "volatility5",
    "volatility20_rank",
    "rebound_from_60d_low",
    "volatility20",
    "macd_histogram",
    "bb_width",
    "trend_strength60",
    "macd_signal",
    "ma60",
    "rebound_from_20d_low",
    "bb_lower",
    "ma60_slope",
]

RANDOM_FOREST_FEATURES = [
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "momentum20",
    "volatility5",
    "volatility20",
    "trend_strength20",
    "trend_strength60",
    "rebound_from_20d_low",
    "rebound_from_60d_low",
    "macd_signal",
    "macd_histogram",
    "ma20_slope",
    "ma60_slope",
    "close_vs_ma60",
]

GRADIENT_BOOSTING_FEATURES = FEATURE_COLUMNS
HIST_GRADIENT_BOOSTING_FEATURES = FEATURE_COLUMNS
HIST_GRADIENT_BOOSTING_ALIGNMENT_CONFIRMATION_FEATURES = FEATURE_COLUMNS
LIGHTGBM_FEATURES = FEATURE_COLUMNS

MODEL_FEATURE_COLUMNS = {
    "logistic_regression": LOGISTIC_REGRESSION_FEATURES,
    "random_forest": RANDOM_FOREST_FEATURES,
    "gradient_boosting": GRADIENT_BOOSTING_FEATURES,
    "hist_gradient_boosting": HIST_GRADIENT_BOOSTING_FEATURES,
    "hist_gradient_boosting_alignment_confirmation": HIST_GRADIENT_BOOSTING_ALIGNMENT_CONFIRMATION_FEATURES,
    "lightgbm": LIGHTGBM_FEATURES,
}


def validate_model_feature_columns(model_names: Collection[str], available_columns: Collection[str]) -> None:
    configured_models = set(MODEL_FEATURE_COLUMNS)
    expected_models = set(model_names)
    errors: list[str] = []

    missing_configs = sorted(expected_models - configured_models)
    if missing_configs:
        errors.append(f"missing feature configs for models: {', '.join(missing_configs)}")

    unused_configs = sorted(configured_models - expected_models)
    if unused_configs:
        errors.append(f"feature configs without matching models: {', '.join(unused_configs)}")

    available_column_set = set(available_columns)
    for model_name in sorted(configured_models & expected_models):
        missing_features = [
            feature
            for feature in MODEL_FEATURE_COLUMNS[model_name]
            if feature not in available_column_set
        ]
        if missing_features:
            errors.append(f"{model_name} missing dataset columns: {', '.join(missing_features)}")

    if errors:
        raise ValueError("Invalid model feature configuration: " + "; ".join(errors))

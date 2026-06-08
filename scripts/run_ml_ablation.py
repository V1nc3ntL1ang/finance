from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ml_baselines import (
    RETURN_WEIGHT,
    SELECTION_OBJECTIVE,
    SHARPE_WEIGHT,
    TEST_START,
    VALID_START,
    choose_exposure_mapping,
    load_dataset,
    safe_auc,
)
from src.backtest import compute_metrics, run_backtest, write_metrics_csv
from src.feature_groups import BASIC_FEATURES, FEATURE_GROUPS
from src.ml_models import get_ml_models
from src.paths import ML_ABLATION_METRICS_CSV, ensure_output_dirs
from src.position_policy import build_position


FEATURE_SETS = {
    "market": BASIC_FEATURES,
    "composite": FEATURE_GROUPS["composite"],
}


def main() -> None:
    ensure_output_dirs()
    df = load_dataset()
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    test = df[df["split"] == "test"]

    y_train = train["future_up_5d"]
    y_valid = valid["future_up_5d"]
    y_test = test["future_up_5d"]

    buy_hold_position = pd.Series(1.0, index=df.index)
    valid_buy_hold_metrics = compute_metrics(
        run_backtest(df, buy_hold_position, test_start=VALID_START, test_end=TEST_START)
    )
    buy_hold_return = compute_metrics(run_backtest(df, buy_hold_position, test_start=TEST_START))["cumulative_return"]

    metrics_rows: list[dict[str, float | str]] = []

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        for model_name, model in get_ml_models().items():
            model.fit(train[feature_columns], y_train)

            probability = pd.Series(model.predict_proba(df[feature_columns])[:, 1], index=df.index)
            valid_probability = probability.loc[valid.index]
            test_probability = probability.loc[test.index]

            mapping_params, valid_backtest_metrics, valid_selection_scores = choose_exposure_mapping(
                df,
                probability,
                buy_hold_valid_metrics=valid_buy_hold_metrics,
            )
            position = build_position(probability, mapping_params)
            test_equity = run_backtest(df, position, test_start=TEST_START)
            test_backtest_metrics = compute_metrics(test_equity)

            valid_pred = (valid_probability >= 0.5).astype(int)
            test_pred = (test_probability >= 0.5).astype(int)

            row: dict[str, float | str] = {
                "feature_set": feature_set_name,
                "feature_count": len(feature_columns),
                "model": model_name,
                "selection_objective": SELECTION_OBJECTIVE,
                "return_weight": RETURN_WEIGHT,
                "sharpe_weight": SHARPE_WEIGHT,
                **valid_selection_scores,
                **mapping_params,
                "valid_accuracy": float(accuracy_score(y_valid, valid_pred)),
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
                "valid_auc": safe_auc(y_valid, valid_probability),
                "test_auc": safe_auc(y_test, test_probability),
                "valid_cumulative_return": valid_backtest_metrics["cumulative_return"],
                "valid_max_drawdown": valid_backtest_metrics["max_drawdown"],
                "valid_sharpe": valid_backtest_metrics["sharpe"],
                **test_backtest_metrics,
            }
            row["excess_return_vs_buy_hold"] = row["cumulative_return"] - buy_hold_return
            metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    write_metrics_csv(metrics, ML_ABLATION_METRICS_CSV)
    print(f"wrote {ML_ABLATION_METRICS_CSV} rows={len(metrics)}")


if __name__ == "__main__":
    main()

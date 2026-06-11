from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score


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
from src.feature_groups import FEATURE_GROUPS
from src.ml_models import get_ml_models
from src.paths import FEATURE_GROUP_ABLATION_METRICS_CSV, ensure_output_dirs
from src.position_policy import build_position


def get_worker_count() -> int:
    return int(os.environ.get("FINANCE_WORKERS", "-1"))


def evaluate_combination(
    group_name: str,
    feature_columns: list[str],
    model_name: str,
    df: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_buy_hold_metrics: dict[str, float],
    buy_hold_return: float,
) -> dict[str, float | int | str]:
    model = get_ml_models(n_jobs=1)[model_name]
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
    test_metrics = compute_metrics(test_equity)

    valid_pred = (valid_probability >= 0.5).astype(int)
    test_pred = (test_probability >= 0.5).astype(int)

    row: dict[str, float | int | str] = {
        "feature_group": group_name,
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
        **test_metrics,
    }
    row["excess_return_vs_buy_hold"] = row["cumulative_return"] - buy_hold_return
    return row


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

    tasks = [
        (group_name, feature_columns, model_name)
        for group_name, feature_columns in FEATURE_GROUPS.items()
        for model_name in get_ml_models(n_jobs=1)
    ]
    worker_count = get_worker_count()
    rows = Parallel(n_jobs=worker_count)(
        delayed(evaluate_combination)(
            group_name,
            feature_columns,
            model_name,
            df,
            train,
            valid,
            test,
            y_train,
            y_valid,
            y_test,
            valid_buy_hold_metrics,
            buy_hold_return,
        )
        for group_name, feature_columns, model_name in tasks
    )

    metrics = pd.DataFrame(rows)
    write_metrics_csv(metrics, FEATURE_GROUP_ABLATION_METRICS_CSV)
    print(f"wrote {FEATURE_GROUP_ABLATION_METRICS_CSV} rows={len(metrics)} workers={worker_count}")


if __name__ == "__main__":
    main()

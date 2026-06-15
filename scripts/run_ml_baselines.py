from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest, write_equity_csv, write_metrics_csv
from src.experiment_config import RETURN_WEIGHT, SELECTION_OBJECTIVE, SHARPE_WEIGHT, TEST_START, VALID_START
from src.ml_models import get_ml_models
from src.model_features import MODEL_FEATURE_COLUMNS
from src.paths import ML_DATASET_CSV, ML_EQUITY_CSV, ML_METRICS_CSV, ensure_output_dirs
from src.position_policy import build_position, iter_position_policy_candidates


def get_worker_count() -> int:
    return int(os.environ.get("FINANCE_WORKERS", os.cpu_count() or 1))


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ML_DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in df.columns if column not in {"date", "split"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df.sort_values("date").reset_index(drop=True)


def safe_auc(y_true: pd.Series, probability: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def choose_exposure_mapping(
    df: pd.DataFrame,
    probability: pd.Series,
    *,
    model_name: str,
    buy_hold_valid_metrics: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    buy_hold_return = buy_hold_valid_metrics["cumulative_return"]
    buy_hold_sharpe = buy_hold_valid_metrics["sharpe"]

    def evaluate_params(params: dict[str, float | int | str]) -> tuple[dict[str, float | int | str], dict[str, float], dict[str, float]]:
        position = build_position(probability, params)
        valid_equity = run_backtest(df, position, test_start=VALID_START, test_end=TEST_START)
        metrics = compute_metrics(valid_equity)

        return_score = metrics["cumulative_return"] / buy_hold_return if buy_hold_return != 0 else 0.0
        sharpe_score = metrics["sharpe"] / buy_hold_sharpe if buy_hold_sharpe != 0 else 0.0
        selection_score = RETURN_WEIGHT * return_score + SHARPE_WEIGHT * sharpe_score
        scores = {
            "valid_selection_score": selection_score,
            "valid_return_score": return_score,
            "valid_sharpe_score": sharpe_score,
        }
        return params.copy(), metrics, scores

    candidates = list(iter_position_policy_candidates())
    result_iter = Parallel(n_jobs=get_worker_count(), prefer="threads", return_as="generator")(
        delayed(evaluate_params)(params) for params in candidates
    )
    results = list(
        tqdm(
            result_iter,
            total=len(candidates),
            desc=f"{model_name} policy",
            unit="policy",
            dynamic_ncols=True,
        )
    )

    best_params, best_metrics, best_scores = max(
        results,
        key=lambda item: (item[2]["valid_selection_score"], item[1]["cumulative_return"]),
    )

    return best_params, best_metrics, best_scores


def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_buy_hold_metrics: dict[str, float],
) -> tuple[dict[str, float | str], pd.DataFrame]:
    model = get_ml_models(n_jobs=get_worker_count())[model_name]
    feature_columns = MODEL_FEATURE_COLUMNS[model_name]
    model.fit(train[feature_columns], y_train)

    probability = pd.Series(model.predict_proba(df[feature_columns])[:, 1], index=df.index)
    valid_probability = probability.loc[valid.index]
    test_probability = probability.loc[test.index]

    mapping_params, valid_backtest_metrics, valid_selection_scores = choose_exposure_mapping(
        df,
        probability,
        model_name=model_name,
        buy_hold_valid_metrics=valid_buy_hold_metrics,
    )
    position = build_position(probability, mapping_params)
    test_equity = run_backtest(df, position, test_start=TEST_START)
    test_backtest_metrics = compute_metrics(test_equity)

    valid_pred = (valid_probability >= 0.5).astype(int)
    test_pred = (test_probability >= 0.5).astype(int)

    metric_row: dict[str, float | str] = {
        "model": model_name,
        "feature_count": len(feature_columns),
        "features": ", ".join(feature_columns),
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

    test_equity.insert(0, "model", model_name)
    return metric_row, test_equity


def main() -> None:
    ensure_output_dirs()
    worker_count = get_worker_count()
    print(f"aggressive CPU mode: workers={worker_count}", flush=True)

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

    model_names = list(get_ml_models(n_jobs=worker_count))
    results = []
    for model_name in tqdm(model_names, desc="models", unit="model", dynamic_ncols=True):
        results.append(
            evaluate_model(
                model_name,
                df,
                train,
                valid,
                test,
                y_train,
                y_valid,
                y_test,
                valid_buy_hold_metrics,
            )
        )

    metrics_rows = [row for row, _equity in results]
    equity_frames = [equity for _row, equity in results]

    metrics = pd.DataFrame(metrics_rows)
    buy_hold_return = compute_metrics(run_backtest(df, buy_hold_position, test_start=TEST_START))["cumulative_return"]
    metrics["excess_return_vs_buy_hold"] = metrics["cumulative_return"] - buy_hold_return

    write_metrics_csv(metrics, ML_METRICS_CSV)
    write_equity_csv(pd.concat(equity_frames, ignore_index=True), ML_EQUITY_CSV)
    print(f"wrote {ML_METRICS_CSV} models={len(metrics)}")
    print(f"wrote {ML_EQUITY_CSV} rows={sum(len(frame) for frame in equity_frames)}")


if __name__ == "__main__":
    main()

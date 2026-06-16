from __future__ import annotations

import argparse
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
from src.model_features import MODEL_FEATURE_COLUMNS, validate_model_feature_columns
from src.paths import ML_DATASET_CSV, ML_EQUITY_CSV, ML_METRICS_CSV, ensure_output_dirs
from src.position_policy import build_policy_position, get_locked_policy_for_model, iter_position_policy_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML timing baselines.")
    parser.add_argument("--models", nargs="+", help="Optional model names to run instead of all models.")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge selected model outputs into existing ML metrics/equity files.",
    )
    return parser.parse_args()


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


def score_valid_metrics(
    metrics: dict[str, float],
    buy_hold_valid_metrics: dict[str, float],
) -> dict[str, float]:
    buy_hold_return = buy_hold_valid_metrics["cumulative_return"]
    buy_hold_sharpe = buy_hold_valid_metrics["sharpe"]

    return_score = metrics["cumulative_return"] / buy_hold_return if buy_hold_return != 0 else 0.0
    sharpe_score = metrics["sharpe"] / buy_hold_sharpe if buy_hold_sharpe != 0 else 0.0
    selection_score = RETURN_WEIGHT * return_score + SHARPE_WEIGHT * sharpe_score
    return {
        "valid_selection_score": selection_score,
        "valid_return_score": return_score,
        "valid_sharpe_score": sharpe_score,
    }


def choose_exposure_mapping(
    df: pd.DataFrame,
    probability: pd.Series,
    *,
    model_name: str,
    buy_hold_valid_metrics: dict[str, float],
) -> tuple[dict[str, float | int | str], dict[str, float], dict[str, float]]:
    def evaluate_params(params: dict[str, float | int | str]) -> tuple[dict[str, float | int | str], dict[str, float], dict[str, float]]:
        position = build_policy_position(df, probability, params)
        valid_equity = run_backtest(df, position, test_start=VALID_START, test_end=TEST_START)
        metrics = compute_metrics(valid_equity)
        scores = score_valid_metrics(metrics, buy_hold_valid_metrics)
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

    locked_policy = get_locked_policy_for_model(model_name)
    if locked_policy is None:
        mapping_params, valid_backtest_metrics, valid_selection_scores = choose_exposure_mapping(
            df,
            probability,
            model_name=model_name,
            buy_hold_valid_metrics=valid_buy_hold_metrics,
        )
        policy_selection = "valid_grid_search"
    else:
        mapping_params = locked_policy
        valid_position = build_policy_position(df, probability, mapping_params)
        valid_backtest_metrics = compute_metrics(
            run_backtest(df, valid_position, test_start=VALID_START, test_end=TEST_START)
        )
        valid_selection_scores = score_valid_metrics(valid_backtest_metrics, valid_buy_hold_metrics)
        policy_selection = "locked_multifold_min_score"

    position = build_policy_position(df, probability, mapping_params)
    test_equity = run_backtest(df, position, test_start=TEST_START)
    test_backtest_metrics = compute_metrics(test_equity)

    valid_pred = (valid_probability >= 0.5).astype(int)
    test_pred = (test_probability >= 0.5).astype(int)

    metric_row: dict[str, float | str] = {
        "model": model_name,
        "feature_count": len(feature_columns),
        "features": ", ".join(feature_columns),
        "selection_objective": SELECTION_OBJECTIVE,
        "policy_selection": policy_selection,
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
    args = parse_args()
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

    all_model_names = list(get_ml_models(n_jobs=worker_count))
    if args.models:
        unknown_models = sorted(set(args.models) - set(all_model_names))
        if unknown_models:
            raise ValueError(f"Unknown models: {', '.join(unknown_models)}")
        model_names = args.models
    else:
        model_names = all_model_names
    validate_model_feature_columns(all_model_names, df.columns)

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

    new_metrics = pd.DataFrame(metrics_rows)
    buy_hold_return = compute_metrics(run_backtest(df, buy_hold_position, test_start=TEST_START))["cumulative_return"]
    if args.merge_existing and ML_METRICS_CSV.exists():
        previous_metrics = pd.read_csv(ML_METRICS_CSV)
        previous_metrics = previous_metrics[~previous_metrics["model"].isin(model_names)]
        metrics = pd.concat([previous_metrics, new_metrics], ignore_index=True, sort=False)
    else:
        metrics = new_metrics
    if "policy_selection" in metrics.columns:
        metrics["policy_selection"] = metrics["policy_selection"].fillna("valid_grid_search")
    metrics["excess_return_vs_buy_hold"] = metrics["cumulative_return"] - buy_hold_return

    new_equity = pd.concat(equity_frames, ignore_index=True)
    if args.merge_existing and ML_EQUITY_CSV.exists():
        previous_equity = pd.read_csv(ML_EQUITY_CSV)
        previous_equity["date"] = pd.to_datetime(previous_equity["date"], format="%Y/%m/%d")
        previous_equity = previous_equity[~previous_equity["model"].isin(model_names)]
        equity = pd.concat([previous_equity, new_equity], ignore_index=True, sort=False)
    else:
        equity = new_equity

    write_metrics_csv(metrics, ML_METRICS_CSV)
    write_equity_csv(equity, ML_EQUITY_CSV)
    print(f"wrote {ML_METRICS_CSV} models={len(metrics)}")
    print(f"wrote {ML_EQUITY_CSV} rows={len(equity)}")


if __name__ == "__main__":
    main()

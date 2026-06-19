from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest, write_equity_csv, write_metrics_csv
from src.experiment_config import RETURN_WEIGHT, SHARPE_WEIGHT, TEST_START
from src.ml_dataset import FEATURE_COLUMNS
from src.ml_models import get_ml_models
from src.experiment_protocol import (
    FINAL_MODEL_TRAIN_CUTOFF,
    FORMAL_SELECTION_RULE,
    FORMAL_SELECTION_RULE_DESCRIPTION,
    INVESTMENT_END,
    VALIDATION_FOLDS,
    get_worker_count,
    load_formal_frames,
    policy_to_json,
    safe_auc,
    score_metrics,
    select_policy,
)
from src.paths import (
    ML_BASELINE_EQUITY_CSV,
    ML_BASELINE_METRICS_CSV,
    ML_BASELINE_VALIDATION_CSV,
    ensure_output_dirs,
)
from src.position_policy import build_policy_position


ML_BASELINE_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "lightgbm",
]


def list_policy_candidates() -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []

    min_positions = [0.0, 0.10, 0.20, 0.30]
    max_positions = [0.70, 0.80, 0.90, 1.0]
    smoothing_windows = [1, 3, 5]
    smoothing_methods = ["sma"]

    for min_position in min_positions:
        for max_position in max_positions:
            if min_position > max_position:
                continue
            for smoothing_window in smoothing_windows:
                for smoothing_method in smoothing_methods:
                    common = {
                        "min_position": min_position,
                        "max_position": max_position,
                        "smoothing_window": smoothing_window,
                        "smoothing_method": smoothing_method,
                    }

                    for lower_prob in [0.35, 0.40, 0.45, 0.50, 0.55]:
                        for upper_prob in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                            if lower_prob >= upper_prob:
                                continue
                            candidates.append(
                                {
                                    "mapping_type": "linear_clipped",
                                    "lower_prob": lower_prob,
                                    "upper_prob": upper_prob,
                                    **common,
                                }
                            )

                    for lower_rank in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                        for upper_rank in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
                            candidates.append(
                                {
                                    "mapping_type": "rank_linear",
                                    "lower_rank": lower_rank,
                                    "upper_rank": upper_rank,
                                    **common,
                                }
                            )

                    for center_prob in [0.40, 0.45, 0.50, 0.55, 0.60]:
                        for sharpness in [6.0, 8.0, 10.0, 12.0, 16.0, 20.0]:
                            candidates.append(
                                {
                                    "mapping_type": "sigmoid",
                                    "center_prob": center_prob,
                                    "sharpness": sharpness,
                                    **common,
                                }
                            )

                    for lower_prob in [0.35, 0.40, 0.45, 0.50, 0.55]:
                        for upper_prob in [0.55, 0.60, 0.65, 0.70, 0.75]:
                            if lower_prob >= upper_prob:
                                continue
                            for power in [1.0, 1.5, 2.0, 3.0]:
                                candidates.append(
                                    {
                                        "mapping_type": "power",
                                        "lower_prob": lower_prob,
                                        "upper_prob": upper_prob,
                                        "power": power,
                                        **common,
                                    }
                                )

                    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
                        candidates.append(
                            {
                                "mapping_type": "threshold",
                                "threshold": threshold,
                                **common,
                            }
                        )

    disallowed_mapping_types = {"relative_signal_stabilizer"}
    if any(candidate["mapping_type"] in disallowed_mapping_types for candidate in candidates):
        raise AssertionError("Panel B policy candidates must not include StableHGB mappings")
    if any("trend_guard_feature" in candidate for candidate in candidates):
        raise AssertionError("Panel B policy candidates must not include Trend Position Guard rules")

    return candidates


def evaluate_policy_candidate(
    df: pd.DataFrame,
    probability: pd.Series,
    candidate_index: int,
    params: dict[str, float | int | str],
    fold_name: str,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    buy_hold_metrics: dict[str, float],
) -> dict[str, float | int | str]:
    position = build_policy_position(df, probability, params)
    equity = run_backtest(df, position, test_start=fold_start, test_end=fold_end)
    metrics = compute_metrics(equity)
    scores = score_metrics(metrics, buy_hold_metrics)

    return {
        "fold": fold_name,
        "candidate_index": candidate_index,
        "policy_params": policy_to_json(params),
        "mapping_type": str(params["mapping_type"]),
        **scores,
        "valid_cumulative_return": metrics["cumulative_return"],
        "valid_annualized_return": metrics["annualized_return"],
        "valid_max_drawdown": metrics["max_drawdown"],
        "valid_sharpe": metrics["sharpe"],
    }


def evaluate_validation_fold(
    model_name: str,
    labeled: pd.DataFrame,
    candidates: list[dict[str, float | int | str]],
    fold_name: str,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    worker_count: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    train = labeled[labeled["target_end_date"] < fold_start]
    valid_labels = labeled[
        (labeled["date"] >= fold_start)
        & (labeled["date"] < fold_end)
        & (labeled["target_end_date"] < fold_end)
    ]
    if train.empty or valid_labels.empty:
        raise ValueError(f"{fold_name} has empty train or validation labels")

    print(
        f"[{model_name}] {fold_name}: train_labels={len(train)} "
        f"valid_labels={len(valid_labels)} candidates={len(candidates)}",
        flush=True,
    )
    model = get_ml_models(n_jobs=worker_count)[model_name]
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])
    probability = pd.Series(model.predict_proba(labeled[FEATURE_COLUMNS])[:, 1], index=labeled.index)

    buy_hold_position = pd.Series(1.0, index=labeled.index)
    buy_hold_metrics = compute_metrics(
        run_backtest(labeled, buy_hold_position, test_start=fold_start, test_end=fold_end)
    )

    result_iter = Parallel(n_jobs=worker_count, prefer="threads", return_as="generator", batch_size=16)(
        delayed(evaluate_policy_candidate)(
            labeled,
            probability,
            candidate_index,
            params,
            fold_name,
            fold_start,
            fold_end,
            buy_hold_metrics,
        )
        for candidate_index, params in enumerate(candidates)
    )
    rows = list(
        tqdm(
            result_iter,
            total=len(candidates),
            desc=f"{model_name} {fold_name}",
            unit="policy",
            dynamic_ncols=True,
        )
    )

    valid_probability = probability.loc[valid_labels.index]
    label_metrics = {
        f"{fold_name}_accuracy": float(
            accuracy_score(valid_labels["future_up_5d"], (valid_probability >= 0.5).astype(int))
        ),
        f"{fold_name}_auc": safe_auc(valid_labels["future_up_5d"], valid_probability),
    }
    for row in rows:
        row["model"] = model_name
        row["buy_hold_cumulative_return"] = buy_hold_metrics["cumulative_return"]
        row["buy_hold_sharpe"] = buy_hold_metrics["sharpe"]

    return rows, label_metrics


def evaluate_final_model(
    model_name: str,
    labeled: pd.DataFrame,
    trading: pd.DataFrame,
    params: dict[str, float | int | str],
    selected_candidate_index: int,
    validation_summary: pd.Series,
    validation_label_metrics: dict[str, float],
    worker_count: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    train = labeled[labeled["target_end_date"] < FINAL_MODEL_TRAIN_CUTOFF]
    test_labels = labeled[(labeled["date"] >= TEST_START) & (labeled["target_end_date"] <= INVESTMENT_END)]
    if train.empty or test_labels.empty:
        raise ValueError(f"{model_name} has empty final train or test labels")

    model = get_ml_models(n_jobs=worker_count)[model_name]
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])

    trading_probability = pd.Series(model.predict_proba(trading[FEATURE_COLUMNS])[:, 1], index=trading.index)
    position = build_policy_position(trading, trading_probability, params)
    equity = run_backtest(trading, position, test_start=TEST_START)
    metrics = compute_metrics(equity)

    buy_hold_position = pd.Series(1.0, index=trading.index)
    buy_hold_metrics = compute_metrics(run_backtest(trading, buy_hold_position, test_start=TEST_START))

    test_probability = pd.Series(model.predict_proba(test_labels[FEATURE_COLUMNS])[:, 1], index=test_labels.index)
    test_pred = (test_probability >= 0.5).astype(int)

    metric_row: dict[str, float | int | str] = {
        "model": model_name,
        "panel": "Panel B",
        "feature_count": len(FEATURE_COLUMNS),
        "features": ", ".join(FEATURE_COLUMNS),
        "validation_method": "rolling_folds_2022_2024",
        "policy_selection": FORMAL_SELECTION_RULE,
        "selection_rule": FORMAL_SELECTION_RULE_DESCRIPTION,
        "return_weight": RETURN_WEIGHT,
        "sharpe_weight": SHARPE_WEIGHT,
        "train_label_start": train["date"].min().strftime("%Y/%m/%d"),
        "train_label_end": train["date"].max().strftime("%Y/%m/%d"),
        "train_target_end": train["target_end_date"].max().strftime("%Y/%m/%d"),
        "test_label_start": test_labels["date"].min().strftime("%Y/%m/%d"),
        "test_label_end": test_labels["date"].max().strftime("%Y/%m/%d"),
        "test_target_end": test_labels["target_end_date"].max().strftime("%Y/%m/%d"),
        "investment_start": TEST_START.strftime("%Y/%m/%d"),
        "investment_end": INVESTMENT_END.strftime("%Y/%m/%d"),
        "trading_days": len(equity),
        "selected_candidate_index": selected_candidate_index,
        "selected_mapping_type": str(params["mapping_type"]),
        "selected_policy_params": policy_to_json(params),
        "valid_score": float(validation_summary["valid_score"]),
        "valid_min_score": float(validation_summary["valid_min_score"]),
        "valid_mean_score": float(validation_summary["valid_mean_score"]),
        "valid_mean_return": float(validation_summary["valid_mean_return"]),
        "valid_return_std": float(validation_summary["valid_return_std"]),
        "valid_mean_sharpe": float(validation_summary["valid_mean_sharpe"]),
        "valid_worst_drawdown": float(validation_summary["valid_worst_drawdown"]),
        **validation_label_metrics,
        "test_accuracy": float(accuracy_score(test_labels["future_up_5d"], test_pred)),
        "test_auc": safe_auc(test_labels["future_up_5d"], test_probability),
        "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
        "excess_return_pp_vs_buy_hold": metrics["cumulative_return"] - buy_hold_metrics["cumulative_return"],
        "relative_wealth_ratio_vs_buy_hold": (1 + metrics["cumulative_return"])
        / (1 + buy_hold_metrics["cumulative_return"]),
        **metrics,
    }

    equity.insert(0, "model", model_name)
    return metric_row, equity


def main() -> None:
    ensure_output_dirs()
    worker_count = get_worker_count()
    candidates = list_policy_candidates()
    labeled, trading = load_formal_frames()

    print(
        f"ml baseline protocol: workers={worker_count} models={len(ML_BASELINE_MODEL_NAMES)} "
        f"folds={len(VALIDATION_FOLDS)} candidates_per_fold={len(candidates)}",
        flush=True,
    )
    print(
        f"labeled: {labeled['date'].min().date()}..{labeled['date'].max().date()} "
        f"rows={len(labeled)} target_end_max={labeled['target_end_date'].max().date()}",
        flush=True,
    )
    print(
        f"trading: {trading['date'].min().date()}..{trading['date'].max().date()} rows={len(trading)}",
        flush=True,
    )

    unknown_models = sorted(set(ML_BASELINE_MODEL_NAMES) - set(get_ml_models(n_jobs=worker_count)))
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")

    metric_rows: list[dict[str, float | int | str]] = []
    equity_frames: list[pd.DataFrame] = []
    all_validation_rows: list[dict[str, float | int | str]] = []

    for model_name in ML_BASELINE_MODEL_NAMES:
        print(f"\n[{model_name}] validation search started", flush=True)
        model_validation_rows: list[dict[str, float | int | str]] = []
        validation_label_metrics: dict[str, float] = {}

        for fold_name, fold_start, fold_end in VALIDATION_FOLDS:
            fold_rows, fold_label_metrics = evaluate_validation_fold(
                model_name,
                labeled,
                candidates,
                fold_name,
                fold_start,
                fold_end,
                worker_count,
            )
            model_validation_rows.extend(fold_rows)
            validation_label_metrics.update(fold_label_metrics)

        best_index, validation_summary = select_policy(
            model_validation_rows,
            expected_candidate_indices=range(len(candidates)),
        )
        best_params = candidates[best_index]
        print(
            f"[{model_name}] selected candidate={best_index} mapping={best_params['mapping_type']} "
            f"valid_score={validation_summary['valid_score']:.6f} "
            f"mean_return={validation_summary['valid_mean_return']:.6f}",
            flush=True,
        )

        metric_row, equity = evaluate_final_model(
            model_name,
            labeled,
            trading,
            best_params,
            best_index,
            validation_summary,
            validation_label_metrics,
            worker_count,
        )
        metric_rows.append(metric_row)
        equity_frames.append(equity)
        all_validation_rows.extend(model_validation_rows)
        print(
            f"[{model_name}] final_return={metric_row['cumulative_return']:.6f} "
            f"excess_pp={metric_row['excess_return_pp_vs_buy_hold']:.6f} "
            f"max_dd={metric_row['max_drawdown']:.6f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    validation = pd.DataFrame(all_validation_rows)
    equity = pd.concat(equity_frames, ignore_index=True)

    write_metrics_csv(metrics, ML_BASELINE_METRICS_CSV)
    write_metrics_csv(validation, ML_BASELINE_VALIDATION_CSV)
    write_equity_csv(equity, ML_BASELINE_EQUITY_CSV)

    print(f"\nwrote {ML_BASELINE_METRICS_CSV} models={len(metrics)}", flush=True)
    print(f"wrote {ML_BASELINE_VALIDATION_CSV} rows={len(validation)}", flush=True)
    print(f"wrote {ML_BASELINE_EQUITY_CSV} rows={len(equity)}", flush=True)


if __name__ == "__main__":
    main()

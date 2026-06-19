from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
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
from src.stable_hgb import (
    STABLE_HGB_MODEL_NAME,
    STABLE_HGB_MODEL_PARAMS,
    STABLE_HGB_STRATEGY_NAME,
    list_stable_hgb_policy_candidates,
)
from src.paths import (
    STABLE_HGB_EQUITY_CSV,
    STABLE_HGB_METRICS_CSV,
    STABLE_HGB_VALIDATION_CSV,
    ensure_output_dirs,
)
from src.position_policy import build_policy_position


MODEL_NAME = STABLE_HGB_MODEL_NAME
STRATEGY_NAME = STABLE_HGB_STRATEGY_NAME
MODEL_PARAMS = STABLE_HGB_MODEL_PARAMS


def evaluate_validation_fold(
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
        f"[{MODEL_NAME}] {fold_name}: train_labels={len(train)} "
        f"valid_labels={len(valid_labels)} candidates={len(candidates)}",
        flush=True,
    )

    model = get_ml_models(n_jobs=worker_count)[MODEL_NAME]
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])
    probability = pd.Series(model.predict_proba(labeled[FEATURE_COLUMNS])[:, 1], index=labeled.index)

    buy_hold_position = pd.Series(1.0, index=labeled.index)
    buy_hold_metrics = compute_metrics(
        run_backtest(labeled, buy_hold_position, test_start=fold_start, test_end=fold_end)
    )

    rows: list[dict[str, float | int | str]] = []
    for candidate_index, policy in tqdm(
        list(enumerate(candidates)),
        desc=f"{MODEL_NAME} {fold_name}",
        unit="policy",
        dynamic_ncols=True,
    ):
        position = build_policy_position(labeled, probability, policy)
        metrics = compute_metrics(run_backtest(labeled, position, test_start=fold_start, test_end=fold_end))
        scores = score_metrics(metrics, buy_hold_metrics)
        rows.append(
            {
                "model": MODEL_NAME,
                "strategy": STRATEGY_NAME,
                "fold": fold_name,
                "candidate_index": candidate_index,
                "policy_params": policy_to_json(policy),
                "mapping_type": str(policy["mapping_type"]),
                **scores,
                "valid_cumulative_return": metrics["cumulative_return"],
                "valid_annualized_return": metrics["annualized_return"],
                "valid_max_drawdown": metrics["max_drawdown"],
                "valid_sharpe": metrics["sharpe"],
                "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
                "buy_hold_sharpe": buy_hold_metrics["sharpe"],
            }
        )

    valid_probability = probability.loc[valid_labels.index]
    label_metrics = {
        f"{fold_name}_accuracy": float(
            accuracy_score(valid_labels["future_up_5d"], (valid_probability >= 0.5).astype(int))
        ),
        f"{fold_name}_auc": safe_auc(valid_labels["future_up_5d"], valid_probability),
    }
    return rows, label_metrics


def evaluate_final_strategy(
    labeled: pd.DataFrame,
    trading: pd.DataFrame,
    policy: dict[str, float | int | str],
    selected_candidate_index: int,
    validation_summary: dict[str, float],
    validation_label_metrics: dict[str, float],
    worker_count: int,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    train = labeled[labeled["target_end_date"] < FINAL_MODEL_TRAIN_CUTOFF]
    test_labels = labeled[(labeled["date"] >= TEST_START) & (labeled["target_end_date"] <= INVESTMENT_END)]
    if train.empty or test_labels.empty:
        raise ValueError("Panel C has empty final train or test labels")

    model = get_ml_models(n_jobs=worker_count)[MODEL_NAME]
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])

    trading_probability = pd.Series(model.predict_proba(trading[FEATURE_COLUMNS])[:, 1], index=trading.index)
    position = build_policy_position(trading, trading_probability, policy)
    equity = run_backtest(trading, position, test_start=TEST_START)
    metrics = compute_metrics(equity)
    cost_sensitivity: dict[str, float] = {}
    for cost_bps in [5, 10, 20]:
        cost_equity = run_backtest(
            trading,
            position,
            test_start=TEST_START,
            transaction_cost_bps=float(cost_bps),
        )
        cost_metrics = compute_metrics(cost_equity)
        cost_sensitivity[f"cost_{cost_bps}bps_cumulative_return"] = cost_metrics["cumulative_return"]
        cost_sensitivity[f"cost_{cost_bps}bps_max_drawdown"] = cost_metrics["max_drawdown"]

    buy_hold_position = pd.Series(1.0, index=trading.index)
    buy_hold_metrics = compute_metrics(run_backtest(trading, buy_hold_position, test_start=TEST_START))

    test_probability = pd.Series(model.predict_proba(test_labels[FEATURE_COLUMNS])[:, 1], index=test_labels.index)
    test_pred = (test_probability >= 0.5).astype(int)

    metric_row: dict[str, float | int | str | bool] = {
        "strategy": STRATEGY_NAME,
        "model": MODEL_NAME,
        "panel": "Panel C",
        "feature_count": len(FEATURE_COLUMNS),
        "features": ", ".join(FEATURE_COLUMNS),
        "model_params": policy_to_json(MODEL_PARAMS),
        "validation_method": "rolling_folds_2022_2024",
        "policy_family": "relative_signal_stabilizer_with_trend_position_guard",
        "policy_selection": FORMAL_SELECTION_RULE,
        "selection_rule": FORMAL_SELECTION_RULE_DESCRIPTION,
        "return_weight": RETURN_WEIGHT,
        "sharpe_weight": SHARPE_WEIGHT,
        "uses_moving_average_alignment_as_model_feature": False,
        "uses_trend_position_guard": True,
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
        "selected_mapping_type": str(policy["mapping_type"]),
        "selected_policy_params": policy_to_json(policy),
        **validation_summary,
        **validation_label_metrics,
        "test_accuracy": float(accuracy_score(test_labels["future_up_5d"], test_pred)),
        "test_auc": safe_auc(test_labels["future_up_5d"], test_probability),
        "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
        "excess_return_pp_vs_buy_hold": metrics["cumulative_return"] - buy_hold_metrics["cumulative_return"],
        "relative_wealth_ratio_vs_buy_hold": (1 + metrics["cumulative_return"])
        / (1 + buy_hold_metrics["cumulative_return"]),
        **cost_sensitivity,
        **metrics,
    }

    equity.insert(0, "strategy", STRATEGY_NAME)
    equity.insert(1, "model", MODEL_NAME)
    return metric_row, equity


def main() -> None:
    ensure_output_dirs()
    worker_count = get_worker_count()
    labeled, trading = load_formal_frames()
    candidates = list_stable_hgb_policy_candidates()

    print(
        f"StableHGB protocol: workers={worker_count} model={MODEL_NAME} "
        f"folds={len(VALIDATION_FOLDS)} candidates={len(candidates)} "
        f"final_train_cutoff={FINAL_MODEL_TRAIN_CUTOFF.date()}",
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
    print("StableHGB policy search: fixed 108-candidate local grid", flush=True)

    model_names = set(get_ml_models(n_jobs=worker_count))
    if MODEL_NAME not in model_names:
        raise ValueError(f"Unknown model: {MODEL_NAME}")

    validation_rows: list[dict[str, float | int | str]] = []
    validation_label_metrics: dict[str, float] = {}
    for fold_name, fold_start, fold_end in VALIDATION_FOLDS:
        fold_rows, fold_label_metrics = evaluate_validation_fold(
            labeled,
            candidates,
            fold_name,
            fold_start,
            fold_end,
            worker_count,
        )
        validation_rows.extend(fold_rows)
        validation_label_metrics.update(fold_label_metrics)
        best_fold_row = max(fold_rows, key=lambda item: item["valid_selection_score"])
        print(
            f"[{MODEL_NAME}] {fold_name}: best_candidate={best_fold_row['candidate_index']} "
            f"return={best_fold_row['valid_cumulative_return']:.6f} "
            f"score={best_fold_row['valid_selection_score']:.6f}",
            flush=True,
        )

    best_index, validation_summary = select_policy(
        validation_rows,
        expected_candidate_indices=range(len(candidates)),
    )
    policy = candidates[best_index]
    print(
        f"[{MODEL_NAME}] selected candidate={best_index} "
        f"valid_score={validation_summary['valid_score']:.6f} "
        f"worst_drawdown={validation_summary['valid_worst_drawdown']:.6f} "
        f"policy={policy}",
        flush=True,
    )
    metric_row, equity = evaluate_final_strategy(
        labeled,
        trading,
        policy,
        best_index,
        dict(validation_summary),
        validation_label_metrics,
        worker_count,
    )

    metrics = pd.DataFrame([metric_row])
    validation = pd.DataFrame(validation_rows)

    write_metrics_csv(metrics, STABLE_HGB_METRICS_CSV)
    write_metrics_csv(validation, STABLE_HGB_VALIDATION_CSV)
    write_equity_csv(equity, STABLE_HGB_EQUITY_CSV)

    print(
        f"[{MODEL_NAME}] final_return={metric_row['cumulative_return']:.6f} "
        f"excess_pp={metric_row['excess_return_pp_vs_buy_hold']:.6f} "
        f"max_dd={metric_row['max_drawdown']:.6f}",
        flush=True,
    )
    print(f"wrote {STABLE_HGB_METRICS_CSV} rows={len(metrics)}", flush=True)
    print(f"wrote {STABLE_HGB_VALIDATION_CSV} rows={len(validation)}", flush=True)
    print(f"wrote {STABLE_HGB_EQUITY_CSV} rows={len(equity)}", flush=True)


if __name__ == "__main__":
    main()

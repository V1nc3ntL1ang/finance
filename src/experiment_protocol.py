from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiment_config import RETURN_WEIGHT, SHARPE_WEIGHT
from src.features import read_daily_csv
from src.ml_dataset import FEATURE_COLUMNS, build_ml_dataset, build_ml_feature_frame
from src.paths import DAILY_CSV


INVESTMENT_END = pd.Timestamp("2026-05-06")
FINAL_MODEL_TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
PREDICTION_HORIZON_DAYS = 5
VALID_RETURN_STD_PENALTY = 0.25
VALID_SCORE_ABSOLUTE_TOLERANCE = 0.02
FORMAL_SELECTION_RULE = "return_stability_then_worst_drawdown"
FORMAL_SELECTION_RULE_DESCRIPTION = (
    "screen candidates within 0.02 of the best "
    "mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return), "
    "then prefer the least severe worst-fold drawdown; tie break by valid score, "
    "mean valid return, mean selection score, and mean Sharpe"
)
VALIDATION_FOLDS = [
    ("valid_2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")),
    ("valid_2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
    ("valid_2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
]


def get_worker_count() -> int:
    worker_count = int(os.environ.get("FINANCE_WORKERS", os.cpu_count() or 1))
    for thread_variable in [
        "LOKY_MAX_CPU_COUNT",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(thread_variable, str(worker_count))
    return worker_count


def safe_auc(y_true: pd.Series, probability: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def score_metrics(metrics: dict[str, float], buy_hold_metrics: dict[str, float]) -> dict[str, float]:
    buy_hold_return = buy_hold_metrics["cumulative_return"]
    buy_hold_sharpe = buy_hold_metrics["sharpe"]

    return_score = metrics["cumulative_return"] / buy_hold_return if buy_hold_return != 0 else 0.0
    sharpe_score = metrics["sharpe"] / buy_hold_sharpe if buy_hold_sharpe != 0 else 0.0
    selection_score = RETURN_WEIGHT * return_score + SHARPE_WEIGHT * sharpe_score

    return {
        "valid_return_score": return_score,
        "valid_sharpe_score": sharpe_score,
        "valid_selection_score": selection_score,
    }


def policy_to_json(params: Mapping[str, float | int | str]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def add_formal_validation_score(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    scored["valid_return_std"] = scored["valid_return_std"].fillna(0.0)
    scored["valid_score"] = (
        scored["valid_mean_return"] - VALID_RETURN_STD_PENALTY * scored["valid_return_std"]
    )
    return scored


def sort_by_formal_validation_score(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    if "valid_score" not in scored.columns:
        scored = add_formal_validation_score(scored)

    best_score = float(scored["valid_score"].max())
    retained = scored[scored["valid_score"] >= best_score - VALID_SCORE_ABSOLUTE_TOLERANCE].copy()
    if retained.empty:
        retained = scored.copy()

    return retained.sort_values(
        [
            "valid_worst_drawdown",
            "valid_score",
            "valid_mean_return",
            "valid_mean_score",
            "valid_mean_sharpe",
        ],
        ascending=[False, False, False, False, False],
    )


def add_target_end_date(labeled: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if "target_end_date" in labeled.columns:
        result = labeled.copy()
        result["target_end_date"] = pd.to_datetime(result["target_end_date"])
        return result.sort_values("date").reset_index(drop=True)

    target_dates = daily[["date"]].copy()
    target_dates["target_end_date"] = target_dates["date"].shift(-PREDICTION_HORIZON_DAYS)

    merged = labeled.merge(target_dates, on="date", how="left", validate="one_to_one")
    if merged["target_end_date"].isna().any():
        raise ValueError("labeled data contains rows without a target_end_date")
    return merged.sort_values("date").reset_index(drop=True)


def load_formal_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = read_daily_csv(DAILY_CSV)
    max_date = daily["date"].max()
    if max_date < INVESTMENT_END:
        raise ValueError(f"daily data end at {max_date.date()}, expected at least {INVESTMENT_END.date()}")

    daily = daily[daily["date"] <= INVESTMENT_END].reset_index(drop=True)
    labeled = add_target_end_date(build_ml_dataset(daily), daily)
    trading = build_ml_feature_frame(daily)

    missing_features = sorted(set(FEATURE_COLUMNS) - set(labeled.columns))
    if missing_features:
        raise ValueError(f"labeled frame missing features: {', '.join(missing_features)}")
    missing_trading_features = sorted(set(FEATURE_COLUMNS) - set(trading.columns))
    if missing_trading_features:
        raise ValueError(f"trading frame missing features: {', '.join(missing_trading_features)}")
    if "ma_alignment" in FEATURE_COLUMNS:
        raise AssertionError("ma_alignment must remain a position-only feature, not a model feature")

    return labeled, trading


def select_policy(
    validation_rows: list[dict[str, float | int | str]],
    *,
    expected_candidate_indices: Collection[int] | None = None,
    expected_fold_names: Collection[str] | None = None,
) -> tuple[int, pd.Series]:
    validation = pd.DataFrame(validation_rows)
    if validation.empty:
        raise ValueError("validation_rows must not be empty")

    expected_fold_names = expected_fold_names or [fold_name for fold_name, _, _ in VALIDATION_FOLDS]
    expected_fold_set = set(expected_fold_names)
    actual_fold_set = set(validation["fold"].astype(str))
    if actual_fold_set != expected_fold_set:
        raise ValueError(
            "validation rows must contain exactly the expected folds; "
            f"expected={sorted(expected_fold_set)}, actual={sorted(actual_fold_set)}"
        )

    if expected_candidate_indices is not None:
        expected_candidate_set = {int(index) for index in expected_candidate_indices}
        actual_candidate_set = {int(index) for index in validation["candidate_index"]}
        if actual_candidate_set != expected_candidate_set:
            missing = sorted(expected_candidate_set - actual_candidate_set)[:5]
            extra = sorted(actual_candidate_set - expected_candidate_set)[:5]
            raise ValueError(
                "validation rows must contain exactly the expected candidate indices; "
                f"missing={missing}, extra={extra}"
            )

    expected_folds = len(expected_fold_set)
    fold_counts = validation.groupby("candidate_index")["fold"].nunique()
    incomplete = fold_counts[fold_counts != expected_folds]
    if not incomplete.empty:
        bad = ", ".join(str(index) for index in incomplete.index[:5])
        raise ValueError(
            f"validation rows must contain exactly {expected_folds} folds per candidate; "
            f"incomplete candidate_index values: {bad}"
        )
    duplicate_rows = validation.duplicated(subset=["candidate_index", "fold"])
    if duplicate_rows.any():
        raise ValueError("validation rows contain duplicate candidate_index/fold pairs")
    inconsistent_params = validation.groupby("candidate_index")["policy_params"].nunique(dropna=False)
    inconsistent_params = inconsistent_params[inconsistent_params != 1]
    if not inconsistent_params.empty:
        bad = ", ".join(str(index) for index in inconsistent_params.index[:5])
        raise ValueError(f"validation rows contain inconsistent policy_params for candidates: {bad}")
    inconsistent_mapping = validation.groupby("candidate_index")["mapping_type"].nunique(dropna=False)
    inconsistent_mapping = inconsistent_mapping[inconsistent_mapping != 1]
    if not inconsistent_mapping.empty:
        bad = ", ".join(str(index) for index in inconsistent_mapping.index[:5])
        raise ValueError(f"validation rows contain inconsistent mapping_type for candidates: {bad}")

    summary = (
        validation.groupby("candidate_index")
        .agg(
            policy_params=("policy_params", "first"),
            mapping_type=("mapping_type", "first"),
            valid_min_score=("valid_selection_score", "min"),
            valid_mean_score=("valid_selection_score", "mean"),
            valid_mean_return=("valid_cumulative_return", "mean"),
            valid_return_std=("valid_cumulative_return", "std"),
            valid_mean_sharpe=("valid_sharpe", "mean"),
            valid_worst_drawdown=("valid_max_drawdown", "min"),
        )
    )
    summary = sort_by_formal_validation_score(add_formal_validation_score(summary))
    best_index = int(summary.index[0])
    return best_index, summary.iloc[0]

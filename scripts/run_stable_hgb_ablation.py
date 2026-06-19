from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ml_baselines import evaluate_policy_candidate, list_policy_candidates
from src.backtest import compute_metrics, run_backtest, write_metrics_csv
from src.experiment_config import TEST_START
from src.ml_dataset import FEATURE_COLUMNS
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
    sort_by_formal_validation_score,
)
from src.stable_hgb import (
    STABLE_HGB_MODEL_NAME,
    STABLE_HGB_MODEL_PARAMS,
)
from src.paths import (
    ML_BASELINE_METRICS_CSV,
    STABLE_HGB_METRICS_CSV,
    STABLE_HGB_ABLATION_CSV,
    STABLE_HGB_ABLATION_MD,
    STABLE_HGB_ABLATION_VALIDATION_CSV,
    ensure_output_dirs,
)
from src.position_policy import build_policy_position


def make_stable_hgb_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**STABLE_HGB_MODEL_PARAMS)


def without_trend_position_guard(policy: Mapping[str, float | int | str]) -> dict[str, float | int | str]:
    result = dict(policy)
    result.pop("trend_guard_feature")
    result.pop("trend_guard_threshold")
    result.pop("trend_guard_min_position")
    return result


def with_trend_position_guard(policy: Mapping[str, float | int | str]) -> dict[str, float | int | str]:
    result = dict(policy)
    result["trend_guard_feature"] = "ma_alignment"
    result["trend_guard_threshold"] = 0.5
    result["trend_guard_min_position"] = 1.0
    return result


def load_panel_c_selected_policy() -> dict[str, float | int | str]:
    metrics = pd.read_csv(STABLE_HGB_METRICS_CSV)
    if len(metrics) != 1:
        raise ValueError(f"expected exactly one StableHGB metrics row in {STABLE_HGB_METRICS_CSV}")
    return json.loads(str(metrics.iloc[0]["selected_policy_params"]))


def summarize_validation_rows(rows: list[dict[str, float | int | str]]) -> pd.Series:
    _, summary = select_policy(rows)
    return summary


def search_standard_policy_for_stable_hgb(
    labeled: pd.DataFrame,
    worker_count: int,
) -> tuple[dict[str, float | int | str], pd.Series, pd.DataFrame]:
    candidates = list_policy_candidates()
    all_rows: list[dict[str, float | int | str]] = []

    print(
        f"[component-ablation] standard policy search: folds={len(VALIDATION_FOLDS)} "
        f"candidates_per_fold={len(candidates)} workers={worker_count}",
        flush=True,
    )
    for fold_name, fold_start, fold_end in VALIDATION_FOLDS:
        train = labeled[labeled["target_end_date"] < fold_start]
        valid_labels = labeled[
            (labeled["date"] >= fold_start)
            & (labeled["date"] < fold_end)
            & (labeled["target_end_date"] < fold_end)
        ]
        if train.empty or valid_labels.empty:
            raise ValueError(f"{fold_name} has empty train or validation labels")

        print(
            f"[component-ablation] {fold_name}: train_labels={len(train)} "
            f"valid_labels={len(valid_labels)}",
            flush=True,
        )
        model = make_stable_hgb_model()
        model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])
        probability = pd.Series(model.predict_proba(labeled[FEATURE_COLUMNS])[:, 1], index=labeled.index)

        buy_hold_position = pd.Series(1.0, index=labeled.index)
        buy_hold_metrics = compute_metrics(
            run_backtest(labeled, buy_hold_position, test_start=fold_start, test_end=fold_end)
        )

        iterator = Parallel(n_jobs=worker_count, prefer="threads", return_as="generator", batch_size=16)(
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
        fold_rows = list(
            tqdm(
                iterator,
                total=len(candidates),
                desc=f"standard policy {fold_name}",
                unit="policy",
                dynamic_ncols=True,
            )
        )
        for row in fold_rows:
            row["component_experiment"] = "standard_policy_search"
            row["ablation_row"] = "hgb_standard_policy"
            row["model"] = STABLE_HGB_MODEL_NAME
        all_rows.extend(fold_rows)

    best_index, summary = select_policy(
        all_rows,
        expected_candidate_indices=range(len(candidates)),
    )
    best_policy = candidates[best_index]
    print(
        f"[component-ablation] selected standard candidate={best_index} "
        f"mapping={best_policy['mapping_type']} valid_score={summary['valid_score']:.6f}",
        flush=True,
    )
    return best_policy, summary, pd.DataFrame(all_rows)


def evaluate_fixed_policies_on_validation(
    labeled: pd.DataFrame,
    policies: list[tuple[str, str, Mapping[str, float | int | str]]],
    worker_count: int,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []

    print(
        f"[component-ablation] fixed-policy validation: policies={len(policies)} "
        f"folds={len(VALIDATION_FOLDS)}",
        flush=True,
    )
    for fold_name, fold_start, fold_end in VALIDATION_FOLDS:
        train = labeled[labeled["target_end_date"] < fold_start]
        model = make_stable_hgb_model()
        model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])
        probability = pd.Series(model.predict_proba(labeled[FEATURE_COLUMNS])[:, 1], index=labeled.index)

        buy_hold_position = pd.Series(1.0, index=labeled.index)
        buy_hold_metrics = compute_metrics(
            run_backtest(labeled, buy_hold_position, test_start=fold_start, test_end=fold_end)
        )

        iterator = Parallel(n_jobs=worker_count, prefer="threads", return_as="generator", batch_size=1)(
            delayed(evaluate_policy_candidate)(
                labeled,
                probability,
                candidate_index,
                dict(policy),
                fold_name,
                fold_start,
                fold_end,
                buy_hold_metrics,
            )
            for candidate_index, (_, _, policy) in enumerate(policies)
        )
        fold_rows = list(
            tqdm(
                iterator,
                total=len(policies),
                desc=f"fixed policies {fold_name}",
                unit="policy",
                dynamic_ncols=True,
            )
        )
        for row, (row_id, label, _) in zip(fold_rows, policies, strict=True):
            row["component_experiment"] = "fixed_policy_validation"
            row["ablation_row"] = row_id
            row["ablation_label"] = label
            row["model"] = STABLE_HGB_MODEL_NAME
            rows.append(row)

    validation = pd.DataFrame(rows)
    summaries: dict[str, pd.Series] = {}
    for row_id in validation["ablation_row"].unique():
        row_records = validation[validation["ablation_row"] == row_id].to_dict("records")
        summaries[str(row_id)] = summarize_validation_rows(row_records)
    return summaries, validation


def evaluate_policy_on_test(
    trading: pd.DataFrame,
    test_labels: pd.DataFrame,
    trading_probability: pd.Series,
    test_probability: pd.Series,
    policy: Mapping[str, float | int | str],
    buy_hold_metrics: dict[str, float],
) -> tuple[dict[str, float], pd.Series]:
    position = build_policy_position(trading, trading_probability, policy)
    equity = run_backtest(trading, position, test_start=TEST_START)
    metrics = compute_metrics(equity)
    scores = score_metrics(metrics, buy_hold_metrics)
    test_pred = (test_probability >= 0.5).astype(int)

    metrics.update(
        {
            "test_accuracy": float(accuracy_score(test_labels["future_up_5d"], test_pred)),
            "test_auc": safe_auc(test_labels["future_up_5d"], test_probability),
            "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
            "excess_return_pp_vs_buy_hold": metrics["cumulative_return"] - buy_hold_metrics["cumulative_return"],
            "relative_wealth_ratio_vs_buy_hold": (1 + metrics["cumulative_return"])
            / (1 + buy_hold_metrics["cumulative_return"]),
            "test_return_score": scores["valid_return_score"],
            "test_sharpe_score": scores["valid_sharpe_score"],
            "test_selection_score": scores["valid_selection_score"],
            "mean_position": float(equity["position"].mean()),
            "turnover": float(equity["position"].diff().abs().fillna(0.0).sum()),
            "full_position_days": int((equity["position"] >= 0.999999).sum()),
            "zero_position_days": int((equity["position"] <= 0.000001).sum()),
        }
    )
    return metrics, position


def make_reference_rows(buy_hold_metrics: dict[str, float]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = [
        {
            "row_order": 1,
            "ablation_row": "buy_hold",
            "experiment": "Buy-and-hold reference",
            "model": "none",
            "position_rule": "constant_full_position",
            "uses_relative_signal_stabilizer": False,
            "uses_trend_position_guard": False,
            "policy_selection": "not_applicable",
            "selected_mapping_type": "constant",
            "selected_policy_params": "{}",
            "valid_score": pd.NA,
            "valid_mean_return": pd.NA,
            "valid_return_std": pd.NA,
            "test_accuracy": pd.NA,
            "test_auc": pd.NA,
            "buy_hold_cumulative_return": buy_hold_metrics["cumulative_return"],
            "excess_return_pp_vs_buy_hold": 0.0,
            "relative_wealth_ratio_vs_buy_hold": 1.0,
            **buy_hold_metrics,
        }
    ]

    ml_metrics = pd.read_csv(ML_BASELINE_METRICS_CSV)
    best_ml = sort_by_formal_validation_score(ml_metrics).iloc[0]
    rows.append(
        {
            "row_order": 2,
            "ablation_row": "best_ml_baseline",
            "experiment": "Best ML baseline reference",
            "model": str(best_ml["model"]),
            "position_rule": f"Panel B {best_ml['selected_mapping_type']}",
            "uses_relative_signal_stabilizer": False,
            "uses_trend_position_guard": False,
            "policy_selection": "selected_in_panel_b",
            "selected_mapping_type": str(best_ml["selected_mapping_type"]),
            "selected_policy_params": str(best_ml["selected_policy_params"]),
            "valid_score": float(best_ml["valid_score"]),
            "valid_mean_return": float(best_ml["valid_mean_return"]),
            "valid_return_std": float(best_ml["valid_return_std"]),
            "test_accuracy": float(best_ml["test_accuracy"]),
            "test_auc": float(best_ml["test_auc"]),
            "buy_hold_cumulative_return": float(best_ml["buy_hold_cumulative_return"]),
            "excess_return_pp_vs_buy_hold": float(best_ml["excess_return_pp_vs_buy_hold"]),
            "relative_wealth_ratio_vs_buy_hold": float(best_ml["relative_wealth_ratio_vs_buy_hold"]),
            "cumulative_return": float(best_ml["cumulative_return"]),
            "annualized_return": float(best_ml["annualized_return"]),
            "max_drawdown": float(best_ml["max_drawdown"]),
            "sharpe": float(best_ml["sharpe"]),
        }
    )
    return rows


def build_markdown_table(table: pd.DataFrame) -> str:
    display = table[
        [
            "experiment",
            "position_rule",
            "valid_score",
            "cumulative_return",
            "excess_return_pp_vs_buy_hold",
            "max_drawdown",
            "sharpe",
            "test_auc",
        ]
    ].copy()

    def fmt_percent(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.2f}%"

    def fmt_percentage_points(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.2f} pp"

    def fmt_float(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value):.4f}"

    display["valid_score"] = display["valid_score"].map(fmt_float)
    display["cumulative_return"] = display["cumulative_return"].map(fmt_percent)
    display["excess_return_pp_vs_buy_hold"] = display["excess_return_pp_vs_buy_hold"].map(fmt_percentage_points)
    display["max_drawdown"] = display["max_drawdown"].map(fmt_percent)
    display["sharpe"] = display["sharpe"].map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    display["test_auc"] = display["test_auc"].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    headers = [
        "Experiment",
        "Position rule",
        "Valid score",
        "Cumulative return",
        "Excess vs buy-hold (percentage points)",
        "Max drawdown",
        "Sharpe",
        "Test AUC",
    ]
    display.columns = headers
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(rows)


def main() -> None:
    ensure_output_dirs()
    worker_count = get_worker_count()
    labeled, trading = load_formal_frames()

    print(
        f"StableHGB component ablation: workers={worker_count} "
        f"labeled_rows={len(labeled)} trading_rows={len(trading)}",
        flush=True,
    )

    train = labeled[labeled["target_end_date"] < FINAL_MODEL_TRAIN_CUTOFF]
    test_labels = labeled[(labeled["date"] >= TEST_START) & (labeled["target_end_date"] <= INVESTMENT_END)]
    if train.empty or test_labels.empty:
        raise ValueError("component ablation has empty final train or test labels")

    buy_hold_position = pd.Series(1.0, index=trading.index)
    buy_hold_metrics = compute_metrics(run_backtest(trading, buy_hold_position, test_start=TEST_START))

    standard_policy, standard_validation_summary, search_validation = search_standard_policy_for_stable_hgb(
        labeled,
        worker_count,
    )

    stable_hgb_policy = load_panel_c_selected_policy()
    signal_stabilizer_only = without_trend_position_guard(stable_hgb_policy)
    standard_with_guard = with_trend_position_guard(standard_policy)

    fixed_policies = [
        ("hgb_standard_policy", "HGB + standard policy", standard_policy),
        (
            "hgb_standard_policy_with_trend_position_guard",
            "HGB + standard policy + Trend Position Guard",
            standard_with_guard,
        ),
        (
            "hgb_relative_signal_stabilizer_without_trend_position_guard",
            "HGB + Relative Signal Stabilizer",
            signal_stabilizer_only,
        ),
        ("stable_hgb", "StableHGB", stable_hgb_policy),
    ]
    fixed_validation_summaries, fixed_validation = evaluate_fixed_policies_on_validation(
        labeled,
        fixed_policies,
        worker_count,
    )
    fixed_validation = pd.concat([search_validation, fixed_validation], ignore_index=True)

    print("[component-ablation] fitting final StableHGB model", flush=True)
    model = make_stable_hgb_model()
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])
    trading_probability = pd.Series(model.predict_proba(trading[FEATURE_COLUMNS])[:, 1], index=trading.index)
    test_probability = pd.Series(model.predict_proba(test_labels[FEATURE_COLUMNS])[:, 1], index=test_labels.index)

    rows = make_reference_rows(buy_hold_metrics)
    hgb_rows: list[dict[str, float | int | str]] = []
    for offset, (row_id, label, policy) in enumerate(fixed_policies, start=3):
        print(f"[component-ablation] final test: {label}", flush=True)
        metrics, _ = evaluate_policy_on_test(
            trading,
            test_labels,
            trading_probability,
            test_probability,
            policy,
            buy_hold_metrics,
        )
        validation_summary = (
            standard_validation_summary if row_id == "hgb_standard_policy" else fixed_validation_summaries[row_id]
        )
        uses_relative_signal_stabilizer = str(policy["mapping_type"]) == "relative_signal_stabilizer"
        uses_trend_position_guard = "trend_guard_feature" in policy
        hgb_rows.append(
            {
                "row_order": offset,
                "ablation_row": row_id,
                "experiment": label,
                "model": STABLE_HGB_MODEL_NAME,
                "position_rule": (
                    "Relative Signal Stabilizer + Trend Position Guard"
                    if row_id == "stable_hgb"
                    else (
                        f"{policy['mapping_type']} + Trend Position Guard"
                        if uses_trend_position_guard
                        else (
                            "Relative Signal Stabilizer"
                            if uses_relative_signal_stabilizer
                            else str(policy["mapping_type"])
                        )
                    )
                ),
                "uses_relative_signal_stabilizer": uses_relative_signal_stabilizer,
                "uses_trend_position_guard": uses_trend_position_guard,
                "policy_selection": (
                    "selected_in_panel_c"
                    if row_id == "stable_hgb"
                    else (
                        "selected_from_panel_b_standard_grid_then_trend_guard_added"
                        if row_id == "hgb_standard_policy_with_trend_position_guard"
                        else (
                            "stable_hgb_policy_without_trend_position_guard"
                            if row_id == "hgb_relative_signal_stabilizer_without_trend_position_guard"
                            else "selected_from_panel_b_standard_grid"
                        )
                    )
                ),
                "selected_mapping_type": str(policy["mapping_type"]),
                "selected_policy_params": policy_to_json(policy),
                "model_params": policy_to_json(STABLE_HGB_MODEL_PARAMS),
                "validation_method": "rolling_folds_2022_2024",
                "selection_rule": FORMAL_SELECTION_RULE_DESCRIPTION,
                "valid_score": float(validation_summary["valid_score"]),
                "valid_mean_return": float(validation_summary["valid_mean_return"]),
                "valid_return_std": float(validation_summary["valid_return_std"]),
                "valid_mean_sharpe": float(validation_summary["valid_mean_sharpe"]),
                "valid_worst_drawdown": float(validation_summary["valid_worst_drawdown"]),
                **metrics,
            }
        )
    rows.extend(hgb_rows)

    table = pd.DataFrame(rows).sort_values("row_order").reset_index(drop=True)
    write_metrics_csv(table, STABLE_HGB_ABLATION_CSV)
    write_metrics_csv(fixed_validation, STABLE_HGB_ABLATION_VALIDATION_CSV)
    STABLE_HGB_ABLATION_MD.write_text(build_markdown_table(table) + "\n", encoding="utf-8")

    print(f"wrote {STABLE_HGB_ABLATION_CSV} rows={len(table)}", flush=True)
    print(f"wrote {STABLE_HGB_ABLATION_VALIDATION_CSV} rows={len(fixed_validation)}", flush=True)
    print(f"wrote {STABLE_HGB_ABLATION_MD}", flush=True)
    print(build_markdown_table(table), flush=True)

    stable_hgb = table[table["ablation_row"] == "stable_hgb"].iloc[0]
    stabilizer_only = table[
        table["ablation_row"] == "hgb_relative_signal_stabilizer_without_trend_position_guard"
    ].iloc[0]
    standard = table[table["ablation_row"] == "hgb_standard_policy"].iloc[0]
    print(
        f"summary: stable_hgb_return={stable_hgb['cumulative_return']:.6f} "
        f"vs_stabilizer_only_delta={stable_hgb['cumulative_return'] - stabilizer_only['cumulative_return']:.6f} "
        f"vs_standard_delta={stable_hgb['cumulative_return'] - standard['cumulative_return']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

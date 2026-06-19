from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stable_hgb import (
    evaluate_final_strategy,
    evaluate_validation_fold,
)
from src.experiment_config import TEST_START
from src.ml_dataset import FEATURE_COLUMNS
from src.experiment_protocol import (
    FINAL_MODEL_TRAIN_CUTOFF,
    FORMAL_SELECTION_RULE_DESCRIPTION,
    INVESTMENT_END,
    VALIDATION_FOLDS,
    get_worker_count,
    load_formal_frames,
    select_policy,
)
from src.stable_hgb import (
    STABLE_HGB_MODEL_NAME,
    STABLE_HGB_MODEL_PARAMS,
    STABLE_HGB_STRATEGY_NAME,
    list_stable_hgb_policy_candidates,
)
from src.paths import DAILY_CSV


EXPECTED_DAILY_SHA256 = "815f616c7cb99bc322d3f77f8ed4f3cc4eff099466b084116ff1c6ec689fe622"
EXPECTED_LABELED_ROWS = 2162
EXPECTED_TRADING_ROWS = 2167
EXPECTED_LABELED_START = pd.Timestamp("2017-06-01")
EXPECTED_LABELED_END = pd.Timestamp("2026-04-24")
EXPECTED_TARGET_END = pd.Timestamp("2026-05-06")
EXPECTED_TRADING_END = pd.Timestamp("2026-05-06")
EXPECTED_SELECTED_CANDIDATE_INDEX = 78
EXPECTED_RESULTS = {
    "valid_score": 0.3164202841731396,
    "valid_mean_return": 0.3385039581775708,
    "valid_return_std": 0.0883346960177248,
    "test_auc": 0.6348990683229814,
    "cumulative_return": 1.5343532904463069,
    "annualized_return": 1.0751743442601005,
    "max_drawdown": -0.0983705650725493,
    "sharpe": 3.376286206095246,
    "buy_hold_cumulative_return": 1.0921233506284724,
    "excess_return_pp_vs_buy_hold": 0.4422299398178344,
    "relative_wealth_ratio_vs_buy_hold": 1.2113785211015349,
}
RTOL = 1e-6
ATOL = 1e-8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{name} mismatch: expected {expected}, got {actual}")


def assert_close(name: str, actual: float, expected: float) -> None:
    if abs(actual - expected) > ATOL + RTOL * abs(expected):
        raise AssertionError(f"{name} mismatch: expected {expected:.12f}, got {actual:.12f}")


def check_input_contract(labeled: pd.DataFrame, trading: pd.DataFrame) -> None:
    actual_hash = file_sha256(DAILY_CSV)
    assert_equal("daily.csv sha256", actual_hash, EXPECTED_DAILY_SHA256)
    assert_equal("labeled rows", len(labeled), EXPECTED_LABELED_ROWS)
    assert_equal("trading rows", len(trading), EXPECTED_TRADING_ROWS)
    assert_equal("labeled start", labeled["date"].min(), EXPECTED_LABELED_START)
    assert_equal("labeled end", labeled["date"].max(), EXPECTED_LABELED_END)
    assert_equal("target end max", labeled["target_end_date"].max(), EXPECTED_TARGET_END)
    assert_equal("trading end", trading["date"].max(), EXPECTED_TRADING_END)
    if "ma_alignment" in FEATURE_COLUMNS:
        raise AssertionError("ma_alignment must remain position-only and must not be a model feature")


def main() -> None:
    worker_count = get_worker_count()
    labeled, trading = load_formal_frames()
    check_input_contract(labeled, trading)

    candidates = list_stable_hgb_policy_candidates()

    print(f"dataset: {DAILY_CSV}", flush=True)
    print(f"dataset hash ok: {EXPECTED_DAILY_SHA256}", flush=True)
    print(
        f"strategy={STABLE_HGB_STRATEGY_NAME} model={STABLE_HGB_MODEL_NAME} "
        f"workers={worker_count}",
        flush=True,
    )
    print(f"model_params: {STABLE_HGB_MODEL_PARAMS}", flush=True)
    print(f"policy_candidates: {len(candidates)}", flush=True)
    print(f"selection_rule: {FORMAL_SELECTION_RULE_DESCRIPTION}", flush=True)
    print(
        f"train_labels_end_before={FINAL_MODEL_TRAIN_CUTOFF.date()} "
        f"investment={TEST_START.date()}..{INVESTMENT_END.date()}",
        flush=True,
    )

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
            f"[{fold_name}] best_candidate={best_fold_row['candidate_index']} "
            f"return={best_fold_row['valid_cumulative_return']:.12f} "
            f"score={best_fold_row['valid_selection_score']:.12f}",
            flush=True,
        )

    best_index, validation_summary = select_policy(
        validation_rows,
        expected_candidate_indices=range(len(candidates)),
    )
    assert_equal("selected candidate index", best_index, EXPECTED_SELECTED_CANDIDATE_INDEX)
    policy = candidates[best_index]
    print(f"selected_candidate={best_index} policy_params={policy}", flush=True)
    metric_row, equity = evaluate_final_strategy(
        labeled,
        trading,
        policy,
        best_index,
        dict(validation_summary),
        validation_label_metrics,
        worker_count,
    )

    assert_equal("trading days", len(equity), 321)
    assert_equal(
        "metric selected candidate index",
        int(metric_row["selected_candidate_index"]),
        EXPECTED_SELECTED_CANDIDATE_INDEX,
    )
    for name, expected in EXPECTED_RESULTS.items():
        assert_close(name, float(metric_row[name]), expected)

    print("\nStableHGB reproduction passed", flush=True)
    for name in [
        "valid_score",
        "cumulative_return",
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "test_auc",
        "buy_hold_cumulative_return",
        "excess_return_pp_vs_buy_hold",
        "relative_wealth_ratio_vs_buy_hold",
    ]:
        print(f"{name}: {float(metric_row[name]):.12f}", flush=True)


if __name__ == "__main__":
    main()

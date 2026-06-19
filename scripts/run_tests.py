from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.features import read_daily_csv
from src.ml_dataset import build_ml_dataset
from src.paths import DAILY_CSV
from src.stable_hgb import get_stable_hgb_policy_params, list_stable_hgb_policy_candidates


def test_ml_dataset_split_uses_target_end_date() -> None:
    dataset = build_ml_dataset(read_daily_csv(DAILY_CSV))
    if "target_end_date" not in dataset.columns:
        raise AssertionError("ml dataset must include target_end_date")

    train = dataset[dataset["split"] == "train"]
    valid = dataset[dataset["split"] == "valid"]
    test = dataset[dataset["split"] == "test"]
    if train["target_end_date"].max() >= pd.Timestamp("2024-01-01"):
        raise AssertionError("train split contains labels ending in validation period")
    if valid["target_end_date"].min() < pd.Timestamp("2024-01-01"):
        raise AssertionError("valid split contains labels ending before validation period")
    if valid["target_end_date"].max() >= pd.Timestamp("2025-01-01"):
        raise AssertionError("valid split contains labels ending in test period")
    if test["target_end_date"].min() < pd.Timestamp("2025-01-01"):
        raise AssertionError("test split contains labels ending before test period")


def test_backtest_checks_index_and_initial_drawdown() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "ret_1d": [-0.10, 0.05, 0.02],
        }
    )
    bad_position = pd.Series([1.0, 1.0, 1.0], index=[10, 11, 12])
    try:
        run_backtest(df, bad_position, test_start=pd.Timestamp("2025-01-01"))
    except ValueError:
        pass
    else:
        raise AssertionError("backtest must reject position with a mismatched index")

    position = pd.Series([1.0, 1.0, 1.0], index=df.index)
    equity = run_backtest(df, position, test_start=pd.Timestamp("2025-01-01"), execution_lag=0)
    if float(equity["drawdown"].iloc[0]) >= 0:
        raise AssertionError("drawdown must include the initial capital point")


def test_stable_hgb_policy_candidates_are_clean() -> None:
    policy = get_stable_hgb_policy_params()
    if "exit_gap" in policy:
        raise AssertionError("StableHGB policy must not contain unused exit_gap")

    candidates = list_stable_hgb_policy_candidates()
    if len(candidates) != 108:
        raise AssertionError(f"expected 108 StableHGB candidates, got {len(candidates)}")
    if policy not in candidates:
        raise AssertionError("selected StableHGB policy must be part of the validation-search grid")


def main() -> None:
    tests = [
        test_ml_dataset_split_uses_target_end_date,
        test_backtest_checks_index_and_initial_drawdown,
        test_stable_hgb_policy_candidates_are_clean,
    ]
    for test in tests:
        test()
        print(f"passed {test.__name__}", flush=True)
    print(f"passed {len(tests)} tests", flush=True)


if __name__ == "__main__":
    main()

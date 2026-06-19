from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest, write_equity_csv, write_metrics_csv
from src.experiment_config import TEST_START
from src.paths import DAILY_FEATURES_CSV, BASELINE_EQUITY_CSV, BASELINE_METRICS_CSV, ensure_output_dirs
from src.strategies.baselines import buy_hold, ma20_momentum20_combo, ma20_timing, momentum20_timing


INVESTMENT_END = pd.Timestamp("2026-05-06")
BASELINE_STRATEGIES = OrderedDict(
    [
        ("buy_hold", buy_hold),
        ("ma20_timing", ma20_timing),
        ("momentum20_timing", momentum20_timing),
        ("ma20_momentum20_combo", ma20_momentum20_combo),
    ]
)


def load_trading_features() -> pd.DataFrame:
    df = pd.read_csv(DAILY_FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in df.columns if column != "date"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df = df.sort_values("date").reset_index(drop=True)

    max_date = df["date"].max()
    if max_date < INVESTMENT_END:
        raise ValueError(f"daily features end at {max_date.date()}, expected at least {INVESTMENT_END.date()}")

    return df[df["date"] <= INVESTMENT_END].reset_index(drop=True)


def main() -> None:
    ensure_output_dirs()
    df = load_trading_features()
    print(
        f"baseline protocol: investment={TEST_START.date()}..{INVESTMENT_END.date()} "
        f"rows={len(df)}",
        flush=True,
    )

    metrics_rows: list[dict[str, float | str]] = []
    equity_frames: list[pd.DataFrame] = []

    for strategy_name, strategy_func in BASELINE_STRATEGIES.items():
        position = strategy_func(df)
        equity = run_backtest(df, position, test_start=TEST_START)
        metrics_rows.append({"strategy": strategy_name, **compute_metrics(equity)})
        equity.insert(0, "strategy", strategy_name)
        equity_frames.append(equity)
        print(f"evaluated {strategy_name}: days={len(equity)}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    buy_hold_return = float(metrics.loc[metrics["strategy"] == "buy_hold", "cumulative_return"].iloc[0])
    metrics["excess_return_pp_vs_buy_hold"] = metrics["cumulative_return"] - buy_hold_return
    metrics["relative_wealth_ratio_vs_buy_hold"] = (1 + metrics["cumulative_return"]) / (1 + buy_hold_return)

    write_metrics_csv(metrics, BASELINE_METRICS_CSV)
    write_equity_csv(pd.concat(equity_frames, ignore_index=True), BASELINE_EQUITY_CSV)
    print(f"wrote {BASELINE_METRICS_CSV} strategies={len(metrics)}", flush=True)
    print(f"wrote {BASELINE_EQUITY_CSV} rows={sum(len(frame) for frame in equity_frames)}", flush=True)


if __name__ == "__main__":
    main()

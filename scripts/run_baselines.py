from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest, write_equity_csv, write_metrics_csv
from src.paths import BASELINE_EQUITY_CSV, BASELINE_METRICS_CSV, ML_DATASET_CSV, ensure_output_dirs
from src.strategies.baselines import BASELINE_STRATEGIES


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ML_DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in df.columns if column not in {"date", "split"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    ensure_output_dirs()
    df = load_dataset()
    metrics_rows: list[dict[str, float | str]] = []
    equity_frames: list[pd.DataFrame] = []

    for strategy_name, strategy_func in BASELINE_STRATEGIES.items():
        position = strategy_func(df)
        equity = run_backtest(df, position)
        metrics_rows.append({"strategy": strategy_name, **compute_metrics(equity)})
        equity.insert(0, "strategy", strategy_name)
        equity_frames.append(equity)

    metrics = pd.DataFrame(metrics_rows)
    buy_hold_return = float(metrics.loc[metrics["strategy"] == "buy_hold", "cumulative_return"].iloc[0])
    metrics["excess_return_vs_buy_hold"] = metrics["cumulative_return"] - buy_hold_return

    write_metrics_csv(metrics, BASELINE_METRICS_CSV)
    write_equity_csv(pd.concat(equity_frames, ignore_index=True), BASELINE_EQUITY_CSV)
    print(f"wrote {BASELINE_METRICS_CSV} strategies={len(metrics)}")
    print(f"wrote {BASELINE_EQUITY_CSV} rows={sum(len(frame) for frame in equity_frames)}")


if __name__ == "__main__":
    main()

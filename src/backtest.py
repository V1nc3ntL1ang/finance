from __future__ import annotations

from pathlib import Path

import pandas as pd


INITIAL_CAPITAL = 100000.0
TEST_START = pd.Timestamp("2025-01-01")


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    *,
    test_start: pd.Timestamp = TEST_START,
    test_end: pd.Timestamp | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    execution_lag: int = 1,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    if len(df) != len(position):
        raise ValueError("df and position must have the same length")
    if not position.index.equals(df.index):
        raise ValueError("df and position must have the same index")
    if execution_lag < 0:
        raise ValueError("execution_lag must be non-negative")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if df["date"].duplicated().any():
        raise ValueError("df date column must not contain duplicates")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("df date column must be sorted in increasing order")

    result = df[["date", "ret_1d"]].copy()
    result["position"] = position.astype(float).clip(lower=0.0, upper=1.0).fillna(0.0).to_numpy()
    result["executed_position"] = result["position"].shift(execution_lag).fillna(0.0)
    result["turnover"] = result["executed_position"].diff().abs().fillna(result["executed_position"].abs())
    cost_rate = transaction_cost_bps / 10000.0
    result["strategy_return"] = result["executed_position"] * result["ret_1d"].fillna(0.0) - result["turnover"] * cost_rate
    mask = result["date"] >= test_start
    if test_end is not None:
        mask &= result["date"] < test_end
    result = result[mask].copy()

    result["equity"] = initial_capital * (1 + result["strategy_return"]).cumprod()
    running_max = pd.concat(
        [
            pd.Series([initial_capital], index=[-1], dtype=float),
            result["equity"],
        ]
    ).cummax().iloc[1:]
    result["drawdown"] = result["equity"] / running_max.to_numpy() - 1
    return result.drop(columns=["ret_1d"]).reset_index(drop=True)


def compute_metrics(backtest_df: pd.DataFrame, *, initial_capital: float = INITIAL_CAPITAL) -> dict[str, float]:
    if backtest_df.empty:
        raise ValueError("backtest_df is empty")

    daily_returns = backtest_df["strategy_return"]
    final_equity = float(backtest_df["equity"].iloc[-1])
    periods = len(backtest_df)
    cumulative_return = final_equity / initial_capital - 1
    annualized_return = (final_equity / initial_capital) ** (252 / periods) - 1
    max_drawdown = float(backtest_df["drawdown"].min())

    std_return = float(daily_returns.std(ddof=1))
    sharpe = float(daily_returns.mean() / std_return * (252**0.5)) if std_return > 0 else 0.0

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def write_metrics_csv(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False, lineterminator="\r\n")


def write_equity_csv(equity: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = equity.copy()
    output["date"] = output["date"].dt.strftime("%Y/%m/%d")
    output.to_csv(path, index=False, lineterminator="\r\n")

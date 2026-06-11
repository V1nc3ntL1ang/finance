from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import BASELINE_EQUITY_CSV, BASELINE_METRICS_CSV, BASELINE_PLOTS_DIR, ensure_output_dirs

INITIAL_CAPITAL = 100000.0

TRADABLE_BASELINES = [
    "buy_hold",
    "ma20_timing",
    "momentum20_timing",
    "ma20_momentum20_combo",
]


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = pd.read_csv(BASELINE_EQUITY_CSV)
    equity["date"] = pd.to_datetime(equity["date"], format="%Y/%m/%d")
    metrics = pd.read_csv(BASELINE_METRICS_CSV)
    return equity, metrics


def plot_equity_curves(equity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))

    for strategy in TRADABLE_BASELINES:
        strategy_df = equity[equity["strategy"] == strategy]
        ax.plot(strategy_df["date"], strategy_df["equity"] / INITIAL_CAPITAL, label=strategy)

    ax.set_title("Equity Curves of Tradable Baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity Multiple")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = BASELINE_PLOTS_DIR / "equity_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_drawdowns(equity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))

    for strategy in TRADABLE_BASELINES:
        strategy_df = equity[equity["strategy"] == strategy]
        ax.plot(strategy_df["date"], strategy_df["drawdown"], label=strategy)

    ax.set_title("Drawdowns of Tradable Baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = BASELINE_PLOTS_DIR / "drawdowns.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_theoretical_optimal(equity: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    tradable_metrics = metrics[metrics["strategy"].isin(TRADABLE_BASELINES)].copy()
    best_tradable = tradable_metrics.sort_values("sharpe", ascending=False).iloc[0]["strategy"]
    strategies = ["buy_hold", best_tradable, "theoretical_optimal"]

    fig, ax = plt.subplots(figsize=(11, 6))

    for strategy in strategies:
        strategy_df = equity[equity["strategy"] == strategy]
        style = "--" if strategy == "theoretical_optimal" else "-"
        ax.plot(strategy_df["date"], strategy_df["equity"] / INITIAL_CAPITAL, linestyle=style, label=strategy)

    ax.set_title("Theoretical Optimal Upper Bound")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity Multiple")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = BASELINE_PLOTS_DIR / "theoretical_optimal.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    ensure_output_dirs()
    equity, metrics = load_outputs()

    paths = [
        plot_equity_curves(equity),
        plot_drawdowns(equity),
        plot_theoretical_optimal(equity, metrics),
    ]

    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

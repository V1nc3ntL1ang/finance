from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import BASELINE_EQUITY_CSV, ML_EQUITY_CSV, ML_METRICS_CSV, ML_PLOTS_DIR, ensure_output_dirs

INITIAL_CAPITAL = 100000.0


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_equity = pd.read_csv(BASELINE_EQUITY_CSV)
    baseline_equity["date"] = pd.to_datetime(baseline_equity["date"], format="%Y/%m/%d")

    ml_equity = pd.read_csv(ML_EQUITY_CSV)
    ml_equity["date"] = pd.to_datetime(ml_equity["date"], format="%Y/%m/%d")

    ml_metrics = pd.read_csv(ML_METRICS_CSV)
    return baseline_equity, ml_equity, ml_metrics


def plot_ml_equity_curves(ml_equity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))

    for model in ml_equity["model"].unique():
        model_df = ml_equity[ml_equity["model"] == model]
        ax.plot(model_df["date"], model_df["equity"] / INITIAL_CAPITAL, label=model)

    ax.set_title("Equity Curves of ML Baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity Multiple")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = ML_PLOTS_DIR / "equity_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ml_drawdowns(ml_equity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))

    for model in ml_equity["model"].unique():
        model_df = ml_equity[ml_equity["model"] == model]
        ax.plot(model_df["date"], model_df["drawdown"], label=model)

    ax.set_title("Drawdowns of ML Baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = ML_PLOTS_DIR / "drawdowns.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_ml_vs_baselines(baseline_equity: pd.DataFrame, ml_equity: pd.DataFrame, ml_metrics: pd.DataFrame) -> Path:
    best_model = ml_metrics.sort_values("sharpe", ascending=False).iloc[0]["model"]
    max_ml_date = ml_equity["date"].max()

    fig, ax = plt.subplots(figsize=(11, 6))

    for strategy in ["buy_hold", "ma20_timing"]:
        strategy_df = baseline_equity[
            (baseline_equity["strategy"] == strategy) & (baseline_equity["date"] <= max_ml_date)
        ]
        ax.plot(strategy_df["date"], strategy_df["equity"] / INITIAL_CAPITAL, label=strategy)

    best_model_df = ml_equity[ml_equity["model"] == best_model]
    ax.plot(best_model_df["date"], best_model_df["equity"] / INITIAL_CAPITAL, label=f"best_ml: {best_model}")

    ax.set_title("Best ML Baseline vs Key Tradable Baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity Multiple")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = ML_PLOTS_DIR / "best_ml_vs_baselines.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    ensure_output_dirs()
    baseline_equity, ml_equity, ml_metrics = load_outputs()

    paths = [
        plot_ml_equity_curves(ml_equity),
        plot_ml_drawdowns(ml_equity),
        plot_best_ml_vs_baselines(baseline_equity, ml_equity, ml_metrics),
    ]

    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

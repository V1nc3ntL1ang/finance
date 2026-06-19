from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import INITIAL_CAPITAL
from src.experiment_protocol import sort_by_formal_validation_score
from src.paths import (
    BASELINE_EQUITY_CSV,
    BASELINE_METRICS_CSV,
    BASELINES_PLOT,
    DRAWDOWN_REFERENCE_PLOT,
    ML_BASELINES_PLOT,
    ML_BASELINE_EQUITY_CSV,
    ML_BASELINE_METRICS_CSV,
    RISK_RETURN_PLOT,
    STABLE_HGB_EQUITY_CSV,
    STABLE_HGB_METRICS_CSV,
    STABLE_HGB_POSITION_PLOT,
    STABLE_HGB_REFERENCE_PLOT,
    ensure_output_dirs,
)


BASELINE_LABELS = {
    "buy_hold": "Buy and hold",
    "ma20_timing": "MA20 timing",
    "momentum20_timing": "20-day momentum timing",
    "ma20_momentum20_combo": "MA20 and momentum combo",
}
ML_LABELS = {
    "logistic_regression": "Logistic reg.",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "hist_gradient_boosting": "Histogram GB",
    "lightgbm": "LightGBM",
}
STABLE_HGB_LABEL = "StableHGB"

# Matplotlib/Seaborn-like Tableau palette for a cleaner default visual style.
PALETTE = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
    "grid": "#e6e6e6",
    "text": "#262626",
    "reference": "#bdbdbd",
}
PLOT_STYLE = {
    "axes.edgecolor": PALETTE["text"],
    "axes.labelcolor": PALETTE["text"],
    "axes.titlecolor": PALETTE["text"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
}
plt.rcParams.update(PLOT_STYLE)

BASELINE_STYLES = {
    "buy_hold": {"color": PALETTE["blue"], "linestyle": "-", "linewidth": 1.45},
    "ma20_timing": {"color": PALETTE["orange"], "linestyle": "-", "linewidth": 1.35},
    "momentum20_timing": {"color": PALETTE["green"], "linestyle": "-", "linewidth": 1.35},
    "ma20_momentum20_combo": {"color": PALETTE["purple"], "linestyle": "-", "linewidth": 1.35},
}
ML_STYLES = {
    "logistic_regression": {"color": PALETTE["orange"], "linestyle": "-", "linewidth": 1.35},
    "random_forest": {"color": PALETTE["green"], "linestyle": "-", "linewidth": 1.35},
    "gradient_boosting": {"color": PALETTE["purple"], "linestyle": "-", "linewidth": 1.35},
    "hist_gradient_boosting": {"color": PALETTE["pink"], "linestyle": "-", "linewidth": 1.35},
    "lightgbm": {"color": PALETTE["brown"], "linestyle": "-", "linewidth": 1.45},
}
GROUP_COLORS = {
    "Baseline": PALETTE["gray"],
    "ML baseline": PALETTE["blue"],
    "StableHGB": PALETTE["red"],
}
GROUP_MARKERS = {
    "Baseline": "o",
    "ML baseline": "s",
    "StableHGB": "D",
}
STABLE_HGB_STYLE = {"color": PALETTE["red"], "linestyle": "-", "linewidth": 1.8}
REFERENCE_LINE = {"color": PALETTE["reference"], "linewidth": 0.8, "alpha": 0.75}
POSITION_STYLE = {"color": PALETTE["cyan"], "linewidth": 1.2, "alpha": 0.72}
SCATTER_LABEL_OFFSETS = {
    "StableHGB": (18, 0),
    "Buy and hold": (10, -2),
    "Logistic reg.": (10, -2),
    "Random forest": (13, -4),
    "Histogram GB": (13, -2),
    "Gradient boosting": (10, -2),
    "LightGBM": (13, -2),
    "MA20 timing": (10, -2),
    "MA20 and momentum combo": (10, -2),
    "20-day momentum timing": (-120, -4),
}


def read_equity_csv(path: Path) -> pd.DataFrame:
    equity = pd.read_csv(path)
    equity["date"] = pd.to_datetime(equity["date"], format="%Y/%m/%d")
    return equity.sort_values("date").reset_index(drop=True)


def format_return(value: float) -> str:
    return f"{value:.2%}"


def label_with_return(name: str, cumulative_return: float) -> str:
    return f"{name} ({format_return(cumulative_return)})"


def finish_equity_plot(fig: plt.Figure, ax: plt.Axes, title: str, path: Path, ncol: int = 1) -> Path:
    ax.axhline(1.0, **REFERENCE_LINE)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity multiple")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=False, ncol=ncol)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def finish_percent_plot(fig: plt.Figure, ax: plt.Axes, title: str, path: Path) -> Path:
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_all_baselines(equity: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ordered_strategies = [strategy for strategy in BASELINE_LABELS if strategy in set(equity["strategy"])]

    for strategy in ordered_strategies:
        strategy_df = equity[equity["strategy"] == strategy]
        metric_row = metrics[metrics["strategy"] == strategy].iloc[0]
        label = label_with_return(BASELINE_LABELS[strategy], float(metric_row["cumulative_return"]))
        ax.plot(
            strategy_df["date"],
            strategy_df["equity"] / INITIAL_CAPITAL,
            label=label,
            **BASELINE_STYLES[strategy],
        )

    return finish_equity_plot(fig, ax, "Panel A: Tradable Baseline Strategies", BASELINES_PLOT)


def plot_all_ml_baselines(
    equity: pd.DataFrame,
    metrics: pd.DataFrame,
    baseline_equity: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ordered_models = [model for model in ML_LABELS if model in set(equity["model"])]

    buy_hold_equity = baseline_equity[baseline_equity["strategy"] == "buy_hold"]
    buy_hold_metric = baseline_metrics[baseline_metrics["strategy"] == "buy_hold"].iloc[0]
    ax.plot(
        buy_hold_equity["date"],
        buy_hold_equity["equity"] / INITIAL_CAPITAL,
        label=label_with_return("Buy and hold", float(buy_hold_metric["cumulative_return"])),
        **BASELINE_STYLES["buy_hold"],
    )

    for model in ordered_models:
        model_df = equity[equity["model"] == model]
        metric_row = metrics[metrics["model"] == model].iloc[0]
        label = label_with_return(ML_LABELS[model], float(metric_row["cumulative_return"]))
        ax.plot(
            model_df["date"],
            model_df["equity"] / INITIAL_CAPITAL,
            label=label,
            **ML_STYLES[model],
        )

    return finish_equity_plot(fig, ax, "Panel B: Machine-Learning Baselines vs Buy-and-Hold", ML_BASELINES_PLOT, ncol=2)


def select_best_by_return(metrics: pd.DataFrame, name_column: str) -> str:
    return str(metrics.sort_values("cumulative_return", ascending=False, kind="mergesort").iloc[0][name_column])


def select_best_by_validation(metrics: pd.DataFrame, name_column: str) -> str:
    return str(sort_by_formal_validation_score(metrics).iloc[0][name_column])


def plot_final_comparison(
    baseline_equity: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    ml_equity: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    stable_hgb_equity: pd.DataFrame,
    stable_hgb_metrics: pd.DataFrame,
) -> Path:
    best_baseline = select_best_by_return(baseline_metrics, "strategy")
    best_ml = select_best_by_validation(ml_metrics, "model")

    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    best_baseline_equity = baseline_equity[baseline_equity["strategy"] == best_baseline]
    best_baseline_metric = baseline_metrics[baseline_metrics["strategy"] == best_baseline].iloc[0]
    ax.plot(
        best_baseline_equity["date"],
        best_baseline_equity["equity"] / INITIAL_CAPITAL,
        label=label_with_return(
            f"Best baseline: {BASELINE_LABELS.get(best_baseline, best_baseline)}",
            float(best_baseline_metric["cumulative_return"]),
        ),
        **BASELINE_STYLES.get(best_baseline, BASELINE_STYLES["buy_hold"]),
    )

    best_ml_equity = ml_equity[ml_equity["model"] == best_ml]
    best_ml_metric = ml_metrics[ml_metrics["model"] == best_ml].iloc[0]
    ax.plot(
        best_ml_equity["date"],
        best_ml_equity["equity"] / INITIAL_CAPITAL,
        label=label_with_return(
            f"Validation-selected ML baseline: {ML_LABELS.get(best_ml, best_ml)}",
            float(best_ml_metric["cumulative_return"]),
        ),
        **ML_STYLES.get(best_ml, ML_STYLES["hist_gradient_boosting"]),
    )

    stable_hgb_metric = stable_hgb_metrics.iloc[0]
    ax.plot(
        stable_hgb_equity["date"],
        stable_hgb_equity["equity"] / INITIAL_CAPITAL,
        label=label_with_return(STABLE_HGB_LABEL, float(stable_hgb_metric["cumulative_return"])),
        **STABLE_HGB_STYLE,
    )

    return finish_equity_plot(fig, ax, "Panel C: StableHGB vs Best References", STABLE_HGB_REFERENCE_PLOT)


def plot_drawdown_references(
    baseline_equity: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    ml_equity: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    stable_hgb_equity: pd.DataFrame,
) -> Path:
    best_baseline = select_best_by_return(baseline_metrics, "strategy")
    best_ml = select_best_by_validation(ml_metrics, "model")

    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    reference_lines = [
        (
            baseline_equity[baseline_equity["strategy"] == best_baseline],
            f"Best baseline: {BASELINE_LABELS.get(best_baseline, best_baseline)}",
            BASELINE_STYLES.get(best_baseline, BASELINE_STYLES["buy_hold"]),
        ),
        (
            ml_equity[ml_equity["model"] == best_ml],
            f"Validation-selected ML baseline: {ML_LABELS.get(best_ml, best_ml)}",
            ML_STYLES.get(best_ml, ML_STYLES["hist_gradient_boosting"]),
        ),
        (
            stable_hgb_equity,
            STABLE_HGB_LABEL,
            STABLE_HGB_STYLE,
        ),
    ]

    for equity, label, line_style in reference_lines:
        ax.plot(
            equity["date"],
            equity["drawdown"],
            label=label,
            **line_style,
        )

    ax.axhline(0.0, **REFERENCE_LINE)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", frameon=False)
    fig.autofmt_xdate()
    return finish_percent_plot(fig, ax, "Drawdown: StableHGB vs Best References", DRAWDOWN_REFERENCE_PLOT)


def build_risk_return_frame(
    baseline_metrics: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    stable_hgb_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for _, row in baseline_metrics.iterrows():
        strategy = str(row["strategy"])
        rows.append(
            {
                "group": "Baseline",
                "name": BASELINE_LABELS.get(strategy, strategy),
                "cumulative_return": float(row["cumulative_return"]),
                "max_drawdown_magnitude": abs(float(row["max_drawdown"])),
                "sharpe": float(row["sharpe"]),
            }
        )

    for _, row in ml_metrics.iterrows():
        model = str(row["model"])
        rows.append(
            {
                "group": "ML baseline",
                "name": ML_LABELS.get(model, model),
                "cumulative_return": float(row["cumulative_return"]),
                "max_drawdown_magnitude": abs(float(row["max_drawdown"])),
                "sharpe": float(row["sharpe"]),
            }
        )

    stable_row = stable_hgb_metrics.iloc[0]
    rows.append(
        {
            "group": "StableHGB",
            "name": STABLE_HGB_LABEL,
            "cumulative_return": float(stable_row["cumulative_return"]),
            "max_drawdown_magnitude": abs(float(stable_row["max_drawdown"])),
            "sharpe": float(stable_row["sharpe"]),
        }
    )
    return pd.DataFrame(rows)


def plot_risk_return_scatter(
    baseline_metrics: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    stable_hgb_metrics: pd.DataFrame,
) -> Path:
    scatter = build_risk_return_frame(baseline_metrics, ml_metrics, stable_hgb_metrics)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for group, group_df in scatter.groupby("group", sort=False):
        sizes = 45 + group_df["sharpe"].clip(lower=0) * 38
        ax.scatter(
            group_df["max_drawdown_magnitude"],
            group_df["cumulative_return"],
            s=sizes,
            label=group,
            color=GROUP_COLORS[group],
            marker=GROUP_MARKERS[group],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.9,
        )
        for _, row in group_df.iterrows():
            offset = SCATTER_LABEL_OFFSETS.get(str(row["name"]), (5, 5))
            ax.annotate(
                str(row["name"]),
                (float(row["max_drawdown_magnitude"]), float(row["cumulative_return"])),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5 if row["name"] != STABLE_HGB_LABEL else 10,
                fontweight="bold" if row["name"] == STABLE_HGB_LABEL else "normal",
            )

    ax.set_xlabel("Maximum drawdown magnitude")
    ax.set_ylabel("Cumulative return")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Risk-Return Profile")
    ax.margins(x=0.05, y=0.12)
    fig.tight_layout()
    fig.savefig(RISK_RETURN_PLOT, dpi=220)
    plt.close(fig)
    return RISK_RETURN_PLOT


def plot_stable_hgb_position(stable_hgb_equity: pd.DataFrame) -> Path:
    fig, ax_equity = plt.subplots(figsize=(11.5, 6.4))
    equity_multiple = stable_hgb_equity["equity"] / INITIAL_CAPITAL
    ax_equity.plot(
        stable_hgb_equity["date"],
        equity_multiple,
        label="Equity multiple",
        **STABLE_HGB_STYLE,
    )
    ax_equity.set_ylabel("Equity multiple")
    ax_equity.set_xlabel("Date")
    ax_equity.grid(True, alpha=0.22)

    ax_position = ax_equity.twinx()
    ax_position.step(
        stable_hgb_equity["date"],
        stable_hgb_equity["executed_position"],
        where="post",
        label="Executed position",
        **POSITION_STYLE,
    )
    ax_position.fill_between(
        stable_hgb_equity["date"],
        0,
        stable_hgb_equity["executed_position"],
        step="post",
        color=PALETTE["cyan"],
        alpha=0.10,
    )
    ax_position.set_ylim(-0.04, 1.04)
    ax_position.set_ylabel("Position")

    lines, labels = ax_equity.get_legend_handles_labels()
    position_lines, position_labels = ax_position.get_legend_handles_labels()
    ax_equity.legend(lines + position_lines, labels + position_labels, loc="lower right", frameon=False)
    ax_equity.set_title("StableHGB Position and Equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(STABLE_HGB_POSITION_PLOT, dpi=220)
    plt.close(fig)
    return STABLE_HGB_POSITION_PLOT


def main() -> None:
    ensure_output_dirs()

    baseline_equity = read_equity_csv(BASELINE_EQUITY_CSV)
    baseline_metrics = pd.read_csv(BASELINE_METRICS_CSV)
    ml_equity = read_equity_csv(ML_BASELINE_EQUITY_CSV)
    ml_metrics = pd.read_csv(ML_BASELINE_METRICS_CSV)
    stable_hgb_equity = read_equity_csv(STABLE_HGB_EQUITY_CSV)
    stable_hgb_metrics = pd.read_csv(STABLE_HGB_METRICS_CSV)

    paths = [
        plot_all_baselines(baseline_equity, baseline_metrics),
        plot_all_ml_baselines(ml_equity, ml_metrics, baseline_equity, baseline_metrics),
        plot_final_comparison(
            baseline_equity,
            baseline_metrics,
            ml_equity,
            ml_metrics,
            stable_hgb_equity,
            stable_hgb_metrics,
        ),
        plot_drawdown_references(
            baseline_equity,
            baseline_metrics,
            ml_equity,
            ml_metrics,
            stable_hgb_equity,
        ),
        plot_risk_return_scatter(baseline_metrics, ml_metrics, stable_hgb_metrics),
        plot_stable_hgb_position(stable_hgb_equity),
    ]

    for path in paths:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

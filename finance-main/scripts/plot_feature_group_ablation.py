from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import ABLATION_PLOTS_DIR, FEATURE_GROUP_ABLATION_METRICS_CSV, ensure_output_dirs

GROUP_ORDER = ["market", "regime", "momentum", "composite"]


def plot_model_lines(metrics: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))

    for model in metrics["model"].unique():
        model_df = metrics[metrics["model"] == model].copy()
        model_df["feature_group"] = pd.Categorical(model_df["feature_group"], categories=GROUP_ORDER, ordered=True)
        model_df = model_df.sort_values("feature_group")
        ax.plot(model_df["feature_group"].astype(str), model_df[metric], marker="o", label=model)

    ax.set_title(title)
    ax.set_xlabel("Feature Group")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = ABLATION_PLOTS_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    ensure_output_dirs()
    metrics = pd.read_csv(FEATURE_GROUP_ABLATION_METRICS_CSV)

    paths = [
        plot_model_lines(
            metrics,
            "sharpe",
            "Feature Group Ablation: Test Sharpe",
            "Test Sharpe",
            "feature_group_sharpe.png",
        ),
        plot_model_lines(
            metrics,
            "cumulative_return",
            "Feature Group Ablation: Test Return",
            "Cumulative Return",
            "feature_group_return.png",
        ),
    ]

    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

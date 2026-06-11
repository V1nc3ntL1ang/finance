from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import ABLATION_PLOTS_DIR, ML_ABLATION_METRICS_CSV, ensure_output_dirs


def plot_metric(metrics: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str) -> Path:
    pivot = metrics.pivot(index="model", columns="feature_set", values=metric)
    pivot = pivot.sort_values(by="composite", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Feature set")
    fig.xticks = None
    fig.tight_layout()

    path = ABLATION_PLOTS_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    ensure_output_dirs()
    metrics = pd.read_csv(ML_ABLATION_METRICS_CSV)

    paths = [
        plot_metric(metrics, "sharpe", "Ablation: Test Sharpe by Feature Set", "Test Sharpe", "ml_sharpe.png"),
        plot_metric(
            metrics,
            "cumulative_return",
            "Ablation: Test Return by Feature Set",
            "Cumulative Return",
            "ml_return.png",
        ),
    ]

    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

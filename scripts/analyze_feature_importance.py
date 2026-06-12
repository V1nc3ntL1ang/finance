from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_dataset import FEATURE_COLUMNS
from src.paths import FEATURE_IMPORTANCE_CSV, FEATURE_IMPORTANCE_PLOT, ML_DATASET_CSV, ensure_output_dirs


RANDOM_STATE = 42


def get_worker_count() -> int:
    return int(os.environ.get("FINANCE_WORKERS", "-1"))


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ML_DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in df.columns if column not in {"date", "split"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df.sort_values("date").reset_index(drop=True)


def compute_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=get_worker_count(),
    )
    model.fit(train[FEATURE_COLUMNS], train["future_up_5d"])

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def write_feature_importance(importance: pd.DataFrame) -> None:
    FEATURE_IMPORTANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 20) -> None:
    top_features = importance.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Random Forest importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    FEATURE_IMPORTANCE_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_IMPORTANCE_PLOT, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    df = load_dataset()
    importance = compute_feature_importance(df)
    write_feature_importance(importance)
    plot_feature_importance(importance)

    print(f"wrote {FEATURE_IMPORTANCE_CSV} features={len(importance)}")
    print(f"wrote {FEATURE_IMPORTANCE_PLOT}")


if __name__ == "__main__":
    main()

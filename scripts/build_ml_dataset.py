from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import read_daily_csv
from src.ml_dataset import build_ml_dataset, write_ml_dataset
from src.paths import DAILY_CSV, ML_DATASET_CSV


def main() -> None:
    daily = read_daily_csv(DAILY_CSV)
    dataset = build_ml_dataset(daily)
    write_ml_dataset(dataset, ML_DATASET_CSV)

    split_counts = dataset["split"].value_counts().to_dict()
    print(f"wrote {ML_DATASET_CSV} rows={len(dataset)} split_counts={split_counts}")


if __name__ == "__main__":
    main()

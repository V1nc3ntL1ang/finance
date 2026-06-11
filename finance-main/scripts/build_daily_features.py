from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import add_minimal_features, read_daily_csv, write_feature_csv
from src.paths import DAILY_CSV, DAILY_FEATURES_CSV


def main() -> None:
    rows = read_daily_csv(DAILY_CSV)
    featured = add_minimal_features(rows)
    write_feature_csv(featured, DAILY_FEATURES_CSV)
    print(f"wrote {DAILY_FEATURES_CSV} rows={len(featured)}")


if __name__ == "__main__":
    main()

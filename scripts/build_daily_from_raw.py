from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DAILY_CSV, RAW_DAILY_TXT


DAILY_ROW_PATTERN = re.compile(r"^\d{4}/\d{2}/\d{2}\s+")
DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def parse_raw_daily_text(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with path.open("r", encoding="gb18030", errors="replace") as handle:
        for line in handle:
            if not DAILY_ROW_PATTERN.match(line):
                continue
            parts = line.strip().split()
            if len(parts) != len(DAILY_COLUMNS):
                raise ValueError(f"Unexpected daily row format: {line.rstrip()}")
            rows.append(parts)

    if not rows:
        raise ValueError(f"No daily rows found in {path}")

    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    daily["date"] = pd.to_datetime(daily["date"], format="%Y/%m/%d")
    for column in ["open", "high", "low", "close", "amount"]:
        daily[column] = pd.to_numeric(daily[column])
    daily["volume"] = pd.to_numeric(daily["volume"]).round().astype("Int64")
    return daily.sort_values("date").reset_index(drop=True)


def write_daily_csv(daily: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = daily.copy()
    output["date"] = output["date"].dt.strftime("%Y/%m/%d")
    for column in ["open", "high", "low", "close", "amount"]:
        output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    output.to_csv(path, index=False, lineterminator="\r\n")


def main() -> None:
    daily = parse_raw_daily_text(RAW_DAILY_TXT)
    write_daily_csv(daily, DAILY_CSV)
    print(
        f"wrote {DAILY_CSV} rows={len(daily)} "
        f"date_range={daily['date'].min().date()}..{daily['date'].max().date()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

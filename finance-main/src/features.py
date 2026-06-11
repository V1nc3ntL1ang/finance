from __future__ import annotations

from pathlib import Path

import pandas as pd


DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
FEATURE_COLUMNS = DAILY_COLUMNS + ["ret_1d", "ma20", "momentum20"]


def read_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")

    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    return df.sort_values("date").reset_index(drop=True)


def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["ret_1d"] = featured["close"].pct_change()
    featured["ma20"] = featured["close"].rolling(window=20).mean()
    featured["momentum20"] = featured["close"].pct_change(periods=20)
    return featured[FEATURE_COLUMNS]


def write_feature_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output = df.copy()
    output["date"] = output["date"].dt.strftime("%Y/%m/%d")
    output["volume"] = output["volume"].round().astype("Int64")
    output.to_csv(path, index=False, float_format="%.10g")


def read_feature_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "date"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df.sort_values("date").reset_index(drop=True)

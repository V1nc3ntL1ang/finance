from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "momentum5",
    "momentum10",
    "momentum20",
    "volatility5",
    "volatility20",
    "range_pct",
    "volume_change20",
    "close_vs_ma20",
    "close_vs_ma60",
    "ma20_slope",
    "ma60_slope",
    "drawdown_from_20d_high",
    "drawdown_from_60d_high",
    "volume_zscore20",
    "volatility20_rank",
    "ma_alignment",
    "trend_strength20",
    "trend_strength60",
    "ret5_over_vol20",
    "ret20_over_vol20",
    "rebound_from_20d_low",
    "rebound_from_60d_low",
    "range_zscore20",
]

TARGET_COLUMNS = ["future_ret_5d", "future_up_5d"]

OUTPUT_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
    "split",
]


def build_ml_dataset(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()

    df["ret_1d"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["momentum5"] = df["close"].pct_change(periods=5)
    df["momentum10"] = df["close"].pct_change(periods=10)
    df["momentum20"] = df["close"].pct_change(periods=20)
    df["volatility5"] = df["ret_1d"].rolling(window=5).std()
    df["volatility20"] = df["ret_1d"].rolling(window=20).std()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change20"] = df["volume"].pct_change(periods=20)
    df["close_vs_ma20"] = df["close"] / df["ma20"] - 1
    df["close_vs_ma60"] = df["close"] / df["ma60"] - 1
    df["ma20_slope"] = df["ma20"] / df["ma20"].shift(5) - 1
    df["ma60_slope"] = df["ma60"] / df["ma60"].shift(5) - 1
    df["drawdown_from_20d_high"] = df["close"] / df["close"].rolling(window=20).max() - 1
    df["drawdown_from_60d_high"] = df["close"] / df["close"].rolling(window=60).max() - 1
    volume_mean20 = df["volume"].rolling(window=20).mean()
    volume_std20 = df["volume"].rolling(window=20).std()
    df["volume_zscore20"] = (df["volume"] - volume_mean20) / volume_std20
    df["volatility20_rank"] = df["volatility20"].rolling(window=252, min_periods=60).rank(pct=True)
    df["ma_alignment"] = ((df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(int)
    df["trend_strength20"] = df["ma20_slope"] / df["volatility20"]
    df["trend_strength60"] = df["ma60_slope"] / df["volatility20"]
    df["ret5_over_vol20"] = df["momentum5"] / df["volatility20"]
    df["ret20_over_vol20"] = df["momentum20"] / df["volatility20"]
    df["rebound_from_20d_low"] = df["close"] / df["close"].rolling(window=20).min() - 1
    df["rebound_from_60d_low"] = df["close"] / df["close"].rolling(window=60).min() - 1
    range_mean20 = df["range_pct"].rolling(window=20).mean()
    range_std20 = df["range_pct"].rolling(window=20).std()
    df["range_zscore20"] = (df["range_pct"] - range_mean20) / range_std20

    df["future_ret_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["future_up_5d"] = (df["future_ret_5d"] > 0).astype(int)

    df["split"] = "train"
    df.loc[df["date"] >= pd.Timestamp("2024-01-01"), "split"] = "valid"
    df.loc[df["date"] >= pd.Timestamp("2025-01-01"), "split"] = "test"

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + ["future_ret_5d"]).reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


def write_ml_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output = df.copy()
    output["date"] = output["date"].dt.strftime("%Y/%m/%d")
    output["volume"] = output["volume"].round().astype("Int64")
    for column in ["open", "high", "low", "close", "amount"]:
        output[column] = output[column].map(lambda value: f"{value:.2f}")
    for column in FEATURE_COLUMNS + ["future_ret_5d"]:
        output[column] = output[column].map(lambda value: f"{value:.10g}")
    output.to_csv(path, index=False)

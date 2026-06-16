from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "volatility5",
    "volatility20_rank",
    "rebound_from_60d_low",
    "volatility20",
    "macd_histogram",
    "bb_width",
    "trend_strength60",
    "macd_signal",
    "ma60",
    "rebound_from_20d_low",
    "bb_lower",
    "ma60_slope",
    "ma10",
    "ma20_slope",
    "close_vs_ma60",
    "ma5",
    "bb_upper",
    "trend_strength20",
    "momentum20",
    "volume_zscore20",
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
    "ret_1d",
    "ma20",
    "ma_alignment",
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
    df["momentum20"] = df["close"].pct_change(periods=20)
    df["volatility5"] = df["ret_1d"].rolling(window=5).std()
    df["volatility20"] = df["ret_1d"].rolling(window=20).std()
    df["close_vs_ma60"] = df["close"] / df["ma60"] - 1
    df["ma20_slope"] = df["ma20"] / df["ma20"].shift(5) - 1
    df["ma60_slope"] = df["ma60"] / df["ma60"].shift(5) - 1
    volume_mean20 = df["volume"].rolling(window=20).mean()
    volume_std20 = df["volume"].rolling(window=20).std()
    df["volume_zscore20"] = (df["volume"] - volume_mean20) / volume_std20
    df["volatility20_rank"] = df["volatility20"].rolling(window=252, min_periods=60).rank(pct=True)
    df["ma_alignment"] = ((df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(int)
    df["trend_strength20"] = df["ma20_slope"] / df["volatility20"]
    df["trend_strength60"] = df["ma60_slope"] / df["volatility20"]
    df["rebound_from_20d_low"] = df["close"] / df["close"].rolling(window=20).min() - 1
    df["rebound_from_60d_low"] = df["close"] / df["close"].rolling(window=60).min() - 1
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    bb_middle = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = bb_middle + 2 * bb_std
    df["bb_lower"] = bb_middle - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_middle

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
    output["ma_alignment"] = output["ma_alignment"].astype("Int64")
    for column in ["open", "high", "low", "close", "amount"]:
        output[column] = output[column].map(lambda value: f"{value:.2f}")
    for column in ["ret_1d", "ma20", *FEATURE_COLUMNS, "future_ret_5d"]:
        output[column] = output[column].map(lambda value: f"{value:.10g}")
    output.to_csv(path, index=False)

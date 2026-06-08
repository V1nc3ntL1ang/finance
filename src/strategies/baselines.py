from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import pandas as pd


Strategy = Callable[[pd.DataFrame], pd.Series]


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def ma20_timing(df: pd.DataFrame) -> pd.Series:
    return (df["close"] > df["ma20"]).astype(float)


def momentum20_timing(df: pd.DataFrame) -> pd.Series:
    return (df["momentum20"] > 0).astype(float)


def ma20_momentum20_combo(df: pd.DataFrame) -> pd.Series:
    return (ma20_timing(df) + momentum20_timing(df)) / 2


def theoretical_optimal(df: pd.DataFrame) -> pd.Series:
    return (df["ret_1d"].shift(-1) > 0).astype(float)


BASELINE_STRATEGIES: OrderedDict[str, Strategy] = OrderedDict(
    [
        ("buy_hold", buy_hold),
        ("ma20_timing", ma20_timing),
        ("momentum20_timing", momentum20_timing),
        ("ma20_momentum20_combo", ma20_momentum20_combo),
        ("theoretical_optimal", theoretical_optimal),
    ]
)

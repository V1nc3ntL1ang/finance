from __future__ import annotations

from bisect import bisect_right, insort
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd


LOWER_PROB_CANDIDATES = [0.35, 0.40, 0.45, 0.50]
UPPER_PROB_CANDIDATES = [0.50, 0.55, 0.60, 0.65]
LOWER_RANK_CANDIDATES = [0.20, 0.30, 0.40]
UPPER_RANK_CANDIDATES = [0.60, 0.70, 0.80]
CENTER_PROB_CANDIDATES = [0.45, 0.50, 0.55]
SHARPNESS_CANDIDATES = [8.0, 12.0, 16.0]
MIN_POSITION_CANDIDATES = [0.0, 0.25, 0.50]
MAX_POSITION_CANDIDATES = [0.75, 1.0]
SMOOTHING_WINDOW_CANDIDATES = [1, 3, 5]


def iter_position_policy_candidates() -> Iterator[dict[str, float | int | str]]:
    for min_position in MIN_POSITION_CANDIDATES:
        for max_position in MAX_POSITION_CANDIDATES:
            if min_position > max_position:
                continue
            for smoothing_window in SMOOTHING_WINDOW_CANDIDATES:
                yield from _iter_linear_candidates(min_position, max_position, smoothing_window)
                yield from _iter_rank_candidates(min_position, max_position, smoothing_window)
                yield from _iter_sigmoid_candidates(min_position, max_position, smoothing_window)


def _iter_linear_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
) -> Iterator[dict[str, float | int | str]]:
    for lower_prob in LOWER_PROB_CANDIDATES:
        for upper_prob in UPPER_PROB_CANDIDATES:
            if lower_prob > upper_prob:
                continue
            yield {
                "mapping_type": "linear_clipped",
                "lower_prob": lower_prob,
                "upper_prob": upper_prob,
                "min_position": min_position,
                "max_position": max_position,
                "smoothing_window": smoothing_window,
            }


def _iter_rank_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
) -> Iterator[dict[str, float | int | str]]:
    for lower_rank in LOWER_RANK_CANDIDATES:
        for upper_rank in UPPER_RANK_CANDIDATES:
            if lower_rank > upper_rank:
                continue
            yield {
                "mapping_type": "rank_linear",
                "lower_rank": lower_rank,
                "upper_rank": upper_rank,
                "min_position": min_position,
                "max_position": max_position,
                "smoothing_window": smoothing_window,
            }


def _iter_sigmoid_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
) -> Iterator[dict[str, float | int | str]]:
    for center_prob in CENTER_PROB_CANDIDATES:
        for sharpness in SHARPNESS_CANDIDATES:
            yield {
                "mapping_type": "sigmoid",
                "center_prob": center_prob,
                "sharpness": sharpness,
                "min_position": min_position,
                "max_position": max_position,
                "smoothing_window": smoothing_window,
            }


def build_position(signal: pd.Series, params: Mapping[str, float | int | str]) -> pd.Series:
    mapping_type = str(params["mapping_type"])
    min_position = float(params["min_position"])
    max_position = float(params["max_position"])
    smoothing_window = int(params["smoothing_window"])

    if min_position > max_position:
        raise ValueError("min_position must be <= max_position")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")

    if mapping_type == "linear_clipped":
        raw_position = _linear_clipped(
            signal,
            lower=float(params["lower_prob"]),
            upper=float(params["upper_prob"]),
            min_position=min_position,
            max_position=max_position,
        )
    elif mapping_type == "rank_linear":
        rank_score = expanding_percentile_rank(signal)
        raw_position = _linear_clipped(
            rank_score,
            lower=float(params["lower_rank"]),
            upper=float(params["upper_rank"]),
            min_position=min_position,
            max_position=max_position,
        )
    elif mapping_type == "sigmoid":
        raw_position = _sigmoid(
            signal,
            center=float(params["center_prob"]),
            sharpness=float(params["sharpness"]),
            min_position=min_position,
            max_position=max_position,
        )
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")

    return smooth_position(raw_position, smoothing_window)


def expanding_percentile_rank(signal: pd.Series) -> pd.Series:
    sorted_values: list[float] = []
    scores: list[float] = []

    for value in signal.astype(float):
        if pd.isna(value):
            scores.append(0.5)
            continue

        value_float = float(value)
        insort(sorted_values, value_float)
        scores.append(bisect_right(sorted_values, value_float) / len(sorted_values))

    return pd.Series(scores, index=signal.index, dtype=float)


def smooth_position(position: pd.Series, smoothing_window: int) -> pd.Series:
    smoothed = position.rolling(window=smoothing_window, min_periods=1).mean()
    return smoothed.clip(lower=0.0, upper=1.0).fillna(0.0)


def _linear_clipped(
    score: pd.Series,
    *,
    lower: float,
    upper: float,
    min_position: float,
    max_position: float,
) -> pd.Series:
    if lower > upper:
        raise ValueError("lower must be <= upper")

    if lower == upper:
        position = pd.Series(max_position, index=score.index, dtype=float)
        position[score < lower] = min_position
        return position

    scaled = ((score.astype(float) - lower) / (upper - lower)).clip(lower=0.0, upper=1.0)
    return min_position + scaled * (max_position - min_position)


def _sigmoid(
    signal: pd.Series,
    *,
    center: float,
    sharpness: float,
    min_position: float,
    max_position: float,
) -> pd.Series:
    scaled = 1.0 / (1.0 + np.exp(-sharpness * (signal.astype(float) - center)))
    return min_position + scaled * (max_position - min_position)

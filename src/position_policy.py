from __future__ import annotations

import random
from bisect import bisect_right, insort
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

LOWER_PROB_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50]
UPPER_PROB_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70]
LOWER_RANK_CANDIDATES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
UPPER_RANK_CANDIDATES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
CENTER_PROB_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60]
SHARPNESS_CANDIDATES = [4.0, 8.0, 12.0, 16.0, 20.0]
POWER_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 3.0]
THRESHOLD_CANDIDATES = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
MIN_POSITION_CANDIDATES = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
MAX_POSITION_CANDIDATES = [0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
SMOOTHING_WINDOW_CANDIDATES = [1, 2, 3, 5, 7, 10]
SMOOTHING_METHOD_CANDIDATES = ["sma", "ewma"]


def iter_position_policy_candidates() -> Iterator[dict[str, float | int | str]]:
    for min_position in MIN_POSITION_CANDIDATES:
        for max_position in MAX_POSITION_CANDIDATES:
            if min_position > max_position:
                continue
            for smoothing_window in SMOOTHING_WINDOW_CANDIDATES:
                for smoothing_method in SMOOTHING_METHOD_CANDIDATES:
                    yield from _iter_linear_candidates(min_position, max_position, smoothing_window, smoothing_method)
                    yield from _iter_rank_candidates(min_position, max_position, smoothing_window, smoothing_method)
                    yield from _iter_sigmoid_candidates(min_position, max_position, smoothing_window, smoothing_method)
                    yield from _iter_power_candidates(min_position, max_position, smoothing_window, smoothing_method)
                    yield from _iter_threshold_candidates(min_position, max_position, smoothing_window, smoothing_method)


def list_position_policy_candidates() -> list[dict[str, float | int | str]]:
    return list(iter_position_policy_candidates())


def sample_position_policy_candidates(
    candidates: list[dict[str, float | int | str]],
    sample_per_mapping: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    if sample_per_mapping <= 0:
        raise ValueError("sample_per_mapping must be positive")

    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["mapping_type"]), []).append(candidate)

    rng = random.Random(seed)
    sampled: list[dict[str, float | int | str]] = []
    for mapping_type in sorted(grouped):
        group = grouped[mapping_type]
        if len(group) <= sample_per_mapping:
            sampled.extend(group)
        else:
            sampled.extend(rng.sample(group, sample_per_mapping))
    return sampled


def _iter_linear_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
    smoothing_method: str,
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
                "smoothing_method": smoothing_method,
            }


def _iter_rank_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
    smoothing_method: str,
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
                "smoothing_method": smoothing_method,
            }


def _iter_sigmoid_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
    smoothing_method: str,
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
                "smoothing_method": smoothing_method,
            }


def _iter_power_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
    smoothing_method: str,
) -> Iterator[dict[str, float | int | str]]:
    for lower_prob in LOWER_PROB_CANDIDATES:
        for upper_prob in UPPER_PROB_CANDIDATES:
            if lower_prob >= upper_prob:
                continue
            for power in POWER_CANDIDATES:
                yield {
                    "mapping_type": "power",
                    "lower_prob": lower_prob,
                    "upper_prob": upper_prob,
                    "power": power,
                    "min_position": min_position,
                    "max_position": max_position,
                    "smoothing_window": smoothing_window,
                    "smoothing_method": smoothing_method,
                }


def _iter_threshold_candidates(
    min_position: float,
    max_position: float,
    smoothing_window: int,
    smoothing_method: str,
) -> Iterator[dict[str, float | int | str]]:
    for threshold in THRESHOLD_CANDIDATES:
        yield {
            "mapping_type": "threshold",
            "threshold": threshold,
            "min_position": min_position,
            "max_position": max_position,
            "smoothing_window": smoothing_window,
            "smoothing_method": smoothing_method,
        }


def build_position(signal: pd.Series, params: Mapping[str, float | int | str]) -> pd.Series:
    mapping_type = str(params["mapping_type"])
    min_position = float(params["min_position"])
    max_position = float(params["max_position"])
    smoothing_window = int(params["smoothing_window"])
    smoothing_method = str(params.get("smoothing_method", "sma"))

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
    elif mapping_type == "power":
        raw_position = _power_mapping(
            signal,
            lower=float(params["lower_prob"]),
            upper=float(params["upper_prob"]),
            power=float(params["power"]),
            min_position=min_position,
            max_position=max_position,
        )
    elif mapping_type == "threshold":
        raw_position = _threshold_mapping(
            signal,
            threshold=float(params["threshold"]),
            min_position=min_position,
            max_position=max_position,
        )
    elif mapping_type == "relative_signal_stabilizer":
        raw_position = _relative_signal_stabilizer(
            signal,
            entry_rank=float(params["entry_rank"]),
            exit_rank=float(params["exit_rank"]),
            confirm_days=int(params["confirm_days"]),
            min_position=min_position,
            max_position=max_position,
        )
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")

    return smooth_position(raw_position, smoothing_window, smoothing_method)


def build_policy_position(
    df: pd.DataFrame,
    signal: pd.Series,
    params: Mapping[str, float | int | str],
) -> pd.Series:
    position = build_position(signal, params)
    if "trend_guard_feature" not in params:
        return position

    guard_feature_name = str(params["trend_guard_feature"])
    if guard_feature_name not in df.columns:
        raise ValueError(f"Trend Position Guard feature missing from dataset: {guard_feature_name}")

    guard_threshold = float(params["trend_guard_threshold"])
    guard_min_position = float(params["trend_guard_min_position"])
    trend_mask = df[guard_feature_name].astype(float) > guard_threshold
    return position.mask(trend_mask & (position < guard_min_position), guard_min_position).clip(lower=0.0, upper=1.0)


def expanding_percentile_rank(signal: pd.Series) -> pd.Series:
    sorted_values: list[float] = []
    scores: list[float] = []

    for value in signal.astype(float):
        if pd.isna(value):
            scores.append(0.5)
            continue

        value_float = float(value)
        if not sorted_values:
            scores.append(0.5)
        else:
            scores.append(bisect_right(sorted_values, value_float) / len(sorted_values))
        insort(sorted_values, value_float)

    return pd.Series(scores, index=signal.index, dtype=float)


def smooth_position(position: pd.Series, smoothing_window: int, method: str = "sma") -> pd.Series:
    if method == "sma":
        smoothed = position.rolling(window=smoothing_window, min_periods=1).mean()
    elif method == "ewma":
        smoothed = position.ewm(span=smoothing_window, adjust=False).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
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


def _power_mapping(
    signal: pd.Series,
    *,
    lower: float,
    upper: float,
    power: float,
    min_position: float,
    max_position: float,
) -> pd.Series:
    if lower >= upper:
        raise ValueError("lower must be < upper")

    normalized = ((signal.astype(float) - lower) / (upper - lower)).clip(lower=0.0, upper=1.0)
    scaled = normalized ** power
    return min_position + scaled * (max_position - min_position)


def _threshold_mapping(
    signal: pd.Series,
    *,
    threshold: float,
    min_position: float,
    max_position: float,
) -> pd.Series:
    return pd.Series(
        np.where(signal.astype(float) >= threshold, max_position, min_position),
        index=signal.index,
        dtype=float,
    )


def _relative_signal_stabilizer(
    signal: pd.Series,
    *,
    entry_rank: float,
    exit_rank: float,
    confirm_days: int,
    min_position: float,
    max_position: float,
) -> pd.Series:
    if confirm_days < 1:
        raise ValueError("confirm_days must be >= 1")
    if exit_rank > entry_rank:
        raise ValueError("exit_rank must be <= entry_rank")

    rank_score = expanding_percentile_rank(signal)
    state = min_position
    entry_count = 0
    exit_count = 0
    values: list[float] = []

    for value in rank_score:
        entry_count = entry_count + 1 if value >= entry_rank else 0
        exit_count = exit_count + 1 if value <= exit_rank else 0
        if state < max_position and entry_count >= confirm_days:
            state = max_position
        elif state > min_position and exit_count >= confirm_days:
            state = min_position
        values.append(state)

    return pd.Series(values, index=rank_score.index, dtype=float)

from __future__ import annotations

from collections.abc import Iterator


STABLE_HGB_MODEL_NAME = "stable_hgb"
STABLE_HGB_STRATEGY_NAME = "StableHGB"

STABLE_HGB_MODEL_PARAMS = {
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 70,
    "random_state": 42,
}

STABLE_HGB_POLICY_PARAMS = {
    "mapping_type": "relative_signal_stabilizer",
    "entry_rank": 0.55,
    "exit_rank": 0.45,
    "confirm_days": 2,
    "min_position": 0.0,
    "max_position": 1.0,
    "smoothing_window": 1,
    "smoothing_method": "sma",
    "trend_guard_feature": "ma_alignment",
    "trend_guard_threshold": 0.5,
    "trend_guard_min_position": 1.0,
}

STABLE_HGB_ENTRY_RANK_CANDIDATES = [0.500, 0.525, 0.550]
STABLE_HGB_EXIT_RANK_CANDIDATES = [0.450, 0.475, 0.500]
STABLE_HGB_CONFIRM_DAYS_CANDIDATES = [1, 2, 3]
STABLE_HGB_TREND_GUARD_MIN_POSITION_CANDIDATES = [0.8, 1.0]
STABLE_HGB_SMOOTHING_WINDOW_CANDIDATES = [1, 3]


def get_stable_hgb_policy_params() -> dict[str, float | int | str]:
    return STABLE_HGB_POLICY_PARAMS.copy()


def iter_stable_hgb_policy_candidates() -> Iterator[dict[str, float | int | str]]:
    for entry_rank in STABLE_HGB_ENTRY_RANK_CANDIDATES:
        for exit_rank in STABLE_HGB_EXIT_RANK_CANDIDATES:
            if exit_rank > entry_rank:
                continue
            for confirm_days in STABLE_HGB_CONFIRM_DAYS_CANDIDATES:
                for trend_guard_min_position in STABLE_HGB_TREND_GUARD_MIN_POSITION_CANDIDATES:
                    for smoothing_window in STABLE_HGB_SMOOTHING_WINDOW_CANDIDATES:
                        candidate = get_stable_hgb_policy_params()
                        candidate.update(
                            {
                                "entry_rank": entry_rank,
                                "exit_rank": exit_rank,
                                "confirm_days": confirm_days,
                                "trend_guard_min_position": trend_guard_min_position,
                                "smoothing_window": smoothing_window,
                            }
                        )
                        yield candidate


def list_stable_hgb_policy_candidates() -> list[dict[str, float | int | str]]:
    return list(iter_stable_hgb_policy_candidates())

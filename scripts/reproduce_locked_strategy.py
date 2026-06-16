from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest
from src.experiment_config import TEST_START, VALID_START
from src.ml_models import get_ml_models
from src.model_features import MODEL_FEATURE_COLUMNS
from src.paths import ML_DATASET_CSV
from src.position_policy import build_policy_position, get_locked_policy_for_model


MODEL_NAME = "hist_gradient_boosting_alignment_confirmation"
EXPECTED_DATA_SHA256 = "a00bf3d3918e5230e0a4f7906a6b0233d4af93a19837e53ae9973434348fb51e"
EXPECTED_ROWS = 2162
EXPECTED_SPLIT_COUNTS = {"train": 1604, "valid": 242, "test": 316}
EXPECTED_RESULTS = {
    "valid_selection_score": 2.2938318985,
    "valid_cumulative_return": 0.2651946392,
    "valid_sharpe": 0.7619028051,
    "test_auc": 0.6535811335,
    "cumulative_return": 1.632727292589,
    "annualized_return": 1.164015573318,
    "max_drawdown": -0.092726987482,
    "sharpe": 3.535132344229,
    "buy_hold_cumulative_return": 0.992770066097,
    "excess_return_vs_buy_hold": 0.639957226492,
}
TOLERANCE = 1e-10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset() -> pd.DataFrame:
    actual_hash = file_sha256(ML_DATASET_CSV)
    if actual_hash != EXPECTED_DATA_SHA256:
        raise AssertionError(
            f"dataset hash mismatch: expected {EXPECTED_DATA_SHA256}, got {actual_hash}"
        )

    df = pd.read_csv(ML_DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in df.columns if column not in {"date", "split"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) != EXPECTED_ROWS:
        raise AssertionError(f"dataset row mismatch: expected {EXPECTED_ROWS}, got {len(df)}")
    if "ma_alignment" not in df.columns:
        raise AssertionError("dataset missing required policy feature: ma_alignment")

    split_counts = df["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"split count mismatch: expected {EXPECTED_SPLIT_COUNTS}, got {split_counts}")

    return df


def safe_auc(y_true: pd.Series, probability: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def score_valid_metrics(
    metrics: dict[str, float],
    buy_hold_metrics: dict[str, float],
) -> dict[str, float]:
    return_score = metrics["cumulative_return"] / buy_hold_metrics["cumulative_return"]
    sharpe_score = metrics["sharpe"] / buy_hold_metrics["sharpe"]
    return {
        "valid_selection_score": 0.5 * return_score + 0.5 * sharpe_score,
        "valid_return_score": return_score,
        "valid_sharpe_score": sharpe_score,
    }


def assert_close(name: str, actual: float, expected: float) -> None:
    if abs(actual - expected) > TOLERANCE:
        raise AssertionError(f"{name} mismatch: expected {expected:.12f}, got {actual:.12f}")


def main() -> None:
    print(f"dataset: {ML_DATASET_CSV}", flush=True)
    df = load_dataset()
    print(f"dataset hash ok: {EXPECTED_DATA_SHA256}", flush=True)

    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    test = df[df["split"] == "test"]
    print(
        "splits: "
        f"train={train['date'].min().date()}..{train['date'].max().date()} ({len(train)}) "
        f"valid={valid['date'].min().date()}..{valid['date'].max().date()} ({len(valid)}) "
        f"test={test['date'].min().date()}..{test['date'].max().date()} ({len(test)})",
        flush=True,
    )

    feature_columns = MODEL_FEATURE_COLUMNS[MODEL_NAME]
    if "ma_alignment" in feature_columns:
        raise AssertionError("ma_alignment must not be used as a model feature")
    print(f"model: {MODEL_NAME}", flush=True)
    print(f"model feature_count={len(feature_columns)} uses_ma_alignment=False", flush=True)

    model = get_ml_models(n_jobs=1)[MODEL_NAME]
    model.fit(train[feature_columns], train["future_up_5d"])
    probability = pd.Series(model.predict_proba(df[feature_columns])[:, 1], index=df.index)

    policy = get_locked_policy_for_model(MODEL_NAME)
    if policy is None:
        raise AssertionError(f"missing locked policy for model: {MODEL_NAME}")
    print(f"policy: {policy}", flush=True)

    position = build_policy_position(df, probability, policy)
    buy_hold_position = pd.Series(1.0, index=df.index)

    valid_buy_hold = compute_metrics(run_backtest(df, buy_hold_position, test_start=VALID_START, test_end=TEST_START))
    test_buy_hold = compute_metrics(run_backtest(df, buy_hold_position, test_start=TEST_START))
    valid_metrics = compute_metrics(run_backtest(df, position, test_start=VALID_START, test_end=TEST_START))
    test_metrics = compute_metrics(run_backtest(df, position, test_start=TEST_START))
    valid_scores = score_valid_metrics(valid_metrics, valid_buy_hold)

    valid_probability = probability.loc[valid.index]
    test_probability = probability.loc[test.index]
    results = {
        **valid_scores,
        "valid_accuracy": float(accuracy_score(valid["future_up_5d"], (valid_probability >= 0.5).astype(int))),
        "test_accuracy": float(accuracy_score(test["future_up_5d"], (test_probability >= 0.5).astype(int))),
        "valid_auc": safe_auc(valid["future_up_5d"], valid_probability),
        "test_auc": safe_auc(test["future_up_5d"], test_probability),
        "valid_cumulative_return": valid_metrics["cumulative_return"],
        "valid_max_drawdown": valid_metrics["max_drawdown"],
        "valid_sharpe": valid_metrics["sharpe"],
        **test_metrics,
        "buy_hold_cumulative_return": test_buy_hold["cumulative_return"],
        "excess_return_vs_buy_hold": test_metrics["cumulative_return"] - test_buy_hold["cumulative_return"],
    }

    for name, expected in EXPECTED_RESULTS.items():
        assert_close(name, float(results[name]), expected)

    print("\nreproduction passed", flush=True)
    for name in [
        "cumulative_return",
        "max_drawdown",
        "sharpe",
        "test_auc",
        "buy_hold_cumulative_return",
        "excess_return_vs_buy_hold",
    ]:
        print(f"{name}: {results[name]:.12f}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ml_baselines import (
    RETURN_WEIGHT,
    SELECTION_OBJECTIVE,
    SHARPE_WEIGHT,
    TEST_START,
    VALID_START,
    load_dataset,
    safe_auc,
)
from src.model_config import get_model_config
from src.backtest import compute_metrics, run_backtest, write_metrics_csv
from src.feature_groups import BASIC_FEATURES, FEATURE_GROUPS
from src.ml_models import get_ml_models
from src.paths import ML_ABLATION_METRICS_CSV, ensure_output_dirs
from src.position_policy import build_position


FEATURE_SETS = {
    "market": BASIC_FEATURES,
    "composite": FEATURE_GROUPS["composite"],
}


def get_worker_count() -> int:
    return int(os.environ.get("FINANCE_WORKERS", "-1"))


def evaluate_combination(
    feature_set_name: str,
    feature_columns: list[str],
    model_name: str,
    df: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_buy_hold_metrics: dict[str, float],
    buy_hold_return: float,
) -> dict[str, float | int | str]:
    model = get_ml_models(n_jobs=1)[model_name]
    model.fit(train[feature_columns], y_train)

    probability = pd.Series(model.predict_proba(df[feature_columns])[:, 1], index=df.index)
    valid_probability = probability.loc[valid.index]
    test_probability = probability.loc[test.index]

    # 使用预定义配置，跳过参数搜索
    mapping_params = get_model_config(model_name)
    
    position = build_position(probability, mapping_params)
    
    # 计算验证集指标
    valid_equity = run_backtest(df, position, test_start=VALID_START, test_end=TEST_START)
    valid_backtest_metrics = compute_metrics(valid_equity)
    
    # 计算选择分数（保持输出格式一致）
    bh_return = valid_buy_hold_metrics["cumulative_return"]
    bh_sharpe = valid_buy_hold_metrics["sharpe"]
    return_score = valid_backtest_metrics["cumulative_return"] / bh_return if bh_return != 0 else 0.0
    sharpe_score = valid_backtest_metrics["sharpe"] / bh_sharpe if bh_sharpe != 0 else 0.0
    valid_selection_scores = {
        "valid_selection_score": RETURN_WEIGHT * return_score + SHARPE_WEIGHT * sharpe_score,
        "valid_return_score": return_score,
        "valid_sharpe_score": sharpe_score,
    }
    
    test_equity = run_backtest(df, position, test_start=TEST_START)
    test_backtest_metrics = compute_metrics(test_equity)

    valid_pred = (valid_probability >= 0.5).astype(int)
    test_pred = (test_probability >= 0.5).astype(int)

    row: dict[str, float | str] = {
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "model": model_name,
        "selection_objective": SELECTION_OBJECTIVE,
        "return_weight": RETURN_WEIGHT,
        "sharpe_weight": SHARPE_WEIGHT,
        **valid_selection_scores,
        **mapping_params,
        "valid_accuracy": float(accuracy_score(y_valid, valid_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "valid_auc": safe_auc(y_valid, valid_probability),
        "test_auc": safe_auc(y_test, test_probability),
        "valid_cumulative_return": valid_backtest_metrics["cumulative_return"],
        "valid_max_drawdown": valid_backtest_metrics["max_drawdown"],
        "valid_sharpe": valid_backtest_metrics["sharpe"],
        **test_backtest_metrics,
    }
    row["excess_return_vs_buy_hold"] = row["cumulative_return"] - buy_hold_return
    return row


def main() -> None:
    ensure_output_dirs()
    print("ML Ablation - 加载数据集...")
    df = load_dataset()
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    test = df[df["split"] == "test"]

    y_train = train["future_up_5d"]
    y_valid = valid["future_up_5d"]
    y_test = test["future_up_5d"]

    buy_hold_position = pd.Series(1.0, index=df.index)
    valid_buy_hold_metrics = compute_metrics(
        run_backtest(df, buy_hold_position, test_start=VALID_START, test_end=TEST_START)
    )
    buy_hold_return = compute_metrics(run_backtest(df, buy_hold_position, test_start=TEST_START))["cumulative_return"]

    tasks = [
        (feature_set_name, feature_columns, model_name)
        for feature_set_name, feature_columns in FEATURE_SETS.items()
        for model_name in get_ml_models(n_jobs=1)
    ]
    
    worker_count = get_worker_count()
    total_tasks = len(tasks)
    
    print(f"\n开始评估 {total_tasks} 个组合 (workers={worker_count})...")
    
    rows = Parallel(n_jobs=worker_count)(
        delayed(evaluate_combination)(
            feature_set_name,
            feature_columns,
            model_name,
            df,
            train,
            valid,
            test,
            y_train,
            y_valid,
            y_test,
            valid_buy_hold_metrics,
            buy_hold_return,
        )
        for feature_set_name, feature_columns, model_name in tqdm(tasks, desc="组合评估", unit="组合")
    )

    metrics = pd.DataFrame(rows)
    write_metrics_csv(metrics, ML_ABLATION_METRICS_CSV)
    
    print(f"\n完成!")
    print(f"  写入 {ML_ABLATION_METRICS_CSV} rows={len(metrics)} workers={worker_count}")
    
    # 显示结果摘要
    print("\n结果摘要:")
    for _, row in metrics.iterrows():
        print(f"  {row['model']}/{row['feature_set']}: test_return={row['cumulative_return']:.2%}, "
              f"sharpe={row['sharpe']:.2f}, excess={row['excess_return_vs_buy_hold']:.2%}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import compute_metrics, run_backtest, write_equity_csv, write_metrics_csv
from src.feature_groups import FEATURE_GROUPS, MODEL_FEATURE_GROUPS
from src.ml_models import get_ml_models
from src.paths import ML_DATASET_CSV, ML_EQUITY_CSV, ML_METRICS_CSV, ensure_output_dirs
from src.position_policy import build_position, iter_position_policy_candidates
from src.model_config import get_model_config


VALID_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")
SELECTION_OBJECTIVE = "weighted_valid_return_sharpe"
RETURN_WEIGHT = 0.5
SHARPE_WEIGHT = 0.5


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ML_DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    numeric_columns = [column for column in df.columns if column not in {"date", "split"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df.sort_values("date").reset_index(drop=True)


def safe_auc(y_true: pd.Series, probability: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def choose_exposure_mapping(
    df: pd.DataFrame,
    probability: pd.Series,
    *,
    buy_hold_valid_metrics: dict[str, float],
    model_name: str = "",
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    best_params: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    best_scores: dict[str, float] | None = None

    buy_hold_return = buy_hold_valid_metrics["cumulative_return"]
    buy_hold_sharpe = buy_hold_valid_metrics["sharpe"]

    # 计算参数总数用于进度显示
    total_params = sum(1 for _ in iter_position_policy_candidates())
    
    print(f"\n  [{model_name}] 搜索 {total_params} 种参数组合...")
    
    for params in tqdm(iter_position_policy_candidates(), total=total_params, desc=f"  参数搜索", unit="组合", leave=False):
        position = build_position(probability, params)
        valid_equity = run_backtest(df, position, test_start=VALID_START, test_end=TEST_START)
        metrics = compute_metrics(valid_equity)

        return_score = metrics["cumulative_return"] / buy_hold_return if buy_hold_return != 0 else 0.0
        sharpe_score = metrics["sharpe"] / buy_hold_sharpe if buy_hold_sharpe != 0 else 0.0
        selection_score = RETURN_WEIGHT * return_score + SHARPE_WEIGHT * sharpe_score
        scores = {
            "valid_selection_score": selection_score,
            "valid_return_score": return_score,
            "valid_sharpe_score": sharpe_score,
        }

        if (
            best_scores is None
            or selection_score > best_scores["valid_selection_score"]
            or (
                selection_score == best_scores["valid_selection_score"]
                and metrics["cumulative_return"] > (best_metrics or {})["cumulative_return"]
            )
        ):
            best_params = params.copy()
            best_metrics = metrics
            best_scores = scores

    if best_params is None or best_metrics is None or best_scores is None:
        raise RuntimeError("No valid exposure mapping was evaluated")
    
    # 显示最优结果
    print(f"  [{model_name}] 最优参数: mapping={best_params['mapping_type']}, "
          f"smooth={best_params['smoothing_window']}, "
          f"valid_return={best_metrics['cumulative_return']:.2%}")
    
    return best_params, best_metrics, best_scores


def main() -> None:
    ensure_output_dirs()
    print("加载数据集...")
    df = load_dataset()
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    test = df[df["split"] == "test"]

    y_train = train["future_up_5d"]
    y_valid = valid["future_up_5d"]
    y_test = test["future_up_5d"]

    metrics_rows: list[dict[str, float | str]] = []
    equity_frames: list[pd.DataFrame] = []
    buy_hold_position = pd.Series(1.0, index=df.index)
    valid_buy_hold_metrics = compute_metrics(
        run_backtest(df, buy_hold_position, test_start=VALID_START, test_end=TEST_START)
    )

    models = get_ml_models()
    print(f"\n开始训练 {len(models)} 个模型...")
    
    for model_name, model in tqdm(models.items(), desc="模型训练", unit="模型"):
        print(f"\n[{model_name}] 开始处理...")
        feature_group = MODEL_FEATURE_GROUPS.get(model_name, "composite")
        feature_columns = FEATURE_GROUPS[feature_group]
        x_train = train[feature_columns]
        x_valid = valid[feature_columns]
        x_test = test[feature_columns]
        
        print(f"  [{model_name}] 训练模型 (特征: {feature_group}, {len(feature_columns)}个)...")
        model.fit(x_train, y_train)

        probability = pd.Series(model.predict_proba(df[feature_columns])[:, 1], index=df.index)
        valid_probability = probability.loc[valid.index]
        test_probability = probability.loc[test.index]

        # 使用预定义的模型配置
        mapping_params = get_model_config(model_name)
        print(f"  [{model_name}] 使用预定义配置: mapping={mapping_params['mapping_type']}, "
              f"smooth={mapping_params['smoothing_window']}")
        
        position = build_position(probability, mapping_params)
        
        # 计算验证集指标
        valid_equity = run_backtest(df, position, test_start=VALID_START, test_end=TEST_START)
        valid_backtest_metrics = compute_metrics(valid_equity)
        
        # 计算选择分数（用于保持输出格式一致）
        buy_hold_return = valid_buy_hold_metrics["cumulative_return"]
        buy_hold_sharpe = valid_buy_hold_metrics["sharpe"]
        return_score = valid_backtest_metrics["cumulative_return"] / buy_hold_return if buy_hold_return != 0 else 0.0
        sharpe_score = valid_backtest_metrics["sharpe"] / buy_hold_sharpe if buy_hold_sharpe != 0 else 0.0
        valid_selection_scores = {
            "valid_selection_score": RETURN_WEIGHT * return_score + SHARPE_WEIGHT * sharpe_score,
            "valid_return_score": return_score,
            "valid_sharpe_score": sharpe_score,
        }
        
        test_equity = run_backtest(df, position, test_start=TEST_START)
        test_backtest_metrics = compute_metrics(test_equity)

        valid_pred = (valid_probability >= 0.5).astype(int)
        test_pred = (test_probability >= 0.5).astype(int)

        metric_row: dict[str, float | str] = {
            "model": model_name,
            "feature_group": feature_group,
            "feature_count": len(feature_columns),
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
        metrics_rows.append(metric_row)
        
        print(f"  [{model_name}] 测试结果: return={test_backtest_metrics['cumulative_return']:.2%}, "
              f"sharpe={test_backtest_metrics['sharpe']:.2f}")

        test_equity.insert(0, "model", model_name)
        equity_frames.append(test_equity)

    metrics = pd.DataFrame(metrics_rows)
    buy_hold_return = compute_metrics(run_backtest(df, buy_hold_position, test_start=TEST_START))["cumulative_return"]
    metrics["excess_return_vs_buy_hold"] = metrics["cumulative_return"] - buy_hold_return

    write_metrics_csv(metrics, ML_METRICS_CSV)
    write_equity_csv(pd.concat(equity_frames, ignore_index=True), ML_EQUITY_CSV)
    
    print(f"\n完成!")
    print(f"  写入 {ML_METRICS_CSV} models={len(metrics)}")
    print(f"  写入 {ML_EQUITY_CSV} rows={sum(len(frame) for frame in equity_frames)}")
    
    # 显示最终结果摘要
    print("\n最终结果摘要:")
    for _, row in metrics.iterrows():
        print(f"  {row['model']}: test_return={row['cumulative_return']:.2%}, "
              f"sharpe={row['sharpe']:.2f}, excess={row['excess_return_vs_buy_hold']:.2%}")


if __name__ == "__main__":
    main()

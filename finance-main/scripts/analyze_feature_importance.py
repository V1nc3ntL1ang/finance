"""特征重要性分析脚本"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent


def load_data():
    """加载ML数据集"""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_feature_importance(df):
    """计算特征重要性"""
    # 获取特征和目标
    feature_columns = [
        "ret_1d", "ma5", "ma10", "ma20", "ma60",
        "momentum5", "momentum10", "momentum20",
        "volatility5", "volatility20",
        "range_pct", "volume_change20",
        "close_vs_ma20", "close_vs_ma60",
        "ma20_slope", "ma60_slope",
        "drawdown_from_20d_high", "drawdown_from_60d_high",
        "volume_zscore20", "volatility20_rank",
        "ma_alignment", "trend_strength20", "trend_strength60",
        "ret5_over_vol20", "ret20_over_vol20",
        "rebound_from_20d_low", "rebound_from_60d_low",
        "range_zscore20",
        # 新增特征
        "macd", "macd_signal", "macd_histogram",
        "rsi14",
        "bb_upper", "bb_lower", "bb_width", "bb_position"
    ]
    
    # 只使用训练集数据
    train_df = df[df["split"] == "train"]
    
    X = train_df[feature_columns]
    y = train_df["future_up_5d"]
    
    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # 获取特征重要性
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance


def plot_feature_importance(importance):
    """绘制特征重要性图"""
    plt.figure(figsize=(12, 10))
    
    # 取前20个最重要的特征
    top_features = importance.head(20)
    
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('特征重要性', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.title('特征重要性排名（前20）', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "outputs" / "plots" / "feature_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存: {output_path}")


def analyze_new_features(importance):
    """分析新增特征的重要性"""
    new_features = [
        "macd", "macd_signal", "macd_histogram",
        "rsi14",
        "bb_upper", "bb_lower", "bb_width", "bb_position"
    ]
    
    new_feature_importance = importance[importance['feature'].isin(new_features)]
    new_feature_importance['rank'] = importance['importance'].rank(ascending=False)
    
    print("\n=== 新增特征重要性分析 ===")
    print(new_feature_importance[['feature', 'importance', 'rank']].to_string(index=False))
    
    return new_feature_importance


def main():
    print("=== 特征重要性分析 ===")
    
    # 加载数据
    df = load_data()
    print(f"数据集大小: {len(df)} 行")
    
    # 计算特征重要性
    importance = get_feature_importance(df)
    
    # 保存完整的特征重要性排名
    output_path = PROJECT_ROOT / "outputs" / "metrics" / "feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    print(f"\n特征重要性排名已保存: {output_path}")
    
    # 显示前10个最重要的特征
    print("\n=== 特征重要性排名（前10）===")
    print(importance.head(10).to_string(index=False))
    
    # 分析新增特征
    analyze_new_features(importance)
    
    # 绘制特征重要性图
    plot_feature_importance(importance)
    
    print("\n=== 分析完成 ===")


if __name__ == "__main__":
    main()

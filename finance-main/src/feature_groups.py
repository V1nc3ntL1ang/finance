from __future__ import annotations

from src.ml_dataset import FEATURE_COLUMNS


# ==================== 特征组定义 ====================

# 最小特征组（用于逻辑回归，12个最重要的特征）
MINIMAL_FEATURES = [
    "volatility5",           # 最重要：5日波动率
    "volatility20_rank",     # 第二重要：20日波动率排名
    "rebound_from_60d_low",  # 第三重要：从60日低点反弹
    "volatility20",          # 第四重要：20日波动率
    "macd_histogram",        # 第五重要：MACD柱状图
    "bb_width",              # 第六重要：布林带宽度
    "trend_strength60",      # 第七重要：60日趋势强度
    "macd_signal",           # 第八重要：MACD信号线
    "ma60",                 # 第九重要：60日均线
    "rebound_from_20d_low",  # 第十重要：从20日低点反弹
    "bb_lower",              # 第十一重要：布林带下轨
    "ma60_slope",            # 第十二重要：60日均线斜率
]

# 基础特征
BASIC_FEATURES = [
    "ma5",
    "ma10",
    "ma60",
    "momentum20",
    "volatility5",
    "volatility20",
]

# 水平特征（价格位置相关）
LEVEL_FEATURES = [
    "close_vs_ma60",
    "ma20_slope",
    "ma60_slope",
    "volume_zscore20",
    "volatility20_rank",
    "bb_upper",
    "bb_lower",
    "bb_width",
]

# 强度特征（趋势强度相关）
STRENGTH_FEATURES = [
    "trend_strength20",
    "trend_strength60",
    "rebound_from_20d_low",
    "rebound_from_60d_low",
    "macd_signal",
    "macd_histogram",
]

# 扩展动量特征组（用于随机森林，动量特征更丰富）
EXTENDED_MOMENTUM_FEATURES = [
    # 基础特征
    "ma5",
    "ma10",
    "ma20",  # 添加ma20
    "ma60",
    "momentum20",
    "volatility5",
    "volatility20",
    # 强度特征
    "trend_strength20",
    "trend_strength60",
    "rebound_from_20d_low",
    "rebound_from_60d_low",
    "macd_signal",
    "macd_histogram",
    # 额外动量特征
    "ma20_slope",
    "ma60_slope",
    "close_vs_ma60",
]

# 特征组字典
FEATURE_GROUPS = {
    "minimal": MINIMAL_FEATURES,                    # 最小特征组（12个）
    "market": BASIC_FEATURES,                       # 基础特征组（6个）
    "regime": BASIC_FEATURES + LEVEL_FEATURES,      # 状态特征组（14个）
    "momentum": EXTENDED_MOMENTUM_FEATURES,         # 扩展动量特征组（17个）
    "composite": FEATURE_COLUMNS,                   # 综合特征组（20个）
}

# ==================== 模型专属特征组配置 ====================
# 根据每个模型的特点选择最合适的特征组

MODEL_FEATURE_GROUPS = {
    # 逻辑回归：线性模型，特征过多容易过拟合，使用最精简的特征
    "logistic_regression": "minimal",
    
    # 随机森林：擅长处理非线性关系，需要更多动量特征
    "random_forest": "momentum",
    
    # 梯度提升：中等复杂度，使用综合特征
    "gradient_boosting": "composite",
    
    # 直方图梯度提升：表现最好，使用综合特征
    "hist_gradient_boosting": "composite",
    
    # LightGBM：工业级模型，使用综合特征
    "lightgbm": "composite",
}

# 模型特征数量统计
MODEL_FEATURE_COUNTS = {
    model: len(FEATURE_GROUPS[group]) 
    for model, group in MODEL_FEATURE_GROUPS.items()
}

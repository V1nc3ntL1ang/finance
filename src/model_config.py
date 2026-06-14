from __future__ import annotations

from typing import Dict, Any

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 保持不变：线性映射已最优
    "logistic_regression": {
        "mapping_type": "linear_clipped",
        "lower_prob": 0.5,
        "upper_prob": 0.5,
        "min_position": 0.0,
        "max_position": 1.0,
        "smoothing_window": 3,
        "smoothing_method": "sma"
    },
    
    # 保留改动：power映射效果更好
    "random_forest": {
        "mapping_type": "power",
        "lower_prob": 0.5,
        "upper_prob": 0.65,
        "power": 3.0,
        "min_position": 0.0,
        "max_position": 1.0,
        "smoothing_window": 5,
        "smoothing_method": "sma"
    },
    
    # 恢复旧配置：linear_clipped表现更好
    "gradient_boosting": {
        "mapping_type": "linear_clipped",
        "lower_prob": 0.45,
        "upper_prob": 0.55,
        "min_position": 0.0,
        "max_position": 1.0,
        "smoothing_window": 5,
        "smoothing_method": "sma"
    },
    
    # 恢复旧配置：linear_clipped表现更好
    "hist_gradient_boosting": {
        "mapping_type": "linear_clipped",
        "lower_prob": 0.45,
        "upper_prob": 0.55,
        "min_position": 0.0,
        "max_position": 1.0,
        "smoothing_window": 5,
        "smoothing_method": "sma"
    },
    
    # 恢复旧配置：linear_clipped表现更好
    "lightgbm": {
        "mapping_type": "linear_clipped",
        "lower_prob": 0.4,
        "upper_prob": 0.6,
        "min_position": 0.0,
        "max_position": 1.0,
        "smoothing_window": 3,
        "smoothing_method": "sma"
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """获取指定模型的最优配置"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["logistic_regression"])

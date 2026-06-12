# 指数择时实验结果表

数据标的：`SH#880823` 微盘股指数  
初始资金：100000  
测试区间：2025-01-01 至数据末尾  
说明：非 LSTM 结果来自 `python scripts/run_all.py` 重新生成的实验输出；LSTM 结果直接照搬 `feature/lstm` 分支已有实验结果，本次没有重跑 LSTM。

## 基准策略

| 策略 | 累计收益 | 最大回撤 | 夏普 | 相对买入持有超额 |
|---|---:|---:|---:|---:|
| buy_hold | 109.21% | -17.26% | 2.30 | 0.00% |
| ma20_timing | 77.64% | -10.90% | 2.76 | -31.58% |
| momentum20_timing | 55.73% | -17.91% | 1.77 | -53.48% |
| ma20_momentum20_combo | 66.94% | -11.69% | 2.38 | -42.27% |
| theoretical_optimal | 917.68% | 0.00% | 12.47 | 808.46% |

## 默认机器学习模型

| 模型 | 特征组 | 特征数 | 累计收益 | 最大回撤 | 夏普 | Test AUC | 相对买入持有超额 |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | composite | 28 | 107.35% | -14.62% | 2.69 | 0.634 | 8.07% |
| random_forest | momentum | 20 | 87.07% | -7.94% | 2.97 | 0.639 | -12.21% |
| gradient_boosting | composite | 28 | 87.01% | -13.48% | 2.50 | 0.600 | -12.27% |
| hist_gradient_boosting | composite | 28 | 80.73% | -14.70% | 2.17 | 0.623 | -18.54% |
| lightgbm | composite | 28 | 86.00% | -12.80% | 2.45 | 0.650 | -13.28% |
| LSTM (GRU) | composite | 28 | 84.73% | -15.30% | 2.18 | 0.486 | -14.55% |

## 特征组消融 Top 10

| 排名 | 模型 | 特征组 | 特征数 | 累计收益 | 最大回撤 | 夏普 | Test AUC | 相对买入持有超额 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | logistic_regression | composite | 28 | 107.35% | -14.62% | 2.69 | 0.634 | 8.07% |
| 2 | lightgbm | momentum | 20 | 105.13% | -11.95% | 2.77 | 0.612 | 5.85% |
| 3 | random_forest | regime | 20 | 103.54% | -11.89% | 2.62 | 0.646 | 4.26% |
| 4 | lightgbm | regime | 20 | 102.07% | -15.70% | 2.36 | 0.635 | 2.79% |
| 5 | random_forest | composite | 28 | 90.31% | -10.86% | 2.67 | 0.647 | -8.96% |
| 6 | logistic_regression | momentum | 20 | 87.31% | -14.84% | 2.10 | 0.620 | -11.97% |
| 7 | random_forest | momentum | 20 | 87.07% | -7.94% | 2.97 | 0.639 | -12.21% |
| 8 | gradient_boosting | composite | 28 | 87.01% | -13.48% | 2.50 | 0.600 | -12.27% |
| 9 | lightgbm | composite | 28 | 86.00% | -12.80% | 2.45 | 0.650 | -13.28% |
| 10 | logistic_regression | market | 12 | 84.69% | -13.03% | 2.29 | 0.618 | -14.58% |

## 特征重要性 Top 15

| 排名 | 特征 | 重要性 |
|---:|---|---:|
| 1 | volatility20_rank | 0.0615 |
| 2 | volatility5 | 0.0585 |
| 3 | ma60 | 0.0483 |
| 4 | rebound_from_60d_low | 0.0480 |
| 5 | volatility20 | 0.0479 |
| 6 | rebound_from_20d_low | 0.0465 |
| 7 | ma10 | 0.0432 |
| 8 | ma20 | 0.0415 |
| 9 | close_vs_ma60 | 0.0410 |
| 10 | trend_strength60 | 0.0409 |
| 11 | ma5 | 0.0401 |
| 12 | ma60_slope | 0.0360 |
| 13 | momentum20 | 0.0352 |
| 14 | ma20_slope | 0.0340 |
| 15 | volume_zscore20 | 0.0337 |

## LSTM 对照结果

此部分直接照搬 `feature/lstm` 分支结果。

| 特征组 | 累计收益 | 最大回撤 | 夏普 | Test AUC |
|---|---:|---:|---:|---:|
| market(12) | 84.73% | -15.30% | 2.18 | 0.500 |
| regime(20) | 84.73% | -15.30% | 2.18 | 0.500 |
| momentum(20) | 84.73% | -15.30% | 2.18 | 0.500 |
| composite(28) | 84.73% | -15.30% | 2.18 | 0.486 |

LSTM 使用 `lookback=15`、`gru_units=32`、`dropout=0.3`，仓位映射为 `sigmoid`，`min_position=0.5`、`max_position=1.0`、`smoothing_window=5`。该分支结论是 LSTM 在当前小样本金融数据上基本没有学到有效排序信号，AUC 约等于随机猜测。

## 结论

默认模型中，`logistic_regression + composite` 的累计收益最高，并且相对买入持有有 8.07% 超额。完整特征组消融里，`lightgbm + momentum` 和 `random_forest + regime` 也能跑出正超额。LSTM 结果作为对照保留，但不作为主方案。

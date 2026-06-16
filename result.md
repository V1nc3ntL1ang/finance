# 指数择时实验结果表

## 实验口径

最终比较统一使用 `data/processed/ml_dataset.csv`，不再混用更长的日频全周期 baseline。

| 区间 | 日期 | 用途 |
|---|---|---|
| Train | 2017-06-01 到 2023-12-29 | 训练分类模型 |
| Valid | 2024-01-02 到 2024-12-31 | 搜索仓位映射参数 |
| Test | 2025-01-02 到 2026-04-24 | 最终报告结果 |

调参只使用 Valid。普通 ML 模型先在 Train 上训练，然后在 Valid 上从 107352 个仓位候选里选择参数，最后只在 Test 上报告结果。

锁定策略 `hist_gradient_boosting_alignment_confirmation` 使用多折验证探索得到的固定组合，本次只重新训练模型并重算这一行，不重新搜索旧 baseline 或普通 ML 模型。

仓位参数选择目标：

```text
valid_selection_score = 0.5 * valid_return_score + 0.5 * valid_sharpe_score
valid_return_score = 策略 Valid 累计收益 / Valid buy_hold 累计收益
valid_sharpe_score = 策略 Valid 夏普 / Valid buy_hold 夏普
```

Test 同周期 buy_hold 收益为 99.28%。所有“相对买入持有超额”都以这个数为基准。

## 最终同口径结果

| 方法 | 累计收益 | 最大回撤 | 夏普 | Test AUC | 相对买入持有超额 | 特征数 | 仓位策略 |
|---|---:|---:|---:|---:|---:|---:|---|
| buy_hold | 99.28% | -17.26% | 2.18 | - | 0.00% | - | 固定满仓 |
| ma20_timing | 69.20% | -10.90% | 2.57 | - | -30.08% | - | 收盘价高于 MA20 满仓，否则空仓 |
| momentum20_timing | 48.34% | -17.91% | 1.61 | - | -50.94% | - | 20日动量为正满仓，否则空仓 |
| ma20_momentum20_combo | 59.01% | -11.69% | 2.19 | - | -40.27% | - | MA20 与 momentum20 各给 50% 仓位 |
| theoretical_optimal | 868.50% | 0.00% | 12.38 | - | 769.22% | - | 使用未来涨跌的理论上界 |
| logistic_regression | 79.74% | -14.25% | 2.30 | 0.623 | -19.54% | 12 | `threshold` threshold=0.60, min=0.00, max=1.00, smooth=2, method=sma |
| random_forest | 90.73% | -6.57% | 3.10 | 0.641 | -8.55% | 16 | `threshold` threshold=0.60, min=0.00, max=1.00, smooth=5, method=sma |
| gradient_boosting | 92.25% | -7.40% | 3.11 | 0.617 | -7.03% | 20 | `threshold` threshold=0.60, min=0.00, max=1.00, smooth=5, method=sma |
| hist_gradient_boosting | 110.16% | -5.85% | 3.83 | 0.653 | 10.88% | 20 | `rank_linear` lower_rank=0.40, upper_rank=0.85, min=0.00, max=1.00, smooth=1, method=sma |
| lightgbm | 96.01% | -13.69% | 2.61 | 0.630 | -3.27% | 20 | `power` lower=0.50, upper=0.70, power=3.0, min=0.00, max=1.00, smooth=3, method=sma |
| hist_gradient_boosting_alignment_confirmation | 163.27% | -9.27% | 3.54 | 0.654 | 64.00% | 20 | `rank_confirmation` entry_rank=0.50, exit_rank=0.4875, confirm_days=2, min=0.00, max=1.00, smooth=1, method=sma, ma_alignment floor=1.00 |

当前最好的可交易策略是 `hist_gradient_boosting_alignment_confirmation`。它在同一 Test 区间内比 buy_hold 多 64.00 个百分点收益，最大回撤为 -9.27%，夏普为 3.54。

## 干净复现方式

给老师或组员复现时，只运行这一条命令：

```bash
python scripts/reproduce_locked_strategy.py
```

这个脚本只验证最终锁定策略，不重跑旧 baseline，不重搜 107352 个仓位候选，也不写新的输出文件。若数据或结果不一致，脚本会直接报错。

复现数据口径：

```text
data file = data/processed/ml_dataset.csv
sha256 = a00bf3d3918e5230e0a4f7906a6b0233d4af93a19837e53ae9973434348fb51e
rows = 2162
train rows = 1604
valid rows = 242
test rows = 316
model features = 原 20 个 FEATURE_COLUMNS
ma_alignment = 只用于仓位地板，不进入模型特征
transaction cost = 0 bps
random_state = 42
```

预期输出核心数值：

```text
reproduction passed
cumulative_return: 1.632727292589
max_drawdown: -0.092726987482
sharpe: 3.535132344229
test_auc: 0.653581133540
buy_hold_cumulative_return: 0.992770066097
excess_return_vs_buy_hold: 0.639957226492
```

## 当前 ML 特征

| 模型 | 特征 |
|---|---|
| logistic_regression | `volatility5`, `volatility20_rank`, `rebound_from_60d_low`, `volatility20`, `macd_histogram`, `bb_width`, `trend_strength60`, `macd_signal`, `ma60`, `rebound_from_20d_low`, `bb_lower`, `ma60_slope` |
| random_forest | `ma5`, `ma10`, `ma20`, `ma60`, `momentum20`, `volatility5`, `volatility20`, `trend_strength20`, `trend_strength60`, `rebound_from_20d_low`, `rebound_from_60d_low`, `macd_signal`, `macd_histogram`, `ma20_slope`, `ma60_slope`, `close_vs_ma60` |
| gradient_boosting | `volatility5`, `volatility20_rank`, `rebound_from_60d_low`, `volatility20`, `macd_histogram`, `bb_width`, `trend_strength60`, `macd_signal`, `ma60`, `rebound_from_20d_low`, `bb_lower`, `ma60_slope`, `ma10`, `ma20_slope`, `close_vs_ma60`, `ma5`, `bb_upper`, `trend_strength20`, `momentum20`, `volume_zscore20` |
| hist_gradient_boosting | `volatility5`, `volatility20_rank`, `rebound_from_60d_low`, `volatility20`, `macd_histogram`, `bb_width`, `trend_strength60`, `macd_signal`, `ma60`, `rebound_from_20d_low`, `bb_lower`, `ma60_slope`, `ma10`, `ma20_slope`, `close_vs_ma60`, `ma5`, `bb_upper`, `trend_strength20`, `momentum20`, `volume_zscore20` |
| hist_gradient_boosting_alignment_confirmation | `volatility5`, `volatility20_rank`, `rebound_from_60d_low`, `volatility20`, `macd_histogram`, `bb_width`, `trend_strength60`, `macd_signal`, `ma60`, `rebound_from_20d_low`, `bb_lower`, `ma60_slope`, `ma10`, `ma20_slope`, `close_vs_ma60`, `ma5`, `bb_upper`, `trend_strength20`, `momentum20`, `volume_zscore20` |
| lightgbm | `volatility5`, `volatility20_rank`, `rebound_from_60d_low`, `volatility20`, `macd_histogram`, `bb_width`, `trend_strength60`, `macd_signal`, `ma60`, `rebound_from_20d_low`, `bb_lower`, `ma60_slope`, `ma10`, `ma20_slope`, `close_vs_ma60`, `ma5`, `bb_upper`, `trend_strength20`, `momentum20`, `volume_zscore20` |

## 历史参考

以下结果只作为历史记录，不参与最终同口径排名。

| 来源 | 最好结果 | 说明 |
|---|---|---|
| 旧 feature-engineering 显式特征结果 | `hist_gradient_boosting` 115.36%，夏普 3.84 | 使用较小仓位搜索空间，旧参数 `rank_linear upper_rank=0.80` 在 Test 上更高，但不是扩大搜索后 Valid 选择出的参数 |
| 压缩包 position-policy ablation | `logistic_regression + composite` 107.35%，超额 8.07% | 旧 feature-group 口径，证明 `threshold` 值得纳入搜索，但不作为当前主结果 |
| LSTM 对照 | 84.73%，夏普 2.18 | 直接照搬 `feature/lstm` 分支结果，本次没有重跑 LSTM |

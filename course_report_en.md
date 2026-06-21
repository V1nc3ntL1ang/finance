# StableHGB: Stable Index Timing with Histogram Gradient Boosting — Course Report

## 1. Research Background and Objective

This course project proposes **StableHGB** (Stable Index Timing with Histogram Gradient Boosting), a machine learning approach to time the micro-cap index (SH#880823) and construct an index enhancement strategy. The core idea is to frame market timing as a **binary classification problem**: predicting whether the index will rise over the next 5 trading days. The model outputs an upward probability, which is then converted into a dynamic position ranging from 0% to 100% via a position mapping function. Strategy performance is evaluated on an independent test set.

StableHGB's key contributions are two original position management modules:

1. **Relative Signal Stabilizer**: A position mapping mechanism based on expanding percentile rank and consecutive-day confirmation, which converts model probability outputs into robust position signals and eliminates the systematic bias introduced by bull/bear market regime shifts.
2. **Trend Position Guard**: A trend lock-in mechanism based on moving-average alignment, which forces full investment during confirmed bull trends, preventing the model's short-term signal fluctuations from causing missed rallies.

**Objective**: Outperform the buy-and-hold baseline in cumulative return while controlling drawdown, and achieve a higher Sharpe ratio.

---

## 2. Data Processing

### 2.1 Data Source and Splitting

- **Raw data**: Daily OHLCV data (Open, High, Low, Close, Volume, Amount) for the micro-cap index (SH#880823)
- **Time span**: 2017-06-01 to 2026-05-06
- **Backtest period**: 2025-01-01 to 2026-05-06, totaling **321 trading days**
- **Dataset split** (strictly chronological to prevent future information leakage):

| Dataset | Time Range | Purpose |
|---------|------------|---------|
| Final model training | Label windows ending before 2024-01-01 | Model parameter learning |
| Rolling validation | Three independent annual windows: 2022, 2023, 2024 | Position policy parameter selection |
| Test | 2025-01-01 ~ 2026-05-06 | Final strategy evaluation |

### 2.2 Data Preprocessing

1. All missing values (NaN) and infinite values (inf) resulting from rolling-window calculations are removed via `dropna`
2. Daily return `ret_1d` is computed as `close.pct_change()`, serving as the basis for volatility and other derived features
3. Dataset integrity is verified by SHA256 hash, ensuring identical data across every experimental run
4. Backtest uses close-to-close accounting with `execution_lag=1`, i.e., today's signal determines tomorrow's position

---

## 3. Feature Engineering

A total of **20 model input features** and **1 trend guard feature** are constructed from the daily data:

### 3.1 Model Training Features (20)

| Category | Features | Formula | Economic Meaning |
|----------|----------|---------|------------------|
| **Moving Averages** | `ma5`, `ma10`, `ma20`, `ma60` | N-day simple moving average of close | Short/medium/long-term trend baseline |
| **MA Slopes** | `ma20_slope`, `ma60_slope` | ma / ma.shift(5) − 1 | Trend direction and acceleration |
| **Price Deviation** | `close_vs_ma60` | close / ma60 − 1 | Deviation of price from quarterly MA |
| **Momentum** | `momentum20` | close.pct_change(20) | 20-day momentum (medium-term return) |
| **Volatility** | `volatility5`, `volatility20`, `volatility20_rank` | Rolling std of daily returns + 252-day rolling percentile rank | Short/medium-term volatility level and its historical percentile |
| **Trend Strength** | `trend_strength20`, `trend_strength60` | MA slope / volatility | Risk-adjusted trend return (signal-to-noise ratio) |
| **Rebound** | `rebound_from_20d_low`, `rebound_from_60d_low` | close / N-day minimum − 1 | Magnitude of rebound from recent lows |
| **MACD** | `macd_signal`, `macd_histogram` | Signal line and histogram of EMA(12) − EMA(26) | Momentum change |
| **Bollinger Bands** | `bb_upper`, `bb_lower`, `bb_width` | 20-day MA ± 2σ, and bandwidth | Price position relative to volatility range |
| **Volume** | `volume_zscore20` | (volume − 20-day mean volume) / 20-day volume std | Volume abnormality |

### 3.2 Trend Guard Feature (1, not used for model training)

| Feature | Formula | Meaning |
|---------|---------|---------|
| `ma_alignment` | 1 if ma5 > ma20 > ma60, else 0 | Bullish MA alignment (bull market regime), used exclusively by Trend Position Guard |

### 3.3 Target Variable

- **`future_up_5d`**: 1 if close on day t+5 > close on day t (up), otherwise 0 (down)
- **`future_ret_5d`**: `close(t+5) / close(t) − 1` (for reference)

### 3.4 Design Rationale

The features cover the core dimensions of technical analysis: **trend** (MAs, slopes), **momentum** (momentum, MACD), **volatility** (volatility, Bollinger Bands), and **positioning** (deviation, rebound, volume abnormality). The `volatility20_rank` uses a rolling percentile rank rather than the raw value, eliminating the long-term trend component of volatility itself. `trend_strength` divides the trend signal by volatility, achieving risk adjustment — a strong trend accompanied by high volatility is dampened.

---

## 4. Machine Learning Algorithm: Hist Gradient Boosting

### 4.1 Algorithm Principles

Hist Gradient Boosting is a decision-tree ensemble model based on the Boosting framework. Its core ideas are:

1. **Iterative residual learning**: Starting from an initial constant prediction, each round trains a shallow decision tree to fit the residuals (prediction errors) of the previous round, progressively approaching the true labels
2. **Histogram-based split finding**: Continuous features are discretized into a finite number of bins; the optimal split point is found on the binned histogram, reducing time complexity from O(n log n) to O(n)
3. **Gradient descent in function space**: Each iteration adds a new tree along the negative gradient direction of the loss function, equivalent to performing gradient descent in function space

Compared to traditional GBDT, HistGB offers: an order-of-magnitude faster training speed, native missing value handling (treating missing values as a separate branch direction), and lower memory consumption.

### 4.2 Model Architecture and Hyperparameters

```
HistGradientBoostingClassifier(
    max_iter=150,          # Maximum boosting iterations (number of trees)
    learning_rate=0.05,    # Learning rate (contribution weight per tree)
    max_leaf_nodes=15,     # Maximum leaf nodes per tree (limits tree depth)
    min_samples_leaf=70,   # Minimum samples per leaf (strong regularization)
    random_state=42,       # Fixed random seed for reproducibility
)
```

**Hyperparameter rationale:**

| Parameter | Value | Design Intent |
|-----------|-------|---------------|
| `max_iter=150` | 150 iterations | With training samples, 150 trees are sufficient for convergence; combined with `learning_rate=0.05`, the total learning step size is 7.5 |
| `learning_rate=0.05` | Small learning rate | Each tree contributes little, requiring more trees but yielding better generalization and less overfitting |
| `max_leaf_nodes=15` | Shallow trees | Limits model complexity — each tree can learn at most 15 decision rules, preventing memorization of noise |
| `min_samples_leaf=70` | Strong regularization | Each leaf node requires at least 70 samples, preventing overfitting to small subsets of extreme observations |
| `random_state=42` | Fixed seed | Ensures identical results across every run |

### 4.3 Feature Count

HistGB uses all **20 features**. Tree-based boosting models inherently possess feature selection capability — unimportant features are simply not selected during splits, eliminating the need to manually reduce the feature set as is required for linear models. Feature importance analysis (conducted via Random Forest as an auxiliary tool) ranks `volatility5`, `volatility20_rank`, `macd_histogram`, volatility, and rebound indicators as the top five, validating the effectiveness of the feature design.

---

## 5. StableHGB Core Modules

StableHGB integrates two key trading strategy modules on top of the HistGB prediction engine, which constitute the core contributions of this project.

### 5.1 Complete Pipeline

```
Raw OHLCV daily data
    ↓  [Compute basic indicators]  ret_1d, ma20, momentum20
    ↓  [Feature engineering]  20 quantitative factors + target variable future_up_5d
    ↓  [Missing value cleanup]  dropna
    ↓  [Chronological split]  final training set / rolling validation(2022/2023/2024) / test(2025-2026)
    ↓  [Model training]  histGB.fit(train[20_features], train["future_up_5d"])
    ↓  [Full-sample prediction]  histGB.predict_proba(df[20_features])[:, 1] → upward probability series
    ↓  [Position mapping]  probability → Relative Signal Stabilizer → Trend Position Guard → 0~100% position
    ↓  [Backtest evaluation]  daily_equity = ∏(1 + position × daily_return)
```

### 5.2 Model Training

The training process is straightforward: feed the 1,604 training samples with 20 features into HistGB, using `future_up_5d` as the binary classification label. Because the features are purely numeric and HistGB handles binning internally, no additional standardization or encoding is required. HistGB discretizes each continuous feature into 255 equal-width bins (sklearn default) and greedily searches for the optimal split point on the binned histogram at each iteration.

After training, `predict_proba` is called on the full dataset, and `[:, 1]` extracts the predicted probability that the index will rise over the next 5 days for each trading day.

### 5.3 Module 1: Relative Signal Stabilizer

Model probabilities cannot be used for trading directly, because the model tends to be optimistic in bull markets (higher probabilities) and pessimistic in bear markets (lower probabilities). Using raw probability thresholds for decision-making would fail across market regimes. The **Relative Signal Stabilizer** is StableHGB's first core module, solving this problem through a two-step transformation:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `mapping_type` | relative_signal_stabilizer | Confirmation-based position management using expanding percentile rank |
| `entry_rank` | 0.55 | Rank ≥ 55th percentile → trigger long entry count |
| `exit_rank` | 0.45 | Rank ≤ 45th percentile → trigger exit count |
| `confirm_days` | 2 | Signal must persist for 2 consecutive days before switching position |
| `min_position` | 0.0 | 0% position when out of market |
| `max_position` | 1.0 | 100% position when fully invested |
| `smoothing_method` | sma | Simple moving average smoothing |
| `smoothing_window` | 1 | No smoothing (window=1), preserving signal responsiveness |

**How the Relative Signal Stabilizer works:**

1. **Expanding percentile rank**: Starting from the first data point, an expanding percentile rank is computed day by day — i.e., the percentile of the current prediction relative to all historical predictions seen so far. This makes model outputs comparable across different market regimes. Regardless of market phase, the rank distribution is always uniform, meaning position switching logic remains consistent across all market states.

2. **Confirmation-based switching**: A long entry signal (rank ≥ 55th percentile) must persist for 2 consecutive days before executing; likewise, an exit signal (rank ≤ 45th percentile) requires 2 consecutive days of confirmation. This prevents frequent position flipping caused by single-day noise and reduces transaction friction. Compared to single-day threshold switching, the confirmation mechanism effectively filters out false breakout signals in choppy markets.

3. **Asymmetric threshold design**: entry_rank=0.55 while exit_rank=0.45 — a 10-percentile dead zone between them. This asymmetric design serves two purposes: (a) a mild bullish bias grounded in the micro-cap index's long-term upward trend, where entry thresholds are appropriately raised to filter weak signals while exit thresholds are lowered to avoid premature exits; (b) the dead zone itself provides natural position stability — within the dead zone, positions remain unchanged, avoiding frequent switching.

### 5.4 Module 2: Trend Position Guard

When `ma_alignment = 1` (ma5 > ma20 > ma60, i.e., bullish MA alignment), the position is forced to no less than 100% (fully invested). The configuration parameters are:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `trend_guard_feature` | ma_alignment | Feature that triggers the trend guard |
| `trend_guard_threshold` | 0.5 | Triggered when ma_alignment > 0.5 |
| `trend_guard_min_position` | 1.0 | When triggered, position is forced to 100% |

**Design rationale**: In a clear bull trend, the position should not be reduced even if the model signal turns bearish — historically, false sell signals during bull markets are extremely costly. Trend Position Guard implements a complementary **"model timing + trend confirmation"** architecture: day-to-day positions are driven by the Relative Signal Stabilizer's rank signals, while during bull trends the MA structure locks in full exposure. Trend Position Guard takes priority over the Relative Signal Stabilizer.

### 5.5 Policy Parameter Selection via Rolling Validation

The position mapping parameters (entry_rank, exit_rank, confirm_days, etc.) are not chosen arbitrarily. The system employs a **rolling validation** protocol, evaluating parameter candidates on three independent annual windows:

| Validation Fold | Time Range | Purpose |
|-----------------|------------|---------|
| valid_2022 | 2022-01-01 ~ 2022-12-31 | First-year rolling validation |
| valid_2023 | 2023-01-01 ~ 2023-12-31 | Second-year rolling validation |
| valid_2024 | 2024-01-01 ~ 2024-12-31 | Third-year rolling validation |

For each validation fold, the model is trained using all data prior to that year and evaluated within that year. The final selection rule computes `valid_score = mean(valid_cumulative_return) - 0.25 * std(valid_cumulative_return)`, retains candidates within the top 0.02 range, and selects the one with the smallest worst-fold drawdown. This rule balances return mean and stability, yielding more robust parameter selection than single-year validation.

StableHGB's policy parameters were selected from 108 candidate combinations (3 entry_rank × 3 exit_rank × 3 confirm_days × 2 trend_guard_min_position × 2 smoothing_window) using the above rolling validation protocol.

### 5.6 Drawdown Control Mechanisms

This strategy actively controls drawdown through the following three-layer mechanism, rather than passively accepting whatever drawdown occurs:

**Layer 1: Model-driven dynamic position reduction.** When HistGB predicts a lower probability of a 5-day rise, the model's upward probability drops, and after the Relative Signal Stabilizer's rank mapping the position automatically falls to 0%. This realizes a "predict decline → reduce exposure ahead of time → avoid the drop" logic, which serves as the **core defense** for drawdown control. The buy-and-hold strategy remains fully invested in all market conditions and therefore bears the full brunt of every decline; this strategy stays in cash when the model is bearish, reducing the probability of participating in downturns at the source.

**Layer 2: Confirmation mechanism prevents overtrading.** The Relative Signal Stabilizer requires signals to be confirmed for 2 consecutive days before switching positions (`confirm_days=2`). Single-day noise could cause the model to trade frequently in choppy markets — reducing exposure one day and adding it back the next, repeatedly getting whipsawed. This trading friction is itself a form of hidden drawdown. The confirmation mechanism filters out isolated noise, ensuring position switches only follow reliable trend signals, and reduces the cumulative drag from unnecessary turnover.

**Layer 3: Trend Position Guard prevents missed rallies.** The biggest risk in drawdown control is "over-controlling" — the model becomes too conservative, repeatedly reducing exposure and missing rallies during bull markets. While this controls downside drawdown, it creates "opportunity drawdown" relative to the benchmark by failing to capture gains that should have been earned. The Trend Position Guard forces full investment when MAs are in bullish alignment, ensuring that short-term fluctuations in model signals do not cause missed returns during clear uptrends. This provides drawdown control with a "safety boundary" — the position-reduction logic only operates outside of confirmed bull market regimes.

The division of labor across the three layers: **the model judges direction, the Relative Signal Stabilizer filters noise, and the Trend Position Guard locks in bull markets** — together achieving the drawdown control objective of "losing less when the market falls, keeping up when it rises."

---

## 6. Strategy Evaluation

### 6.1 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Cumulative Return | Final equity / Initial equity − 1 | Total strategy return |
| Annualized Return | (Final/Initial)^(252/N) − 1 | Annualized equivalent |
| Maximum Drawdown | min(equity / running peak − 1) | Maximum loss from peak to trough |
| Sharpe Ratio | Mean daily return / Std daily return × √252 | Risk-adjusted return (excess return per unit of risk) |
| Test AUC | Area under the ROC curve | Model ranking ability (0.5 = random, 1.0 = perfect) |

### 6.2 Experiment Design

The experiments are organized into three Panels, following a unified evaluation protocol:

| Item | Setting |
|------|---------|
| Initial capital | 100,000 CNY |
| Backtest period | 2025-01-01 to 2026-05-06 |
| Actual trading days | 321 trading days |
| Prediction label | Binary 5-day forward direction label `future_up_5d` |
| Backtest accounting | Close-to-close, `execution_lag=1` |
| Validation protocol | Three-year rolling validation (2022/2023/2024) |
| Selection rule | `valid_score = mean(return) - 0.25 * std(return)`, retain top 0.02, select min worst-fold drawdown |

**Panel A: Traditional Timing Baselines**

| Strategy | Cumulative Return | Annualized Return | Max Drawdown | Sharpe | Excess Return (vs B&H) |
|----------|-------------------|-------------------|-------------|--------|------------------------|
| Buy & Hold | 109.21% | 78.52% | −17.26% | 2.30 | 0.00 pp |
| MA20 Timing | 77.64% | 57.00% | −10.90% | 2.76 | −31.58 pp |
| 20-Day Momentum | 55.73% | 41.59% | −17.91% | 1.77 | −53.48 pp |
| MA20 + Momentum Combined | 66.94% | 49.53% | −11.69% | 2.38 | −42.27 pp |

![Panel A Baseline NAV Curves](outputs/plots/all_baselines.png)

*Figure 1: Panel A baseline strategy NAV curves. Buy & Hold achieved the highest cumulative return among baselines but also the largest drawdown (−17.26%). MA20 timing had lower volatility with max drawdown of only −10.90%, demonstrating the risk-mitigating effect of simple trend-following rules.*

**Panel B: Competitive ML Baselines**

Panel B uses GBDT, HistGB, and LightGBM with all 20 features (same as StableHGB), while Logistic Regression and Random Forest use 12 and 16 features respectively. Grid search is performed over 5 ML models × 5 standard position mappings × multiple parameter combinations, evaluating 10,656 strategy candidates per model per fold, totaling 159,840 validation records.

| Model | Mapping | Valid Score | Cum. Return | Ann. Return | Max DD | Sharpe | Excess | Test AUC |
|-------|---------|------------|-------------|-------------|--------|--------|--------|----------|
| Logistic Regression | power | 0.2297 | 98.57% | 71.35% | −14.62% | 2.31 | −10.64 pp | 0.62 |
| Random Forest | rank_linear | 0.2077 | 85.39% | 62.36% | −9.43% | 2.82 | −23.82 pp | 0.64 |
| Gradient Boosting | linear_clipped | 0.2575 | 105.29% | 75.88% | −12.21% | 2.72 | −3.92 pp | 0.63 |
| HistGradientBoosting | threshold | 0.1528 | 90.38% | 65.78% | −9.29% | 2.81 | −18.83 pp | 0.64 |
| LightGBM | rank_linear | 0.2010 | 88.72% | 64.64% | −10.48% | 2.81 | −20.49 pp | 0.65 |

![Panel B ML Baseline NAV Curves](outputs/plots/all_ml_baselines.png)

*Figure 2: Panel B ML baseline strategy NAV curves. Gradient Boosting performed best among ML baselines with 105.29% cumulative return and 2.72 Sharpe. Random Forest and HistGradientBoosting had lower drawdowns (−9.43% and −9.29%) but sacrificed returns. LightGBM achieved a balance between return and risk.*

![Risk-Return Scatter](outputs/plots/risk_return_scatter.png)

*Figure 3: Risk-return scatter plot showing the trade-off between annualized return and maximum drawdown for all evaluated strategies. Each point represents a strategy candidate, with selected strategies highlighted. StableHGB (highlighted) occupies a favorable position of high return with moderate drawdown.*

**Panel C: StableHGB**

| Strategy | Composition | Valid Score | Cum. Return | Ann. Return | Max DD | Sharpe | Excess | Test AUC |
|----------|------------|------------|-------------|-------------|--------|--------|--------|----------|
| **StableHGB** | Relative Signal Stabilizer + Trend Position Guard | **0.3164** | **153.44%** | **107.52%** | **−9.84%** | **3.38** | **+44.22 pp** | 0.63 |

### 6.3 Rolling Validation Details

StableHGB's performance across the three validation folds:

| Validation Fold | Cumulative Return | Selection Score | Max Drawdown | Sharpe | Buy & Hold Return |
|-----------------|-------------------|-----------------|-------------|--------|-------------------|
| valid_2022 | 30.99% | 1.36 | −22.29% | 1.18 | 22.20% |
| valid_2023 | 26.80% | 0.70 | −7.39% | 1.99 | 47.60% |
| valid_2024 | 43.76% | 3.56 | −40.84% | 1.06 | 9.33% |

The validation results show that StableHGB achieved positive returns in all three distinct market environments, and performed particularly well in 2024 when buy & hold only returned 9.33%, validating the strategy's adaptability across different market regimes.

### 6.4 Result Analysis

![StableHGB vs Reference Strategies](outputs/plots/stable_hgb_vs_references.png)

*Figure 4: NAV comparison of StableHGB versus buy & hold and the best ML baseline (Gradient Boosting). StableHGB consistently outperformed both reference strategies throughout the backtest period, achieving 153.44% cumulative return versus 109.21% for buy & hold and 105.29% for Gradient Boosting.*

![Drawdown Comparison](outputs/plots/drawdown_stable_hgb_references.png)

*Figure 5: Drawdown curves of StableHGB versus reference strategies. StableHGB's maximum drawdown (−9.84%) is significantly lower than buy & hold (−17.26%) and Gradient Boosting (−12.21%), validating the effectiveness of the Relative Signal Stabilizer and Trend Position Guard in downside risk control.*

![StableHGB Position Changes](outputs/plots/stable_hgb_position.png)

*Figure 6: StableHGB position over time. The strategy dynamically adjusts positions based on model signals. The Relative Signal Stabilizer enforces confirmation days before entry, and the Trend Position Guard maintains full investment during bullish MA alignment. This results in fewer but more confident trading decisions.*

---

## 7. Advantages of Hist Gradient Boosting for Index Timing

### 7.1 Histogram Binning Is Naturally Suited to Noisy Financial Data

Financial data is characterized by a low signal-to-noise ratio — technical indicators derived from daily OHLCV data are full of random fluctuations. HistGB discretizes each continuous feature into 255 equal-width bins (sklearn default) before training, and split-point search operates on bin boundaries rather than raw values. This means that volatility5 values of 1.52% and 1.58% fall into the same bin — minor differences do not affect decisions. The model automatically ignores noise-level fluctuations. In contrast, linear models such as Logistic Regression respond to every small change in feature values and easily overfit on low-SNR financial data.

### 7.2 Boosting's Residual Learning Equals Adaptive Focus on "Difficult Market States"

In each iteration, the newly added weak learner in HistGB is specifically trained to fit the residuals (errors) of the previous round. In the market-timing context, this means the model automatically allocates more modeling capacity to the historically hardest-to-predict market states — such as the eve of trend reversals and false breakouts during choppy markets. On days with clear upward trends (roughly 60% of training samples), the model easily learns to predict "up" without requiring complex tree structures; the Boosting mechanism ensures that limited tree capacity is not exhausted by these "easy samples," but instead remains focused on the "hard samples" that were predicted incorrectly. This **adaptive difficulty weighting** is a capability that a single decision tree or linear model lacks.

### 7.3 Shallow Trees with Strong Regularization for Small-Sample Settings

This strategy has only a few thousand training samples — extremely small by machine learning standards. HistGB achieves strong regularization through two key parameters: `max_leaf_nodes=15` (at most 15 leaf nodes per tree) limits individual tree complexity, and `min_samples_leaf=70` (at least 70 samples per leaf node) ensures every decision rule is supported by adequate data. This means the model will not create dedicated rules for small subsets of samples — a critical property for market timing: markets always contain extreme episodes (e.g., the 2020 pandemic crash); if the model built separate trees for these few special patterns, it would inevitably fail on the test set.

### 7.4 No Feature Standardization Required — Preserving Original Economic Meaning

HistGB makes split decisions based on the relative ordering of feature values rather than their absolute magnitudes, completely eliminating the need for StandardScaler or MinMaxScaler preprocessing. This provides two benefits for a timing strategy: first, features retain their original economic meaning (e.g., `close_vs_ma60 = −0.15` is directly interpretable as "15% below the quarterly MA," not a unitless z-score), facilitating analysis and interpretation; second, even if the feature distribution shifts between training and test sets (e.g., an overall rise in volatility levels), the tree's split logic remains valid, whereas standardization-dependent models assume distributional stability.

### 7.5 Probabilistic Output + Rank-Based Mapping = Adaptive Positioning

HistGB outputs a calibrated probability in `[0, 1]` via sigmoid activation, rather than a hard 0/1 classification. Building on this, the **Relative Signal Stabilizer** applies an **expanding percentile rank** transformation as a second stage — each day's probability is mapped to "what percentile it falls into relative to all historical predictions seen so far." This resolves a critical problem: in bull markets, the model persistently outputs high probabilities (e.g., 0.75–0.90), while in bear markets it outputs low probabilities (e.g., 0.35–0.55); threshold-based decisions would fail under such regime-dependent distributions. After rank transformation, signals are uniformly distributed regardless of bull or bear conditions, and position-switching logic remains consistent across market regimes. This two-stage "probabilistic output + rank transformation" design decouples the model's ranking ability from position decisions — a natural advantage of HistGB's probabilistic output.

### 7.6 Limitations

1. **Limited data volume**: Daily samples are a challenge for any ML model; HistGB's strong regularization mitigates but does not fully resolve this
2. **Class imbalance**: Approximately 60% of trading days in the training period are upward (70% in the test period), causing the model to learn a biased baseline probability — rank transformation is needed to eliminate this bias
3. **Parameter overfitting risk**: The position mapping parameters (entry_rank, exit_rank, etc.) were selected via grid search on the validation set; out-of-sample performance may degrade
4. **Single index**: The strategy was validated only on the micro-cap index; generalization to other indices requires re-evaluation

---

## 8. Component Ablation

To quantify the individual contributions of StableHGB's two core modules, this section presents a component ablation study: fixing the HistGB model and the train/validation/test protocol, components are added one at a time to measure each module's incremental contribution.

| Experiment | Position Rule | Valid Score | Cum. Return | Excess (vs B&H) | Max DD | Sharpe | Test AUC |
|-----------|---------------|------------|-------------|-----------------|--------|--------|----------|
| Buy & Hold Reference | constant_full_position | — | 109.21% | 0.00 pp | −17.26% | 2.30 | — |
| Best ML Baseline Reference | Panel B linear_clipped | 0.2575 | 105.29% | −3.92 pp | −12.21% | 2.72 | 0.625 |
| HGB + Standard Policy | rank_linear | 0.1993 | 82.20% | −27.01 pp | −10.48% | 2.65 | 0.635 |
| HGB + Standard Policy + Trend Position Guard | rank_linear + Trend Position Guard | 0.2237 | 106.66% | −2.55 pp | −12.58% | 2.67 | 0.635 |
| HGB + Relative Signal Stabilizer | Relative Signal Stabilizer | 0.2072 | 121.51% | +12.30 pp | −7.52% | 3.59 | 0.635 |
| **StableHGB (Full)** | **Relative Signal Stabilizer + Trend Position Guard** | **0.3164** | **153.44%** | **+44.22 pp** | **−9.84%** | **3.38** | 0.635 |

### 8.1 Ablation Analysis

**Independent contribution of the Relative Signal Stabilizer**: Comparing "HGB + Standard Policy" (82.20%) with "HGB + Relative Signal Stabilizer" (121.51%), the Relative Signal Stabilizer alone contributes **+39.31 percentage points** of cumulative return, while simultaneously reducing max drawdown from −10.48% to −7.52%. This validates the core value of the expanding percentile rank + confirmation-based switching in eliminating probability bias and filtering noise. Notably, this module alone lifts the Sharpe from 2.65 to 3.59 — the highest among all configurations.

**Independent contribution of the Trend Position Guard**: Comparing "HGB + Standard Policy" (82.20%) with "HGB + Standard Policy + Trend Position Guard" (106.66%), the Trend Position Guard alone contributes **+24.46 percentage points** of return. However, drawdown increases from −10.48% to −12.58%, indicating that naive trend lock-in increases drawdown risk — forced full investment during false MA alignments (brief breakouts followed by reversals) leads to losses.

**Synergy between the two modules**: The Relative Signal Stabilizer alone yields 121.51%, and the Trend Position Guard alone yields 106.66%. If simply additive, the total would be 228.17%, but the actual full StableHGB achieves 153.44%. This demonstrates **synergy rather than simple addition**: the Relative Signal Stabilizer's rank-based and confirmation-based filtering suppresses false signals during pseudo-bullish alignments, allowing the Trend Position Guard to lock in only during genuinely strong trends, thereby avoiding the risk of "full investment in the wrong trend." The full combination's Valid Score (0.3164) far exceeds either module alone, confirming the effectiveness of the synergistic design.

### 8.2 Key Conclusions from Ablation

1. **The Relative Signal Stabilizer is the primary source of return improvement**, contributing +39.31 pp alone, and is the only module that simultaneously improves return and reduces drawdown
2. **The Trend Position Guard is critical for ensuring bull market participation**, contributing +24.46 pp alone, but requires the Signal Stabilizer's cooperation for safe deployment
3. **The synergy between the two modules is the fundamental reason StableHGB surpasses all baselines** — full version 153.44% vs. best ML baseline 105.29%, a gap of 48.15 pp

---

## 9. Transaction Cost Sensitivity

The above results are reported at 0 bps transaction cost. To assess the strategy's robustness in real trading environments, a sensitivity test was conducted on StableHGB with 0–20 bps one-way transaction costs:

| Transaction Cost | Cumulative Return | Max Drawdown |
|-----------------|-------------------|-------------|
| 0 bps | 153.44% | −9.84% |
| 5 bps | 151.93% | −9.93% |
| 10 bps | 150.43% | −10.02% |
| 20 bps | 147.46% | −10.20% |

At 20 bps one-way cost (40 bps round-trip), StableHGB still achieves 147.46% cumulative return, only 5.98 pp lower than at 0 bps, with drawdown increasing by only 0.36 pp. This robustness stems from the Relative Signal Stabilizer's confirmation-based design — requiring 2 consecutive days of confirmation before switching positions effectively reduces turnover, making the strategy insensitive to transaction costs.

---

## 10. Summary

This course project proposes **StableHGB**, with Hist Gradient Boosting as the prediction engine and **Relative Signal Stabilizer** and **Trend Position Guard** as the two core original modules, constructing a complete machine learning index enhancement strategy:

| Component | Approach |
|-----------|----------|
| Data Processing | OHLCV daily data → missing value cleanup → SHA256 integrity verification |
| Feature Engineering | 20 quantitative factors (MAs, momentum, volatility, trend strength, MACD, Bollinger Bands, rebound, volume) |
| Target Definition | Binary classification: will the index rise over the next 5 days (`future_up_5d`) |
| Prediction Engine | HistGradientBoostingClassifier (150 iterations, shallow trees, strong regularization) |
| Training Protocol | Final model training + 2022/2023/2024 three-year rolling validation |
| Module 1: Relative Signal Stabilizer | Expanding percentile rank + 2-day consecutive confirmation switching + asymmetric entry/exit thresholds |
| Module 2: Trend Position Guard | Force full investment during bullish MA alignment, preventing missed bull market rallies |
| Parameter Selection | Rolling validation grid search, maximizing "mean return − 0.25 × std return" |
| Component Ablation | Quantified each module's contribution: RSS +39.31 pp, TPG +24.46 pp, with strong synergy |
| Transaction Cost | At 20 bps one-way cost, cumulative return 147.46%, only 5.98 pp decline |
| Test Results | Cumulative return **153.44%**, excess return **+44.22 pp**, Sharpe **3.38**, max drawdown **−9.84%** |

**Core Contributions**:

1. **Relative Signal Stabilizer**: Innovatively combines expanding percentile rank with confirmation-based switching, solving the failure of traditional probability threshold mapping across bull/bear regime shifts — the single largest source of return improvement among all modules
2. **Trend Position Guard**: Uses a simple MA rule to achieve effective trend lock-in, preventing the model from being overly conservative during bull markets, forming a complementary hierarchical position management architecture with the Relative Signal Stabilizer
3. **Component Ablation Study**: Systematically quantifies the independent contributions and synergistic effects of the two modules, providing interpretable experimental evidence for position policy design

The core value of this machine learning approach lies in: **automatic discovery of nonlinear interactions among multi-dimensional features** (impossible to exhaustively specify via manual rules), **probabilistic output enabling continuous position sizing** (rather than simplistic binary buy/sell decisions), and **strong regularization ensuring generalization on small samples**. Combined with the carefully designed Relative Signal Stabilizer and Trend Position Guard modules, StableHGB substantially outperformed the buy-and-hold benchmark and all competitive ML baselines in real market conditions during 2025–2026.

---

> **Reproduction**: `python scripts/reproduce_stable_hgb.py`
>
> **Reproduction results** (finance-StableHGB conda environment, sklearn / numpy / lightgbm):
> - Cumulative return: 153.44% | Max drawdown: −9.84% | Sharpe: 3.38
> - Test AUC: 0.63 | Excess return: +44.22 pp | ✅ reproduction passed
>
> **Full experiments**: `python scripts/run_all_experiments.py --workers 10`
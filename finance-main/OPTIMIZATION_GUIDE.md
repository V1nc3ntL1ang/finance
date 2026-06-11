# 机器学习优化方案指南

本文档整理了对金融时序择时项目的全面优化建议。

---

## 一、模型层面优化

### 1. 引入更强大的模型

**推荐新增模型：**

| 模型 | 特点 | 预期收益 |
|------|------|----------|
| XGBoost | 业界标杆，正则化能力强 | 高 |
| CatBoost | 自动处理类别特征，抗过拟合 | 中-高 |
| SVM | 处理非线性边界效果好 | 中 |

**实现示例：**

```python
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

# XGBoost
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# CatBoost
cat_clf = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_state=42,
    verbose=False
)
```

### 2. 集成学习（Stacking/Blending）

**原理**：将多个模型的预测结果融合，发挥各自优势。

```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=200, random_state=42))
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    stack_method='predict_proba'  # 使用概率预测
)
```

**预期收益**：提升预测稳定性，降低单一模型风险。

---

## 二、特征工程优化

### 1. 新增特征类型

**① 技术指标扩展**：
- MACD（指数平滑异同移动平均线）
- RSI（相对强弱指数）
- 布林带（Bollinger Bands）
- 情绪指标（涨跌家数比、赚钱效应）

**② 时序特征**：
- 滑动窗口统计特征（均值、标准差、偏度、峰度）
- 时间序列分解特征（趋势、周期、残差）
- 多日滞后收益率

**③ 高阶特征**：
- 特征交叉（如 `momentum * volatility`）
- 非线性变换（log、sqrt、平方）
- 特征差分（变化率）

### 2. 特征选择方法

| 方法 | 适用场景 | 实现复杂度 |
|------|----------|------------|
| 基于树模型的特征重要性 | 非线性模型 | 低 |
| 递归特征消除（RFE） | 需要明确特征数量 | 中 |
| L1正则化 | 线性模型 | 低 |
| 互信息/相关性分析 | 筛选冗余特征 | 低 |

**实现示例：**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# 基于树模型选择特征
selector = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    threshold='median'  # 保留重要性高于中位数的特征
)
X_selected = selector.fit_transform(X, y)
```

---

## 三、超参数调优

### 1. 系统化调参策略

**推荐使用贝叶斯优化**（效果优于网格搜索和随机搜索）：

```python
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def lgbm_objective(n_estimators, learning_rate, num_leaves, min_child_samples):
    model = LGBMClassifier(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    return cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

# 定义参数搜索空间
param_bounds = {
    'n_estimators': (100, 500),
    'learning_rate': (0.01, 0.1),
    'num_leaves': (10, 50),
    'min_child_samples': (10, 100)
}

# 执行贝叶斯优化
optimizer = BayesianOptimization(lgbm_objective, param_bounds, random_state=42)
optimizer.maximize(n_iter=50, init_points=10)

# 获取最佳参数
best_params = optimizer.max['params']
```

**搜索空间建议：**

| 参数 | 搜索范围 | 说明 |
|------|----------|------|
| n_estimators | 100-500 | 树的数量 |
| learning_rate | 0.01-0.1 | 学习率 |
| num_leaves | 10-50 | 叶子节点数 |
| min_child_samples | 10-100 | 最小样本数 |
| subsample | 0.6-1.0 | 样本采样比例 |
| colsample_bytree | 0.6-1.0 | 特征采样比例 |
| reg_alpha | 0-1 | L1正则 |
| reg_lambda | 0-5 | L2正则 |

---

## 四、策略层面优化

### 1. 多时间尺度融合

**原理**：同时使用日线、周线、月线特征，捕捉不同周期的信息。

```python
# 特征融合示例
def create_multi_timeframe_features(df):
    # 日线特征
    daily_features = df[['close', 'volume', 'ma20']]
    
    # 周线特征（重采样）
    weekly_df = df.resample('W').agg({
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    }).shift(1)  # 避免未来数据泄露
    
    # 月线特征
    monthly_df = df.resample('M').agg({
        'close': lambda x: x.iloc[-1] / x.iloc[0] - 1,  # 月度收益率
        'volume': 'mean'
    }).shift(1)
    
    # 合并特征
    return pd.concat([daily_features, weekly_df, monthly_df], axis=1)
```

### 2. 动态阈值调整

**原理**：根据市场状态动态调整仓位阈值。

```python
def dynamic_threshold_strategy(probability, volatility, market_regime):
    """
    根据市场状态动态调整仓位
    
    Args:
        probability: 模型预测上涨概率
        volatility: 当前波动率
        market_regime: 市场状态（'bull'/'bear'/'sideways'）
    """
    base_position = probability  # 基础仓位
    
    # 高波动时期：降低仓位
    if volatility > high_vol_threshold:
        base_position *= 0.7
    
    # 熊市时期：提高买入门槛
    if market_regime == 'bear':
        if probability < 0.6:
            return 0.0  # 低于60%概率不买入
        base_position *= 0.8
    
    return min(max(base_position, 0.0), 1.0)
```

### 3. 止损止盈机制

```python
def apply_risk_management(position, portfolio_value, max_drawdown=0.1, take_profit=0.15):
    """
    应用止损止盈规则
    
    Args:
        position: 当前仓位
        portfolio_value: 组合价值（相对于初始值）
        max_drawdown: 最大允许回撤
        take_profit: 止盈阈值
    """
    # 止损：亏损超过max_drawdown时强制减仓
    if portfolio_value < (1 - max_drawdown):
        return position * 0.5
    
    # 止盈：盈利超过take_profit时锁定部分利润
    if portfolio_value > (1 + take_profit):
        return position * 0.8
    
    return position
```

---

## 五、高级技术

### 1. 时序交叉验证

**重要性**：避免数据泄露，更准确评估模型真实性能。

```python
from sklearn.model_selection import TimeSeriesSplit

# 时间序列专用交叉验证
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred = model.predict_proba(X_test)[:, 1]
```

### 2. 概率校准

**重要性**：模型输出的概率可能未校准，导致仓位决策错误。

```python
from sklearn.calibration import CalibratedClassifierCV

# 使用 Platt Scaling 校准概率
base_model = LGBMClassifier(n_estimators=200)
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Platt Scaling
    cv=5
)
calibrated_model.fit(X_train, y_train)

# 校准后的概率更可靠
probabilities = calibrated_model.predict_proba(X_test)[:, 1]
```

**校准前后对比**：

| 指标 | 未校准 | 校准后 |
|------|--------|--------|
| Brier Score | 0.24 | 0.21 |
| 可靠性曲线 | 偏离对角线 | 接近对角线 |

### 3. 在线学习（增量更新）

**原理**：定期用新数据更新模型，适应市场变化。

```python
from sklearn.linear_model import SGDClassifier

# 初始化在线学习模型
online_clf = SGDClassifier(
    loss='log_loss',  # 对数损失，输出概率
    penalty='l2',
    learning_rate='adaptive',
    eta0=0.1,
    random_state=42
)

# 初始训练
online_clf.fit(X_initial, y_initial)

# 增量更新（例如每月）
for batch_data in monthly_data_batches:
    X_batch, y_batch = batch_data
    online_clf.partial_fit(X_batch, y_batch, classes=[0, 1])
```

---

## 六、优化优先级矩阵

| 优先级 | 优化项 | 预期收益 | 实施难度 | 推荐指数 |
|--------|--------|----------|----------|----------|
| ⭐⭐⭐ | 概率校准 | 高 | 低 | ★★★★★ |
| ⭐⭐⭐ | 时序交叉验证 | 中 | 低 | ★★★★★ |
| ⭐⭐⭐ | 贝叶斯超参数调优 | 高 | 中 | ★★★★☆ |
| ⭐⭐ | 引入XGBoost/CatBoost | 中 | 低 | ★★★★☆ |
| ⭐⭐ | 新增技术指标特征 | 中 | 中 | ★★★☆☆ |
| ⭐⭐ | 动态阈值调整 | 中 | 中 | ★★★☆☆ |
| ⭐ | 集成学习 | 低-中 | 高 | ★★☆☆☆ |
| ⭐ | 在线学习 | 低 | 高 | ★★☆☆☆ |

---

## 七、实施路线图

### 第一阶段（基础优化）
1. ✅ 添加概率校准
2. ✅ 实施时序交叉验证
3. ✅ 对LightGBM进行贝叶斯调优

**预期时间**：1-2周

### 第二阶段（模型扩展）
1. 添加XGBoost和CatBoost模型
2. 对比不同模型的表现
3. 选择最优模型组合

**预期时间**：1周

### 第三阶段（特征工程）
1. 扩展技术指标特征
2. 实施特征选择
3. 尝试多时间尺度融合

**预期时间**：2-3周

### 第四阶段（策略优化）
1. 实现动态阈值调整
2. 添加止损止盈机制
3. 测试不同风险控制策略

**预期时间**：1-2周

### 第五阶段（高级优化）
1. 实现集成学习
2. 探索在线学习方案
3. 构建完整的策略框架

**预期时间**：2-3周

---

## 八、预期效果评估

### 优化前（当前状态）
| 指标 | 数值 |
|------|------|
| 累计收益 | 107.35% |
| 最大回撤 | -14.62% |
| 夏普比率 | 2.69 |

### 优化后（预期）
| 指标 | 预期提升 |
|------|----------|
| 累计收益 | +5-15% |
| 最大回撤 | -2-5%（绝对值） |
| 夏普比率 | +0.3-0.5 |

---

## 九、注意事项

1. **数据泄露**：确保特征计算不使用未来数据
2. **过拟合风险**：使用交叉验证和正则化控制
3. **交易成本**：实际回测中需考虑手续费和滑点
4. **样本外验证**：保留独立测试集评估最终性能
5. **稳健性测试**：在不同市场环境下验证策略

---

*文档版本：v1.0*
*创建日期：2026年6月*

| Experiment | Position rule | Valid score | Cumulative return | Excess vs buy-hold (percentage points) | Max drawdown | Sharpe | Test AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Buy-and-hold reference | constant_full_position |  | 109.21% | 0.00 pp | -17.26% | 2.30 |  |
| Validation-selected Gradient Boosting | Panel B linear_clipped | 0.2575 | 105.29% | -3.92 pp | -12.21% | 2.72 | 0.625 |
| StableHGB classifier + standard policy | rank_linear | 0.1993 | 82.20% | -27.01 pp | -10.48% | 2.65 | 0.635 |
| StableHGB classifier + standard policy + Trend Position Guard | rank_linear + Trend Position Guard | 0.2237 | 106.66% | -2.55 pp | -12.58% | 2.67 | 0.635 |
| StableHGB classifier + Relative Signal Stabilizer | Relative Signal Stabilizer | 0.2072 | 121.51% | 12.30 pp | -7.52% | 3.59 | 0.635 |
| StableHGB | Relative Signal Stabilizer + Trend Position Guard | 0.3164 | 153.44% | 44.22 pp | -9.84% | 3.38 | 0.635 |

All ablation headline results use 0 bps transaction cost.

from __future__ import annotations

import pandas as pd


VALID_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

SELECTION_OBJECTIVE = "weighted_valid_return_sharpe"
RETURN_WEIGHT = 0.5
SHARPE_WEIGHT = 0.5

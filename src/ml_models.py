from __future__ import annotations

import os
from collections import OrderedDict

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def get_ml_models(n_jobs: int | None = None) -> OrderedDict[str, object]:
    worker_count = n_jobs if n_jobs is not None else int(os.environ.get("FINANCE_WORKERS", os.cpu_count() or 1))

    return OrderedDict(
        [
            (
                "logistic_regression",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                    ]
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=8,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=worker_count,
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "hist_gradient_boosting_alignment_confirmation",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=8,
                    min_samples_leaf=38,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "lightgbm",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=30,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                    n_jobs=worker_count,
                ),
            ),
        ]
    )

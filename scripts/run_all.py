from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("daily features", "build_daily_features.py"),
    ("ml dataset", "build_ml_dataset.py"),
    ("baseline metrics", "run_baselines.py"),
    ("baseline plots", "plot_baselines.py"),
    ("ml metrics", "run_ml_baselines.py"),
    ("ml plots", "plot_ml_baselines.py"),
    ("ml ablation metrics", "run_ml_ablation.py"),
    ("ml ablation plots", "plot_ml_ablation.py"),
    ("feature group ablation metrics", "run_feature_group_ablation.py"),
    ("feature group ablation plots", "plot_feature_group_ablation.py"),
]


def main() -> None:
    for step_name, script_name in STEPS:
        print(f"\n== {step_name} ==", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()

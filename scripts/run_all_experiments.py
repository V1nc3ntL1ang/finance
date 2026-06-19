from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_protocol import sort_by_formal_validation_score

STEPS = [
    ("Environment check", "scripts/check_environment.py"),
    ("Smoke tests", "scripts/run_tests.py"),
    ("Build daily data from raw text", "scripts/build_daily_from_raw.py"),
    ("Build daily features", "scripts/build_daily_features.py"),
    ("Build machine-learning dataset", "scripts/build_ml_dataset.py"),
    ("Panel A baselines", "scripts/run_baselines.py"),
    ("Panel B machine-learning baselines", "scripts/run_ml_baselines.py"),
    ("Panel C StableHGB", "scripts/run_stable_hgb.py"),
    ("StableHGB component ablation", "scripts/run_stable_hgb_ablation.py"),
    ("Figures", "scripts/plot_results.py"),
    ("StableHGB reproduction check", "scripts/reproduce_stable_hgb.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full experiment workflow.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FINANCE_WORKERS", os.cpu_count() or 1)),
        help="Worker count passed to scripts that parallelize policy evaluation.",
    )
    return parser.parse_args()


def run_step(label: str, script: str, env: dict[str, str]) -> None:
    command = [sys.executable, "-u", script]
    print(f"\n=== {label} ===", flush=True)
    print(f"command: {' '.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    elapsed = time.perf_counter() - started
    print(f"completed {label} in {elapsed:.1f}s", flush=True)


def print_summary() -> None:
    baseline = pd.read_csv(PROJECT_ROOT / "outputs/metrics/baselines.csv")
    ml = pd.read_csv(PROJECT_ROOT / "outputs/metrics/ml_baselines.csv")
    stable_hgb = pd.read_csv(PROJECT_ROOT / "outputs/metrics/stable_hgb_metrics.csv")

    best_baseline = baseline.sort_values("cumulative_return", ascending=False).iloc[0]
    best_ml = sort_by_formal_validation_score(ml).iloc[0]
    stable = stable_hgb.iloc[0]

    print("\n=== Formal result summary ===", flush=True)
    print(
        f"best baseline: {best_baseline['strategy']} "
        f"return={best_baseline['cumulative_return']:.4f} "
        f"max_dd={best_baseline['max_drawdown']:.4f} "
        f"sharpe={best_baseline['sharpe']:.4f}",
        flush=True,
    )
    print(
        f"best ML baseline: {best_ml['model']} "
        f"return={best_ml['cumulative_return']:.4f} "
        f"max_dd={best_ml['max_drawdown']:.4f} "
        f"sharpe={best_ml['sharpe']:.4f}",
        flush=True,
    )
    print(
        f"StableHGB: {stable['strategy']} "
        f"return={stable['cumulative_return']:.4f} "
        f"excess_pp={stable['excess_return_pp_vs_buy_hold']:.4f} "
        f"max_dd={stable['max_drawdown']:.4f} "
        f"sharpe={stable['sharpe']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FINANCE_WORKERS"] = str(args.workers)
    for thread_variable in [
        "LOKY_MAX_CPU_COUNT",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[thread_variable] = str(args.workers)

    print(
        f"experiment workflow started: workers={args.workers} cwd={PROJECT_ROOT}",
        flush=True,
    )
    started = time.perf_counter()
    for label, script in STEPS:
        run_step(label, script, env)
    print_summary()
    elapsed = time.perf_counter() - started
    print(f"\nexperiment workflow completed in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

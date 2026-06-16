from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("daily features", "build_daily_features.py"),
    ("ml dataset", "build_ml_dataset.py"),
    ("baseline metrics", "run_baselines.py"),
    ("baseline plots", "plot_baselines.py"),
    ("ml metrics", "run_ml_baselines.py"),
    ("ml plots", "plot_ml_baselines.py"),
    ("feature importance", "analyze_feature_importance.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full finance timing workflow.")
    parser.add_argument("--record", action="store_true", help="Record this run after the workflow finishes.")
    parser.add_argument("--name", help="Experiment name for non-interactive recording.")
    parser.add_argument("--purpose", help="Experiment purpose for non-interactive recording.")
    parser.add_argument("--optimization", action="append", default=[], help="Optimization label; can be repeated.")
    parser.add_argument("--notes", default="", help="Optional experiment notes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_steps = len(STEPS)
    run_started = time.perf_counter()
    worker_count = os.cpu_count() or 1
    child_env = os.environ.copy()
    child_env.setdefault("FINANCE_WORKERS", str(worker_count))
    child_env.setdefault("OMP_NUM_THREADS", str(worker_count))
    child_env.setdefault("OPENBLAS_NUM_THREADS", str(worker_count))
    child_env.setdefault("MKL_NUM_THREADS", str(worker_count))
    child_env.setdefault("VECLIB_MAXIMUM_THREADS", str(worker_count))
    child_env.setdefault("NUMEXPR_NUM_THREADS", str(worker_count))

    print(f"aggressive CPU mode: workers={child_env['FINANCE_WORKERS']}", flush=True)

    with tqdm(total=total_steps, desc="run_all", unit="step") as progress:
        for step_name, script_name in STEPS:
            step_started = time.perf_counter()
            progress.set_description(step_name)

            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
                cwd=PROJECT_ROOT,
                env=child_env,
                check=True,
            )

            step_elapsed = time.perf_counter() - step_started
            progress.set_postfix_str(f"{step_elapsed:.1f}s")
            progress.update(1)

    run_elapsed = time.perf_counter() - run_started
    print(f"\ncompleted {total_steps} steps in {run_elapsed:.1f}s", flush=True)

    if not args.record:
        return

    try:
        from src.experiment_recorder import ExperimentRecorder
        recorder = ExperimentRecorder()

        if args.name and args.purpose:
            exp_id = recorder.record(
                name=args.name,
                purpose=args.purpose,
                optimizations=args.optimization,
                notes=args.notes,
                runtime=run_elapsed,
            )
            print(f"recorded experiment: {exp_id}", flush=True)
        else:
            recorder.record_interactive(runtime=run_elapsed)
    except ImportError as e:
        print(f"警告: 无法导入实验记录器 - {e}")


if __name__ == "__main__":
    main()

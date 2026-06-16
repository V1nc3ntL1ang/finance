from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_recorder import ExperimentRecorder


def list_experiments(recorder: ExperimentRecorder):
    df = recorder.list_experiments()
    if df.empty:
        print("暂无实验记录")
        return
    print(df.to_string(index=False))


def show_experiment(recorder: ExperimentRecorder, exp_id: str):
    exp = recorder.get_experiment(exp_id)
    if not exp:
        print(f"未找到实验: {exp_id}")
        return

    print("\n" + "=" * 50)
    print(f"实验详情: {exp['exp_id']}")
    print("=" * 50)
    print(f"名称: {exp['name']}")
    print(f"时间: {exp['timestamp']}")
    print(f"运行时间: {exp['runtime']:.1f}s")
    print(f"Git版本: {exp['git_commit']}")
    print(f"\n优化目的:")
    print(f"  {exp['purpose']}")
    print(f"\n优化内容:")
    for opt in exp["optimizations"]:
        print(f"  - {opt}")
    if exp["notes"]:
        print(f"\n备注: {exp['notes']}")
    if exp["metrics"]:
        print("\n最佳结果（按 Valid 综合分选择）:")
        print(f"  模型: {exp['metrics']['best_model']}")
        print(f"  Valid综合分: {exp['metrics'].get('valid_selection_score', 0):.4f}")
        print(f"  Valid累计收益: {exp['metrics'].get('valid_cumulative_return', 0):.2%}")
        print(f"  Valid夏普: {exp['metrics'].get('valid_sharpe', 0):.2f}")
        print(f"  累计收益: {exp['metrics']['cumulative_return']:.2%}")
        print(f"  最大回撤: {exp['metrics']['max_drawdown']:.2%}")
        print(f"  夏普比率: {exp['metrics']['sharpe']:.2f}")
        print(f"  超额收益: {exp['metrics']['excess_return']:.2%}")


def compare_experiments(recorder: ExperimentRecorder, exp_ids: list[str]):
    df = recorder.compare_experiments(exp_ids)
    if df.empty:
        print("未找到实验")
        return
    columns = [
        "exp_id",
        "name",
        "valid_selection_score",
        "valid_cumulative_return",
        "cumulative_return",
        "max_drawdown",
        "sharpe",
    ]
    existing_columns = [column for column in columns if column in df.columns]
    print(df[existing_columns].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="实验记录管理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="列出所有实验")

    show_parser = subparsers.add_parser("show", help="查看实验详情")
    show_parser.add_argument("exp_id", help="实验ID")

    compare_parser = subparsers.add_parser("compare", help="对比多个实验")
    compare_parser.add_argument("exp_ids", nargs="+", help="实验ID列表")

    args = parser.parse_args()

    recorder = ExperimentRecorder()

    if args.command == "list":
        list_experiments(recorder)
    elif args.command == "show":
        show_experiment(recorder, args.exp_id)
    elif args.command == "compare":
        compare_experiments(recorder, args.exp_ids)


if __name__ == "__main__":
    main()

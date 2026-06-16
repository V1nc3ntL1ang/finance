from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ExperimentRecorder:
    def __init__(self, experiment_dir: Path | None = None):
        if experiment_dir is None:
            self.experiment_dir = Path("experiments")
        else:
            self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.experiment_dir / "experiments_summary.csv"
        self.experiments = self._load_experiments()

    def _load_experiments(self) -> list[dict]:
        experiments = []
        for file in sorted(self.experiment_dir.glob("exp_*.json"), reverse=True):
            with open(file, "r", encoding="utf-8") as f:
                experiments.append(json.load(f))
        return experiments

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parents[1],
            ).strip()
        except Exception:
            return "unknown"

    def _extract_best_model(self) -> dict[str, Any]:
        from src.paths import ML_METRICS_CSV

        if ML_METRICS_CSV.exists():
            df = pd.read_csv(ML_METRICS_CSV)
            required_columns = {"valid_selection_score", "valid_cumulative_return"}
            if not required_columns.issubset(df.columns):
                return {}

            locked = (
                df[df["policy_selection"] == "locked_multifold_min_score"]
                if "policy_selection" in df.columns
                else pd.DataFrame()
            )
            if locked.empty:
                best_idx = df.sort_values(
                    ["valid_selection_score", "valid_cumulative_return"],
                    ascending=[False, False],
                    kind="mergesort",
                ).index[0]
                selection_rule = "max(valid_selection_score, valid_cumulative_return)"
            else:
                best_idx = locked.sort_values("cumulative_return", ascending=False, kind="mergesort").index[0]
                selection_rule = "locked_multifold_min_score"

            best = df.iloc[best_idx]
            return {
                "selection_rule": selection_rule,
                "best_model": str(best["model"]),
                "valid_selection_score": float(best["valid_selection_score"]),
                "valid_cumulative_return": float(best["valid_cumulative_return"]),
                "valid_sharpe": float(best.get("valid_sharpe", 0)),
                "cumulative_return": float(best["cumulative_return"]),
                "max_drawdown": float(best["max_drawdown"]),
                "sharpe": float(best["sharpe"]),
                "excess_return": float(best.get("excess_return_vs_buy_hold", 0)),
                "test_accuracy": float(best.get("test_accuracy", 0)),
                "test_auc": float(best.get("test_auc", 0)),
            }
        return {}

    def _interactive_input(self) -> dict[str, Any]:
        print("\n" + "=" * 50)
        print("📝 实验记录器")
        print("=" * 50)

        name = input("1. 实验名称 (必填): ").strip()
        while not name:
            name = input("   请输入实验名称: ").strip()

        purpose = input("2. 优化目的 (必填): ").strip()
        while not purpose:
            purpose = input("   请输入优化目的: ").strip()

        options = [
            "参数搜索空间扩展",
            "波动率目标调整",
            "模型架构改进",
            "特征工程优化",
            "仓位策略优化",
            "代码性能优化",
            "数据质量改进",
            "可视化增强",
            "其他",
        ]
        print("\n3. 优化内容 (多选，输入数字，用空格分隔):")
        for i, opt in enumerate(options, 1):
            print(f"   [{i}] {opt}")

        selected = input("   选择: ").strip()
        optimizations = []
        if selected:
            for num in selected.split():
                if num.isdigit() and 1 <= int(num) <= len(options):
                    optimizations.append(options[int(num) - 1])

        if "其他" in optimizations:
            other = input("   请补充其他优化内容: ").strip()
            if other:
                optimizations[optimizations.index("其他")] = other

        notes = input("\n4. 备注 (可选): ").strip()

        print("\n" + "-" * 50)
        print("预览实验信息:")
        print(f"   实验名称: {name}")
        print(f"   优化目的: {purpose}")
        print(f"   优化内容: {optimizations}")
        print(f"   备注: {notes if notes else '无'}")
        print("-" * 50)

        confirm = input("\n确认保存以上信息？ [Y/N]: ").strip().upper()
        while confirm not in ["Y", "N"]:
            confirm = input("   请输入 Y 或 N: ").strip().upper()

        if confirm == "N":
            print("已取消保存")
            return {}

        return {
            "name": name,
            "purpose": purpose,
            "optimizations": optimizations,
            "notes": notes,
        }

    def record(
        self,
        *,
        name: str,
        purpose: str,
        optimizations: list[str] | None = None,
        runtime: float | None = None,
        notes: str = "",
        params: dict[str, Any] | None = None,
    ) -> str:
        return self.create_experiment(
            name=name,
            purpose=purpose,
            optimizations=optimizations or [],
            notes=notes,
            runtime=runtime or 0,
            params=params,
        )

    def create_experiment(
        self,
        name: str,
        purpose: str,
        optimizations: list[str],
        runtime: float,
        notes: str = "",
        params: dict[str, Any] | None = None,
    ) -> str:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment = {
            "exp_id": exp_id,
            "name": name,
            "purpose": purpose,
            "optimizations": optimizations,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "runtime": runtime,
            "git_commit": self._get_git_commit(),
            "metrics": self._extract_best_model(),
            "params": params or {},
        }

        with open(self.experiment_dir / f"{exp_id}.json", "w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2, ensure_ascii=False)

        self._update_summary(experiment)

        self.experiments.insert(0, experiment)

        return exp_id

    def _update_summary(self, experiment: dict):
        data = {
            "exp_id": experiment["exp_id"],
            "name": experiment["name"],
            "timestamp": experiment["timestamp"],
            "runtime": experiment["runtime"],
            "best_model": experiment["metrics"].get("best_model", ""),
            "valid_selection_score": experiment["metrics"].get("valid_selection_score", ""),
            "valid_cumulative_return": experiment["metrics"].get("valid_cumulative_return", ""),
            "cumulative_return": experiment["metrics"].get("cumulative_return", ""),
            "max_drawdown": experiment["metrics"].get("max_drawdown", ""),
            "sharpe": experiment["metrics"].get("sharpe", ""),
            "excess_return": experiment["metrics"].get("excess_return", ""),
            "optimizations": ", ".join(experiment["optimizations"]),
            "purpose": experiment["purpose"],
        }

        if self.summary_path.exists():
            df = pd.read_csv(self.summary_path)
            df = pd.concat([pd.DataFrame([data]), df], ignore_index=True)
        else:
            df = pd.DataFrame([data])

        df.to_csv(self.summary_path, index=False, encoding="utf-8")

    def record_interactive(self, runtime: float | None = None) -> str | None:
        user_input = self._interactive_input()
        if not user_input:
            return None

        exp_id = self.create_experiment(
            name=user_input["name"],
            purpose=user_input["purpose"],
            optimizations=user_input["optimizations"],
            notes=user_input["notes"],
            runtime=runtime or 0,
        )

        print(f"\n✅ 实验记录成功!")
        print(f"   实验ID: {exp_id}")
        print(f"   保存位置: {self.experiment_dir / f'{exp_id}.json'}")

        exp = self.experiments[0]
        if exp and exp["metrics"]:
            print("\n📊 本次实验最佳结果（按 Valid 综合分选择）:")
            print(f"   最佳模型: {exp['metrics']['best_model']}")
            print(f"   Valid综合分: {exp['metrics']['valid_selection_score']:.4f}")
            print(f"   累计收益: {exp['metrics']['cumulative_return']:.2%}")
            print(f"   最大回撤: {exp['metrics']['max_drawdown']:.2%}")
            print(f"   夏普比率: {exp['metrics']['sharpe']:.2f}")
            print(f"   超额收益: {exp['metrics']['excess_return']:.2%}")

        return exp_id

    def get_experiment(self, exp_id: str) -> dict | None:
        for exp in self.experiments:
            if exp["exp_id"] == exp_id:
                return exp
        return None

    def list_experiments(self) -> pd.DataFrame:
        if not self.experiments:
            return pd.DataFrame()

        data = []
        for exp in self.experiments:
            data.append({
                "exp_id": exp["exp_id"],
                "name": exp["name"],
                "timestamp": exp["timestamp"],
                "runtime": f"{exp['runtime']:.1f}s" if exp["runtime"] else "N/A",
                "optimizations": ", ".join(exp["optimizations"]),
                "best_model": exp["metrics"].get("best_model", "N/A"),
                "valid_selection_score": f"{exp['metrics']['valid_selection_score']:.4f}" if exp["metrics"].get("valid_selection_score") else "N/A",
                "cumulative_return": f"{exp['metrics']['cumulative_return']:.2%}" if exp["metrics"].get("cumulative_return") else "N/A",
                "sharpe": f"{exp['metrics']['sharpe']:.2f}" if exp["metrics"].get("sharpe") else "N/A",
            })
        return pd.DataFrame(data)

    def compare_experiments(self, exp_ids: list[str]) -> pd.DataFrame:
        data = []
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                data.append({
                    "exp_id": exp["exp_id"],
                    "name": exp["name"],
                    "timestamp": exp["timestamp"],
                    **exp["metrics"],
                })
        return pd.DataFrame(data)

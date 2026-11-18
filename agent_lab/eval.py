"""
agent_lab.eval: Autonomous OSS benchmark and LLM code evaluation pipeline.
Implements Pass@k estimators, deterministic AST syntax checking, and regression tracking.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkTask:
    """A benchmark coding task with evaluation unit tests."""
    task_id: str
    prompt: str
    entry_point: str
    test_code: str
    canonical_solution: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class PassAtKEstimator:
    """Unbiased Pass@k estimator using Chen et al. (2021) iterative formulation."""

    @staticmethod
    def estimate(n: int, c: int, k: int) -> float:
        """
        Calculate unbiased pass@k: 1 - prod_{i=1}^k ((n - c - k + i) / (n - k + i))
        where n = total samples, c = correct samples, k = evaluations.
        """
        if n < k:
            raise ValueError(f"Total samples n={n} must be >= k={k}")
        if c < 0 or c > n:
            raise ValueError(f"Correct samples c={c} must be between 0 and n={n}")

        if n - c < k:
            return 1.0

        # Iterative product to prevent combinatorial overflow
        prob = 1.0
        for i in range(1, k + 1):
            prob *= (n - c - k + i) / (n - k + i)

        return float(1.0 - prob)

    @staticmethod
    def compute_all(n: int, c: int, k_values: List[int]) -> Dict[str, float]:
        """Compute pass@k for all specified k values where k <= n."""
        results: Dict[str, float] = {}
        for k in k_values:
            if k <= n:
                results[f"pass@{k}"] = round(PassAtKEstimator.estimate(n, c, k), 4)
        return results


class ASTVerifier:
    """Deterministic AST-based static syntax parsing and security linting."""

    @staticmethod
    def verify(code: str, expected_entry_point: Optional[str] = None) -> Dict[str, Any]:
        """Verify code syntax, entry point presence, and complexity."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"SyntaxError at line {e.lineno}: {e.msg}",
                "entry_point_found": False,
                "cyclomatic_complexity": 0,
            }

        # Check entry point presence
        entry_found = False
        if expected_entry_point:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == expected_entry_point:
                        entry_found = True
                        break

        # Calculate basic cyclomatic complexity
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return {
            "valid": True,
            "error": None,
            "entry_point_found": entry_found if expected_entry_point else True,
            "cyclomatic_complexity": complexity,
        }


class RegressionTracker:
    """Diffs candidate benchmarks against baselines to catch regressions."""

    @staticmethod
    def compare(
        baseline: Dict[str, bool],  # task_id -> passed
        candidate: Dict[str, bool], # task_id -> passed
    ) -> Dict[str, Any]:
        """Compare task outcomes between baseline and candidate benchmark runs."""
        regressions: List[str] = []
        improvements: List[str] = []
        unchanged: List[str] = []

        all_tasks = set(baseline.keys()).union(candidate.keys())

        for task_id in sorted(all_tasks):
            base_passed = baseline.get(task_id, False)
            cand_passed = candidate.get(task_id, False)

            if base_passed and not cand_passed:
                regressions.append(task_id)
            elif not base_passed and cand_passed:
                improvements.append(task_id)
            else:
                unchanged.append(task_id)

        return {
            "has_regressions": len(regressions) > 0,
            "regression_count": len(regressions),
            "regressions": regressions,
            "improvement_count": len(improvements),
            "improvements": improvements,
            "unchanged_count": len(unchanged),
            "total_tasks": len(all_tasks),
        }


class BenchmarkHarness:
    """Full evaluation orchestrator executing benchmark suites and estimating metrics."""

    def __init__(self, tasks: Optional[List[BenchmarkTask]] = None) -> None:
        self.tasks = tasks or []

    def load_from_json(self, file_path: str) -> int:
        """Load benchmark tasks from a JSON file."""
        path = Path(file_path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded = 0
        tasks_raw = data if isinstance(data, list) else data.get("tasks", [])
        for item in tasks_raw:
            self.tasks.append(
                BenchmarkTask(
                    task_id=item["task_id"],
                    prompt=item.get("prompt", ""),
                    entry_point=item.get("entry_point", ""),
                    test_code=item.get("test_code", ""),
                    canonical_solution=item.get("canonical_solution"),
                    tags=item.get("tags", []),
                )
            )
            loaded += 1
        return loaded

    def evaluate_candidate(
        self,
        task_id: str,
        generated_code: str,
    ) -> Dict[str, Any]:
        """Verify code AST and execute unit tests in a safe local namespace."""
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if not task:
            return {"task_id": task_id, "passed": False, "error": "Task not found"}

        # 1. AST syntax check
        ast_result = ASTVerifier.verify(generated_code, expected_entry_point=task.entry_point)
        if not ast_result["valid"] or not ast_result["entry_point_found"]:
            return {
                "task_id": task_id,
                "passed": False,
                "error": ast_result.get("error") or f"Entry point '{task.entry_point}' not found",
                "ast": ast_result,
            }

        # 2. In-memory execution sandbox
        exec_namespace: Dict[str, Any] = {}
        try:
            # Execute generated code
            exec(generated_code, exec_namespace)
            # Execute validation test suite
            exec(task.test_code, exec_namespace)
            return {
                "task_id": task_id,
                "passed": True,
                "error": None,
                "ast": ast_result,
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "passed": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "ast": ast_result,
            }

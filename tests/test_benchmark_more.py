import json
from pathlib import Path
from agent_lab.eval import BenchmarkHarness, BenchmarkTask, ASTVerifier


def test_benchmark_load_and_run(tmp_path: Path):
    tasks_file = tmp_path / "benchmarks.json"
    tasks_data = {
        "tasks": [
            {
                "task_id": "core/fib",
                "prompt": "Write fibonacci function",
                "entry_point": "fib",
                "test_code": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(5) == 5\n",
            },
            {
                "task_id": "core/reverse",
                "prompt": "Write string reversal",
                "entry_point": "rev",
                "test_code": "assert rev('abc') == 'cba'\n",
            }
        ]
    }
    tasks_file.write_text(json.dumps(tasks_data), encoding="utf-8")

    harness = BenchmarkHarness()
    count = harness.load_from_json(str(tasks_file))
    assert count == 2

    # Test fib success
    fib_code = "def fib(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n"
    res = harness.evaluate_candidate("core/fib", fib_code)
    assert res["passed"] is True

    # Test rev syntax fail
    res_err = harness.evaluate_candidate("core/reverse", "def rev(s): return s[::")
    assert res_err["passed"] is False
    assert "SyntaxError" in res_err["error"]

    # Test non-existent task
    res_none = harness.evaluate_candidate("non/existent", "pass")
    assert res_none["passed"] is False
    assert res_none["error"] == "Task not found"


def test_ast_verifier_complex_branches():
    code = """
def complex_func(x):
    if x > 10:
        for i in range(x):
            while i > 5:
                try:
                    pass
                except Exception:
                    pass
    return x
"""
    res = ASTVerifier.verify(code, expected_entry_point="complex_func")
    assert res["valid"] is True
    assert res["cyclomatic_complexity"] >= 5

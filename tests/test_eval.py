import pytest
from agent_lab.eval import (
    PassAtKEstimator,
    ASTVerifier,
    RegressionTracker,
    BenchmarkHarness,
    BenchmarkTask,
)


def test_pass_at_k_estimator():
    # If all samples correct, pass@k is 1.0
    assert PassAtKEstimator.estimate(n=10, c=10, k=1) == 1.0
    assert PassAtKEstimator.estimate(n=10, c=10, k=5) == 1.0

    # If no samples correct, pass@k is 0.0
    assert PassAtKEstimator.estimate(n=10, c=0, k=1) == 0.0

    # Intermediate calculation
    score = PassAtKEstimator.estimate(n=10, c=5, k=1)
    assert 0.49 <= score <= 0.51

    multi = PassAtKEstimator.compute_all(n=10, c=5, k_values=[1, 5])
    assert "pass@1" in multi
    assert "pass@5" in multi

    with pytest.raises(ValueError):
        PassAtKEstimator.estimate(n=5, c=2, k=10)


def test_ast_verifier():
    valid_code = "def add(a, b):\n    if a > 0:\n        return a + b\n    return b\n"
    res = ASTVerifier.verify(valid_code, expected_entry_point="add")
    assert res["valid"] is True
    assert res["entry_point_found"] is True
    assert res["cyclomatic_complexity"] >= 2

    # Missing entry point
    res_missing = ASTVerifier.verify(valid_code, expected_entry_point="subtract")
    assert res_missing["valid"] is True
    assert res_missing["entry_point_found"] is False

    # Syntax error
    invalid_code = "def broken(a: return a +"
    res_invalid = ASTVerifier.verify(invalid_code)
    assert res_invalid["valid"] is False
    assert "SyntaxError" in res_invalid["error"]


def test_regression_tracker():
    baseline = {"task_1": True, "task_2": True, "task_3": False}
    candidate = {"task_1": True, "task_2": False, "task_3": True}

    diff = RegressionTracker.compare(baseline, candidate)
    assert diff["has_regressions"] is True
    assert diff["regression_count"] == 1
    assert "task_2" in diff["regressions"]
    assert diff["improvement_count"] == 1
    assert "task_3" in diff["improvements"]


def test_benchmark_harness_evaluation():
    task = BenchmarkTask(
        task_id="math/sum",
        prompt="Write a function `sum_two(a, b)` returning the sum of two integers.",
        entry_point="sum_two",
        test_code="assert sum_two(2, 3) == 5\nassert sum_two(-1, 1) == 0\n",
    )
    harness = BenchmarkHarness(tasks=[task])

    # Correct solution
    correct_code = "def sum_two(a, b):\n    return a + b\n"
    eval_res = harness.evaluate_candidate(task_id="math/sum", generated_code=correct_code)
    assert eval_res["passed"] is True
    assert eval_res["error"] is None

    # Incorrect solution
    buggy_code = "def sum_two(a, b):\n    return a - b\n"
    eval_buggy = harness.evaluate_candidate(task_id="math/sum", generated_code=buggy_code)
    assert eval_buggy["passed"] is False
    assert "AssertionError" in eval_buggy["error"]

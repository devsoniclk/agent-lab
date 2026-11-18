"""
Unit tests for TrajectoryEvaluator, StatisticalInterval, and BenchmarkHarness.run_suite.
"""

from __future__ import annotations

import os
import pytest
from agent_lab.eval import TrajectoryEvaluator, StatisticalInterval, BenchmarkHarness


def test_trajectory_evaluator_metrics() -> None:
    trajectory = [
        {"tool": "agentlab_checkpoint_create", "args": {"label": "start"}},
        {"tool": "file_write", "args": {"path": "src/main.py"}},
        {"tool": "pytest_run", "args": {}},
    ]
    expected = ["agentlab_checkpoint_create", "file_write", "pytest_run"]

    res = TrajectoryEvaluator.evaluate(trajectory=trajectory, expected_tools=expected)
    assert res["valid"] is True
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["step_count"] == 3
    assert res["step_efficiency"] == 1.0
    assert res["has_loop"] is False


def test_trajectory_evaluator_extra_and_missing() -> None:
    # 2 correct, 1 unexpected -> precision = 2/3, recall = 2/2 = 1.0
    trajectory = [
        {"tool": "tool_a", "args": {}},
        {"tool": "tool_b", "args": {}},
        {"tool": "tool_unexpected", "args": {}},
    ]
    expected = ["tool_a", "tool_b"]
    res = TrajectoryEvaluator.evaluate(trajectory=trajectory, expected_tools=expected)
    assert res["recall"] == 1.0
    assert res["precision"] == round(2 / 3, 4)
    assert 0.7 < res["f1_score"] < 0.9


def test_trajectory_sequence_rules() -> None:
    # Rule: checkpoint must be called before file_write
    rules = [("agentlab_checkpoint_create", "file_write")]

    # 1. Valid order
    valid_traj = [
        {"tool": "agentlab_checkpoint_create", "args": {}},
        {"tool": "file_write", "args": {}},
    ]
    r1 = TrajectoryEvaluator.evaluate(
        trajectory=valid_traj,
        expected_tools=["agentlab_checkpoint_create", "file_write"],
        sequence_rules=rules,
    )
    assert r1["valid"] is True
    assert len(r1["sequence_violations"]) == 0

    # 2. Inverted order (file_write before checkpoint)
    invalid_traj = [
        {"tool": "file_write", "args": {}},
        {"tool": "agentlab_checkpoint_create", "args": {}},
    ]
    r2 = TrajectoryEvaluator.evaluate(
        trajectory=invalid_traj,
        expected_tools=["agentlab_checkpoint_create", "file_write"],
        sequence_rules=rules,
    )
    assert r2["valid"] is False
    assert len(r2["sequence_violations"]) == 1
    assert "Order violation" in r2["sequence_violations"][0]

    # 3. Missing prerequisite
    missing_prereq_traj = [{"tool": "file_write", "args": {}}]
    r3 = TrajectoryEvaluator.evaluate(
        trajectory=missing_prereq_traj,
        expected_tools=["file_write"],
        sequence_rules=rules,
    )
    assert r3["valid"] is False
    assert len(r3["sequence_violations"]) == 1
    assert "without required prerequisite" in r3["sequence_violations"][0]


def test_trajectory_loop_detection() -> None:
    # 3 identical consecutive calls -> loop
    loop_traj = [
        {"tool": "poll_status", "args": {"job_id": 123}},
        {"tool": "poll_status", "args": {"job_id": 123}},
        {"tool": "poll_status", "args": {"job_id": 123}},
    ]
    res = TrajectoryEvaluator.evaluate(trajectory=loop_traj, expected_tools=["poll_status"])
    assert res["has_loop"] is True
    assert res["valid"] is False
    assert len(res["loop_details"]) == 1

    # 2 identical calls -> no loop
    ok_traj = [
        {"tool": "poll_status", "args": {"job_id": 123}},
        {"tool": "poll_status", "args": {"job_id": 123}},
        {"tool": "other_tool", "args": {}},
    ]
    res2 = TrajectoryEvaluator.evaluate(trajectory=ok_traj, expected_tools=["poll_status", "other_tool"])
    assert res2["has_loop"] is False


def test_trajectory_max_steps() -> None:
    traj = [{"tool": f"step_{i}", "args": {}} for i in range(10)]
    res = TrajectoryEvaluator.evaluate(trajectory=traj, expected_tools=["step_0"], max_steps=5)
    assert res["exceeded_max_steps"] is True
    assert res["valid"] is False


def test_statistical_interval_wilson() -> None:
    # Zero trials
    assert StatisticalInterval.wilson_score_interval(0, 0) == (0.0, 0.0)

    # Perfect pass rate
    low, high = StatisticalInterval.wilson_score_interval(10, 10, confidence=0.95)
    assert low > 0.65
    assert high == 1.0

    # 50% pass rate
    low, high = StatisticalInterval.wilson_score_interval(50, 100, confidence=0.95)
    assert 0.35 < low < 0.45
    assert 0.55 < high < 0.65

    # Invalid input
    with pytest.raises(ValueError):
        StatisticalInterval.wilson_score_interval(15, 10)


def test_benchmark_harness_run_suite() -> None:
    suite_path = os.path.join(
        os.path.dirname(__file__), "..", "benchmarks", "humaneval_mini.json"
    )
    harness = BenchmarkHarness()
    res = harness.run_suite(tasks_file=suite_path)

    assert res["total"] == 10
    assert res["passed"] == 10
    assert res["failed"] == 0
    assert res["pass_rate"] == 1.0
    assert len(res["confidence_interval_95"]) == 2
    assert res["confidence_interval_95"][0] > 0.65

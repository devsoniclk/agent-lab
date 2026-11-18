"""
Additional edge-case tests to achieve maximum test coverage for Agent-Lab v0.2.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from click.testing import CliRunner
from agent_lab.cli import main
from agent_lab.eval import StatisticalInterval, TrajectoryEvaluator, BenchmarkHarness, BenchmarkTask
from agent_lab.telemetry import TelemetryCollector
from agent_lab.mcp_server import agentlab_eval_trajectory, agentlab_benchmark_run_suite


def test_statistical_interval_different_confidences() -> None:
    # 99% confidence
    low99, high99 = StatisticalInterval.wilson_score_interval(8, 10, confidence=0.99)
    assert low99 < 0.8 < high99

    # 90% confidence
    low90, high90 = StatisticalInterval.wilson_score_interval(8, 10, confidence=0.90)
    assert low90 < 0.8 < high90

    # < 90% fallback
    low_fallback, high_fallback = StatisticalInterval.wilson_score_interval(8, 10, confidence=0.80)
    assert low_fallback < 0.8 < high_fallback


def test_trajectory_evaluator_empty_cases() -> None:
    # Empty trajectory and empty expected
    res_empty = TrajectoryEvaluator.evaluate(trajectory=[], expected_tools=[])
    assert res_empty["precision"] == 1.0
    assert res_empty["recall"] == 1.0
    assert res_empty["f1_score"] == 1.0
    assert res_empty["step_efficiency"] == 1.0

    # Malformed non-dict items in trajectory
    res_malformed = TrajectoryEvaluator.evaluate(
        trajectory=["not-a-dict", 123, None, {"tool": "valid_tool", "args": {}}],  # type: ignore
        expected_tools=["valid_tool"],
    )
    assert res_malformed["precision"] == 1.0
    assert res_malformed["recall"] == 1.0
    assert res_malformed["f1_score"] == 1.0


def test_benchmark_harness_empty_and_custom_generator() -> None:
    harness = BenchmarkHarness()
    # Empty suite
    res_empty = harness.run_suite()
    assert res_empty["total"] == 0

    # Custom candidate generator
    task = BenchmarkTask(
        task_id="Custom/1",
        prompt="def add_one(x: int) -> int:\n    pass\n",
        entry_point="add_one",
        test_code="assert add_one(5) == 6\n",
        canonical_solution=None,
    )
    harness.tasks = [task]

    def generator(t: BenchmarkTask) -> str:
        return "def add_one(x: int) -> int:\n    return x + 1\n"

    res = harness.run_suite(candidate_generator=generator)
    assert res["passed"] == 1

    # Task without canonical solution fallback
    harness2 = BenchmarkHarness(tasks=[task])
    res_no_solution = harness2.run_suite()
    assert res_no_solution["failed"] == 1


def test_mcp_server_v2_error_paths() -> None:
    # Invalid JSON in trajectory
    err_traj = agentlab_eval_trajectory(trajectory_json="invalid json", expected_tools_json="[]")
    assert json.loads(err_traj)["status"] == "error"

    # Invalid file in run suite
    err_suite = agentlab_benchmark_run_suite(tasks_file="/nonexistent/path/tasks.json")
    assert json.loads(err_suite)["status"] == "error"


def test_cli_trajectory_error_paths(tmp_path: Path) -> None:
    runner = CliRunner()
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json content", encoding="utf-8")

    res = runner.invoke(main, ["eval", "trajectory", "--trajectory", str(bad_json), "--expected", "tool_a"])
    assert res.exit_code == 1
    assert "Error:" in res.output

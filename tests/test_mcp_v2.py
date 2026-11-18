"""
Unit tests for expanded FastMCP server tools in Agent-Lab v0.2.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from agent_lab.mcp_server import (
    agentlab_eval_trajectory,
    agentlab_telemetry_get_summary,
    agentlab_benchmark_run_suite,
)


def test_mcp_eval_trajectory() -> None:
    traj = [
        {"tool": "agentlab_checkpoint_create", "args": {}},
        {"tool": "file_write", "args": {}},
    ]
    expected = ["agentlab_checkpoint_create", "file_write"]

    res_raw = agentlab_eval_trajectory(
        trajectory_json=json.dumps(traj),
        expected_tools_json=json.dumps(expected),
    )
    res = json.loads(res_raw)
    assert res["status"] == "success"
    assert res["evaluation"]["valid"] is True
    assert res["evaluation"]["f1_score"] == 1.0


def test_mcp_telemetry_summary() -> None:
    res_raw = agentlab_telemetry_get_summary()
    res = json.loads(res_raw)
    assert res["status"] == "success"
    assert "telemetry" in res
    assert "sessions_created" in res["telemetry"]


def test_mcp_benchmark_run_suite() -> None:
    suite_path = str(Path(__file__).parent.parent / "benchmarks" / "sample_humaneval_task.json")
    res_raw = agentlab_benchmark_run_suite(tasks_file=suite_path)
    res = json.loads(res_raw)
    assert res["status"] == "success"
    assert res["summary"]["total"] == 2
    assert res["summary"]["passed"] == 2

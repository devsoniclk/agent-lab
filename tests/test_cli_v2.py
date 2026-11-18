"""
Unit tests for expanded CLI commands in Agent-Lab v0.2.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from click.testing import CliRunner
from agent_lab.cli import main


def test_cli_eval_pass_at_k_with_ci() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["eval", "pass-at-k", "-n", "10", "-c", "8", "-k", "1"])
    assert res.exit_code == 0
    assert "Pass@1:" in res.output
    assert "95% Confidence Interval (Wilson):" in res.output


def test_cli_eval_trajectory_success(tmp_path: Path) -> None:
    traj_file = tmp_path / "trajectory.json"
    traj_data = [
        {"tool": "agentlab_checkpoint_create", "args": {}},
        {"tool": "file_write", "args": {}},
    ]
    traj_file.write_text(json.dumps(traj_data), encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        main,
        ["eval", "trajectory", "--trajectory", str(traj_file), "--expected", "agentlab_checkpoint_create,file_write"],
    )
    assert res.exit_code == 0
    assert "✓ Valid" in res.output
    assert "Tool-Selection F1" in res.output


def test_cli_eval_trajectory_failure_on_loop(tmp_path: Path) -> None:
    traj_file = tmp_path / "bad_trajectory.json"
    traj_data = [
        {"tool": "bad_loop", "args": {"id": 1}},
        {"tool": "bad_loop", "args": {"id": 1}},
        {"tool": "bad_loop", "args": {"id": 1}},
    ]
    traj_file.write_text(json.dumps(traj_data), encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(
        main,
        ["eval", "trajectory", "--trajectory", str(traj_file), "--expected", "bad_loop"],
    )
    assert res.exit_code == 1
    assert "✗ Failed Rules" in res.output
    assert "Loop Detected" in res.output


def test_cli_eval_suite() -> None:
    suite_path = Path(__file__).parent.parent / "benchmarks" / "sample_humaneval_task.json"
    runner = CliRunner()
    res = runner.invoke(main, ["eval", "suite", "--tasks", str(suite_path)])
    assert res.exit_code == 0
    assert "PASSED" in res.output
    assert "Pass Rate: 100.0%" in res.output


def test_cli_telemetry_commands(tmp_path: Path) -> None:
    runner = CliRunner()

    # Test show
    res_show = runner.invoke(main, ["telemetry", "show"])
    assert res_show.exit_code == 0
    assert "Agent Lifecycle & Safety Metrics" in res_show.output

    # Test export markdown
    out_md = tmp_path / "report.md"
    res_exp_md = runner.invoke(main, ["telemetry", "export", "--format", "markdown", "--output", str(out_md)])
    assert res_exp_md.exit_code == 0
    assert out_md.exists()
    assert "# Agent-Lab Telemetry" in out_md.read_text(encoding="utf-8")

    # Test export json
    out_json = tmp_path / "report.json"
    res_exp_json = runner.invoke(main, ["telemetry", "export", "--format", "json", "--output", str(out_json)])
    assert res_exp_json.exit_code == 0
    assert out_json.exists()
    assert "sessions_created" in out_json.read_text(encoding="utf-8")

"""
tests/test_cli_v9.py: CLI test suite for Multi-Agent Arena and Swarm Workflow Commands.
"""

import json
from pathlib import Path
from click.testing import CliRunner

from agent_lab.cli import main


def test_cli_arena_help():
    runner = CliRunner()
    result = runner.invoke(main, ["arena", "--help"])
    assert result.exit_code == 0
    assert "Multi-Agent Tournament Arena" in result.output
    assert "match" in result.output
    assert "tournament" in result.output
    assert "ratings" in result.output
    assert "benchmark-swarm" in result.output


def test_cli_arena_match_and_ratings(tmp_path: Path):
    runner = CliRunner()
    json_out = tmp_path / "match.json"

    result = runner.invoke(main, [
        "arena", "match",
        "--agent-a", "claude-3-5-sonnet",
        "--agent-b", "gpt-4o",
        "--passed-a", "9",
        "--passed-b", "7",
        "--total", "10",
        "--tokens-a", "1200",
        "--tokens-b", "1800",
        "--json-out", str(json_out),
    ])
    assert result.exit_code == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["agent_a"] == "claude-3-5-sonnet"
    assert data["result_a"] == "WIN"

    # Ratings command
    res_ratings = runner.invoke(main, ["arena", "ratings"])
    assert res_ratings.exit_code == 0
    assert "gpt-4o" in res_ratings.output


def test_cli_arena_tournament(tmp_path: Path):
    runner = CliRunner()
    json_out = tmp_path / "standings.json"

    result = runner.invoke(main, [
        "arena", "tournament",
        "--agents", "agent-x,agent-y,agent-z",
        "--rounds", "1",
        "--json-out", str(json_out),
    ])
    assert result.exit_code == 0
    assert json_out.exists()
    standings = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(standings) >= 3


def test_cli_arena_benchmark_swarm(tmp_path: Path):
    runner = CliRunner()
    json_out = tmp_path / "swarm_report.json"

    result = runner.invoke(main, [
        "arena", "benchmark-swarm",
        "--workflow-id", "wf-e2e-01",
        "--workflow-name", "Enterprise Integration",
        "--planner", "p-1",
        "--coder", "c-1",
        "--reviewer", "r-1",
        "--auditor", "a-1",
        "--json-out", str(json_out),
    ])
    assert result.exit_code == 0
    assert json_out.exists()
    rep = json.loads(json_out.read_text(encoding="utf-8"))
    assert rep["workflow_id"] == "wf-e2e-01"
    assert rep["overall_success"] is True
    assert len(rep["stages"]) == 4


def test_cli_arena_export_import(tmp_path: Path):
    runner = CliRunner()
    export_file = tmp_path / "exported_arena.json"

    # Export
    res_exp = runner.invoke(main, ["arena", "export", "--out", str(export_file)])
    assert res_exp.exit_code == 0
    assert export_file.exists()

    # Import
    res_imp = runner.invoke(main, ["arena", "import", "--file", str(export_file)])
    assert res_imp.exit_code == 0
    assert "Successfully imported" in res_imp.output

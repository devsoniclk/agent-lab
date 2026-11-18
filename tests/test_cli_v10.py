"""
tests/test_cli_v10.py: CLI test suite for Fleet Diagnostics, Missions & Benchmark Commands.
"""

import json
from pathlib import Path
from click.testing import CliRunner

from agent_lab.cli import main


def test_cli_fleet_help():
    runner = CliRunner()
    result = runner.invoke(main, ["fleet", "--help"])
    assert result.exit_code == 0
    assert "Full-Lifecycle Multi-Agent Fleet Diagnostics" in result.output
    assert "doctor" in result.output
    assert "run-mission" in result.output
    assert "benchmark" in result.output


def test_cli_fleet_doctor_table():
    runner = CliRunner()
    result = runner.invoke(main, ["fleet", "doctor"])
    assert result.exit_code == 0
    assert "Fleet Diagnostics (HEALTHY)" in result.output
    assert "semantic_memory" in result.output
    assert "bft_consensus" in result.output
    assert "All sub-engines operational and healthy." in result.output


def test_cli_fleet_doctor_json():
    runner = CliRunner()
    result = runner.invoke(main, ["fleet", "doctor", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "HEALTHY"
    assert data["all_healthy"] is True
    assert len(data["checks"]) == 6


def test_cli_fleet_run_mission():
    runner = CliRunner()
    result = runner.invoke(main, [
        "fleet", "run-mission",
        "--id", "cli_mission_001",
        "--goal", "Test CLI mission execution",
        "--agents", "agent-x,agent-y,agent-z",
    ])
    assert result.exit_code == 0
    assert "Fleet Mission Execution Receipt: cli_mission_001" in result.output
    assert "Status: SUCCESS" in result.output
    assert "BFT Consensus: APPROVED" in result.output
    assert "ZK Proof Verified: YES (SP1 zkVM)" in result.output


def test_cli_fleet_run_mission_json():
    runner = CliRunner()
    result = runner.invoke(main, [
        "fleet", "run-mission",
        "--id", "cli_mission_json_002",
        "--json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["mission_id"] == "cli_mission_json_002"
    assert data["status"] == "SUCCESS"
    assert "sha256_seal" in data


def test_cli_fleet_run_mission_insufficient_agents():
    runner = CliRunner()
    result = runner.invoke(main, [
        "fleet", "run-mission",
        "--agents", "solo-agent",
    ])
    assert result.exit_code != 0
    assert "require at least 3 agents" in result.output


def test_cli_fleet_benchmark():
    runner = CliRunner()
    result = runner.invoke(main, [
        "fleet", "benchmark",
        "--count", "3",
    ])
    assert result.exit_code == 0
    assert "Throughput" in result.output
    assert "Benchmark" in result.output
    assert "Total Missions" in result.output


def test_cli_fleet_benchmark_json():
    runner = CliRunner()
    result = runner.invoke(main, [
        "fleet", "benchmark",
        "--count", "2",
        "--json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["num_missions"] == 2
    assert data["success_count"] == 2
    assert data["success_rate"] == 1.0

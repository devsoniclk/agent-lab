"""
tests/test_cli_v11.py: CLI test suite for Portfolio Ecosystem Commands.
"""

import json
from click.testing import CliRunner
from agent_lab.cli import main


def test_cli_portfolio_help():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "doctor" in result.output
    assert "simulate" in result.output
    assert "benchmark" in result.output


def test_cli_portfolio_status_table():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "status"])
    assert result.exit_code == 0
    assert "Portfolio Status" in result.output
    assert "rollback" in result.output
    assert "solana-tx-lens" in result.output
    assert "agent-witness" in result.output
    assert "agent-lab" in result.output


def test_cli_portfolio_status_json():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "rollback" in data
    assert "solana-tx-lens" in data
    assert "agent-witness" in data
    assert "agent-lab" in data


def test_cli_portfolio_doctor_table():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "doctor"])
    assert result.exit_code == 0
    assert "Portfolio Doctor Diagnostics" in result.output
    assert "Ecosystem Health Status" in result.output
    assert "All 4 portfolio repositories operational" in result.output


def test_cli_portfolio_doctor_json():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "doctor", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["all_healthy"] is True
    assert len(data["checks"]) == 4


def test_cli_portfolio_simulate_panel():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "simulate", "--id", "cli_sim_01", "--goal", "CLI Test Goal"])
    assert result.exit_code == 0
    assert "Unified Portfolio Simulation Receipt" in result.output
    assert "cli_sim_01" in result.output
    assert "SUCCESS" in result.output
    assert "Atomic Rollback Applied" in result.output
    assert "SP1 zkVM Receipt Verified" in result.output


def test_cli_portfolio_simulate_json():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "simulate", "--id", "cli_sim_json", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["simulation_id"] == "cli_sim_json"
    assert data["status"] == "SUCCESS"
    assert data["rollback_applied"] is True
    assert len(data["sha256_seal"]) == 64


def test_cli_portfolio_benchmark_table():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "benchmark", "-n", "3"])
    assert result.exit_code == 0
    assert "Portfolio" in result.output
    assert "Benchmark" in result.output
    assert "Total Iterations" in result.output
    assert "Success Rate" in result.output


def test_cli_portfolio_benchmark_json():
    runner = CliRunner()
    result = runner.invoke(main, ["portfolio", "benchmark", "-n", "2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["num_iterations"] == 2
    assert data["success_count"] == 2
    assert data["success_rate"] == 1.0

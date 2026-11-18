"""
Integration tests for Agent-Lab CLI v0.7.0:
- agent-lab router status (table & JSON output)
- agent-lab router route (direct primary hit & simulated failover)
- agent-lab router reset-circuit (single model & all models)
- agent-lab router benchmark-resilience
"""

from __future__ import annotations

import json
from pathlib import Path
from click.testing import CliRunner

from agent_lab.cli import main


def test_cli_router_help():
    runner = CliRunner()
    result = runner.invoke(main, ["router", "--help"])
    assert result.exit_code == 0
    assert "Adaptive dynamic token rate-limiter" in result.output
    assert "status" in result.output
    assert "route" in result.output
    assert "reset-circuit" in result.output
    assert "benchmark-resilience" in result.output


def test_cli_router_status_table_and_json(tmp_path: Path):
    runner = CliRunner()
    # Table output
    res = runner.invoke(main, ["router", "status"])
    assert res.exit_code == 0
    assert "Model Fleet Circuit Breakers" in res.output
    assert "claude-3-5-sonnet" in res.output
    assert "gpt-4o" in res.output
    assert "Configured Cross-Model Fallback Cascades" in res.output

    # JSON export
    json_path = tmp_path / "fleet_status.json"
    res_json = runner.invoke(main, ["router", "status", "--json-out", str(json_path)])
    assert res_json.exit_code == 0
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "models" in data
    assert "routes" in data
    assert "claude-3-5-sonnet" in data["models"]
    assert data["models"]["claude-3-5-sonnet"]["circuit_state"] == "CLOSED"


def test_cli_router_route_direct_primary():
    runner = CliRunner()
    res = runner.invoke(main, [
        "router", "route",
        "--prompt", "Generate binary search algorithm in Rust",
        "--route", "code-generation",
        "--priority", "HIGH",
        "--agent", "agent-builder",
        "--tokens", "300",
    ])
    assert res.exit_code == 0
    assert "Adaptive Route Execution Receipt" in res.output
    assert "DIRECT PRIMARY HIT" in res.output
    assert "Cascade Attempt Breakdown" in res.output
    assert "claude-3-5-sonnet" in res.output


def test_cli_router_route_with_simulated_failover():
    runner = CliRunner()
    res = runner.invoke(main, [
        "router", "route",
        "--prompt", "Generate binary search algorithm in Rust",
        "--route", "code-generation",
        "--simulate-fail-primary",
        "--tokens", "250",
    ])
    assert res.exit_code == 0
    assert "AUTOMATIC FALLBACK TRIGGERED" in res.output
    assert "deepseek-coder-v2" in res.output


def test_cli_router_reset_circuit():
    runner = CliRunner()
    # Reset single model
    res1 = runner.invoke(main, ["router", "reset-circuit", "--model", "claude-3-5-sonnet"])
    assert res1.exit_code == 0
    assert "Successfully reset circuit breaker for: claude-3-5-sonnet" in res1.output

    # Reset all
    res2 = runner.invoke(main, ["router", "reset-circuit", "--all"])
    assert res2.exit_code == 0
    assert "Successfully reset all circuit breakers to CLOSED" in res2.output

    # Invalid options
    res3 = runner.invoke(main, ["router", "reset-circuit"])
    assert res3.exit_code == 0
    assert "Please specify --model <model_id> or --all" in res3.output


def test_cli_router_benchmark_resilience():
    runner = CliRunner()
    res = runner.invoke(main, [
        "router", "benchmark-resilience",
        "--requests", "10",
        "--fault-rate", "0.4",
        "--route", "code-generation",
    ])
    assert res.exit_code == 0
    assert "Running Resilience Benchmark" in res.output
    assert "Swarm Fault-Tolerance & Resilience Metrics" in res.output
    assert "System Resilience Rate" in res.output

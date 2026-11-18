"""
tests/test_mcp_v11.py: Test suite for FastMCP Portfolio Ecosystem Tools.
"""

import json
from agent_lab.mcp_server import (
    agentlab_portfolio_get_status,
    agentlab_portfolio_run_doctor,
    agentlab_portfolio_simulate_mission,
    agentlab_portfolio_benchmark,
)


def test_mcp_portfolio_get_status():
    res = json.loads(agentlab_portfolio_get_status())
    assert res["status"] == "success"
    portfolio = res["portfolio"]
    assert "rollback" in portfolio
    assert "solana-tx-lens" in portfolio
    assert "agent-witness" in portfolio
    assert "agent-lab" in portfolio


def test_mcp_portfolio_run_doctor():
    res = json.loads(agentlab_portfolio_run_doctor())
    assert res["status"] == "success"
    report = res["report"]
    assert report["all_healthy"] is True
    assert len(report["checks"]) == 4


def test_mcp_portfolio_simulate_mission():
    res = json.loads(agentlab_portfolio_simulate_mission(
        simulation_id="mcp_sim_test",
        task_goal="MCP cross-repository attestation mission",
    ))
    assert res["status"] == "success"
    receipt = res["receipt"]
    assert receipt["simulation_id"] == "mcp_sim_test"
    assert receipt["status"] == "SUCCESS"
    assert receipt["rollback_applied"] is True
    assert receipt["zkvm_receipt_verified"] is True
    assert len(receipt["sha256_seal"]) == 64


def test_mcp_portfolio_benchmark():
    res = json.loads(agentlab_portfolio_benchmark(num_iterations=2))
    assert res["status"] == "success"
    bench = res["benchmark"]
    assert bench["num_iterations"] == 2
    assert bench["success_count"] == 2
    assert bench["success_rate"] == 1.0
    assert bench["simulations_per_second"] > 0.0

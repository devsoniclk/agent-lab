"""
Integration tests for Agent-Lab FastMCP v0.7.0 tools:
- agentlab_router_route
- agentlab_router_get_fleet_status
- agentlab_router_reset_circuit
- agentlab_router_benchmark_resilience
"""

from __future__ import annotations

import json
from agent_lab.mcp_server import (
    agentlab_router_benchmark_resilience,
    agentlab_router_get_fleet_status,
    agentlab_router_reset_circuit,
    agentlab_router_route,
)


def test_mcp_router_route_default():
    raw = agentlab_router_route(
        prompt="Write a hello world in Python",
        route="code-generation",
        priority="NORMAL",
        agent_id="test-mcp-agent",
        tokens=100,
    )
    data = json.loads(raw)
    assert data["status"] == "success"
    receipt = data["receipt"]
    assert receipt["status"] == "success"
    assert receipt["primary_model"] == "claude-3-5-sonnet"
    assert receipt["resolved_model"] == "claude-3-5-sonnet"
    assert receipt["fallback_occurred"] is False
    assert receipt["total_tokens"] > 0
    assert len(receipt["attempts"]) >= 1


def test_mcp_router_route_with_failover():
    raw = agentlab_router_route(
        prompt="Write an LRU cache in Go",
        route="code-generation",
        priority="HIGH",
        agent_id="test-mcp-agent",
        tokens=200,
        simulate_fail_primary=True,
    )
    data = json.loads(raw)
    assert data["status"] == "success"
    receipt = data["receipt"]
    assert receipt["fallback_occurred"] is True
    assert receipt["resolved_model"] == "deepseek-coder-v2"


def test_mcp_router_get_fleet_status():
    # Ensure circuits are clean
    agentlab_router_reset_circuit(reset_all=True)

    raw = agentlab_router_get_fleet_status()
    data = json.loads(raw)
    assert data["status"] == "success"
    fleet = data["fleet_status"]
    assert "models" in fleet
    assert "routes" in fleet
    assert "claude-3-5-sonnet" in fleet["models"]
    assert fleet["models"]["claude-3-5-sonnet"]["circuit_state"] == "CLOSED"

    # Single model query
    raw_single = agentlab_router_get_fleet_status(model_id="gpt-4o")
    data_single = json.loads(raw_single)
    assert data_single["status"] == "success"
    assert data_single["model_id"] == "gpt-4o"
    assert "circuit_state" in data_single["model_status"]


def test_mcp_router_reset_circuit():
    # Reset all
    raw_all = agentlab_router_reset_circuit(reset_all=True)
    data_all = json.loads(raw_all)
    assert data_all["status"] == "success"

    # Reset single model
    raw_one = agentlab_router_reset_circuit(model_id="claude-3-5-sonnet", reset_all=False)
    data_one = json.loads(raw_one)
    assert data_one["status"] == "success"


def test_mcp_router_benchmark_resilience():
    raw = agentlab_router_benchmark_resilience(
        requests=8,
        fault_rate=0.25,
        route="code-generation",
    )
    data = json.loads(raw)
    assert data["status"] == "success"
    assert data["requests"] == 8
    assert data["resilience_rate_pct"] >= 90.0
    assert data["primary_hits"] + data["fallback_recoveries"] == 8
    assert data["unhandled_failures"] == 0

"""
tests/test_mcp_v10.py: Test suite for FastMCP Fleet Diagnostics, Mission Execution & Receipt Tools.
"""

import json
from agent_lab.mcp_server import (
    agentlab_fleet_run_diagnostics,
    agentlab_fleet_run_mission,
    agentlab_fleet_benchmark,
    agentlab_fleet_verify_receipt,
)


def test_mcp_fleet_run_diagnostics():
    res_str = agentlab_fleet_run_diagnostics()
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert "report" in res
    report = res["report"]
    assert report["status"] == "HEALTHY"
    assert report["all_healthy"] is True
    assert len(report["checks"]) == 6


def test_mcp_fleet_run_mission():
    res_str = agentlab_fleet_run_mission(
        mission_id="mcp_mission_001",
        goal="MCP cryptographic attestation mission",
        agents_csv="agent-1,agent-2,agent-3",
        simulate_failover=False,
    )
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert "receipt" in res
    receipt = res["receipt"]
    assert receipt["mission_id"] == "mcp_mission_001"
    assert receipt["status"] == "SUCCESS"
    assert receipt["consensus_approved"] is True
    assert receipt["zk_verified"] is True
    assert "sha256_seal" in receipt


def test_mcp_fleet_benchmark():
    res_str = agentlab_fleet_benchmark(num_missions=2)
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert "benchmark" in res
    bench = res["benchmark"]
    assert bench["num_missions"] == 2
    assert bench["success_count"] == 2
    assert bench["success_rate"] == 1.0


def test_mcp_fleet_verify_receipt():
    # Run mission first to get valid receipt JSON
    mission_res = json.loads(agentlab_fleet_run_mission(mission_id="mcp_v_01"))
    receipt_data = mission_res["receipt"]
    receipt_json = json.dumps(receipt_data)

    # Valid receipt verification
    verify_res_str = agentlab_fleet_verify_receipt(receipt_json=receipt_json)
    verify_res = json.loads(verify_res_str)
    assert verify_res["status"] == "success"
    assert verify_res["valid"] is True
    assert verify_res["claimed_seal"] == verify_res["computed_seal"]

    # Tampered receipt verification
    tampered_data = dict(receipt_data)
    tampered_data["status"] = "FAILED"
    verify_tampered_str = agentlab_fleet_verify_receipt(receipt_json=json.dumps(tampered_data))
    verify_tampered = json.loads(verify_tampered_str)
    assert verify_tampered["status"] == "success"
    assert verify_tampered["valid"] is False

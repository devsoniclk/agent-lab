"""
tests/test_fleet.py: Comprehensive test suite for Fleet Diagnostics, Missions & Full-Lifecycle Execution.
"""

import json
from pathlib import Path
import pytest

from agent_lab.fleet import (
    FleetDiagnostics,
    FleetHealthReport,
    FleetHealthStatus,
    FleetMissionRunner,
    FleetExecutionReceipt,
    MissionConfig,
)


def test_fleet_diagnostics_subsystems():
    diag = FleetDiagnostics()

    mem_check = diag.check_memory_engine()
    assert mem_check.status == FleetHealthStatus.HEALTHY
    assert mem_check.latency_ms >= 0.0
    assert "Vector hypersphere" in mem_check.message

    router_check = diag.check_router_engine()
    assert router_check.status == FleetHealthStatus.HEALTHY
    assert "closed and ready" in router_check.message

    bft_check = diag.check_consensus_engine()
    assert bft_check.status == FleetHealthStatus.HEALTHY
    assert "supermajority quorum verified" in bft_check.message

    zk_check = diag.check_zk_engine()
    assert zk_check.status == FleetHealthStatus.HEALTHY
    assert "Zero-Knowledge" in zk_check.message

    arena_check = diag.check_arena_engine()
    assert arena_check.status == FleetHealthStatus.HEALTHY
    assert "Adaptive Elo" in arena_check.message

    eval_check = diag.check_eval_engine()
    assert eval_check.status == FleetHealthStatus.HEALTHY
    assert "metrics operational" in eval_check.message


def test_fleet_diagnostics_run_diagnostics():
    diag = FleetDiagnostics(version="1.0.0")
    report = diag.run_diagnostics()

    assert report.all_healthy is True
    assert report.status == FleetHealthStatus.HEALTHY
    assert len(report.checks) == 6
    assert report.version == "1.0.0"

    data = report.to_dict()
    assert data["status"] == "HEALTHY"
    assert len(data["checks"]) == 6

    json_str = report.to_json()
    assert "semantic_memory" in json_str
    assert "bft_consensus" in json_str


def test_fleet_mission_execution_success():
    runner = FleetMissionRunner()
    cfg = MissionConfig(
        mission_id="m_test_success_01",
        goal="Synthesize Fibonacci dynamic programming solution",
        agents=["agent-c", "agent-r", "agent-a"],
        target_file="src/fib.py",
        code_proposal="def fib(n: int) -> int:\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n",
    )

    receipt = runner.execute_mission(cfg)

    assert receipt.mission_id == "m_test_success_01"
    assert receipt.status == "SUCCESS"
    assert receipt.consensus_approved is True
    assert receipt.zk_verified is True
    assert receipt.syntax_valid is True
    assert receipt.trajectory_f1 == 1.0
    assert receipt.loop_detected is False
    assert receipt.arena_match_score >= 80.0
    assert receipt.verify_seal() is True


def test_fleet_mission_execution_failover():
    runner = FleetMissionRunner()
    cfg = MissionConfig(
        mission_id="m_test_failover_02",
        goal="Synthesize matrix transpose kernel under provider outage",
        agents=["agent-c", "agent-r", "agent-a"],
        simulate_failover=True,
    )

    receipt = runner.execute_mission(cfg)

    assert receipt.mission_id == "m_test_failover_02"
    assert receipt.status == "SUCCESS"
    assert receipt.routing_attempts >= 2
    assert receipt.routed_model != "claude-3-5-sonnet"
    assert receipt.verify_seal() is True


def test_fleet_execution_receipt_seal_tamper_detection():
    receipt = FleetExecutionReceipt(
        mission_id="m_tamper_01",
        timestamp="2026-11-18T00:00:00Z",
        status="SUCCESS",
        memory_hits=2,
        routed_model="deepseek-coder-v2",
        routing_attempts=1,
        consensus_approved=True,
        consensus_merkle_root="merkle_123",
        zk_verified=True,
        zk_journal_digest="zk_digest_456",
        syntax_valid=True,
        trajectory_f1=1.0,
        loop_detected=False,
        arena_match_score=95.0,
        elo_updated=True,
    )
    receipt.sha256_seal = receipt.compute_seal()
    assert receipt.verify_seal() is True

    # Tamper with status
    receipt.status = "FAILED"
    assert receipt.verify_seal() is False

    # Restore and tamper with arena score
    receipt.status = "SUCCESS"
    receipt.arena_match_score = 99.0
    assert receipt.verify_seal() is False


def test_fleet_benchmark():
    runner = FleetMissionRunner()
    res = runner.benchmark_fleet(num_missions=3)

    assert res["num_missions"] == 3
    assert res["success_count"] == 3
    assert res["success_rate"] == 1.0
    assert res["elapsed_seconds"] >= 0.0
    assert res["missions_per_second"] > 0.0
    assert res["avg_latency_ms"] >= 0.0

"""
tests/test_v10_coverage_edges.py: Edge cases and branch coverage tests for Fleet and v1.0.0 components.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from agent_lab.fleet import (
    FleetDiagnostics,
    FleetExecutionReceipt,
    FleetHealthReport,
    FleetHealthStatus,
    FleetMissionRunner,
    FleetSubsystemCheck,
    MissionConfig,
)
from agent_lab.mcp_server import (
    agentlab_fleet_benchmark,
    agentlab_fleet_run_diagnostics,
    agentlab_fleet_run_mission,
    agentlab_fleet_verify_receipt,
)


def test_fleet_subsystem_check_to_dict():
    check = FleetSubsystemCheck(
        subsystem="test_engine",
        status=FleetHealthStatus.HEALTHY,
        latency_ms=1.2345,
        message="Test check passed",
        details={"key": "val"},
    )
    d = check.to_dict()
    assert d["subsystem"] == "test_engine"
    assert d["status"] == "HEALTHY"
    assert d["latency_ms"] == 1.23
    assert d["message"] == "Test check passed"
    assert d["details"]["key"] == "val"


def test_fleet_diagnostics_unhealthy_branches():
    diag = FleetDiagnostics()

    # Memory failure branch
    with patch("agent_lab.fleet.CrossAgentMemoryStore") as mock_store:
        mock_store.side_effect = RuntimeError("Memory failure injected")
        res = diag.check_memory_engine()
        assert res.status == FleetHealthStatus.UNHEALTHY
        assert "Memory engine verification failed" in res.message

    # Router missing route branch
    with patch("agent_lab.fleet.AdaptiveModelRouter") as mock_router:
        mock_instance = MagicMock()
        mock_instance.get_fleet_status.return_value = {"routes": {}, "models": {}}
        mock_router.return_value = mock_instance
        res = diag.check_router_engine()
        assert res.status == FleetHealthStatus.UNHEALTHY
        assert "Default code-generation cascade route missing" in res.message

    # Router tripped circuit branch
    with patch("agent_lab.fleet.AdaptiveModelRouter") as mock_router:
        mock_instance = MagicMock()
        mock_instance.get_fleet_status.return_value = {
            "routes": {"code-generation": {}},
            "models": {"faulty-model": {"circuit_state": "OPEN"}},
        }
        mock_router.return_value = mock_instance
        res = diag.check_router_engine()
        assert res.status == FleetHealthStatus.DEGRADED
        assert "faulty-model" in res.message

    # Consensus failure branch
    with patch("agent_lab.fleet.SwarmCoordinator") as mock_coord:
        mock_instance = MagicMock()
        mock_res = MagicMock()
        mock_res.committed = False
        mock_res.quorum_reached = False
        mock_instance.run_consensus_round.return_value = mock_res
        mock_coord.return_value = mock_instance
        res = diag.check_consensus_engine()
        assert res.status == FleetHealthStatus.UNHEALTHY
        assert "Consensus verification failed" in res.message

    # ZK failure branch
    with patch("agent_lab.fleet.ZKTrajectoryEngine.verify_receipt") as mock_verify:
        mock_report = MagicMock()
        mock_report.is_valid = False
        mock_report.error_message = "Proof validation failure"
        mock_verify.return_value = mock_report
        res = diag.check_zk_engine()
        assert res.status == FleetHealthStatus.UNHEALTHY
        assert "Proof validation failure" in res.message

    # Arena failure branch
    with patch("agent_lab.fleet.EloRatingSystem") as mock_elo:
        mock_instance = MagicMock()
        mock_instance.update_ratings.return_value = (1400.0, 1600.0, -100.0, 100.0)
        mock_elo.return_value = mock_instance
        res = diag.check_arena_engine()
        assert res.status == FleetHealthStatus.UNHEALTHY
        assert "Elo rating update invariant failed" in res.message

    # Eval failure branch
    with patch("agent_lab.fleet.ASTVerifier.verify") as mock_ast:
        mock_ast.return_value = {"valid": False, "error": "SyntaxError"}
        res = diag.check_eval_engine()
        assert res.status == FleetHealthStatus.UNHEALTHY
        assert "SyntaxError" in res.message


def test_fleet_health_report_degraded_status():
    diag = FleetDiagnostics()
    # Mock router to degraded
    with patch.object(diag, "check_router_engine") as mock_r:
        mock_r.return_value = FleetSubsystemCheck(
            subsystem="adaptive_router",
            status=FleetHealthStatus.DEGRADED,
            latency_ms=0.5,
            message="Circuits tripped",
        )
        report = diag.run_diagnostics()
        assert report.status == FleetHealthStatus.DEGRADED
        assert report.all_healthy is False


def test_fleet_health_report_unhealthy_status():
    diag = FleetDiagnostics()
    # Mock memory to unhealthy
    with patch.object(diag, "check_memory_engine") as mock_m:
        mock_m.return_value = FleetSubsystemCheck(
            subsystem="semantic_memory",
            status=FleetHealthStatus.UNHEALTHY,
            latency_ms=0.5,
            message="Database down",
        )
        report = diag.run_diagnostics()
        assert report.status == FleetHealthStatus.UNHEALTHY
        assert report.all_healthy is False


def test_fleet_mission_custom_invoker_and_failure():
    runner = FleetMissionRunner()

    # Custom invoker that always succeeds
    def custom_ok(model: str, prompt: str):
        return True, "def custom(): pass\n", 25, 50

    cfg_ok = MissionConfig(
        mission_id="m_custom_invoker",
        goal="Test custom invoker hook",
    )
    rec_ok = runner.execute_mission(cfg_ok, provider_invoker=custom_ok)
    assert rec_ok.status == "SUCCESS"

    # Invalid syntax code proposal -> status FAILED
    cfg_bad = MissionConfig(
        mission_id="m_bad_syntax",
        goal="Test invalid syntax handling",
        code_proposal="def broken_syntax(:\n",
    )
    rec_bad = runner.execute_mission(cfg_bad)
    assert rec_bad.status == "FAILED"
    assert rec_bad.syntax_valid is False
    assert rec_bad.verify_seal() is True


def test_mcp_fleet_error_handlers():
    # Invalid JSON receipt in mcp
    res = json.loads(agentlab_fleet_verify_receipt("not-a-json"))
    assert res["status"] == "error"

    # Exception in diagnostics
    with patch("agent_lab.mcp_server._fleet_diagnostics.run_diagnostics", side_effect=ValueError("Simulated mcp diag error")):
        res = json.loads(agentlab_fleet_run_diagnostics())
        assert res["status"] == "error"
        assert "Simulated mcp diag error" in res["error"]

    # Exception in mission runner
    with patch("agent_lab.mcp_server._fleet_runner.execute_mission", side_effect=RuntimeError("Simulated mcp mission error")):
        res = json.loads(agentlab_fleet_run_mission())
        assert res["status"] == "error"
        assert "Simulated mcp mission error" in res["error"]

    # Exception in benchmark
    with patch("agent_lab.mcp_server._fleet_runner.benchmark_fleet", side_effect=RuntimeError("Simulated mcp bench error")):
        res = json.loads(agentlab_fleet_benchmark())
        assert res["status"] == "error"
        assert "Simulated mcp bench error" in res["error"]

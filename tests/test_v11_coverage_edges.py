"""
tests/test_v11_coverage_edges.py: Edge cases and branch coverage tests for Portfolio Ecosystem components.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from agent_lab.portfolio import (
    PortfolioDiscovery,
    PortfolioDoctor,
    PortfolioHealthStatus,
    PortfolioSubsystemCheck,
    PortfolioDoctorReport,
    CrossRepoFleetSimulator,
    UnifiedPortfolioReceipt,
)
import agent_lab.mcp_server as mcp


def test_portfolio_subsystem_check_to_dict():
    c = PortfolioSubsystemCheck(
        subsystem="test_repo",
        status=PortfolioHealthStatus.HEALTHY,
        latency_ms=12.345,
        message="operational",
        version="0.1.0",
        details={"key": "val"},
    )
    d = c.to_dict()
    assert d["subsystem"] == "test_repo"
    assert d["status"] == "HEALTHY"
    assert d["latency_ms"] == 12.35
    assert d["version"] == "0.1.0"


def test_portfolio_doctor_unhealthy_branches():
    doc = PortfolioDoctor()

    with patch.object(doc, "check_rollback", return_value=PortfolioSubsystemCheck(
        subsystem="rollback",
        status=PortfolioHealthStatus.UNHEALTHY,
        latency_ms=10.0,
        message="Simulated error",
        version="0.10.0",
    )):
        rep = doc.run_doctor()
        assert rep.status == PortfolioHealthStatus.UNHEALTHY
        assert rep.all_healthy is False

    with patch.object(doc, "check_solana_tx_lens", return_value=PortfolioSubsystemCheck(
        subsystem="solana-tx-lens",
        status=PortfolioHealthStatus.DEGRADED,
        latency_ms=10.0,
        message="Simulated degraded",
        version="0.10.0",
    )):
        rep = doc.run_doctor()
        assert rep.status == PortfolioHealthStatus.DEGRADED
        assert rep.all_healthy is False


def test_portfolio_doctor_check_exceptions():
    doc = PortfolioDoctor()

    # Exception in check_rollback
    with patch("rollback.core.create_session", side_effect=RuntimeError("Rollback crash")):
        check = doc.check_rollback()
        assert check.status == PortfolioHealthStatus.UNHEALTHY
        assert "Rollback crash" in check.message

    # Exception in check_solana_tx_lens
    with patch("solana_tx_lens.simulator.SolanaSimulator.parse_raw_simulation", side_effect=RuntimeError("Solana crash")):
        check = doc.check_solana_tx_lens()
        assert check.status == PortfolioHealthStatus.UNHEALTHY
        assert "Solana crash" in check.message

    # Exception in check_agent_witness
    with patch("agent_witness.zkvm.ZKVMDeliberationEngine.generate_receipt", side_effect=RuntimeError("Witness crash")):
        check = doc.check_agent_witness()
        assert check.status == PortfolioHealthStatus.UNHEALTHY
        assert "Witness crash" in check.message

    # Exception in check_agent_lab
    with patch("agent_lab.fleet.FleetDiagnostics.run_diagnostics", side_effect=RuntimeError("Lab crash")):
        check = doc.check_agent_lab()
        assert check.status == PortfolioHealthStatus.UNHEALTHY
        assert "Lab crash" in check.message


def test_portfolio_simulator_fallbacks():
    sim = CrossRepoFleetSimulator()

    # Fallback in solana simulation
    with patch("solana_tx_lens.simulator.SolanaSimulator.parse_raw_simulation", side_effect=RuntimeError("Sim fail")):
        r = sim.run_simulation(simulation_id="sim_fb_sol")
        assert r.status == "SUCCESS"
        assert r.verify_seal() is True

    # Fallback in witness
    with patch("agent_witness.receipt.WitnessEngine.record_step", side_effect=RuntimeError("Witness fail")):
        r = sim.run_simulation(simulation_id="sim_fb_wit")
        assert r.status == "SUCCESS"
        assert r.verify_seal() is True


def test_mcp_portfolio_error_handlers():
    with patch("agent_lab.mcp_server._portfolio_doctor.run_doctor", side_effect=RuntimeError("Doctor failure")):
        res = json.loads(mcp.agentlab_portfolio_run_doctor())
        assert res["status"] == "error"
        assert "Doctor failure" in res["error"]

    with patch("agent_lab.mcp_server._portfolio_simulator.run_simulation", side_effect=RuntimeError("Sim failure")):
        res = json.loads(mcp.agentlab_portfolio_simulate_mission())
        assert res["status"] == "error"
        assert "Sim failure" in res["error"]

    with patch("agent_lab.mcp_server._portfolio_simulator.benchmark_portfolio", side_effect=RuntimeError("Bench failure")):
        res = json.loads(mcp.agentlab_portfolio_benchmark())
        assert res["status"] == "error"
        assert "Bench failure" in res["error"]

    with patch("agent_lab.portfolio.PortfolioDiscovery.discover_all", side_effect=RuntimeError("Disco failure")):
        res = json.loads(mcp.agentlab_portfolio_get_status())
        assert res["status"] == "error"
        assert "Disco failure" in res["error"]

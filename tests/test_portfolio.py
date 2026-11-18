"""
tests/test_portfolio.py: Comprehensive unit and integration test suite for Portfolio Ecosystem.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest

from agent_lab.portfolio import (
    PortfolioDiscovery,
    PortfolioDoctor,
    PortfolioHealthStatus,
    CrossRepoFleetSimulator,
    UnifiedPortfolioReceipt,
    PortfolioComponentInfo,
)


def test_portfolio_discovery():
    disco = PortfolioDiscovery.discover_all()
    assert len(disco) == 4
    assert "rollback" in disco
    assert "solana-tx-lens" in disco
    assert "agent-witness" in disco
    assert "agent-lab" in disco

    for name, info in disco.items():
        assert isinstance(info, PortfolioComponentInfo)
        assert info.name == name
        assert info.is_available is True
        assert len(info.capabilities) >= 5
        d = info.to_dict()
        assert d["name"] == name


def test_portfolio_doctor():
    doctor = PortfolioDoctor()
    report = doctor.run_doctor()

    assert report.status == PortfolioHealthStatus.HEALTHY
    assert report.all_healthy is True
    assert len(report.checks) == 4

    subsystems = [c.subsystem for c in report.checks]
    assert "rollback" in subsystems
    assert "solana-tx-lens" in subsystems
    assert "agent-witness" in subsystems
    assert "agent-lab" in subsystems

    for check in report.checks:
        assert check.status == PortfolioHealthStatus.HEALTHY
        assert check.latency_ms >= 0.0
        assert len(check.message) > 0
        assert check.version != "unavailable"

    # JSON export
    j = report.to_json()
    parsed = json.loads(j)
    assert parsed["all_healthy"] is True
    assert len(parsed["checks"]) == 4


def test_cross_repo_simulator_execution():
    simulator = CrossRepoFleetSimulator()
    receipt = simulator.run_simulation(
        simulation_id="sim_test_exec_001",
        task_goal="Verify full cross-repository integration",
    )

    assert receipt.simulation_id == "sim_test_exec_001"
    assert receipt.status == "SUCCESS"
    assert receipt.rollback_applied is True
    assert receipt.corrupt_files_reverted == 1
    assert len(receipt.clean_workspace_hash) == 64
    assert receipt.solana_sim_success is True
    assert receipt.solana_cu_consumed > 0
    assert receipt.solana_priority_fee_micro_lamports > 0
    assert len(receipt.witness_merkle_root) == 64
    assert receipt.witness_step_count == 5
    assert receipt.zkvm_receipt_verified is True
    assert len(receipt.zkvm_journal_digest) == 64
    assert receipt.routed_model != ""
    assert receipt.arena_match_score > 0.0
    assert receipt.latency_ms > 0.0

    # Cryptographic seal verification
    assert receipt.verify_seal() is True


def test_cross_repo_simulator_custom_workspace():
    with tempfile.TemporaryDirectory() as tmp_dir:
        simulator = CrossRepoFleetSimulator()
        receipt = simulator.run_simulation(
            simulation_id="sim_custom_ws",
            task_goal="Custom directory test",
            workspace_dir=tmp_dir,
        )
        assert receipt.status == "SUCCESS"
        assert receipt.verify_seal() is True


def test_unified_portfolio_receipt_seal_tamper():
    simulator = CrossRepoFleetSimulator()
    receipt = simulator.run_simulation(simulation_id="sim_tamper_test")
    assert receipt.verify_seal() is True

    # Tamper with a field
    receipt.arena_match_score = 999.9
    assert receipt.verify_seal() is False

    # Restore seal
    receipt.sha256_seal = receipt.compute_seal()
    assert receipt.verify_seal() is True


def test_portfolio_benchmark():
    simulator = CrossRepoFleetSimulator()
    stats = simulator.benchmark_portfolio(num_iterations=4)

    assert stats["num_iterations"] == 4
    assert stats["success_count"] == 4
    assert stats["success_rate"] == 1.0
    assert stats["simulations_per_second"] > 0.0
    assert stats["avg_latency_ms"] > 0.0
    assert stats["latency_p50_ms"] > 0.0
    assert stats["latency_p95_ms"] >= stats["latency_p50_ms"]
    assert stats["latency_p99_ms"] >= stats["latency_p95_ms"]


def test_portfolio_fallback_emulation():
    simulator = CrossRepoFleetSimulator()

    # Test fallback branch when rollback raises an error
    with patch("rollback.core.create_session", side_effect=ImportError("Mocked rollback missing")):
        receipt = simulator.run_simulation(simulation_id="sim_fallback_test")
        assert receipt.status == "SUCCESS"
        assert receipt.rollback_applied is True
        assert receipt.verify_seal() is True

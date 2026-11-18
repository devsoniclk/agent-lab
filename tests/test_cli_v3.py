"""
Unit tests for distributed swarm CLI commands in Agent-Lab v0.3.0.
"""

from __future__ import annotations

from pathlib import Path
from click.testing import CliRunner
from agent_lab.cli import main


def test_cli_swarm_group_help() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["swarm", "--help"])
    assert res.exit_code == 0
    assert "Distributed WebSocket BFT swarm consensus" in res.output
    assert "serve" in res.output
    assert "worker" in res.output
    assert "broadcast-proposal" in res.output


def test_cli_swarm_serve_help() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["swarm", "serve", "--help"])
    assert res.exit_code == 0
    assert "--host" in res.output
    assert "--port" in res.output
    assert "--quorum" in res.output


def test_cli_swarm_worker_help() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["swarm", "worker", "--help"])
    assert res.exit_code == 0
    assert "--url" in res.output
    assert "--node-id" in res.output
    assert "--weight" in res.output


def test_cli_swarm_broadcast_proposal_flow(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, [
        "swarm", "broadcast-proposal",
        "--workspace", str(tmp_path),
        "--proposal-id", "prop_test_cli",
        "--proposer", "agent-builder",
        "--file", "output.txt",
        "--content", "Hello distributed world",
        "--port", "0",
    ])
    assert res.exit_code == 0
    assert "Round ID:" in res.output
    assert "prop_test_cli" in res.output

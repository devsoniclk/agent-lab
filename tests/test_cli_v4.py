import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from agent_lab.cli import main


def test_cli_registry_group_help():
    runner = CliRunner()
    result = runner.invoke(main, ["registry", "--help"])
    assert result.exit_code == 0
    assert "Solana Anchor On-Chain Agent Registry" in result.output
    assert "register" in result.output
    assert "stake" in result.output
    assert "slash-proof" in result.output
    assert "export-idl" in result.output


def test_cli_registry_lifecycle_flow(tmp_path):
    runner = CliRunner()
    reg_file = str(tmp_path / "cli_registry.json")

    # 1. Register agent
    res = runner.invoke(main, [
        "registry", "register",
        "--pubkey", "cli_agent_pubkey_1",
        "--name", "CliAgentOne",
        "--stake", "150000000",
        "--storage", reg_file,
    ])
    assert res.exit_code == 0
    assert "Agent Registered" in res.output
    assert "cli_agent_pubkey_1" in res.output

    # 2. Deposit stake
    res = runner.invoke(main, [
        "registry", "stake",
        "--pubkey", "cli_agent_pubkey_1",
        "--amount", "50000000",
        "--storage", reg_file,
    ])
    assert res.exit_code == 0
    assert "Stake Deposited" in res.output

    # 3. List agents
    res = runner.invoke(main, [
        "registry", "list",
        "--storage", reg_file,
    ])
    assert res.exit_code == 0
    assert "CliAgentOne" in res.output
    assert "0.200" in res.output

    # 4. Info
    res = runner.invoke(main, [
        "registry", "info",
        "--pubkey", "cli_agent_pubkey_1",
        "--storage", reg_file,
    ])
    assert res.exit_code == 0
    assert "CliAgentOne" in res.output
    assert "Eligible" in res.output


def test_cli_registry_slash_proof_and_export_idl(tmp_path):
    runner = CliRunner()
    reg_file = str(tmp_path / "cli_reg_slash.json")

    # Register offender and reporter
    runner.invoke(main, [
        "registry", "register",
        "--pubkey", "bad_agent",
        "--name", "BadAgent",
        "--stake", "200000000",
        "--storage", reg_file,
    ])
    runner.invoke(main, [
        "registry", "register",
        "--pubkey", "good_agent",
        "--name", "GoodAgent",
        "--stake", "100000000",
        "--storage", reg_file,
    ])

    # Submit slash proof
    res = runner.invoke(main, [
        "registry", "slash-proof",
        "--offender", "bad_agent",
        "--reporter", "good_agent",
        "--proposal", "prop_cli_slash_01",
        "--vote-a", "True",
        "--vote-b", "False",
        "--storage", reg_file,
    ])
    assert res.exit_code == 0
    assert "Equivocation Slashing Executed" in res.output
    assert "JAILED" in res.output

    # Export IDL
    idl_out = str(tmp_path / "idl.json")
    res = runner.invoke(main, ["registry", "export-idl", "--out", idl_out])
    assert res.exit_code == 0
    data = json.loads(Path(idl_out).read_text(encoding="utf-8"))
    assert data["name"] == "agent_registry"
    assert data["version"] == "0.4.0"

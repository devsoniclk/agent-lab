import asyncio
import hashlib
import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from agent_lab.cli import main
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote
from agent_lab.registry import (
    AnchorAgentRegistryClient,
    AgentAccount,
    AgentStatus,
    SlashingProof,
)
from agent_lab.distributed import (
    DistributedCoordinator,
    DistributedSwarmNode,
    NetworkMessage,
    MessageType,
)


def test_swarm_coordinator_with_registry_integration(tmp_path):
    storage = str(tmp_path / "reg_edges.json")
    client = AnchorAgentRegistryClient(storage_path=storage)

    # Register eligible validators
    client.register_agent("val_good_1", "Good 1", initial_stake=200_000_000)
    client.register_agent("val_good_2", "Good 2", initial_stake=200_000_000)
    # Register underfunded validator
    poor = client.register_agent("val_poor", "Poor", initial_stake=100_000_000)
    client.withdraw_stake("val_poor", 50_000_000)  # Drops below min_stake -> INACTIVE

    coordinator = SwarmCoordinator(
        workspace_root=str(tmp_path),
        quorum_ratio=0.5,
        registry_client=client,
    )

    content = b"registry_test = True\n"
    content_hash = hashlib.sha256(content).hexdigest()
    proposal = SwarmProposal(
        proposal_id="prop_reg_01",
        proposer="val_good_1",
        description="Registry consensus test",
        read_set={},
        write_set={"reg_file.py": content_hash},
        file_mutations={"reg_file.py": content},
    )

    # Poor validator's vote should be filtered out
    votes = [
        SwarmVote(voter="val_good_1", proposal_id="prop_reg_01", decision=True),
        SwarmVote(voter="val_poor", proposal_id="prop_reg_01", decision=True),
    ]
    validators = {
        "val_good_1": 1.0,
        "val_poor": 1.0,
    }

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is True
    # total weight only includes val_good_1 since val_poor is filtered
    assert res.total_weight == 1.0


def test_swarm_coordinator_auto_slashing_on_equivocation(tmp_path):
    storage = str(tmp_path / "reg_slash.json")
    client = AnchorAgentRegistryClient(storage_path=storage)

    client.register_agent("val_equivocator", "Equivocator", initial_stake=200_000_000)
    client.register_agent("val_honest", "Honest", initial_stake=200_000_000)

    coordinator = SwarmCoordinator(
        workspace_root=str(tmp_path),
        quorum_ratio=0.5,
        registry_client=client,
    )

    content = b"x = 1\n"
    content_hash = hashlib.sha256(content).hexdigest()
    proposal = SwarmProposal(
        proposal_id="prop_equiv_01",
        proposer="val_honest",
        description="Equivocation test",
        read_set={},
        write_set={"test.py": content_hash},
        file_mutations={"test.py": content},
    )

    # Double vote by val_equivocator
    votes = [
        SwarmVote(voter="val_equivocator", proposal_id="prop_equiv_01", decision=True, signature="sig1"),
        SwarmVote(voter="val_equivocator", proposal_id="prop_equiv_01", decision=False, signature="sig2"),
        SwarmVote(voter="val_honest", proposal_id="prop_equiv_01", decision=True, signature="sig3"),
    ]
    validators = {
        "val_equivocator": 1.0,
        "val_honest": 1.0,
    }

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is False
    assert "val_equivocator" in res.equivocations
    assert len(res.slashing_receipts) == 1
    assert res.slashing_receipts[0]["offender_pubkey"] == "val_equivocator"

    # Verify agent was jailed on-chain
    agent = client.get_agent("val_equivocator")
    assert agent.status == AgentStatus.JAILED


def test_distributed_coordinator_with_registry_eligibility(tmp_path):
    async def run_test():
        storage = str(tmp_path / "reg_dist.json")
        client = AnchorAgentRegistryClient(storage_path=storage)

        # Ineligible agent
        client.register_agent("agent_underfunded", "Underfunded", initial_stake=100_000_000)
        client.withdraw_stake("agent_underfunded", 50_000_000)

        coordinator = DistributedCoordinator(
            workspace_root=str(tmp_path),
            port=0,
            registry_client=client,
        )
        await coordinator.start()
        port = coordinator.server_port
        url = f"ws://127.0.0.1:{port}"

        node = DistributedSwarmNode(
            node_id="agent_underfunded",
            coordinator_url=url,
        )

        # Handshake will be rejected by coordinator
        connected = await node.connect()
        assert connected is False

        await coordinator.stop()

    asyncio.run(run_test())


def test_cli_registry_error_edge_cases(tmp_path):
    storage = str(tmp_path / "err_reg.json")
    runner = CliRunner()

    # 1. Info on missing agent
    res_info = runner.invoke(main, ["registry", "info", "--pubkey", "non_existent", "--storage", storage])
    assert res_info.exit_code != 0
    assert "not found" in res_info.output

    # 2. Register with invalid stake
    res_reg_fail = runner.invoke(main, [
        "registry", "register",
        "--pubkey", "fail_pk",
        "--name", "Fail",
        "--stake", "50",
        "--storage", storage,
    ])
    assert res_reg_fail.exit_code != 0

    # 3. Stake on missing agent
    res_stake_fail = runner.invoke(main, [
        "registry", "stake",
        "--pubkey", "ghost",
        "--amount", "1000",
        "--storage", storage,
    ])
    assert res_stake_fail.exit_code != 0

    # 4. Slashing proof failure
    res_slash_fail = runner.invoke(main, [
        "registry", "slash-proof",
        "--offender", "ghost1",
        "--reporter", "ghost2",
        "--proposal", "p",
        "--vote-a", "True",
        "--vote-b", "False",
        "--storage", storage,
    ])
    assert res_slash_fail.exit_code != 0

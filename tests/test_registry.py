import json
import pytest
from pathlib import Path
from agent_lab.registry import (
    AnchorAgentRegistryClient,
    AgentStatus,
    AgentAccount,
    SlashingProof,
    SlashingReceipt,
    derive_anchor_discriminator,
    derive_pda,
    ANCHOR_AGENT_REGISTRY_IDL,
    DEFAULT_MIN_STAKE_LAMPORTS,
    DEFAULT_SLASHING_PENALTY_BPS,
)
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote


def test_anchor_discriminator_and_pda_derivation():
    disc = derive_anchor_discriminator("global", "register_agent")
    assert len(disc) == 8
    assert isinstance(disc, bytes)

    pda, bump = derive_pda([b"agent", b"agent_test_pubkey_1234"], "AgentReg1111111111111111111111111111111111111")
    assert len(pda) == 44
    assert 0 <= bump <= 255


def test_agent_registration_lifecycle(tmp_path):
    storage = tmp_path / "registry.json"
    client = AnchorAgentRegistryClient(storage_path=str(storage))

    agent = client.register_agent(
        pubkey="validator_alpha_1",
        name="AlphaValidator",
        metadata_uri="ipfs://QmMetadata123",
        initial_stake=200_000_000,
    )
    assert agent.pubkey == "validator_alpha_1"
    assert agent.name == "AlphaValidator"
    assert agent.stake_lamports == 200_000_000
    assert agent.reputation_score == 5000
    assert agent.status == AgentStatus.ACTIVE
    assert len(agent.pda) == 44
    assert client.total_staked_lamports == 200_000_000

    # Query agent
    fetched = client.get_agent("validator_alpha_1")
    assert fetched is not None
    assert fetched.name == "AlphaValidator"


def test_registration_validation_errors():
    client = AnchorAgentRegistryClient()
    client.register_agent(
        pubkey="val_01",
        name="Val01",
        initial_stake=DEFAULT_MIN_STAKE_LAMPORTS,
    )

    with pytest.raises(ValueError, match="already registered"):
        client.register_agent(
            pubkey="val_01",
            name="Val01_Duplicate",
            initial_stake=DEFAULT_MIN_STAKE_LAMPORTS,
        )

    with pytest.raises(ValueError, match="below minimum"):
        client.register_agent(
            pubkey="val_02",
            name="Val02_Poor",
            initial_stake=1_000_000,
        )


def test_stake_deposit_and_withdrawal(tmp_path):
    storage = tmp_path / "reg.json"
    client = AnchorAgentRegistryClient(storage_path=str(storage))
    client.register_agent(
        pubkey="val_stake_test",
        name="Staker",
        initial_stake=100_000_000,
    )

    # Deposit
    updated = client.deposit_stake("val_stake_test", 50_000_000)
    assert updated.stake_lamports == 150_000_000
    assert client.total_staked_lamports == 150_000_000

    # Withdraw partial
    withdrawn = client.withdraw_stake("val_stake_test", 30_000_000)
    assert withdrawn.stake_lamports == 120_000_000
    assert withdrawn.status == AgentStatus.ACTIVE

    # Withdraw below minimum stake -> transitions to INACTIVE
    below_min = client.withdraw_stake("val_stake_test", 30_000_000)
    assert below_min.stake_lamports == 90_000_000
    assert below_min.status == AgentStatus.INACTIVE

    # Deposit back above minimum -> reactivates to ACTIVE
    reactivated = client.deposit_stake("val_stake_test", 20_000_000)
    assert reactivated.stake_lamports == 110_000_000
    assert reactivated.status == AgentStatus.ACTIVE

    # Invalid withdrawals
    with pytest.raises(ValueError, match="Invalid withdrawal"):
        client.withdraw_stake("val_stake_test", 200_000_000)

    with pytest.raises(ValueError, match="not found"):
        client.withdraw_stake("val_unknown", 10_000)


def test_agent_eligibility_verification():
    client = AnchorAgentRegistryClient()
    client.register_agent(
        pubkey="eligible_node",
        name="EligibleNode",
        initial_stake=150_000_000,
    )
    ok, msg = client.verify_agent_eligibility("eligible_node")
    assert ok is True
    assert msg == "Eligible"

    ok, msg = client.verify_agent_eligibility("unknown_node")
    assert ok is False
    assert "not registered" in msg

    client.withdraw_stake("eligible_node", 60_000_000)
    ok, msg = client.verify_agent_eligibility("eligible_node")
    assert ok is False
    assert "not active" in msg


def test_slashing_proof_validation_and_execution(tmp_path):
    storage = tmp_path / "slash_reg.json"
    client = AnchorAgentRegistryClient(storage_path=str(storage))

    client.register_agent(
        pubkey="rogue_validator",
        name="RogueNode",
        initial_stake=200_000_000,
    )
    client.register_agent(
        pubkey="honest_reporter",
        name="HonestReporter",
        initial_stake=100_000_000,
    )

    proof = SlashingProof(
        offender_pubkey="rogue_validator",
        reporter_pubkey="honest_reporter",
        proposal_id="prop_round_42",
        vote_a_decision=True,
        vote_a_sig="sig_decision_true",
        vote_b_decision=False,
        vote_b_sig="sig_decision_false",
    )
    receipt = client.submit_slashing_proof(proof)

    assert receipt.offender_pubkey == "rogue_validator"
    assert receipt.reporter_pubkey == "honest_reporter"
    assert receipt.slashed_lamports == 100_000_000  # 50% of 200M
    assert receipt.burned_lamports == 50_000_000
    assert receipt.reward_lamports == 50_000_000
    assert receipt.reputation_penalty == 2500

    rogue = client.get_agent("rogue_validator")
    assert rogue.status == AgentStatus.JAILED
    assert rogue.stake_lamports == 100_000_000
    assert rogue.reputation_score == 2500
    assert rogue.slashed_count == 1

    reporter = client.get_agent("honest_reporter")
    assert reporter.stake_lamports == 150_000_000  # 100M + 50M reward
    assert reporter.reputation_score == 5500      # 5000 + 500 reward


def test_invalid_slashing_proof_rejection():
    client = AnchorAgentRegistryClient()
    client.register_agent(pubkey="offender_1", name="O1", initial_stake=100_000_000)
    client.register_agent(pubkey="reporter_1", name="R1", initial_stake=100_000_000)

    # Identical votes (no equivocation)
    proof_same = SlashingProof(
        offender_pubkey="offender_1",
        reporter_pubkey="reporter_1",
        proposal_id="p1",
        vote_a_decision=True,
        vote_a_sig="s1",
        vote_b_decision=True,
        vote_b_sig="s2",
    )
    with pytest.raises(ValueError, match="identical decisions"):
        client.submit_slashing_proof(proof_same)

    # Non-existent offender
    proof_ghost = SlashingProof(
        offender_pubkey="ghost_agent",
        reporter_pubkey="reporter_1",
        proposal_id="p1",
        vote_a_decision=True,
        vote_a_sig="s1",
        vote_b_decision=False,
        vote_b_sig="s2",
    )
    with pytest.raises(ValueError, match="Offending agent 'ghost_agent' not found"):
        client.submit_slashing_proof(proof_ghost)


def test_registry_disk_persistence(tmp_path):
    storage = tmp_path / "persistent_reg.json"
    client1 = AnchorAgentRegistryClient(storage_path=str(storage))
    client1.register_agent(pubkey="agent_saved", name="SavedAgent", initial_stake=300_000_000)

    # Reload in new client instance
    client2 = AnchorAgentRegistryClient(storage_path=str(storage))
    agent = client2.get_agent("agent_saved")
    assert agent is not None
    assert agent.name == "SavedAgent"
    assert agent.stake_lamports == 300_000_000
    assert client2.total_staked_lamports == 300_000_000
    assert len(client2.list_agents()) == 1


def test_anchor_idl_structure():
    client = AnchorAgentRegistryClient()
    idl = client.export_idl()
    assert idl["name"] == "agent_registry"
    assert idl["version"] == "0.4.0"
    instruction_names = [inst["name"] for inst in idl["instructions"]]
    assert "register_agent" in instruction_names
    assert "deposit_stake" in instruction_names
    assert "withdraw_stake" in instruction_names
    assert "submit_slashing_proof" in instruction_names


def test_coordinator_with_registry_integration(tmp_path):
    client = AnchorAgentRegistryClient()
    client.register_agent(pubkey="val_coord_1", name="Val1", initial_stake=100_000_000)
    client.register_agent(pubkey="val_coord_2", name="Val2", initial_stake=100_000_000)

    coord = SwarmCoordinator(workspace_root=str(tmp_path), registry_client=client)

    target_file = tmp_path / "consensus_test.txt"
    prop = SwarmProposal(
        proposal_id="prop_int_01",
        proposer="val_coord_1",
        description="Integration test",
        read_set={},
        write_set={"consensus_test.txt": "dummy_sha"},
        file_mutations={"consensus_test.txt": b"verified"},
    )

    # Honest consensus pass
    votes = [
        SwarmVote(voter="val_coord_1", proposal_id="prop_int_01", decision=True),
        SwarmVote(voter="val_coord_2", proposal_id="prop_int_01", decision=True),
    ]
    res = coord.run_consensus_round(
        proposal=prop,
        votes=votes,
        validators={"val_coord_1": 1.0, "val_coord_2": 1.0},
    )
    assert res.committed is True
    assert res.quorum_reached is True

    # Equivocation by val_coord_2 triggers automatic on-chain slashing
    equivocating_votes = [
        SwarmVote(voter="val_coord_1", proposal_id="prop_int_02", decision=True),
        SwarmVote(voter="val_coord_2", proposal_id="prop_int_02", decision=True, signature="sig_yes"),
        SwarmVote(voter="val_coord_2", proposal_id="prop_int_02", decision=False, signature="sig_no"),
    ]
    res2 = coord.run_consensus_round(
        proposal=prop,
        votes=equivocating_votes,
        validators={"val_coord_1": 1.0, "val_coord_2": 1.0},
    )
    assert "val_coord_2" in coord.slashed_validators
    assert len(res2.slashing_receipts) == 1
    slashed_agent = client.get_agent("val_coord_2")
    assert slashed_agent.status == AgentStatus.JAILED

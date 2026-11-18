import time
import pytest
from pathlib import Path

from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote
from agent_lab.distributed import serialize_proposal, deserialize_proposal
from agent_lab.zk import ZKTrajectoryEngine, ZKPolicyConstraints


def test_consensus_with_zk_proof_disabled_by_default(tmp_path: Path):
    coordinator = SwarmCoordinator(workspace_root=str(tmp_path), require_zk_proof=False)

    proposal = SwarmProposal(
        proposal_id="prop_no_zk",
        proposer="agent_1",
        description="Write a file without zk proof",
        file_mutations={"file.txt": b"content_a"},
    )
    votes = [
        SwarmVote(voter="v1", proposal_id="prop_no_zk", decision=True),
        SwarmVote(voter="v2", proposal_id="prop_no_zk", decision=True),
    ]
    validators = {"v1": 1.0, "v2": 1.0}

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is True
    assert (tmp_path / "file.txt").read_bytes() == b"content_a"
    assert res.zk_report is None


def test_consensus_with_zk_proof_missing_fails(tmp_path: Path):
    coordinator = SwarmCoordinator(workspace_root=str(tmp_path), require_zk_proof=True)

    proposal = SwarmProposal(
        proposal_id="prop_missing_zk",
        proposer="agent_1",
        description="Missing zk proof",
        file_mutations={"file.txt": b"content_b"},
        zk_receipt=None,
    )
    votes = [
        SwarmVote(voter="v1", proposal_id="prop_missing_zk", decision=True),
        SwarmVote(voter="v2", proposal_id="prop_missing_zk", decision=True),
    ]
    validators = {"v1": 1.0, "v2": 1.0}

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is False
    assert any("MISSING_ZK_PROOF" in c for c in res.conflicts)
    assert not (tmp_path / "file.txt").exists()


def test_consensus_with_zk_proof_valid_commits(tmp_path: Path):
    coordinator = SwarmCoordinator(
        workspace_root=str(tmp_path),
        require_zk_proof=True,
        zk_constraints=ZKPolicyConstraints(max_steps=10, require_ast_verification=True),
    )

    steps = [
        {"tool": "editor", "content": "def run(): return True\n"},
    ]
    receipt = ZKTrajectoryEngine.generate_receipt(
        steps=steps,
        constraints=coordinator.zk_constraints,
    )

    proposal = SwarmProposal(
        proposal_id="prop_with_zk",
        proposer="agent_1",
        description="With valid zk proof",
        file_mutations={"code.py": b"def run(): return True\n"},
        zk_receipt=receipt.to_dict(),
    )
    votes = [
        SwarmVote(voter="v1", proposal_id="prop_with_zk", decision=True),
        SwarmVote(voter="v2", proposal_id="prop_with_zk", decision=True),
    ]
    validators = {"v1": 1.0, "v2": 1.0}

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is True
    assert (tmp_path / "code.py").read_bytes() == b"def run(): return True\n"
    assert res.zk_report is not None
    assert res.zk_report["is_valid"] is True


def test_consensus_with_zk_proof_invalid_fails(tmp_path: Path):
    coordinator = SwarmCoordinator(
        workspace_root=str(tmp_path),
        require_zk_proof=True,
    )

    steps = [{"tool": "editor", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)
    receipt_dict = receipt.to_dict()
    # Corrupt seal
    receipt_dict["seal_hex"] = "0x" + "0" * 80

    proposal = SwarmProposal(
        proposal_id="prop_bad_zk",
        proposer="agent_1",
        description="With tampered zk proof",
        file_mutations={"file.txt": b"content"},
        zk_receipt=receipt_dict,
    )
    votes = [
        SwarmVote(voter="v1", proposal_id="prop_bad_zk", decision=True),
    ]
    validators = {"v1": 1.0}

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is False
    assert any("INVALID_ZK_PROOF" in c for c in res.conflicts)


def test_distributed_proposal_serialization_with_zk_receipt():
    steps = [{"tool": "test", "content": "pass"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)

    proposal = SwarmProposal(
        proposal_id="p_dist",
        proposer="node_1",
        description="Distributed proposal with ZK",
        file_mutations={"test.txt": b"hello"},
        zk_receipt=receipt.to_dict(),
    )

    serialized = serialize_proposal(proposal)
    assert "zk_receipt" in serialized
    assert serialized["zk_receipt"]["receipt_id"] == receipt.receipt_id

    restored = deserialize_proposal(serialized)
    assert restored.proposal_id == "p_dist"
    assert restored.zk_receipt is not None
    assert restored.zk_receipt["receipt_id"] == receipt.receipt_id
    assert restored.file_mutations["test.txt"] == b"hello"

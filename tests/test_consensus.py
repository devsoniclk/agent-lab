import hashlib
from pathlib import Path
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote


def test_swarm_consensus_approval(tmp_path: Path):
    coordinator = SwarmCoordinator(workspace_root=str(tmp_path), quorum_ratio=2.0 / 3.0)

    target_file = tmp_path / "module.py"
    target_file.write_text("def run(): pass\n", encoding="utf-8")
    initial_hash = hashlib.sha256(b"def run(): pass\n").hexdigest()

    new_content = b"def run(): return 42\n"
    new_hash = hashlib.sha256(new_content).hexdigest()

    proposal = SwarmProposal(
        proposal_id="prop_01",
        proposer="worker-1",
        description="Optimize return value",
        read_set={"module.py": initial_hash},
        write_set={"module.py": new_hash},
        file_mutations={"module.py": new_content},
    )

    validators = {"val-1": 1.0, "val-2": 1.0, "val-3": 1.0}
    votes = [
        SwarmVote(voter="val-1", proposal_id="prop_01", decision=True),
        SwarmVote(voter="val-2", proposal_id="prop_01", decision=True),
        SwarmVote(voter="val-3", proposal_id="prop_01", decision=False),
    ]

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is True
    assert res.quorum_reached is True
    assert res.approved_weight == 2.0
    assert target_file.read_bytes() == new_content


def test_swarm_consensus_rejection_and_rollback(tmp_path: Path):
    coordinator = SwarmCoordinator(workspace_root=str(tmp_path), quorum_ratio=0.67)

    target_file = tmp_path / "protected.py"
    target_file.write_text("# original\n", encoding="utf-8")
    initial_hash = hashlib.sha256(b"# original\n").hexdigest()

    proposal = SwarmProposal(
        proposal_id="prop_02",
        proposer="worker-2",
        description="Malicious or unverified change",
        read_set={"protected.py": initial_hash},
        write_set={"protected.py": "hash"},
        file_mutations={"protected.py": b"# hacked\n"},
    )

    validators = {"val-1": 1.0, "val-2": 1.0}
    votes = [
        SwarmVote(voter="val-1", proposal_id="prop_02", decision=False),
        SwarmVote(voter="val-2", proposal_id="prop_02", decision=False),
    ]

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is False
    assert res.quorum_reached is False
    assert "protected.py" in res.reverted_files
    assert target_file.read_text() == "# original\n"


def test_swarm_occ_conflict_detection(tmp_path: Path):
    coordinator = SwarmCoordinator(workspace_root=str(tmp_path), quorum_ratio=0.5)

    # Simulate another worker having already committed a change to state.py
    coordinator.committed_versions["state.py"] = "newer_hash_abc"

    proposal = SwarmProposal(
        proposal_id="prop_03",
        proposer="worker-3",
        description="Stale branch edit",
        read_set={"state.py": "stale_hash_xyz"},
        write_set={"state.py": "new_hash_123"},
        file_mutations={"state.py": b"state = 3\n"},
    )

    validators = {"val-1": 1.0}
    votes = [SwarmVote(voter="val-1", proposal_id="prop_03", decision=True)]

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is False
    assert len(res.conflicts) > 0
    assert "OCC write-write conflict" in res.conflicts[0]


def test_swarm_equivocation_slashing(tmp_path: Path):
    coordinator = SwarmCoordinator(workspace_root=str(tmp_path), quorum_ratio=0.5)

    proposal = SwarmProposal(
        proposal_id="prop_04",
        proposer="worker-4",
        description="Equivocation test",
        read_set={},
        write_set={},
        file_mutations={"test.txt": b"data"},
    )

    validators = {"rogue-val": 1.0}
    # Double vote (both True and False)
    votes = [
        SwarmVote(voter="rogue-val", proposal_id="prop_04", decision=True),
        SwarmVote(voter="rogue-val", proposal_id="prop_04", decision=False),
    ]

    res = coordinator.run_consensus_round(proposal, votes, validators)
    assert res.committed is False
    assert "rogue-val" in res.equivocations
    assert "rogue-val" in coordinator.slashed_validators

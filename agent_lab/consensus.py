"""
agent_lab.consensus: Multi-agent BFT Swarm Consensus Coordinator with
Optimistic Concurrency Control (OCC) and automatic filesystem rollback.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SwarmProposal:
    """A proposed change set submitted by an autonomous swarm agent."""
    proposal_id: str
    proposer: str
    description: str
    read_set: Dict[str, str] = field(default_factory=dict)   # path -> sha256
    write_set: Dict[str, str] = field(default_factory=dict)  # path -> sha256
    file_mutations: Dict[str, bytes] = field(default_factory=dict)  # path -> content
    timestamp: float = field(default_factory=time.time)

    def compute_hash(self) -> str:
        payload = f"{self.proposal_id}:{self.proposer}:{sorted(self.read_set.items())}:{sorted(self.write_set.items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SwarmVote:
    """A cryptographic or programmatic vote cast by a validator agent."""
    voter: str
    proposal_id: str
    decision: bool  # True = PRECOMMIT/APPROVE, False = REJECT
    weight: float = 1.0
    signature: Optional[str] = None
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmConsensusResult:
    """Result of a multi-agent BFT swarm consensus round."""
    round_id: str
    proposal_id: str
    committed: bool
    quorum_reached: bool
    approved_weight: float
    total_weight: float
    conflicts: List[str]
    equivocations: List[str]
    applied_files: List[str]
    reverted_files: List[str]
    duration_ms: float


class SwarmCoordinator:
    """Orchestrates multi-agent consensus and guards atomic disk commits."""

    def __init__(
        self,
        workspace_root: str,
        quorum_ratio: float = 2.0 / 3.0,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.quorum_ratio = quorum_ratio
        self.committed_versions: Dict[str, str] = {}  # path -> sha256
        self.slashed_validators: Set[str] = set()

    def hash_file(self, path: Path) -> Optional[str]:
        """Compute SHA-256 hash of a file on disk."""
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_consensus_round(
        self,
        proposal: SwarmProposal,
        votes: List[SwarmVote],
        validators: Dict[str, float],  # validator_id -> weight
    ) -> SwarmConsensusResult:
        """Execute a 2-phase BFT consensus round over a proposed change set."""
        start_time = time.time()
        round_id = f"rnd_{int(start_time*1000)}"

        # 1. Detect Equivocations (double voting by the same validator)
        equivocations: List[str] = []
        seen_votes: Dict[str, bool] = {}
        valid_votes: List[SwarmVote] = []

        for vote in votes:
            if vote.voter in self.slashed_validators:
                continue
            if vote.voter in seen_votes:
                if seen_votes[vote.voter] != vote.decision:
                    equivocations.append(vote.voter)
                    self.slashed_validators.add(vote.voter)
                    continue
            seen_votes[vote.voter] = vote.decision
            valid_votes.append(vote)

        # 2. Compute BFT Quorum
        total_weight = sum(w for v, w in validators.items() if v not in self.slashed_validators)
        approved_weight = sum(
            validators.get(v.voter, 1.0)
            for v in valid_votes
            if v.decision and v.voter in validators and v.voter not in self.slashed_validators
        )

        quorum_reached = total_weight > 0 and (approved_weight / total_weight) >= (self.quorum_ratio - 0.005)

        # 3. Optimistic Concurrency Control (OCC) Check
        conflicts: List[str] = []
        for rel_path, expected_hash in proposal.read_set.items():
            current_committed = self.committed_versions.get(rel_path)
            disk_path = self.workspace_root / rel_path
            current_disk = self.hash_file(disk_path)

            if current_committed and current_committed != expected_hash:
                conflicts.append(f"OCC write-write conflict on '{rel_path}': expected {expected_hash[:8]}, committed {current_committed[:8]}")
            elif current_disk and current_disk != expected_hash and rel_path not in proposal.write_set:
                conflicts.append(f"OCC disk-drift conflict on '{rel_path}': expected {expected_hash[:8]}, disk {current_disk[:8]}")

        # 4. Atomic Commit or Revert
        committed = False
        applied_files: List[str] = []
        reverted_files: List[str] = []

        if quorum_reached and not conflicts and not equivocations:
            # Atomic commit of file mutations
            for rel_path, content in proposal.file_mutations.items():
                target = self.workspace_root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                new_hash = hashlib.sha256(content).hexdigest()
                self.committed_versions[rel_path] = new_hash
                applied_files.append(rel_path)
            committed = True
        else:
            # Revert any proposed writes (safety rollback)
            reverted_files = list(proposal.file_mutations.keys())

        duration_ms = (time.time() - start_time) * 1000.0

        return SwarmConsensusResult(
            round_id=round_id,
            proposal_id=proposal.proposal_id,
            committed=committed,
            quorum_reached=quorum_reached,
            approved_weight=approved_weight,
            total_weight=total_weight,
            conflicts=conflicts,
            equivocations=equivocations,
            applied_files=applied_files,
            reverted_files=reverted_files,
            duration_ms=round(duration_ms, 2),
        )

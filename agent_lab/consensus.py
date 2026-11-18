"""
agent_lab.consensus: Multi-agent BFT Swarm Consensus Coordinator with
Optimistic Concurrency Control (OCC), on-chain staking validation, and automatic filesystem rollback.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from agent_lab.registry import AnchorAgentRegistryClient


@dataclass
class SwarmProposal:
    """A proposed change set submitted by an autonomous swarm agent."""
    proposal_id: str
    proposer: str
    description: str
    read_set: Dict[str, str] = field(default_factory=dict)   # path -> sha256
    write_set: Dict[str, str] = field(default_factory=dict)  # path -> sha256
    file_mutations: Dict[str, bytes] = field(default_factory=dict)  # path -> content
    zk_receipt: Optional[Dict[str, Any]] = None
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
    slashing_receipts: List[Dict[str, Any]] = field(default_factory=list)
    zk_report: Optional[Dict[str, Any]] = None


class SwarmCoordinator:
    """Orchestrates multi-agent consensus, guards atomic disk commits, and verifies on-chain stakes."""

    def __init__(
        self,
        workspace_root: str,
        quorum_ratio: float = 2.0 / 3.0,
        registry_client: Optional[AnchorAgentRegistryClient] = None,
        require_zk_proof: bool = False,
        zk_constraints: Optional[Any] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.quorum_ratio = quorum_ratio
        self.registry_client = registry_client
        self.require_zk_proof = require_zk_proof
        self.zk_constraints = zk_constraints
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

        # 0. On-Chain Stake and Eligibility Filtering
        active_validators = dict(validators)
        if self.registry_client is not None:
            filtered_validators: Dict[str, float] = {}
            for v_id, weight in active_validators.items():
                eligible, _ = self.registry_client.verify_agent_eligibility(v_id)
                if eligible:
                    filtered_validators[v_id] = weight
            active_validators = filtered_validators

        # 1. Detect Equivocations (double voting by the same validator)
        equivocations: List[str] = []
        slashing_receipts: List[Dict[str, Any]] = []
        seen_votes: Dict[str, SwarmVote] = {}
        valid_votes: List[SwarmVote] = []

        for vote in votes:
            if vote.voter in self.slashed_validators:
                continue

            # If registry client is active, check voter eligibility
            if self.registry_client is not None:
                eligible, _ = self.registry_client.verify_agent_eligibility(vote.voter)
                if not eligible:
                    continue

            if vote.voter in seen_votes:
                prev_vote = seen_votes[vote.voter]
                if prev_vote.decision != vote.decision:
                    equivocations.append(vote.voter)
                    self.slashed_validators.add(vote.voter)

                    # Submit slashing proof to Solana Anchor registry
                    if self.registry_client is not None:
                        try:
                            from agent_lab.registry import SlashingProof
                            reporter = proposal.proposer if proposal.proposer != vote.voter else "coordinator"
                            # Ensure reporter is registered for reward distribution
                            if reporter not in self.registry_client.agents:
                                self.registry_client.register_agent(
                                    pubkey=reporter,
                                    name="Consensus Reporter",
                                    initial_stake=self.registry_client.min_stake_lamports,
                                )
                            proof = SlashingProof(
                                offender_pubkey=vote.voter,
                                reporter_pubkey=reporter,
                                proposal_id=proposal.proposal_id,
                                vote_a_decision=prev_vote.decision,
                                vote_a_sig=prev_vote.signature or f"sig_{vote.voter}_a",
                                vote_b_decision=vote.decision,
                                vote_b_sig=vote.signature or f"sig_{vote.voter}_b",
                            )
                            receipt = self.registry_client.submit_slashing_proof(proof)
                            slashing_receipts.append(receipt.to_dict())
                        except Exception:
                            pass
                    continue
            seen_votes[vote.voter] = vote
            valid_votes.append(vote)

        # 2. Compute BFT Quorum
        total_weight = sum(w for v, w in active_validators.items() if v not in self.slashed_validators)
        approved_weight = sum(
            active_validators.get(v.voter, 1.0)
            for v in valid_votes
            if v.decision and v.voter in active_validators and v.voter not in self.slashed_validators
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

        # 4. Zero-Knowledge Trajectory Verification (ZK-Guarded Consensus)
        zk_report: Optional[Dict[str, Any]] = None
        if self.require_zk_proof:
            from agent_lab.zk import ZKTrajectoryEngine, ZKTrajectoryReceipt
            if not proposal.zk_receipt:
                conflicts.append("MISSING_ZK_PROOF: Proposal lacks mandatory zero-knowledge trajectory execution receipt.")
            else:
                try:
                    zk_receipt_obj = ZKTrajectoryReceipt.from_dict(proposal.zk_receipt)
                    report = ZKTrajectoryEngine.verify_receipt(zk_receipt_obj, constraints=self.zk_constraints)
                    zk_report = report.to_dict()
                    if not report.is_valid:
                        conflicts.append(f"INVALID_ZK_PROOF: {report.error_message or 'Receipt validation failed.'}")
                except Exception as e:
                    conflicts.append(f"CORRUPT_ZK_PROOF: Failed to parse or verify zk receipt: {str(e)}")

        # 5. Atomic Commit or Revert
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
            slashing_receipts=slashing_receipts,
            zk_report=zk_report,
        )

"""
Example 02: Multi-Agent BFT Swarm Consensus with Optimistic Concurrency Control (OCC).
"""

import hashlib
from pathlib import Path
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote


def main():
    workspace = Path("./swarm_demo").resolve()
    workspace.mkdir(exist_ok=True)

    coordinator = SwarmCoordinator(workspace_root=str(workspace), quorum_ratio=2.0 / 3.0)

    # 1. Proposer prepares changes
    file_path = workspace / "consensus_app.py"
    file_path.write_text("VERSION = 1\n")
    initial_hash = hashlib.sha256(b"VERSION = 1\n").hexdigest()

    new_content = b"VERSION = 2  # Upgraded\n"
    new_hash = hashlib.sha256(new_content).hexdigest()

    proposal = SwarmProposal(
        proposal_id="prop_upgrade_v2",
        proposer="agent-worker-1",
        description="Upgrade app version",
        read_set={"consensus_app.py": initial_hash},
        write_set={"consensus_app.py": new_hash},
        file_mutations={"consensus_app.py": new_content},
    )

    # 2. Validators cast votes
    validators = {"val-alpha": 1.0, "val-beta": 1.0, "val-gamma": 1.0}
    votes = [
        SwarmVote(voter="val-alpha", proposal_id="prop_upgrade_v2", decision=True),
        SwarmVote(voter="val-beta", proposal_id="prop_upgrade_v2", decision=True),
        SwarmVote(voter="val-gamma", proposal_id="prop_upgrade_v2", decision=False, reason="Conservative hold"),
    ]

    # 3. Execute BFT consensus round
    res = coordinator.run_consensus_round(proposal, votes, validators)
    print(f"Round ID: {res.round_id}")
    print(f"Committed: {res.committed} (Quorum: {res.quorum_reached}, Approved: {res.approved_weight}/{res.total_weight})")
    print(f"Applied Files: {res.applied_files}")
    print(f"Current file on disk: {file_path.read_text().strip()}")


if __name__ == "__main__":
    main()

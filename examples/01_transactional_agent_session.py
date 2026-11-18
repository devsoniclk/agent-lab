"""
Example 01: Transactional Autonomous Agent Session with Checkpoints and Merkle Receipts.
"""

from pathlib import Path
from agent_lab.orchestrator import AgentLabOrchestrator


def main():
    workspace = Path("./workspace_demo").resolve()
    workspace.mkdir(exist_ok=True)

    orchestrator = AgentLabOrchestrator()
    orchestrator.config.workspace_root = str(workspace)

    # 1. Start session
    session = orchestrator.start_session(
        agent_name="RefactorBot",
        goal="Safely refactor core service logic",
    )
    print(f"Started session: {session.session_id}")

    # 2. Write initial code
    file_path = workspace / "service.py"
    file_path.write_text("def run():\n    return 'v1'\n")

    # 3. Create transactional micro-checkpoint
    cp_id = session.create_checkpoint("initial_clean_code")
    print(f"Created checkpoint: {cp_id}")

    # 4. Agent simulates a breaking change or hallucinated overwrite
    file_path.write_text("def run():\n    raise SystemError('broken refactor')\n")
    print("Agent wrote broken code.")

    # 5. Programmatic rollback on validation failure
    session.record_step("thought", "Tests failed on service.py; initiating rollback.")
    session.rollback(cp_id)
    print(f"Rolled back to {cp_id}. Current code: {file_path.read_text().strip()}")

    # 6. Finalize session and output Merkle action receipt
    receipt = session.finalize_and_attest()
    print(f"Attested session Merkle Root: {receipt['merkle_root']}")


if __name__ == "__main__":
    main()

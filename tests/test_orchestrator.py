from pathlib import Path
from agent_lab.orchestrator import AgentLabOrchestrator


def test_agent_session_lifecycle(tmp_path: Path):
    storage = tmp_path / ".agentlab"
    orchestrator = AgentLabOrchestrator()
    orchestrator.config.workspace_root = str(tmp_path)
    orchestrator.config.storage_dir = str(storage)

    session = orchestrator.start_session(agent_name="TestAgent", goal="Refactor auth service")
    assert session.session_id.startswith("sess_")
    assert session.agent_name == "TestAgent"

    # 1. Record steps
    session.record_step("thought", "Analyzing dependencies")
    session.record_step("tool_call", "grep_search", {"pattern": "jwt"})
    assert len(session.steps) >= 3  # initial goal + 2 steps

    # 2. Create checkpoint
    file_a = tmp_path / "hello.py"
    file_a.write_text("print('hello v1')", encoding="utf-8")
    cp_id = session.create_checkpoint("initial_code")
    assert cp_id.startswith("cp_")

    # 3. Modify file and add new file
    file_a.write_text("print('corrupted code')", encoding="utf-8")
    file_b = tmp_path / "extra.py"
    file_b.write_text("# unwanted file", encoding="utf-8")
    assert file_a.read_text() == "print('corrupted code')"
    assert file_b.is_file()

    # 4. Rollback to checkpoint
    success = session.rollback(cp_id)
    assert success is True
    assert file_a.read_text() == "print('hello v1')"
    assert not file_b.exists()

    # 5. Solana transaction simulation
    sim = session.simulate_solana_tx("AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==")
    assert sim["success"] is True
    assert "compute_units_consumed" in sim

    # 6. Finalize session and generate Merkle witness receipt
    receipt = session.finalize_and_attest()
    assert "merkle_root" in receipt
    assert receipt["step_count"] == len(session.steps)
    receipt_file = storage / "sessions" / session.session_id / "receipt.witness"
    assert receipt_file.is_file()

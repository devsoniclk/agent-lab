import json
import pytest

from agent_lab.zk import (
    ZKTrajectoryEngine,
    ZKTrajectoryReceipt,
    ZKPolicyConstraints,
    ZKVerificationReport,
    ZKVMSystem,
    ZKProofType,
    DEFAULT_RISC0_IMAGE_ID,
    DEFAULT_SP1_VKEY,
)


def test_merkle_root_computation():
    # Empty trajectory
    empty_root = ZKTrajectoryEngine.compute_action_merkle_root([])
    assert len(empty_root) == 64

    # Single step
    single_step = [{"step_index": 0, "type": "action", "tool": "read", "content": "foo"}]
    single_root = ZKTrajectoryEngine.compute_action_merkle_root(single_step)
    assert len(single_root) == 64
    assert single_root != empty_root

    # Multiple steps
    multi_steps = [
        {"step_index": 0, "type": "action", "tool": "read", "content": "foo"},
        {"step_index": 1, "type": "action", "tool": "write", "content": "bar"},
        {"step_index": 2, "type": "action", "tool": "test", "content": "baz"},
    ]
    multi_root = ZKTrajectoryEngine.compute_action_merkle_root(multi_steps)
    assert len(multi_root) == 64
    assert multi_root != single_root


def test_trajectory_digest_computation():
    steps = [
        {"tool": "bash", "arguments": "echo 1"},
        {"tool": "bash", "arguments": "echo 2"},
    ]
    digest1 = ZKTrajectoryEngine.compute_trajectory_digest(steps, session_id="sess_1")
    digest2 = ZKTrajectoryEngine.compute_trajectory_digest(steps, session_id="sess_1")
    digest3 = ZKTrajectoryEngine.compute_trajectory_digest(steps, session_id="sess_2")
    assert digest1 == digest2
    assert digest1 != digest3
    assert len(digest1) == 64


def test_policy_constraints_digest():
    c1 = ZKPolicyConstraints(max_steps=20, require_ast_verification=True)
    c2 = ZKPolicyConstraints(max_steps=20, require_ast_verification=True)
    c3 = ZKPolicyConstraints(max_steps=30, require_ast_verification=True)

    assert c1.compute_digest() == c2.compute_digest()
    assert c1.compute_digest() != c3.compute_digest()
    assert len(c1.compute_digest()) == 64


def test_generate_receipt_valid_risc0():
    steps = [
        {"step_index": 0, "type": "action", "tool": "file_reader", "content": "cat file.py"},
        {"step_index": 1, "type": "action", "tool": "code_exec", "content": "def test():\n    return 42\n"},
    ]
    constraints = ZKPolicyConstraints(max_steps=10, require_ast_verification=True)

    receipt = ZKTrajectoryEngine.generate_receipt(
        steps=steps,
        session_id="session_risc0_1",
        constraints=constraints,
        zkvm_system="risc0",
    )

    assert receipt.zkvm_system == "risc0"
    assert receipt.program_id == DEFAULT_RISC0_IMAGE_ID
    assert receipt.journal["session_id"] == "session_risc0_1"
    assert receipt.journal["step_count"] == 2
    assert receipt.journal["tool_calls_count"] == 2
    assert receipt.journal["ast_verified"] is True
    assert receipt.journal["status_code"] == 0
    assert len(receipt.journal_digest) == 64
    assert receipt.seal_hex.startswith("0x")


def test_generate_receipt_valid_sp1():
    steps = [
        {"step_index": 0, "type": "action", "tool": "calc", "content": "2 + 2"},
    ]
    receipt = ZKTrajectoryEngine.generate_receipt(
        steps=steps,
        zkvm_system="sp1",
    )
    assert receipt.zkvm_system == "sp1"
    assert receipt.program_id == DEFAULT_SP1_VKEY
    assert receipt.journal["status_code"] == 0


def test_generate_receipt_unsupported_system_raises():
    with pytest.raises(ValueError, match="Unsupported zkVM system"):
        ZKTrajectoryEngine.generate_receipt(steps=[], zkvm_system="unknown_zk")


def test_generate_receipt_policy_violations():
    steps = [
        {"step_index": 0, "type": "action", "tool": "bash_rm_rf", "content": "rm -rf /"},
        {"step_index": 1, "type": "action", "tool": "write", "content": "data"},
        {"step_index": 2, "type": "action", "tool": "unauthorized_tool", "content": "data"},
    ]
    # 1. Exceeded max steps
    c_steps = ZKPolicyConstraints(max_steps=1)
    r_steps = ZKTrajectoryEngine.generate_receipt(steps=steps, constraints=c_steps)
    assert r_steps.journal["status_code"] == 1

    # 2. Forbidden tool
    c_forbidden = ZKPolicyConstraints(forbidden_tools=["bash_rm_rf"])
    r_forbidden = ZKTrajectoryEngine.generate_receipt(steps=steps, constraints=c_forbidden)
    assert r_forbidden.journal["status_code"] == 2

    # 3. Unauthorized tool (allowlist)
    c_allowed = ZKPolicyConstraints(allowed_tools=["write", "read"])
    r_allowed = ZKTrajectoryEngine.generate_receipt(steps=steps, constraints=c_allowed)
    assert r_allowed.journal["status_code"] == 3


def test_generate_receipt_ast_verification():
    # Invalid Python syntax
    bad_steps = [
        {"step_index": 0, "type": "action", "tool": "coder", "content": "def broken_code(:\n    pass"},
    ]
    c_ast = ZKPolicyConstraints(require_ast_verification=True)
    receipt_bad = ZKTrajectoryEngine.generate_receipt(steps=bad_steps, constraints=c_ast)
    assert receipt_bad.journal["ast_verified"] is False
    assert receipt_bad.journal["status_code"] == 4

    # Without enforcement, status code remains 0
    c_no_ast = ZKPolicyConstraints(require_ast_verification=False)
    receipt_lenient = ZKTrajectoryEngine.generate_receipt(steps=bad_steps, constraints=c_no_ast)
    assert receipt_lenient.journal["ast_verified"] is False
    assert receipt_lenient.journal["status_code"] == 0


def test_generate_receipt_with_trajectory_f1():
    executed = [
        {"tool": "read", "content": "a"},
        {"tool": "write", "content": "b"},
    ]
    reference = [
        {"tool": "read", "content": "a"},
        {"tool": "write", "content": "b"},
    ]
    receipt = ZKTrajectoryEngine.generate_receipt(
        steps=executed,
        reference_steps=reference,
        constraints=ZKPolicyConstraints(min_trajectory_f1=0.8),
    )
    assert receipt.trajectory_f1 >= 0.99
    assert receipt.journal["status_code"] == 0


def test_verify_receipt_success():
    steps = [
        {"step_index": 0, "type": "action", "tool": "search", "content": "query"},
        {"step_index": 1, "type": "action", "tool": "python", "content": "def f(): return 1\n"},
    ]
    constraints = ZKPolicyConstraints(max_steps=10, require_ast_verification=True)
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps, constraints=constraints)

    report = ZKTrajectoryEngine.verify_receipt(receipt, constraints=constraints)
    assert report.is_valid is True
    assert report.policy_satisfied is True
    assert report.error_message is None
    assert all(c["passed"] for c in report.checks)


def test_verify_receipt_tampered_journal():
    steps = [{"tool": "read", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)

    # Tamper with journal digest
    receipt.journal_digest = "0" * 64
    report = ZKTrajectoryEngine.verify_receipt(receipt)
    assert report.is_valid is False
    assert any(c["check"] == "journal_digest_match" and not c["passed"] for c in report.checks)


def test_verify_receipt_corrupted_hex():
    steps = [{"tool": "read", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)
    receipt.journal_bytes_hex = "INVALID_HEX_XYZ"
    report = ZKTrajectoryEngine.verify_receipt(receipt)
    assert report.is_valid is False
    assert "Corrupted journal hex payload" in (report.error_message or "")


def test_verify_receipt_program_id_allowlist():
    steps = [{"tool": "read", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps, zkvm_system="risc0")

    # Allowed
    report_ok = ZKTrajectoryEngine.verify_receipt(
        receipt,
        allowed_program_ids=[DEFAULT_RISC0_IMAGE_ID],
    )
    assert report_ok.is_valid is True

    # Disallowed
    report_fail = ZKTrajectoryEngine.verify_receipt(
        receipt,
        allowed_program_ids=["0x1111111111111111111111111111111111111111111111111111111111111111"],
    )
    assert report_fail.is_valid is False
    assert any(c["check"] == "program_id_allowlist" and not c["passed"] for c in report_fail.checks)


def test_verify_receipt_merkle_root_mismatch():
    steps = [{"tool": "read", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)

    report = ZKTrajectoryEngine.verify_receipt(receipt, expected_merkle_root="f" * 64)
    assert report.is_valid is False
    assert any(c["check"] == "merkle_root_match" and not c["passed"] for c in report.checks)


def test_verify_receipt_seal_forgery():
    steps = [{"tool": "read", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)

    # Corrupt seal
    receipt.seal_hex = "0x" + "a" * 64 + "b" * 16
    report = ZKTrajectoryEngine.verify_receipt(receipt)
    assert report.is_valid is False
    assert any(c["check"] == "seal_signature_binding" and not c["passed"] for c in report.checks)


def test_trajectory_receipt_serialization_roundtrip():
    steps = [{"tool": "read", "content": "test"}]
    receipt = ZKTrajectoryEngine.generate_receipt(steps=steps)

    d = receipt.to_dict()
    restored = ZKTrajectoryReceipt.from_dict(d)

    assert restored.receipt_id == receipt.receipt_id
    assert restored.session_id == receipt.session_id
    assert restored.journal_digest == receipt.journal_digest
    assert restored.seal_hex == receipt.seal_hex
    assert restored.journal == receipt.journal

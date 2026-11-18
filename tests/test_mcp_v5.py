import json
from agent_lab.mcp_server import (
    agentlab_zk_generate_receipt,
    agentlab_zk_verify_receipt,
    agentlab_zk_audit_trajectory,
    agentlab_zk_policy_digest,
)


def test_mcp_zk_generate_receipt():
    steps = [
        {"tool": "read", "content": "file.txt"},
        {"tool": "python", "content": "def foo(): return 'bar'\n"},
    ]
    res_str = agentlab_zk_generate_receipt(
        session_id="mcp_sess_1",
        steps_json=json.dumps(steps),
        zkvm_system="risc0",
        require_ast=True,
    )
    res = json.loads(res_str)
    assert res["status"] == "success"
    receipt = res["receipt"]
    assert receipt["session_id"] == "mcp_sess_1"
    assert receipt["zkvm_system"] == "risc0"
    assert receipt["journal"]["step_count"] == 2
    assert receipt["journal"]["ast_verified"] is True


def test_mcp_zk_verify_receipt():
    # First generate
    steps = [{"tool": "read", "content": "file.txt"}]
    gen_str = agentlab_zk_generate_receipt(steps_json=json.dumps(steps))
    gen_data = json.loads(gen_str)
    receipt_dict = gen_data["receipt"]

    # Now verify
    ver_str = agentlab_zk_verify_receipt(receipt_json=json.dumps(receipt_dict))
    ver_data = json.loads(ver_str)
    assert ver_data["status"] == "success"
    assert ver_data["report"]["is_valid"] is True


def test_mcp_zk_audit_trajectory():
    steps = [
        {"tool": "read", "content": "input.json"},
        {"tool": "eval", "content": "output"},
    ]
    ref_steps = [
        {"tool": "read", "content": "input.json"},
        {"tool": "eval", "content": "output"},
    ]
    audit_str = agentlab_zk_audit_trajectory(
        steps_json=json.dumps(steps),
        reference_steps_json=json.dumps(ref_steps),
        require_ast=True,
    )
    audit_data = json.loads(audit_str)
    assert audit_data["status"] == "success"
    assert audit_data["is_valid"] is True
    assert audit_data["report"]["status_code"] == 0
    assert audit_data["receipt"]["trajectory_f1"] >= 0.99


def test_mcp_zk_policy_digest():
    res_str = agentlab_zk_policy_digest(
        max_steps=25,
        require_ast=True,
        allowed_tools_csv="read,write,bash",
        forbidden_tools_csv="rm_rf",
    )
    data = json.loads(res_str)
    assert data["status"] == "success"
    assert len(data["policy_digest"]) == 64
    assert data["constraints"]["max_steps"] == 25


def test_mcp_zk_error_handling():
    # Corrupt JSON in generate
    err_gen = agentlab_zk_generate_receipt(steps_json="INVALID_JSON")
    assert json.loads(err_gen)["status"] == "error"

    # Corrupt JSON in verify
    err_ver = agentlab_zk_verify_receipt(receipt_json="INVALID_JSON")
    assert json.loads(err_ver)["status"] == "error"

    # Corrupt JSON in audit
    err_audit = agentlab_zk_audit_trajectory(steps_json="INVALID_JSON")
    assert json.loads(err_audit)["status"] == "error"

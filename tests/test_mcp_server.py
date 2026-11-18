import json
from agent_lab.mcp_server import (
    agentlab_session_start,
    agentlab_checkpoint_create,
    agentlab_rollback,
    agentlab_session_attest,
    agentlab_simulate_solana_tx,
    agentlab_eval_pass_at_k,
    agentlab_eval_ast_verify,
    agentlab_doctor,
)


def test_mcp_tools_flow():
    # 1. Start session
    start_resp = json.loads(agentlab_session_start(agent_name="MCP_Worker", goal="Unit testing MCP"))
    assert start_resp["status"] == "success"
    session_id = start_resp["session_id"]

    # 2. Checkpoint
    cp_resp = json.loads(agentlab_checkpoint_create(session_id=session_id, label="step_1"))
    assert cp_resp["status"] == "success"
    cp_id = cp_resp["checkpoint_id"]

    # 3. Rollback
    rb_resp = json.loads(agentlab_rollback(session_id=session_id, checkpoint_id=cp_id))
    assert rb_resp["status"] == "success"
    assert rb_resp["rolled_back"] is True

    # 4. Simulate Solana
    sim_resp = json.loads(agentlab_simulate_solana_tx(session_id=session_id, tx_encoded="AQ=="))
    assert sim_resp["status"] == "success"

    # 5. Attest
    attest_resp = json.loads(agentlab_session_attest(session_id=session_id))
    assert attest_resp["status"] == "success"
    assert "merkle_root" in attest_resp["receipt"]

    # 6. Eval pass@k
    eval_resp = json.loads(agentlab_eval_pass_at_k(n=10, c=8, k=1))
    assert eval_resp["status"] == "success"
    assert eval_resp["pass@k"] == 0.8

    # 7. AST verify
    ast_resp = json.loads(agentlab_eval_ast_verify(code="def foo(): return 1", entry_point="foo"))
    assert ast_resp["status"] == "success"
    assert ast_resp["ast"]["valid"] is True

    # 8. Doctor
    doc_resp = json.loads(agentlab_doctor())
    assert doc_resp["status"] == "success"
    assert "portfolio" in doc_resp

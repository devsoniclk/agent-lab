import hashlib
import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from agent_lab.eval import PassAtKEstimator, ASTVerifier, RegressionTracker, BenchmarkHarness
from agent_lab.orchestrator import AgentLabOrchestrator, AgentSession, ActionStep
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote
from agent_lab.mcp_server import (
    agentlab_checkpoint_create,
    agentlab_rollback,
    agentlab_session_attest,
    agentlab_simulate_solana_tx,
    agentlab_eval_pass_at_k,
    agentlab_eval_ast_verify,
    agentlab_doctor,
)
from agent_lab.cli import main


def test_pass_at_k_edge_cases():
    with pytest.raises(ValueError, match="Total samples n=2 must be >= k=5"):
        PassAtKEstimator.estimate(n=2, c=1, k=5)

    with pytest.raises(ValueError, match="Correct samples c=-1 must be between"):
        PassAtKEstimator.estimate(n=10, c=-1, k=1)

    with pytest.raises(ValueError, match="Correct samples c=15 must be between"):
        PassAtKEstimator.estimate(n=10, c=15, k=1)

    # When n - c < k, all samples must contain a correct one
    assert PassAtKEstimator.estimate(n=5, c=5, k=1) == 1.0
    assert PassAtKEstimator.estimate(n=5, c=4, k=2) == 1.0

    # compute_all
    res = PassAtKEstimator.compute_all(n=10, c=5, k_values=[1, 5, 10, 20])
    assert "pass@1" in res
    assert "pass@5" in res
    assert "pass@10" in res
    assert "pass@20" not in res  # k > n is omitted


def test_ast_verifier_edges():
    # Syntax error
    res_err = ASTVerifier.verify("def invalid(:", expected_entry_point="foo")
    assert res_err["valid"] is False
    assert "SyntaxError" in res_err["error"]
    assert res_err["entry_point_found"] is False

    # Valid syntax but missing entry point
    res_missing = ASTVerifier.verify("def actual(): pass", expected_entry_point="target")
    assert res_missing["valid"] is True
    assert res_missing["entry_point_found"] is False

    # Boolean operators in complexity
    res_bool = ASTVerifier.verify("def check(a, b, c): return a and b or c")
    assert res_bool["valid"] is True
    assert res_bool["cyclomatic_complexity"] >= 3


def test_regression_tracker_comprehensive():
    baseline = {"task1": True, "task2": True, "task3": False, "task4": False}
    candidate = {"task1": True, "task2": False, "task3": True, "task4": False, "task5": True}

    report = RegressionTracker.compare(baseline, candidate)
    assert report["has_regressions"] is True
    assert report["regression_count"] == 1
    assert "task2" in report["regressions"]
    assert report["improvement_count"] == 2
    assert "task3" in report["improvements"]
    assert "task5" in report["improvements"]
    assert report["unchanged_count"] == 2  # task1 (True->True), task4 (False->False)
    assert report["total_tasks"] == 5


def test_orchestrator_edge_cases(tmp_path: Path):
    orch = AgentLabOrchestrator()
    sess = orch.start_session(agent_name="Tester", goal="Verify rollback cleanup")
    sess.workspace_root = tmp_path

    # Create initial file
    f1 = tmp_path / "file1.txt"
    f1.write_text("v1", encoding="utf-8")

    cp = sess.create_checkpoint("initial")

    # Mutate f1 and create f2
    f1.write_text("v2", encoding="utf-8")
    f2 = tmp_path / "file2.txt"
    f2.write_text("new", encoding="utf-8")

    assert f1.read_text() == "v2"
    assert f2.exists()

    # Rollback to initial
    success = sess.rollback(cp)
    assert success is True
    assert f1.read_text() == "v1"
    assert not f2.exists()

    # Non-existent checkpoint
    fail = sess.rollback("cp_non_existent")
    assert fail is False

    # Unknown session lookup
    assert orch.get_session("unknown_sess_id") is None

    # Empty action steps hash
    sess_empty = AgentSession("empty", "A", "G", str(tmp_path), str(tmp_path / ".storage"))
    sess_empty.steps = []
    receipt = sess_empty.finalize_and_attest()
    assert receipt["merkle_root"] is not None
    assert receipt["step_count"] == 0


def test_consensus_drift_and_empty_validators(tmp_path: Path):
    coord = SwarmCoordinator(workspace_root=str(tmp_path), quorum_ratio=0.67)

    # Disk drift conflict: file exists on disk with different hash than read_set and not in write_set
    file_on_disk = tmp_path / "drift.py"
    file_on_disk.write_text("disk_content", encoding="utf-8")

    prop = SwarmProposal(
        proposal_id="prop_drift",
        proposer="w1",
        description="drift test",
        read_set={"drift.py": "old_hash_expected"},
        write_set={},
        file_mutations={},
    )
    res = coord.run_consensus_round(prop, [SwarmVote("val1", "prop_drift", True)], {"val1": 1.0})
    assert res.committed is False
    assert any("OCC disk-drift conflict" in c for c in res.conflicts)

    # Empty validators
    res_empty = coord.run_consensus_round(prop, [], {})
    assert res_empty.quorum_reached is False
    assert res_empty.committed is False


def test_mcp_server_error_paths():
    # Non existent session calls
    cp_res = json.loads(agentlab_checkpoint_create("bad_id", "cp"))
    assert cp_res["status"] == "error"

    rb_res = json.loads(agentlab_rollback("bad_id", "cp_1"))
    assert rb_res["status"] == "error"

    attest_res = json.loads(agentlab_session_attest("bad_id"))
    assert attest_res["status"] == "error"

    sim_res = json.loads(agentlab_simulate_solana_tx("bad_id", "raw_tx"))
    assert sim_res["status"] == "error"

    # Pass@k error path
    eval_err = json.loads(agentlab_eval_pass_at_k(n=2, c=5, k=1))
    assert eval_err["status"] == "error"

    # AST verify and doctor
    ast_res = json.loads(agentlab_eval_ast_verify("def hello(): return 1", "hello"))
    assert ast_res["status"] == "success"
    assert ast_res["ast"]["entry_point_found"] is True

    doc_res = json.loads(agentlab_doctor())
    assert doc_res["status"] == "success"
    assert "portfolio" in doc_res


def test_cli_error_flows(tmp_path: Path):
    runner = CliRunner()

    # eval pass-at-k invalid
    res_err = runner.invoke(main, ["eval", "pass-at-k", "-n", "2", "-c", "5", "-k", "1"])
    assert res_err.exit_code == 1
    assert "Error:" in res_err.output

    # eval ast-check syntax error
    bad_syntax = tmp_path / "bad.py"
    bad_syntax.write_text("def error(:", encoding="utf-8")
    res_ast_err = runner.invoke(main, ["eval", "ast-check", str(bad_syntax)])
    assert res_ast_err.exit_code == 1
    assert "AST Error:" in res_ast_err.output

    # eval ast-check missing entry point
    ok_file = tmp_path / "ok.py"
    ok_file.write_text("def actual(): pass", encoding="utf-8")
    res_entry = runner.invoke(main, ["eval", "ast-check", str(ok_file), "--entry", "missing"])
    assert res_entry.exit_code == 0
    assert "Missing" in res_entry.output

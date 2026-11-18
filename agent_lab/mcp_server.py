"""
agent_lab.mcp_server: FastMCP server exposing Agent-Lab tools for Claude, OpenBot, and Cursor.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent_lab.config import AgentLabConfig
from agent_lab.orchestrator import AgentLabOrchestrator
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote
from agent_lab.eval import (
    PassAtKEstimator,
    ASTVerifier,
    TrajectoryEvaluator,
    BenchmarkHarness,
    StatisticalInterval,
)
from agent_lab.telemetry import TelemetryCollector
from agent_lab.distributed import (
    DistributedCoordinator,
    DistributedSwarmNode,
    NetworkMessage,
    MessageType,
)

# Default global orchestrator and collector
_config = AgentLabConfig.load()
_orchestrator = AgentLabOrchestrator(_config)
_coordinator = SwarmCoordinator(workspace_root=_config.workspace_root)
_telemetry = TelemetryCollector()


def agentlab_session_start(agent_name: str, goal: str) -> str:
    """Start a new transactional and attested agent session."""
    session = _orchestrator.start_session(agent_name=agent_name, goal=goal)
    _telemetry.record_session(agent_name=agent_name, goal=goal)
    return json.dumps({
        "status": "success",
        "session_id": session.session_id,
        "agent_name": session.agent_name,
        "goal": session.goal,
    })


def agentlab_checkpoint_create(session_id: str, label: str) -> str:
    """Create a micro-checkpoint of the workspace within an active session."""
    session = _orchestrator.get_session(session_id)
    if not session:
        return json.dumps({"status": "error", "error": f"Session '{session_id}' not found"})
    cp_id = session.create_checkpoint(label=label)
    _telemetry.record_checkpoint(session_id=session_id, label=label)
    return json.dumps({"status": "success", "checkpoint_id": cp_id})


def agentlab_rollback(session_id: str, checkpoint_id: str) -> str:
    """Revert workspace to a previous micro-checkpoint."""
    session = _orchestrator.get_session(session_id)
    if not session:
        return json.dumps({"status": "error", "error": f"Session '{session_id}' not found"})
    success = session.rollback(checkpoint_id=checkpoint_id)
    _telemetry.record_rollback(session_id=session_id, checkpoint_id=checkpoint_id, success=success)
    return json.dumps({"status": "success" if success else "failed", "rolled_back": success})


def agentlab_session_attest(session_id: str) -> str:
    """Finalize an agent session and generate a cryptographic Merkle action receipt."""
    session = _orchestrator.get_session(session_id)
    if not session:
        return json.dumps({"status": "error", "error": f"Session '{session_id}' not found"})
    receipt = session.finalize_and_attest()
    return json.dumps({"status": "success", "receipt": receipt})


def agentlab_simulate_solana_tx(session_id: str, tx_encoded: str) -> str:
    """Pre-flight simulate a Solana transaction within an agent session."""
    session = _orchestrator.get_session(session_id)
    if not session:
        return json.dumps({"status": "error", "error": f"Session '{session_id}' not found"})
    result = session.simulate_solana_tx(tx_encoded=tx_encoded)
    _telemetry.record_solana_sim(success=result.get("success", False), units_consumed=result.get("units_consumed", 0))
    return json.dumps({"status": "success", "result": result})


def agentlab_eval_pass_at_k(n: int, c: int, k: int) -> str:
    """Calculate unbiased pass@k for code generation evaluation."""
    try:
        val = PassAtKEstimator.estimate(n=n, c=c, k=k)
        ci = StatisticalInterval.wilson_score_interval(c, n)
        return json.dumps({
            "status": "success",
            "n": n,
            "c": c,
            "k": k,
            "pass@k": round(val, 4),
            "confidence_interval_95": ci,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_eval_ast_verify(code: str, entry_point: Optional[str] = None) -> str:
    """Verify code AST syntax and entrypoint."""
    res = ASTVerifier.verify(code=code, expected_entry_point=entry_point)
    return json.dumps({"status": "success", "ast": res})


def agentlab_eval_trajectory(
    trajectory_json: str,
    expected_tools_json: str,
    sequence_rules_json: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> str:
    """
    Evaluate agent tool call trajectory for precision, recall, F1, loops, and sequence validity.
    """
    try:
        traj = json.loads(trajectory_json)
        expected = json.loads(expected_tools_json)
        seq_rules = json.loads(sequence_rules_json) if sequence_rules_json else None

        res = TrajectoryEvaluator.evaluate(
            trajectory=traj,
            expected_tools=expected,
            sequence_rules=seq_rules,
            max_steps=max_steps,
        )
        _telemetry.record_trajectory_eval(f1_score=res["f1_score"], has_loop=res["has_loop"])
        return json.dumps({"status": "success", "evaluation": res})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_telemetry_get_summary() -> str:
    """Retrieve aggregated cross-agent telemetry and health metrics."""
    return json.dumps({"status": "success", "telemetry": _telemetry.get_metrics_dict()})


def agentlab_benchmark_run_suite(tasks_file: str) -> str:
    """Run an entire coding benchmark suite file."""
    try:
        harness = BenchmarkHarness()
        suite_res = harness.run_suite(tasks_file=tasks_file)
        for r in suite_res.get("results", []):
            _telemetry.record_eval(task_id=r.get("task_id", ""), passed=r.get("passed", False))
        return json.dumps({"status": "success", "summary": suite_res})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_doctor() -> str:
    """Run an environment and portfolio diagnostic check."""
    from agent_lab.orchestrator import HAS_ROLLBACK, HAS_WITNESS, HAS_SOLANA_LENS
    return json.dumps({
        "status": "success",
        "portfolio": {
            "rollback_installed": HAS_ROLLBACK,
            "agent_witness_installed": HAS_WITNESS,
            "solana_tx_lens_installed": HAS_SOLANA_LENS,
        },
        "config": {
            "workspace": _config.workspace_root,
            "storage_dir": _config.storage_dir,
            "env": _config.env,
        }
    })


def agentlab_swarm_status() -> str:
    """Retrieve active swarm consensus coordinator status, peers, and round history."""
    return json.dumps({
        "status": "success",
        "committed_versions": len(_coordinator.committed_versions),
        "slashed_validators": list(_coordinator.slashed_validators),
        "quorum_ratio": _coordinator.quorum_ratio,
    })


def agentlab_swarm_broadcast_proposal(
    proposal_id: str,
    proposer: str,
    target_file: str,
    content: str,
    description: str = "",
) -> str:
    """Submit a change set proposal for swarm consensus."""
    import hashlib
    content_bytes = content.encode("utf-8")
    content_hash = hashlib.sha256(content_bytes).hexdigest()

    proposal = SwarmProposal(
        proposal_id=proposal_id,
        proposer=proposer,
        description=description,
        read_set={},
        write_set={target_file: content_hash},
        file_mutations={target_file: content_bytes},
    )

    # Execute round with default local validator
    vote = SwarmVote(voter="coordinator", proposal_id=proposal_id, decision=True)
    res = _coordinator.run_consensus_round(
        proposal=proposal,
        votes=[vote],
        validators={"coordinator": 1.0},
    )

    return json.dumps({
        "status": "success",
        "committed": res.committed,
        "round_id": res.round_id,
        "applied_files": res.applied_files,
        "conflicts": res.conflicts,
        "duration_ms": res.duration_ms,
    })

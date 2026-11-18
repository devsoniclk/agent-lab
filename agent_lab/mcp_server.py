"""
agent_lab.mcp_server: FastMCP server exposing Agent-Lab tools for Claude, OpenBot, and Cursor.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent_lab.config import AgentLabConfig
from agent_lab.orchestrator import AgentLabOrchestrator
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote
from agent_lab.eval import PassAtKEstimator, ASTVerifier

# Default global orchestrator
_config = AgentLabConfig.load()
_orchestrator = AgentLabOrchestrator(_config)
_coordinator = SwarmCoordinator(workspace_root=_config.workspace_root)


def agentlab_session_start(agent_name: str, goal: str) -> str:
    """Start a new transactional and attested agent session."""
    session = _orchestrator.start_session(agent_name=agent_name, goal=goal)
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
    return json.dumps({"status": "success", "checkpoint_id": cp_id})


def agentlab_rollback(session_id: str, checkpoint_id: str) -> str:
    """Revert workspace to a previous micro-checkpoint."""
    session = _orchestrator.get_session(session_id)
    if not session:
        return json.dumps({"status": "error", "error": f"Session '{session_id}' not found"})
    success = session.rollback(checkpoint_id=checkpoint_id)
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
    return json.dumps({"status": "success", "result": result})


def agentlab_eval_pass_at_k(n: int, c: int, k: int) -> str:
    """Calculate unbiased pass@k for code generation evaluation."""
    try:
        val = PassAtKEstimator.estimate(n=n, c=c, k=k)
        return json.dumps({"status": "success", "n": n, "c": c, "k": k, "pass@k": round(val, 4)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_eval_ast_verify(code: str, entry_point: Optional[str] = None) -> str:
    """Verify code AST syntax and entrypoint."""
    res = ASTVerifier.verify(code=code, expected_entry_point=entry_point)
    return json.dumps({"status": "success", "ast": res})


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

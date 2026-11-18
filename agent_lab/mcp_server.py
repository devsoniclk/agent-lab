"""
agent_lab.mcp_server: FastMCP server exposing Agent-Lab tools for Claude, OpenBot, and Cursor.
"""

from __future__ import annotations

import json
from pathlib import Path
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
from agent_lab.registry import (
    AnchorAgentRegistryClient,
    AgentStatus,
    SlashingProof,
    DEFAULT_MIN_STAKE_LAMPORTS,
    ANCHOR_AGENT_REGISTRY_IDL,
)
from agent_lab.zk import (
    ZKTrajectoryEngine,
    ZKTrajectoryReceipt,
    ZKPolicyConstraints,
    DEFAULT_RISC0_IMAGE_ID,
    DEFAULT_SP1_VKEY,
)
from agent_lab.leaderboard import (
    AgentEvaluationRecord,
    LeaderboardEngine,
    TokenEconomicsSimulator,
    LeaderboardDashboard,
)
from agent_lab.router import (
    AdaptiveModelRouter,
    AgentPriority,
    CircuitState,
    FallbackRoute,
    RoutingExecutionReceipt,
    TokenGovernor,
)
from agent_lab.memory import (
    CrossAgentMemoryStore,
    MemoryKind,
    MemoryAccessLevel,
    MemoryEntry,
    MemorySearchResult,
)
from agent_lab.arena import (
    AgentMatchScore,
    ArenaLeaderboard,
    EloRatingSystem,
    MatchOutcome,
    MatchResult,
    SwarmWorkflowHarness,
    SwarmWorkflowReport,
    SwarmWorkflowStage,
)

# Default global orchestrator and collector
_config = AgentLabConfig.load()
_orchestrator = AgentLabOrchestrator(_config)
_registry = AnchorAgentRegistryClient()
_coordinator = SwarmCoordinator(workspace_root=_config.workspace_root, registry_client=_registry)
_telemetry = TelemetryCollector()
_router = AdaptiveModelRouter()
_memory_store = CrossAgentMemoryStore()
_arena_leaderboard = ArenaLeaderboard()
_workflow_harness = SwarmWorkflowHarness()


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


def agentlab_registry_register_agent(
    pubkey: str,
    name: str,
    stake_lamports: int = DEFAULT_MIN_STAKE_LAMPORTS,
    metadata_uri: str = "",
) -> str:
    """Register an autonomous agent validator on the Solana Anchor registry."""
    try:
        agent = _registry.register_agent(
            pubkey=pubkey,
            name=name,
            metadata_uri=metadata_uri,
            initial_stake=stake_lamports,
        )
        return json.dumps({
            "status": "success",
            "agent": agent.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_registry_get_agent(pubkey: str) -> str:
    """Query on-chain stake, reputation score, and lifecycle status for an agent."""
    agent = _registry.get_agent(pubkey)
    if not agent:
        return json.dumps({"status": "error", "error": f"Agent '{pubkey}' not found"})
    return json.dumps({
        "status": "success",
        "agent": agent.to_dict(),
    })


def agentlab_registry_verify_stake(pubkey: str) -> str:
    """Verify whether an agent meets the minimum stake threshold for BFT quorum voting."""
    eligible, reason = _registry.verify_agent_eligibility(pubkey)
    agent = _registry.get_agent(pubkey)
    return json.dumps({
        "status": "success",
        "pubkey": pubkey,
        "eligible": eligible,
        "reason": reason,
        "stake_lamports": agent.stake_lamports if agent else 0,
        "min_required_lamports": _registry.min_stake_lamports,
        "reputation_score": agent.reputation_score if agent else 0,
    })


def agentlab_registry_submit_slashing_proof(
    offender_pubkey: str,
    reporter_pubkey: str,
    proposal_id: str,
    vote_a_decision: bool,
    vote_b_decision: bool,
    vote_a_sig: str = "sig_a",
    vote_b_sig: str = "sig_b",
) -> str:
    """Submit a cryptographic equivocation proof to slash offending validator collateral."""
    # Ensure reporter is registered if not already present
    if reporter_pubkey not in _registry.agents:
        _registry.register_agent(
            pubkey=reporter_pubkey,
            name="Slashing Reporter",
            initial_stake=_registry.min_stake_lamports,
        )

    proof = SlashingProof(
        offender_pubkey=offender_pubkey,
        reporter_pubkey=reporter_pubkey,
        proposal_id=proposal_id,
        vote_a_decision=vote_a_decision,
        vote_a_sig=vote_a_sig,
        vote_b_decision=vote_b_decision,
        vote_b_sig=vote_b_sig,
    )
    try:
        receipt = _registry.submit_slashing_proof(proof)
        return json.dumps({
            "status": "success",
            "receipt": receipt.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_registry_list_agents() -> str:
    """List all registered autonomous agents, stake balances, and statuses."""
    agents = _registry.list_agents()
    return json.dumps({
        "status": "success",
        "count": len(agents),
        "total_staked_lamports": _registry.total_staked_lamports,
        "agents": [a.to_dict() for a in agents],
    })


def agentlab_registry_deposit_stake(pubkey: str, amount_lamports: int) -> str:
    """Deposit additional SOL collateral into an agent's on-chain stake account."""
    try:
        agent = _registry.deposit_stake(pubkey, amount_lamports)
        return json.dumps({
            "status": "success",
            "agent": agent.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_registry_withdraw_stake(pubkey: str, amount_lamports: int) -> str:
    """Withdraw SOL collateral from an agent's on-chain stake account."""
    try:
        agent = _registry.withdraw_stake(pubkey, amount_lamports)
        return json.dumps({
            "status": "success",
            "agent": agent.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_registry_export_idl() -> str:
    """Export the Anchor Program IDL specification for agent registry."""
    return json.dumps({
        "status": "success",
        "idl": ANCHOR_AGENT_REGISTRY_IDL,
    })


def agentlab_zk_generate_receipt(
    session_id: str = "",
    steps_json: str = "[]",
    zkvm_system: str = "risc0",
    max_steps: int = 50,
    require_ast: bool = False,
    program_id: str = "",
) -> str:
    """Generate a verifiable zero-knowledge virtual machine (zkVM) receipt for agent trajectory."""
    try:
        steps = json.loads(steps_json) if steps_json else []
        constraints = ZKPolicyConstraints(
            max_steps=max_steps,
            require_ast_verification=require_ast,
        )
        receipt = ZKTrajectoryEngine.generate_receipt(
            steps=steps,
            session_id=session_id or None,
            constraints=constraints,
            zkvm_system=zkvm_system,
            program_id=program_id or None,
        )
        return json.dumps({
            "status": "success",
            "receipt": receipt.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_zk_verify_receipt(
    receipt_json: str,
    require_ast: bool = False,
    allowed_program_id: str = "",
    expected_merkle_root: str = "",
) -> str:
    """Cryptographically verify a zkVM trajectory receipt against policy constraints and seals."""
    try:
        data = json.loads(receipt_json)
        receipt = ZKTrajectoryReceipt.from_dict(data)
        constraints = ZKPolicyConstraints(
            require_ast_verification=require_ast,
            allowed_program_ids=[allowed_program_id] if allowed_program_id else None,
        )
        report = ZKTrajectoryEngine.verify_receipt(
            receipt=receipt,
            constraints=constraints,
            expected_merkle_root=expected_merkle_root or None,
        )
        return json.dumps({
            "status": "success",
            "report": report.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_zk_audit_trajectory(
    session_id: str = "",
    steps_json: str = "[]",
    reference_steps_json: str = "",
    zkvm_system: str = "risc0",
    max_steps: int = 50,
    require_ast: bool = True,
) -> str:
    """End-to-end evaluation, zkVM receipt generation, and cryptographic audit for an agent trajectory."""
    try:
        steps = json.loads(steps_json) if steps_json else []
        ref_steps = json.loads(reference_steps_json) if reference_steps_json else None
        constraints = ZKPolicyConstraints(
            max_steps=max_steps,
            require_ast_verification=require_ast,
        )
        receipt = ZKTrajectoryEngine.generate_receipt(
            steps=steps,
            session_id=session_id or None,
            constraints=constraints,
            zkvm_system=zkvm_system,
            reference_steps=ref_steps,
        )
        report = ZKTrajectoryEngine.verify_receipt(
            receipt=receipt,
            constraints=constraints,
        )
        return json.dumps({
            "status": "success",
            "is_valid": report.is_valid,
            "receipt": receipt.to_dict(),
            "report": report.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_zk_policy_digest(
    max_steps: int = 50,
    require_ast: bool = False,
    allowed_tools_csv: str = "",
    forbidden_tools_csv: str = "",
) -> str:
    """Compute the deterministic SHA-256 digest of a set of zkVM policy constraints."""
    try:
        allowed = [t.strip() for t in allowed_tools_csv.split(",") if t.strip()] if allowed_tools_csv else None
        forbidden = [t.strip() for t in forbidden_tools_csv.split(",") if t.strip()] if forbidden_tools_csv else None
        constraints = ZKPolicyConstraints(
            max_steps=max_steps,
            allowed_tools=allowed,
            forbidden_tools=forbidden,
            require_ast_verification=require_ast,
        )
        return json.dumps({
            "status": "success",
            "policy_digest": constraints.compute_digest(),
            "constraints": constraints.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _get_mcp_sample_records() -> List[AgentEvaluationRecord]:
    return [
        AgentEvaluationRecord(
            agent_id="code-synthesizer-alpha",
            model_id="claude-3-5-sonnet",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=92,
            pass_at_1=0.92,
            tool_f1_score=0.94,
            mean_latency_ms=1180.0,
            p95_latency_ms=1850.0,
            avg_prompt_tokens=1800,
            avg_completion_tokens=650,
        ),
        AgentEvaluationRecord(
            agent_id="code-synthesizer-beta",
            model_id="gpt-4o",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=89,
            pass_at_1=0.89,
            tool_f1_score=0.91,
            mean_latency_ms=1050.0,
            p95_latency_ms=1720.0,
            avg_prompt_tokens=1750,
            avg_completion_tokens=600,
        ),
        AgentEvaluationRecord(
            agent_id="deepseek-coder",
            model_id="deepseek-coder-v2",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=86,
            pass_at_1=0.86,
            tool_f1_score=0.88,
            mean_latency_ms=820.0,
            p95_latency_ms=1340.0,
            avg_prompt_tokens=1700,
            avg_completion_tokens=580,
        ),
        AgentEvaluationRecord(
            agent_id="flash-router",
            model_id="gemini-1.5-flash",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=76,
            pass_at_1=0.76,
            tool_f1_score=0.82,
            mean_latency_ms=410.0,
            p95_latency_ms=690.0,
            avg_prompt_tokens=1500,
            avg_completion_tokens=480,
        ),
    ]


def agentlab_leaderboard_generate(eval_records_json: str = "") -> str:
    """Generate a ranked cross-agent benchmark leaderboard with composite scores and tiers."""
    try:
        if eval_records_json.strip():
            raw = json.loads(eval_records_json)
            records = [AgentEvaluationRecord.from_dict(item) for item in raw]
        else:
            records = _get_mcp_sample_records()

        entries = LeaderboardEngine.generate_leaderboard(records)
        return json.dumps({
            "status": "success",
            "count": len(entries),
            "leaderboard": [e.to_dict() for e in entries],
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_leaderboard_pareto_analysis(eval_records_json: str = "") -> str:
    """Perform Pareto frontier analysis on cross-agent benchmark results, identifying non-dominated models."""
    try:
        if eval_records_json.strip():
            raw = json.loads(eval_records_json)
            records = [AgentEvaluationRecord.from_dict(item) for item in raw]
        else:
            records = _get_mcp_sample_records()

        entries = LeaderboardEngine.generate_leaderboard(records)
        pareto_optimal = [e.to_dict() for e in entries if e.is_pareto_optimal]
        dominated = [e.to_dict() for e in entries if not e.is_pareto_optimal]

        return json.dumps({
            "status": "success",
            "total_models": len(entries),
            "pareto_optimal_count": len(pareto_optimal),
            "pareto_optimal_entries": pareto_optimal,
            "dominated_entries": dominated,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_leaderboard_cost_simulator(
    model_allocations_json: str = "",
    tasks_per_day: int = 10000,
    days: int = 30,
) -> str:
    """Simulate swarm token economics, multi-model cost distribution, and routing downgrade savings."""
    try:
        allocations = json.loads(model_allocations_json) if model_allocations_json.strip() else {
            "claude-3-5-sonnet": 30,
            "gpt-4o": 20,
            "deepseek-coder-v2": 25,
            "gemini-1.5-flash": 25,
        }
        report = TokenEconomicsSimulator.simulate_workload(
            model_allocations=allocations,
            tasks_per_day=tasks_per_day,
            days=days,
        )
        return json.dumps({
            "status": "success",
            "report": report.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_leaderboard_export_html(
    eval_records_json: str = "",
    output_path: str = "leaderboard.html",
    title: str = "Agent-Lab Cross-Agent Leaderboard",
) -> str:
    """Render and export a standalone, dark-theme HTML report with SVG bar and scatter charts."""
    try:
        if eval_records_json.strip():
            raw = json.loads(eval_records_json)
            records = [AgentEvaluationRecord.from_dict(item) for item in raw]
        else:
            records = _get_mcp_sample_records()

        entries = LeaderboardEngine.generate_leaderboard(records)
        econ_report = TokenEconomicsSimulator.simulate_workload(
            model_allocations={"claude-3-5-sonnet": 35, "deepseek-coder-v2": 35, "gemini-1.5-flash": 30},
            tasks_per_day=10000,
            days=30,
        )
        html_out = LeaderboardDashboard.export_html_report(
            entries=entries,
            economics=econ_report,
            title=title,
        )
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_out, encoding="utf-8")

        return json.dumps({
            "status": "success",
            "output_path": str(p),
            "bytes_written": len(html_out),
            "models_ranked": len(entries),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Router & Rate-Limiter Tools (v0.7.0)
# ---------------------------------------------------------------------------


def agentlab_router_route(
    prompt: str = "Write an asynchronous token-bucket rate limiter in Rust.",
    route: str = "code-generation",
    priority: str = "NORMAL",
    agent_id: str = "agent-mcp",
    tokens: int = 500,
    simulate_fail_primary: bool = False,
) -> str:
    """Route an LLM request through adaptive fallback cascade with rate-limiting and circuit breakers."""
    try:
        prio_enum = AgentPriority(priority)
        invoker = None
        if simulate_fail_primary:
            route_obj = _router.routes.get(route)
            prim = route_obj.primary_model if route_obj else "claude-3-5-sonnet"

            def fault_invoker(model_id: str, p: str):
                if model_id == prim:
                    return False, "HTTP 429: Rate limit exceeded (simulated)", 20, 0
                return True, f"Success from fallback model {model_id}: Code generated successfully.", 25, tokens

            invoker = fault_invoker

        receipt = _router.route_and_execute(
            prompt=prompt,
            route_name=route,
            priority=prio_enum,
            agent_id=agent_id,
            estimated_output_tokens=tokens,
            provider_invoker=invoker,
        )

        # Record in telemetry
        _telemetry.record_routing_event(
            primary_model=receipt.primary_model,
            resolved_model=receipt.resolved_model,
            fallback_occurred=receipt.fallback_occurred,
            latency_ms=receipt.total_latency_ms,
            tokens=receipt.total_tokens,
            priority=receipt.priority.value,
            status=receipt.status,
        )

        return json.dumps({
            "status": "success",
            "receipt": receipt.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_router_get_fleet_status(model_id: str = "") -> str:
    """Return circuit breaker states, sliding-window RPM/TPM quotas, and active fallback cascades."""
    try:
        status = _router.get_fleet_status()
        if model_id and model_id in status["models"]:
            return json.dumps({
                "status": "success",
                "model_id": model_id,
                "model_status": status["models"][model_id],
            })
        return json.dumps({
            "status": "success",
            "fleet_status": status,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_router_reset_circuit(model_id: str = "", reset_all: bool = True) -> str:
    """Reset tripped circuit breakers to CLOSED state across the model fleet."""
    try:
        if reset_all or not model_id:
            _router.reset_all_circuits()
            return json.dumps({
                "status": "success",
                "message": "All model circuit breakers reset to CLOSED.",
            })
        success = _router.reset_circuit(model_id)
        if success:
            return json.dumps({
                "status": "success",
                "message": f"Circuit breaker for '{model_id}' reset to CLOSED.",
            })
        return json.dumps({
            "status": "error",
            "error": f"Unknown model '{model_id}'",
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_router_benchmark_resilience(
    requests: int = 20,
    fault_rate: float = 0.35,
    route: str = "code-generation",
) -> str:
    """Benchmark swarm resilience and zero-downtime failover under synthetic 429 faults."""
    import random
    try:
        route_obj = _router.routes.get(route)
        primary = route_obj.primary_model if route_obj else "claude-3-5-sonnet"

        success_count = 0
        fallback_count = 0
        primary_hit_count = 0
        failed_count = 0
        total_latency = 0.0
        total_cost = 0.0

        rng = random.Random(42)

        for i in range(requests):
            should_fail = rng.random() < fault_rate

            def bench_invoker(m_id: str, p: str):
                if m_id == primary and should_fail:
                    return False, "HTTP 429: Rate limit exceeded (synthetic)", 15, 0
                return True, f"Response from {m_id}", 20, 200

            try:
                receipt = _router.route_and_execute(
                    prompt=f"Task {i}",
                    route_name=route,
                    agent_id=f"bench-agent-{i % 4}",
                    estimated_output_tokens=200,
                    provider_invoker=bench_invoker,
                )
                success_count += 1
                total_latency += receipt.total_latency_ms
                total_cost += receipt.total_cost_usd
                if receipt.fallback_occurred:
                    fallback_count += 1
                else:
                    primary_hit_count += 1
            except Exception:
                failed_count += 1

        resilience_rate = (success_count / requests) * 100.0 if requests > 0 else 0.0
        return json.dumps({
            "status": "success",
            "requests": requests,
            "route": route,
            "primary_model": primary,
            "fault_rate": fault_rate,
            "resilience_rate_pct": round(resilience_rate, 2),
            "primary_hits": primary_hit_count,
            "fallback_recoveries": fallback_count,
            "unhandled_failures": failed_count,
            "mean_latency_ms": round(total_latency / max(1, success_count), 2),
            "total_cost_usd": round(total_cost, 6),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Cross-Agent Memory FastMCP Tools
# ---------------------------------------------------------------------------

def agentlab_memory_store(
    content: str,
    agent_id: str,
    kind: str = "SEMANTIC",
    access_level: str = "SWARM",
    tags: Optional[List[str]] = None,
    importance: float = 0.5,
    ttl_seconds: Optional[float] = None,
    namespace: str = "default",
) -> str:
    """Store an episodic, semantic, or procedural memory item with dense vector indexing."""
    try:
        entry = _memory_store.store(
            content=content,
            agent_id=agent_id,
            kind=MemoryKind(kind),
            access_level=MemoryAccessLevel(access_level),
            tags=tags or [],
            importance=importance,
            ttl_seconds=ttl_seconds,
            namespace=namespace,
        )
        return json.dumps({
            "status": "success",
            "memory": entry.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_memory_search(
    query: str,
    agent_id: Optional[str] = None,
    namespace: Optional[str] = None,
    kind: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.0,
    hybrid_alpha: float = 0.70,
) -> str:
    """Search cross-agent semantic memories using hybrid dense vector and keyword retrieval."""
    try:
        kind_enum = MemoryKind(kind) if kind else None
        results = _memory_store.search(
            query=query,
            requesting_agent_id=agent_id,
            namespace=namespace,
            kind=kind_enum,
            top_k=top_k,
            min_score=min_score,
            hybrid_alpha=hybrid_alpha,
        )
        return json.dumps({
            "status": "success",
            "query": query,
            "count": len(results),
            "results": [r.to_dict() for r in results],
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_memory_get_stats() -> str:
    """Retrieve cross-agent memory store health, volume, and partition metrics."""
    try:
        stats = _memory_store.get_stats()
        return json.dumps({
            "status": "success",
            "stats": stats,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_memory_clear_or_evict(
    namespace: str = "default",
    target_capacity: int = 100,
    strategy: str = "DECAY",
) -> str:
    """Evict expired or low-value memories to maintain strict capacity limits."""
    try:
        evicted = _memory_store.evict_to_capacity(
            namespace=namespace,
            target_capacity=target_capacity,
            strategy=strategy,
        )
        return json.dumps({
            "status": "success",
            "namespace": namespace,
            "evicted_count": evicted,
            "target_capacity": target_capacity,
            "strategy": strategy,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_arena_run_match(
    agent_a: str,
    agent_b: str,
    tasks_passed_a: int = 8,
    tasks_passed_b: int = 7,
    total_tasks: int = 10,
    tokens_a: int = 1500,
    tokens_b: int = 2200,
    ast_score_a: float = 1.0,
    ast_score_b: float = 1.0,
    trajectory_score_a: float = 1.0,
    trajectory_score_b: float = 1.0,
) -> str:
    """Execute a competitive head-to-head evaluation match in the multi-agent tournament arena."""
    try:
        pass_rate_a = tasks_passed_a / total_tasks if total_tasks > 0 else 0.0
        pass_rate_b = tasks_passed_b / total_tasks if total_tasks > 0 else 0.0

        score_a = AgentMatchScore(
            agent_id=agent_a,
            tasks_passed=tasks_passed_a,
            total_tasks=total_tasks,
            pass_rate=pass_rate_a,
            avg_latency_ms=150.0,
            total_tokens=tokens_a,
            ast_score=ast_score_a,
            trajectory_score=trajectory_score_a,
        )
        score_b = AgentMatchScore(
            agent_id=agent_b,
            tasks_passed=tasks_passed_b,
            total_tasks=total_tasks,
            pass_rate=pass_rate_b,
            avg_latency_ms=180.0,
            total_tokens=tokens_b,
            ast_score=ast_score_b,
            trajectory_score=trajectory_score_b,
        )

        outcome = _arena_leaderboard.record_match(
            agent_a=agent_a,
            agent_b=agent_b,
            score_a=score_a,
            score_b=score_b,
        )

        return json.dumps({
            "status": "success",
            "match": outcome.to_dict(),
            "new_ratings": {
                agent_a: _arena_leaderboard.ratings[agent_a],
                agent_b: _arena_leaderboard.ratings[agent_b],
            },
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_arena_get_ratings() -> str:
    """Retrieve full Elo leaderboard rankings, match records, and Wilson confidence intervals."""
    try:
        rankings = _arena_leaderboard.get_rankings()
        return json.dumps({
            "status": "success",
            "total_agents": len(rankings),
            "total_matches": len(_arena_leaderboard.match_history),
            "rankings": rankings,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_arena_benchmark_swarm(
    workflow_id: str = "wf-std-dev",
    workflow_name: str = "Standard 4-Stage Multi-Agent Dev Pipeline",
    planner: str = "agent-planner",
    coder: str = "agent-coder",
    reviewer: str = "agent-reviewer",
    auditor: str = "agent-auditor",
) -> str:
    """Execute end-to-end multi-agent collaborative DAG workflow benchmark."""
    try:
        stages = SwarmWorkflowHarness.create_standard_dev_workflow(
            workflow_id=workflow_id,
            planner=planner,
            coder=coder,
            reviewer=reviewer,
            auditor=auditor,
        )
        report = _workflow_harness.execute_workflow(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            stages=stages,
        )
        return json.dumps({
            "status": "success",
            "report": report.to_dict(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def agentlab_arena_export_ratings(export_path: str = "") -> str:
    """Export the arena leaderboard and match history to JSON."""
    try:
        data = _arena_leaderboard.to_dict()
        if export_path:
            p = Path(export_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json.dumps({
            "status": "success",
            "exported_agents": len(_arena_leaderboard.ratings),
            "exported_matches": len(_arena_leaderboard.match_history),
            "path": export_path or "in-memory",
            "data": data if not export_path else None,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})






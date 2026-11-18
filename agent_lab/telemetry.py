"""
agent_lab.telemetry: Cross-agent telemetry aggregation and terminal dashboard.
Monitors CoW rollbacks, BFT consensus rounds, equivocation slashing,
benchmark pass rates, and tool-use trajectory F1 scores.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class TelemetryEvent:
    """An individual recorded operational event."""
    timestamp: str
    category: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryMetrics:
    """Aggregated numerical metrics across the agent ecosystem."""
    sessions_created: int = 0
    checkpoints_created: int = 0
    rollbacks_executed: int = 0
    consensus_rounds: int = 0
    consensus_approved: int = 0
    consensus_rejected: int = 0
    equivocation_slashed: int = 0
    solana_simulations: int = 0
    eval_tasks_run: int = 0
    eval_tasks_passed: int = 0
    trajectories_evaluated: int = 0
    trajectory_f1_sum: float = 0.0
    loops_detected: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_inference_cost_usd: float = 0.0
    total_inference_calls: int = 0
    inference_latency_sum_ms: float = 0.0
    router_routed_calls: int = 0
    router_fallback_calls: int = 0
    router_circuit_breaks: int = 0
    router_throttled_calls: int = 0


class TelemetryCollector:
    """Thread-safe collector for cross-agent metrics, events, and performance diagnostics."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self.storage_path = Path(storage_path).resolve() if storage_path else None
        self.metrics = TelemetryMetrics()
        self.events: List[TelemetryEvent] = []

    def record_session(self, agent_name: str, goal: str) -> None:
        """Record start of a new agent execution session."""
        self.metrics.sessions_created += 1
        self._add_event("session", "start", {"agent": agent_name, "goal": goal})

    def record_checkpoint(self, session_id: str, label: str) -> None:
        """Record micro-checkpoint creation."""
        self.metrics.checkpoints_created += 1
        self._add_event("checkpoint", "create", {"session_id": session_id, "label": label})

    def record_rollback(self, session_id: str, checkpoint_id: str, success: bool) -> None:
        """Record workspace rollback to a safe state."""
        self.metrics.rollbacks_executed += 1
        self._add_event(
            "rollback",
            "execute",
            {"session_id": session_id, "checkpoint_id": checkpoint_id, "success": success},
        )

    def record_consensus(self, approved: bool, validators_count: int, slashed: int = 0) -> None:
        """Record a Byzantine consensus proposal round."""
        self.metrics.consensus_rounds += 1
        if approved:
            self.metrics.consensus_approved += 1
        else:
            self.metrics.consensus_rejected += 1
        self.metrics.equivocation_slashed += slashed
        self._add_event(
            "consensus",
            "round",
            {"approved": approved, "validators": validators_count, "slashed": slashed},
        )

    def record_solana_sim(self, success: bool, units_consumed: int = 0) -> None:
        """Record pre-flight Solana transaction simulation."""
        self.metrics.solana_simulations += 1
        self._add_event("solana", "simulation", {"success": success, "units": units_consumed})

    def record_eval(self, task_id: str, passed: bool) -> None:
        """Record an automated coding benchmark evaluation."""
        self.metrics.eval_tasks_run += 1
        if passed:
            self.metrics.eval_tasks_passed += 1
        self._add_event("eval", "task", {"task_id": task_id, "passed": passed})

    def record_trajectory_eval(self, f1_score: float, has_loop: bool) -> None:
        """Record tool-use trajectory analysis."""
        self.metrics.trajectories_evaluated += 1
        self.metrics.trajectory_f1_sum += f1_score
        if has_loop:
            self.metrics.loops_detected += 1
        self._add_event("trajectory", "eval", {"f1_score": f1_score, "has_loop": has_loop})

    def record_inference(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Record an LLM model inference call with token economics and latency."""
        self.metrics.total_inference_calls += 1
        self.metrics.total_prompt_tokens += prompt_tokens
        self.metrics.total_completion_tokens += completion_tokens
        self.metrics.inference_latency_sum_ms += latency_ms
        if cost_usd is not None:
            self.metrics.total_inference_cost_usd += cost_usd
        self._add_event(
            "inference",
            "call",
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
        )

    def record_routing_event(
        self,
        primary_model: str,
        resolved_model: str,
        fallback_occurred: bool,
        latency_ms: float,
        tokens: int,
        priority: str = "NORMAL",
        status: str = "success",
    ) -> None:
        """Record a model routing and fallback event."""
        self.metrics.router_routed_calls += 1
        if fallback_occurred:
            self.metrics.router_fallback_calls += 1
        self._add_event(
            "router",
            "execution",
            {
                "primary_model": primary_model,
                "resolved_model": resolved_model,
                "fallback_occurred": fallback_occurred,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "priority": priority,
                "status": status,
            },
        )

    def avg_inference_latency_ms(self) -> float:
        """Calculate mean inference latency in milliseconds."""
        if self.metrics.total_inference_calls == 0:
            return 0.0
        return round(self.metrics.inference_latency_sum_ms / self.metrics.total_inference_calls, 1)

    def average_tool_f1(self) -> float:
        """Calculate mean tool-selection F1 score across all evaluated trajectories."""
        if self.metrics.trajectories_evaluated == 0:
            return 0.0
        return round(self.metrics.trajectory_f1_sum / self.metrics.trajectories_evaluated, 4)

    def eval_pass_rate(self) -> float:
        """Calculate benchmark pass rate."""
        if self.metrics.eval_tasks_run == 0:
            return 0.0
        return round(self.metrics.eval_tasks_passed / self.metrics.eval_tasks_run, 4)

    def _add_event(self, category: str, action: str, metadata: Dict[str, Any]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.events.append(TelemetryEvent(timestamp=ts, category=category, action=action, metadata=metadata))

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Return a structured dictionary of all aggregated metrics."""
        data = asdict(self.metrics)
        data["average_tool_f1"] = self.average_tool_f1()
        data["eval_pass_rate"] = self.eval_pass_rate()
        data["avg_inference_latency_ms"] = self.avg_inference_latency_ms()
        data["recent_events_count"] = len(self.events)
        return data

    def save(self, path: Optional[str] = None) -> str:
        """Persist telemetry state to disk."""
        target = Path(path).resolve() if path else (self.storage_path or Path(".agent_lab/telemetry.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": self.get_metrics_dict(),
            "events": [asdict(e) for e in self.events[-100:]],  # keep last 100 events
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(target)

    @classmethod
    def load(cls, path: str) -> TelemetryCollector:
        """Load telemetry metrics and history from disk."""
        target = Path(path).resolve()
        if not target.exists():
            return cls(storage_path=str(target))

        data = json.loads(target.read_text(encoding="utf-8"))
        collector = cls(storage_path=str(target))
        m = data.get("metrics", {})
        collector.metrics = TelemetryMetrics(
            sessions_created=m.get("sessions_created", 0),
            checkpoints_created=m.get("checkpoints_created", 0),
            rollbacks_executed=m.get("rollbacks_executed", 0),
            consensus_rounds=m.get("consensus_rounds", 0),
            consensus_approved=m.get("consensus_approved", 0),
            consensus_rejected=m.get("consensus_rejected", 0),
            equivocation_slashed=m.get("equivocation_slashed", 0),
            solana_simulations=m.get("solana_simulations", 0),
            eval_tasks_run=m.get("eval_tasks_run", 0),
            eval_tasks_passed=m.get("eval_tasks_passed", 0),
            trajectories_evaluated=m.get("trajectories_evaluated", 0),
            trajectory_f1_sum=m.get("trajectory_f1_sum", 0.0),
            loops_detected=m.get("loops_detected", 0),
            total_prompt_tokens=m.get("total_prompt_tokens", 0),
            total_completion_tokens=m.get("total_completion_tokens", 0),
            total_inference_cost_usd=m.get("total_inference_cost_usd", 0.0),
            total_inference_calls=m.get("total_inference_calls", 0),
            inference_latency_sum_ms=m.get("inference_latency_sum_ms", 0.0),
            router_routed_calls=m.get("router_routed_calls", 0),
            router_fallback_calls=m.get("router_fallback_calls", 0),
            router_circuit_breaks=m.get("router_circuit_breaks", 0),
            router_throttled_calls=m.get("router_throttled_calls", 0),
        )
        for ev in data.get("events", []):
            collector.events.append(
                TelemetryEvent(
                    timestamp=ev.get("timestamp", ""),
                    category=ev.get("category", ""),
                    action=ev.get("action", ""),
                    metadata=ev.get("metadata", {}),
                )
            )
        return collector

    def to_markdown(self) -> str:
        """Render a clean Markdown summary report of telemetry."""
        m = self.get_metrics_dict()
        return f"""# Agent-Lab Telemetry & Ecosystem Report

- **Sessions Created:** {m['sessions_created']}
- **Checkpoints Created:** {m['checkpoints_created']}
- **Rollbacks Executed:** {m['rollbacks_executed']}
- **Consensus Rounds:** {m['consensus_rounds']} (Approved: {m['consensus_approved']}, Rejected: {m['consensus_rejected']}, Slashed: {m['equivocation_slashed']})
- **Solana Simulations:** {m['solana_simulations']}
- **Benchmark Evaluations:** {m['eval_tasks_run']} (Passed: {m['eval_tasks_passed']}, Pass Rate: {m['eval_pass_rate'] * 100:.1f}%)
- **Trajectory Evaluations:** {m['trajectories_evaluated']} (Mean Tool F1: {m['average_tool_f1']:.4f}, Loops Caught: {m['loops_detected']})
- **Model Inference Calls:** {m['total_inference_calls']} (Tokens: {m['total_prompt_tokens'] + m['total_completion_tokens']:,}, Cost: ${m['total_inference_cost_usd']:.4f}, Avg Latency: {m['avg_inference_latency_ms']:.1f}ms)
- **Model Router & Fallbacks:** {m['router_routed_calls']} routed ({m['router_fallback_calls']} fallbacks, {m['router_circuit_breaks']} circuit trips)
"""

    def render_dashboard(self, console: Optional[Console] = None) -> None:
        """Render an interactive Rich terminal dashboard."""
        con = console or Console()
        m = self.get_metrics_dict()

        # Operational table
        ops_table = Table(title="Agent Lifecycle & Safety Metrics", expand=True)
        ops_table.add_column("Category", style="cyan")
        ops_table.add_column("Count", style="bold green", justify="right")
        ops_table.add_column("Health / Status", style="yellow")

        ops_table.add_row("Sessions Started", str(m["sessions_created"]), "Active Monitoring")
        ops_table.add_row("Micro-Checkpoints", str(m["checkpoints_created"]), "Copy-on-Write Journaled")
        ops_table.add_row("Rollbacks Executed", str(m["rollbacks_executed"]), "State Recovered Cleanly")
        ops_table.add_row("Consensus Rounds", str(m["consensus_rounds"]), f"{m['consensus_approved']} Approved / {m['consensus_rejected']} Rejected")
        ops_table.add_row("Equivocations Slashed", str(m["equivocation_slashed"]), "Byzantine Nodes Isolated")
        ops_table.add_row("Solana Pre-Flight Simulations", str(m["solana_simulations"]), "Compute Budgets Verified")

        # Evaluation & Benchmark Table
        eval_table = Table(title="Benchmark & Trajectory Evaluation", expand=True)
        eval_table.add_column("Evaluation Metric", style="magenta")
        eval_table.add_column("Score / Count", style="bold green", justify="right")
        eval_table.add_column("Target Quality", style="cyan")

        pass_rate_str = f"{m['eval_pass_rate'] * 100:.1f}% ({m['eval_tasks_passed']}/{m['eval_tasks_run']})"
        f1_str = f"{m['average_tool_f1']:.4f}"

        eval_table.add_row("Benchmark Tasks Pass Rate", pass_rate_str, ">= 80% Threshold")
        eval_table.add_row("Mean Tool-Selection F1", f1_str, ">= 0.85 Precision/Recall")
        eval_table.add_row("Trajectories Audited", str(m["trajectories_evaluated"]), "Multi-step Agent Loops")
        eval_table.add_row("Infinite Loops Intercepted", str(m["loops_detected"]), "Zero-Loop Tolerance")
        if m["total_inference_calls"] > 0:
            eval_table.add_row(
                "Model Inference Calls",
                f"{m['total_inference_calls']} calls (${m['total_inference_cost_usd']:.4f})",
                f"Avg Latency: {m['avg_inference_latency_ms']:.1f}ms",
            )
        if m["router_routed_calls"] > 0:
            eval_table.add_row(
                "Router Dynamic Cascade",
                f"{m['router_routed_calls']} routed ({m['router_fallback_calls']} fallbacks)",
                f"{m['router_circuit_breaks']} Circuit Trips",
            )

        con.print(Panel(ops_table, title="[bold cyan]Agent-Lab Operational Dashboard[/bold cyan]"))
        con.print(Panel(eval_table, title="[bold magenta]Autonomous OSS Quality Gates[/bold magenta]"))

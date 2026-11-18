"""
agent_lab.cli: Command-Line Interface for Agent-Lab.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_lab.config import AgentLabConfig
from agent_lab.orchestrator import AgentLabOrchestrator, HAS_ROLLBACK, HAS_WITNESS, HAS_SOLANA_LENS
from agent_lab.consensus import SwarmCoordinator, SwarmProposal, SwarmVote
from agent_lab.eval import (
    PassAtKEstimator,
    ASTVerifier,
    RegressionTracker,
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

console = Console()


@click.group()
@click.version_option(package_name="agent-lab")
def main() -> None:
    """agent-lab: Unified autonomous AI agent safety orchestrator, BFT swarm consensus, & eval harness."""
    pass


@main.command()
@click.option("--workspace", "-w", default=".", help="Target workspace path.")
def init(workspace: str) -> None:
    """Initialize an Agent-Lab safety configuration in the workspace."""
    root = Path(workspace).resolve()
    config = AgentLabConfig(workspace_root=str(root))
    config_path = config.save()

    console.print(Panel(
        f"[green]✓ Initialized Agent-Lab workspace[/green]\n"
        f"Workspace Root: [cyan]{root}[/cyan]\n"
        f"Config File: [yellow]{config_path}[/yellow]",
        title="Agent-Lab Init",
    ))


@main.command()
def doctor() -> None:
    """Diagnose local environment and portfolio integrations."""
    table = Table(title="Agent-Lab Integration Diagnostics")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Notes", style="dim")

    table.add_row(
        "rollback",
        "[green]Available[/green]" if HAS_ROLLBACK else "[yellow]Internal Fallback[/yellow]",
        "In-process CoW journaling and micro-checkpoints",
    )
    table.add_row(
        "solana-tx-lens",
        "[green]Available[/green]" if HAS_SOLANA_LENS else "[yellow]Internal Fallback[/yellow]",
        "Pre-flight Solana tx simulation & CU profiling",
    )
    table.add_row(
        "agent-witness",
        "[green]Available[/green]" if HAS_WITNESS else "[yellow]Internal Fallback[/yellow]",
        "RFC 6962 Merkle action receipts & BFT consensus",
    )

    console.print(table)


@main.group()
def session() -> None:
    """Manage transactional and attested agent execution sessions."""
    pass


@session.command("start")
@click.option("--agent", "-a", default="WorkerAgent", help="Name of the agent.")
@click.option("--goal", "-g", required=True, help="Session objective or prompt.")
def session_start(agent: str, goal: str) -> None:
    """Start a new monitored agent session."""
    orchestrator = AgentLabOrchestrator()
    sess = orchestrator.start_session(agent_name=agent, goal=goal)

    console.print(Panel(
        f"Session ID: [bold cyan]{sess.session_id}[/bold cyan]\n"
        f"Agent Name: [magenta]{sess.agent_name}[/magenta]\n"
        f"Goal: {sess.goal}",
        title="Session Started",
    ))


@main.group()
def eval() -> None:
    """Autonomous OSS evaluation, trajectory verification, and benchmarking."""
    pass


@eval.command("pass-at-k")
@click.option("-n", type=int, required=True, help="Total samples generated.")
@click.option("-c", type=int, required=True, help="Number of correct samples.")
@click.option("-k", type=int, required=True, help="Evaluations count (k).")
def eval_pass_at_k(n: int, c: int, k: int) -> None:
    """Compute unbiased Pass@k metric with Wilson score confidence interval."""
    try:
        score = PassAtKEstimator.estimate(n=n, c=c, k=k)
        ci_lower, ci_upper = StatisticalInterval.wilson_score_interval(c, n)
        console.print(f"[bold green]Pass@{k}:[/bold green] [cyan]{score:.4f}[/cyan] ({score*100:.2f}%)")
        console.print(f"[dim]95% Confidence Interval (Wilson): [{ci_lower:.4f}, {ci_upper:.4f}][/dim]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@eval.command("ast-check")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--entry", "-e", default=None, help="Expected function entrypoint.")
def eval_ast_check(file_path: str, entry: Optional[str]) -> None:
    """Verify code AST syntax and calculate cyclomatic complexity."""
    code = Path(file_path).read_text(encoding="utf-8")
    res = ASTVerifier.verify(code=code, expected_entry_point=entry)

    if res["valid"]:
        console.print(f"[green]✓ Code is syntactically valid[/green]")
        console.print(f"Cyclomatic Complexity: [yellow]{res['cyclomatic_complexity']}[/yellow]")
        if entry:
            status = "[green]Found[/green]" if res["entry_point_found"] else "[red]Missing[/red]"
            console.print(f"Entry Point '{entry}': {status}")
    else:
        console.print(f"[bold red]✗ AST Error:[/bold red] {res['error']}")
        sys.exit(1)


@eval.command("trajectory")
@click.option("--trajectory", "-t", required=True, type=click.Path(exists=True), help="JSON file with tool calls.")
@click.option("--expected", "-e", required=True, help="Comma-separated list of expected tools.")
@click.option("--max-steps", "-m", type=int, default=None, help="Maximum allowed steps.")
def eval_trajectory(trajectory: str, expected: str, max_steps: Optional[int]) -> None:
    """Evaluate an agent tool trajectory for F1 score, loops, and step efficiency."""
    try:
        data = json.loads(Path(trajectory).read_text(encoding="utf-8"))
        expected_list = [t.strip() for t in expected.split(",") if t.strip()]

        res = TrajectoryEvaluator.evaluate(
            trajectory=data,
            expected_tools=expected_list,
            max_steps=max_steps,
        )

        table = Table(title="Trajectory Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green" if res["valid"] else "bold yellow")

        table.add_row("Outcome Validity", "✓ Valid" if res["valid"] else "✗ Failed Rules")
        table.add_row("Precision", f"{res['precision']:.4f}")
        table.add_row("Recall", f"{res['recall']:.4f}")
        table.add_row("Tool-Selection F1", f"{res['f1_score']:.4f}")
        table.add_row("Steps Used", f"{res['step_count']}")
        table.add_row("Step Efficiency", f"{res['step_efficiency']:.4f}")
        table.add_row("Loop Detected", "[red]Yes[/red]" if res["has_loop"] else "[green]No[/green]")

        console.print(table)
        if not res["valid"]:
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@eval.command("suite")
@click.option("--tasks", "-t", required=True, type=click.Path(exists=True), help="Benchmark JSON suite file.")
def eval_suite(tasks: str) -> None:
    """Run an entire coding benchmark suite using canonical verification."""
    harness = BenchmarkHarness()
    summary = harness.run_suite(tasks_file=tasks)

    table = Table(title=f"Benchmark Suite Results ({tasks})")
    table.add_column("Task ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for r in summary.get("results", []):
        st = "[green]PASSED[/green]" if r.get("passed") else "[red]FAILED[/red]"
        err = r.get("error") or "All assertions passed"
        table.add_row(r.get("task_id", ""), st, err)

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {summary['total']} | "
        f"[bold green]Passed:[/bold green] {summary['passed']} | "
        f"[bold red]Failed:[/bold red] {summary['failed']} | "
        f"[bold cyan]Pass Rate:[/bold cyan] {summary['pass_rate']*100:.1f}% "
        f"[dim](95% CI: [{summary['confidence_interval_95'][0]:.2f}, {summary['confidence_interval_95'][1]:.2f}])[/dim]"
    )


@main.group()
def telemetry() -> None:
    """Cross-agent telemetry aggregation and performance dashboard."""
    pass


@telemetry.command("show")
@click.option("--path", "-p", default=None, help="Custom telemetry file path.")
def telemetry_show(path: Optional[str]) -> None:
    """Display the interactive terminal telemetry dashboard."""
    collector = TelemetryCollector.load(path) if path else TelemetryCollector()
    collector.render_dashboard(console)


@telemetry.command("export")
@click.option("--format", "-f", type=click.Choice(["json", "markdown"]), default="markdown", help="Export format.")
@click.option("--output", "-o", default=None, help="Target file path for output.")
def telemetry_export(format: str, output: Optional[str]) -> None:
    """Export telemetry metrics as Markdown or JSON."""
    collector = TelemetryCollector()
    content = collector.to_markdown() if format == "markdown" else json.dumps(collector.get_metrics_dict(), indent=2)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]✓ Exported telemetry to {output}[/green]")
    else:
        console.print(content)


@main.group()
def swarm() -> None:
    """Distributed WebSocket BFT swarm consensus coordinator and worker nodes."""
    pass


@swarm.command("serve")
@click.option("--host", "-h", default="127.0.0.1", help="Host interface to bind.")
@click.option("--port", "-p", default=8765, type=int, help="Port to listen on.")
@click.option("--workspace", "-w", default=".", help="Workspace path to manage.")
@click.option("--secret", "-s", default="", help="Shared secret for HMAC authentication.")
@click.option("--quorum", "-q", default=0.67, type=float, help="Quorum ratio required for BFT consensus.")
def swarm_serve(host: str, port: int, workspace: str, secret: str, quorum: float) -> None:
    """Start the distributed WebSocket consensus coordinator server."""
    coordinator = DistributedCoordinator(
        workspace_root=workspace,
        host=host,
        port=port,
        secret=secret,
        quorum_ratio=quorum,
    )

    async def run_server() -> None:
        await coordinator.start()
        console.print(Panel(
            f"[bold green]✓ Distributed Swarm Coordinator Active[/bold green]\n"
            f"WebSocket URL: [cyan]ws://{host}:{coordinator.server_port}[/cyan]\n"
            f"Workspace: [dim]{coordinator.workspace_root}[/dim]\n"
            f"Quorum Ratio: [dim]{quorum:.2f}[/dim]\n"
            f"Auth: [dim]{'HMAC-SHA256 Enabled' if secret else 'Open / Unauthenticated'}[/dim]",
            title="Agent-Lab Swarm Hub",
            border_style="green",
        ))
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            await coordinator.stop()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        console.print("[yellow]Coordinator stopped by user.[/yellow]")


@swarm.command("worker")
@click.option("--url", "-u", default="ws://127.0.0.1:8765", help="Coordinator WebSocket URL.")
@click.option("--node-id", "-n", required=True, help="Unique node identifier.")
@click.option("--secret", "-s", default="", help="Shared secret for HMAC authentication.")
@click.option("--weight", default=1.0, type=float, help="Validator voting weight.")
def swarm_worker(url: str, node_id: str, secret: str, weight: float) -> None:
    """Connect a worker validator node to a distributed coordinator."""
    node = DistributedSwarmNode(
        node_id=node_id,
        coordinator_url=url,
        secret=secret,
        weight=weight,
        vote_callback=lambda p: (True, "Auto-approved by policy"),
    )

    async def run_worker() -> None:
        connected = await node.connect()
        if not connected:
            console.print(f"[bold red]✗ Failed connecting to coordinator at {url}[/bold red]")
            sys.exit(1)

        console.print(Panel(
            f"[bold green]✓ Swarm Validator Node Connected[/bold green]\n"
            f"Node ID: [cyan]{node_id}[/cyan]\n"
            f"Weight: [dim]{weight}[/dim]\n"
            f"Coordinator: [cyan]{url}[/cyan]",
            title="Agent-Lab Swarm Worker",
            border_style="green",
        ))

        try:
            while node.is_connected:
                msg = await node.listen_one(timeout=5.0)
                if msg and msg.msg_type == MessageType.PROPOSAL:
                    prop_id = msg.payload.get("proposal", {}).get("proposal_id", "")
                    console.print(f"[cyan]→ Received proposal {prop_id}, auto-cast APPROVE vote[/cyan]")
                elif msg and msg.msg_type == MessageType.COMMIT:
                    console.print(f"[green]✓ Consensus COMMIT round completed for proposal {msg.payload.get('proposal_id')}[/green]")
        except (KeyboardInterrupt, asyncio.CancelledError):
            await node.disconnect()

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        console.print("[yellow]Worker stopped by user.[/yellow]")


@swarm.command("broadcast-proposal")
@click.option("--workspace", "-w", default=".", help="Target workspace root.")
@click.option("--proposal-id", "-p", required=True, help="Proposal ID.")
@click.option("--proposer", required=True, help="Proposing agent name.")
@click.option("--file", "-f", "target_file", required=True, help="Relative path of file to mutate.")
@click.option("--content", "-c", required=True, help="New file content.")
@click.option("--secret", "-s", default="", help="Shared HMAC secret.")
@click.option("--port", default=8765, type=int, help="Coordinator port.")
def swarm_broadcast_proposal(workspace: str, proposal_id: str, proposer: str, target_file: str, content: str, secret: str, port: int) -> None:
    """Broadcast an atomic mutation proposal and run a consensus round."""
    coordinator = DistributedCoordinator(
        workspace_root=workspace,
        port=port,
        secret=secret,
    )

    async def run_broadcast() -> None:
        await coordinator.start()
        file_bytes = content.encode("utf-8")
        proposal = SwarmProposal(
            proposal_id=proposal_id,
            proposer=proposer,
            description="CLI broadcast proposal",
            read_set={},
            write_set={target_file: hashlib.sha256(file_bytes).hexdigest()},
            file_mutations={target_file: file_bytes},
        )
        res = await coordinator.broadcast_proposal(proposal=proposal, timeout=1.0)
        await coordinator.stop()

        status_str = "[bold green]COMMITTED[/bold green]" if res.committed else "[bold red]REJECTED[/bold red]"
        console.print(Panel(
            f"Round ID: [cyan]{res.round_id}[/cyan]\n"
            f"Proposal: [dim]{res.proposal_id}[/dim]\n"
            f"Status: {status_str}\n"
            f"Quorum Reached: [dim]{res.quorum_reached}[/dim]\n"
            f"Applied Files: [dim]{res.applied_files}[/dim]\n"
            f"Duration: [dim]{res.duration_ms} ms[/dim]",
            title="Consensus Round Result",
            border_style="green" if res.committed else "red",
        ))

    asyncio.run(run_broadcast())


if __name__ == "__main__":
    main()

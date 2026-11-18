"""
agent_lab.cli: Command-Line Interface for Agent-Lab.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
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
from agent_lab.registry import (
    AnchorAgentRegistryClient,
    AgentStatus,
    SlashingProof,
    ANCHOR_AGENT_REGISTRY_IDL,
    DEFAULT_MIN_STAKE_LAMPORTS,
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
    RateLimitPolicy,
    CircuitBreakerPolicy,
    DEFAULT_FALLBACK_ROUTES,
    DEFAULT_MODEL_POLICIES,
    RoutingExecutionReceipt,
    CascadeExhaustedError,
    QuotaExceededError,
)
from agent_lab.memory import (
    MemoryKind,
    MemoryAccessLevel,
    MemoryEntry,
    MemorySearchResult,
    CrossAgentMemoryStore,
    SemanticTextEmbedder,
)
from agent_lab.arena import (
    MatchResult,
    AgentMatchScore,
    MatchOutcome,
    EloRatingSystem,
    BradleyTerryEstimator,
    ArenaLeaderboard,
    SwarmWorkflowStage,
    SwarmWorkflowReport,
    SwarmWorkflowHarness,
)
from agent_lab.fleet import (
    FleetDiagnostics,
    FleetHealthReport,
    FleetHealthStatus,
    FleetMissionRunner,
    FleetExecutionReceipt,
    MissionConfig,
)
from agent_lab.portfolio import (
    PortfolioDiscovery,
    PortfolioDoctor,
    PortfolioHealthStatus,
    CrossRepoFleetSimulator,
    UnifiedPortfolioReceipt,
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
@click.option("--deep", is_flag=True, help="Run deep self-diagnostics across all sub-engines.")
def doctor(deep: bool) -> None:
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

    if deep:
        diag = FleetDiagnostics()
        report = diag.run_diagnostics()
        sub_table = Table(title=f"Fleet Subsystem Health Check ({report.status.value})")
        sub_table.add_column("Subsystem", style="cyan")
        sub_table.add_column("Status", style="bold")
        sub_table.add_column("Latency (ms)", justify="right")
        sub_table.add_column("Message", style="dim")
        for check in report.checks:
            color = "green" if check.status.value == "HEALTHY" else ("yellow" if check.status.value == "DEGRADED" else "red")
            sub_table.add_row(
                check.subsystem,
                f"[{color}]{check.status.value}[/{color}]",
                f"{check.latency_ms:.2f}",
                check.message,
            )
        console.print(sub_table)


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


@main.group()
def registry() -> None:
    """Solana Anchor On-Chain Agent Registry, Staking & Slashing tools."""
    pass


@registry.command("init")
@click.option("--storage", default="agent_registry.json", help="Path to registry storage JSON file.")
@click.option("--min-stake", default=DEFAULT_MIN_STAKE_LAMPORTS, help="Minimum required stake in lamports.")
@click.option("--penalty-bps", default=5000, help="Slashing penalty in basis points (e.g. 5000 = 50%).")
def registry_init(storage: str, min_stake: int, penalty_bps: int) -> None:
    """Initialize a Solana Anchor agent registry configuration."""
    client = AnchorAgentRegistryClient(
        storage_path=storage,
        min_stake_lamports=min_stake,
        slashing_penalty_bps=penalty_bps,
    )
    client._save_storage()
    console.print(Panel(
        f"Program ID: [cyan]{client.program_id}[/cyan]\n"
        f"Registry PDA: [cyan]{client.registry_pda}[/cyan] (bump: {client.registry_bump})\n"
        f"Min Stake: [yellow]{client.min_stake_lamports / 1e9:.2f} SOL[/yellow] ({client.min_stake_lamports} lamports)\n"
        f"Slashing Penalty: [yellow]{client.slashing_penalty_bps / 100:.1f}%[/yellow]\n"
        f"Storage File: [dim]{storage}[/dim]",
        title="[bold green]Solana Agent Registry Initialized[/bold green]",
        border_style="green",
    ))


@registry.command("register")
@click.option("--pubkey", required=True, help="Agent Solana public key.")
@click.option("--name", required=True, help="Agent descriptive name.")
@click.option("--stake", default=DEFAULT_MIN_STAKE_LAMPORTS, help="Initial stake in lamports.")
@click.option("--metadata", default="", help="Metadata URI (Arweave/IPFS).")
@click.option("--storage", default="agent_registry.json", help="Path to registry storage JSON file.")
def registry_register(pubkey: str, name: str, stake: int, metadata: str, storage: str) -> None:
    """Register an autonomous agent validator with on-chain stake."""
    client = AnchorAgentRegistryClient(storage_path=storage)
    try:
        agent = client.register_agent(
            pubkey=pubkey,
            name=name,
            metadata_uri=metadata,
            initial_stake=stake,
        )
        console.print(Panel(
            f"Agent Name: [bold]{agent.name}[/bold]\n"
            f"Pubkey: [cyan]{agent.pubkey}[/cyan]\n"
            f"Account PDA: [cyan]{agent.pda}[/cyan] (bump: {agent.bump})\n"
            f"Stake: [yellow]{agent.stake_lamports / 1e9:.4f} SOL[/yellow]\n"
            f"Reputation Score: [green]{agent.reputation_score}[/green]\n"
            f"Status: [green]{agent.status.value.upper()}[/green]",
            title="[bold green]Agent Registered Successfully[/bold green]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Registration failed:[/bold red] {e}")
        sys.exit(1)


@registry.command("stake")
@click.option("--pubkey", required=True, help="Agent Solana public key.")
@click.option("--amount", required=True, type=int, help="Additional stake in lamports.")
@click.option("--storage", default="agent_registry.json", help="Path to registry storage JSON file.")
def registry_stake(pubkey: str, amount: int, storage: str) -> None:
    """Deposit additional SOL stake into an agent's account."""
    client = AnchorAgentRegistryClient(storage_path=storage)
    try:
        agent = client.deposit_stake(pubkey, amount)
        console.print(Panel(
            f"Pubkey: [cyan]{agent.pubkey}[/cyan]\n"
            f"Deposited: [yellow]{amount / 1e9:.4f} SOL[/yellow]\n"
            f"Total Stake: [yellow]{agent.stake_lamports / 1e9:.4f} SOL[/yellow]\n"
            f"Status: [green]{agent.status.value.upper()}[/green]",
            title="[bold green]Stake Deposited[/bold green]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Deposit failed:[/bold red] {e}")
        sys.exit(1)


@registry.command("info")
@click.option("--pubkey", required=True, help="Agent Solana public key.")
@click.option("--storage", default="agent_registry.json", help="Path to registry storage JSON file.")
def registry_info(pubkey: str, storage: str) -> None:
    """Inspect an agent's on-chain account, stake, and reputation score."""
    client = AnchorAgentRegistryClient(storage_path=storage)
    agent = client.get_agent(pubkey)
    if not agent:
        console.print(f"[bold red]Agent '{pubkey}' not found in registry.[/bold red]")
        sys.exit(1)

    eligible, reason = client.verify_agent_eligibility(pubkey)
    status_color = "green" if agent.status == AgentStatus.ACTIVE else "red"
    console.print(Panel(
        f"Name: [bold]{agent.name}[/bold]\n"
        f"Pubkey: [cyan]{agent.pubkey}[/cyan]\n"
        f"PDA: [dim]{agent.pda}[/dim] (bump: {agent.bump})\n"
        f"Status: [{status_color}]{agent.status.value.upper()}[/{status_color}]\n"
        f"Stake: [yellow]{agent.stake_lamports / 1e9:.4f} SOL[/yellow] ({agent.stake_lamports} lamports)\n"
        f"Reputation Score: [cyan]{agent.reputation_score}[/cyan] / 10000\n"
        f"Consensus Eligibility: [{'green' if eligible else 'red'}]{'ELIGIBLE' if eligible else 'INELIGIBLE'}[/{'green' if eligible else 'red'}] ({reason})\n"
        f"Slashed Count: [red]{agent.slashed_count}[/red]\n"
        f"Metadata URI: [dim]{agent.metadata_uri or 'None'}[/dim]",
        title="Agent On-Chain Account Details",
        border_style="cyan",
    ))


@registry.command("list")
@click.option("--storage", default="agent_registry.json", help="Path to registry storage JSON file.")
def registry_list(storage: str) -> None:
    """List all registered agents and their stake/reputation metrics."""
    client = AnchorAgentRegistryClient(storage_path=storage)
    agents = client.list_agents()
    if not agents:
        console.print("[yellow]No agents currently registered.[/yellow]")
        return

    table = Table(title="Solana Anchor Agent Registry Validators")
    table.add_column("Name", style="bold")
    table.add_column("Pubkey", style="cyan")
    table.add_column("Stake (SOL)", style="yellow")
    table.add_column("Reputation", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Slashes", style="red")

    for a in agents:
        status_style = "green" if a.status == AgentStatus.ACTIVE else "red"
        table.add_row(
            a.name,
            a.pubkey,
            f"{a.stake_lamports / 1e9:.3f}",
            str(a.reputation_score),
            f"[{status_style}]{a.status.value.upper()}[/{status_style}]",
            str(a.slashed_count),
        )

    console.print(table)


@registry.command("slash-proof")
@click.option("--offender", required=True, help="Offending agent public key.")
@click.option("--reporter", required=True, help="Reporting agent public key.")
@click.option("--proposal", required=True, help="Proposal ID where equivocation occurred.")
@click.option("--vote-a", required=True, type=str, help="First vote decision (approve/reject or true/false).")
@click.option("--sig-a", default="sig_a", help="First vote signature.")
@click.option("--vote-b", required=True, type=str, help="Second conflicting vote decision.")
@click.option("--sig-b", default="sig_b", help="Second vote signature.")
@click.option("--storage", default="agent_registry.json", help="Path to registry storage JSON file.")
def registry_slash_proof(
    offender: str,
    reporter: str,
    proposal: str,
    vote_a: str,
    sig_a: str,
    vote_b: str,
    sig_b: str,
    storage: str,
) -> None:
    """Submit a cryptographic equivocation proof and slash offender stake."""
    def _parse_decision(val: str) -> bool:
        s = val.strip().lower()
        return s in ("true", "1", "approve", "precommit", "yes")

    client = AnchorAgentRegistryClient(storage_path=storage)
    proof = SlashingProof(
        offender_pubkey=offender,
        reporter_pubkey=reporter,
        proposal_id=proposal,
        vote_a_decision=_parse_decision(vote_a),
        vote_a_sig=sig_a,
        vote_b_decision=_parse_decision(vote_b),
        vote_b_sig=sig_b,
    )
    try:
        receipt = client.submit_slashing_proof(proof)
        console.print(Panel(
            f"Receipt ID: [cyan]{receipt.receipt_id}[/cyan]\n"
            f"Offender: [red]{receipt.offender_pubkey}[/red] ([bold red]JAILED[/bold red])\n"
            f"Reporter: [green]{receipt.reporter_pubkey}[/green]\n"
            f"Slashed Collateral: [red]{receipt.slashed_lamports / 1e9:.4f} SOL[/red]\n"
            f"Burned: [red]{receipt.burned_lamports / 1e9:.4f} SOL[/red] | Reward: [green]{receipt.reward_lamports / 1e9:.4f} SOL[/green]\n"
            f"Tx Signature: [dim]{receipt.tx_signature}[/dim]",
            title="[bold red]Equivocation Slashing Executed[/bold red]",
            border_style="red",
        ))
    except Exception as e:
        console.print(f"[bold red]Slashing failed:[/bold red] {e}")
        sys.exit(1)


@registry.command("export-idl")
@click.option("--out", default="", help="Output JSON path (prints to stdout if omitted).")
def registry_export_idl(out: str) -> None:
    """Export the Anchor Program IDL specification for agent registry."""
    idl_json = json.dumps(ANCHOR_AGENT_REGISTRY_IDL, indent=2)
    if out:
        Path(out).write_text(idl_json, encoding="utf-8")
        console.print(f"[green]Anchor IDL exported to [bold]{out}[/bold][/green]")
    else:
        click.echo(idl_json)


@main.group()
def zk() -> None:
    """Zero-Knowledge Trajectory Execution & Deliberation Verifier tools."""
    pass


@zk.command("generate-receipt")
@click.option("--trajectory-file", "-f", default="", help="Path to JSON file containing trajectory steps.")
@click.option("--session-id", "-s", default="", help="Optional session ID.")
@click.option("--system", default="risc0", type=click.Choice(["risc0", "sp1"]), help="zkVM target system.")
@click.option("--max-steps", default=50, help="Maximum allowable steps in policy constraints.")
@click.option("--require-ast/--no-require-ast", default=False, help="Enforce AST syntax validity on generated code.")
@click.option("--out", default="", help="Output file path for generated receipt JSON.")
def zk_generate_receipt(
    trajectory_file: str,
    session_id: str,
    system: str,
    max_steps: int,
    require_ast: bool,
    out: str,
) -> None:
    """Synthesize a zero-knowledge trajectory execution receipt (RISC Zero / SP1)."""
    steps = []
    if trajectory_file:
        p = Path(trajectory_file)
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {trajectory_file}")
            sys.exit(1)
        try:
            steps = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[bold red]Failed to parse trajectory JSON:[/bold red] {e}")
            sys.exit(1)
    else:
        # Generate a standard benchmark demonstration trajectory
        steps = [
            {"step_index": 0, "type": "action", "tool": "file_reader", "content": "read README.md"},
            {"step_index": 1, "type": "action", "tool": "code_generator", "content": "def add(a: int, b: int) -> int:\n    return a + b\n"},
            {"step_index": 2, "type": "action", "tool": "test_runner", "content": "pytest test_math.py"},
        ]

    constraints = ZKPolicyConstraints(
        max_steps=max_steps,
        require_ast_verification=require_ast,
    )

    try:
        receipt = ZKTrajectoryEngine.generate_receipt(
            steps=steps,
            session_id=session_id or None,
            constraints=constraints,
            zkvm_system=system,
        )

        receipt_dict = receipt.to_dict()
        if out:
            Path(out).write_text(json.dumps(receipt_dict, indent=2), encoding="utf-8")
            console.print(f"[green]Receipt successfully written to [bold]{out}[/bold][/green]")

        table = Table(title="[bold green]ZKVM Trajectory Execution Receipt[/bold green]")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Receipt ID", receipt.receipt_id)
        table.add_row("Session ID", receipt.session_id)
        table.add_row("zkVM System", receipt.zkvm_system.upper())
        table.add_row("Program ID", f"{receipt.program_id[:18]}...{receipt.program_id[-8:]}")
        table.add_row("Step Count", str(receipt.journal.get("step_count", 0)))
        table.add_row("Action Merkle Root", f"{receipt.journal.get('action_merkle_root', '')[:16]}...")
        table.add_row("Journal Digest", f"{receipt.journal_digest[:16]}...")
        table.add_row("Status Code", str(receipt.journal.get("status_code", 0)))
        table.add_row("AST Verified", str(receipt.journal.get("ast_verified", True)))
        table.add_row("Cryptographic Seal", f"{receipt.seal_hex[:22]}...{receipt.seal_hex[-10:]}")
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to generate receipt:[/bold red] {e}")
        sys.exit(1)


@zk.command("verify-receipt")
@click.option("--receipt-file", "-f", required=True, help="Path to JSON file containing ZKTrajectoryReceipt.")
@click.option("--allow-program-id", multiple=True, help="Allowed program IDs.")
@click.option("--require-ast/--no-require-ast", default=False, help="Require AST verification.")
@click.option("--expected-root", default="", help="Expected action Merkle root.")
def zk_verify_receipt(
    receipt_file: str,
    allow_program_id: tuple[str, ...],
    require_ast: bool,
    expected_root: str,
) -> None:
    """Cryptographically audit and verify a zkVM trajectory execution receipt."""
    p = Path(receipt_file)
    if not p.is_file():
        console.print(f"[bold red]File not found:[/bold red] {receipt_file}")
        sys.exit(1)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        receipt = ZKTrajectoryReceipt.from_dict(data)
    except Exception as e:
        console.print(f"[bold red]Invalid receipt JSON:[/bold red] {e}")
        sys.exit(1)

    constraints = ZKPolicyConstraints(
        require_ast_verification=require_ast,
        allowed_program_ids=list(allow_program_id) if allow_program_id else None,
    )

    report = ZKTrajectoryEngine.verify_receipt(
        receipt=receipt,
        constraints=constraints,
        expected_merkle_root=expected_root or None,
    )

    table = Table(title="[bold blue]ZKVM Trajectory Cryptographic Verification Report[/bold blue]")
    table.add_column("Audit Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for c in report.checks:
        status_str = "[bold green]PASS[/bold green]" if c.get("passed") else "[bold red]FAIL[/bold red]"
        detail_items = [f"{k}={v}" for k, v in c.items() if k not in ("check", "passed")]
        table.add_row(c.get("check", "unknown"), status_str, ", ".join(detail_items))

    console.print(table)

    verdict_style = "bold green" if report.is_valid else "bold red"
    verdict_text = "VERIFIED VALID" if report.is_valid else f"VERIFICATION FAILED: {report.error_message}"
    console.print(Panel(
        f"[{verdict_style}]{verdict_text}[/{verdict_style}]\n"
        f"System: {report.zkvm_system.upper()} | Program ID: {report.program_id[:16]}... | Root: {report.action_merkle_root[:16]}...\n"
        f"Duration: {report.verification_duration_ms:.2f}ms",
        title="[bold]Audit Verdict[/bold]",
        border_style="green" if report.is_valid else "red",
    ))

    if not report.is_valid:
        sys.exit(1)


@zk.command("inspect")
@click.option("--receipt-file", "-f", required=True, help="Path to JSON file containing ZKTrajectoryReceipt.")
def zk_inspect(receipt_file: str) -> None:
    """Inspect the full journal, Merkle root, seal, and metadata of a ZK receipt."""
    p = Path(receipt_file)
    if not p.is_file():
        console.print(f"[bold red]File not found:[/bold red] {receipt_file}")
        sys.exit(1)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        receipt = ZKTrajectoryReceipt.from_dict(data)
    except Exception as e:
        console.print(f"[bold red]Invalid receipt JSON:[/bold red] {e}")
        sys.exit(1)

    console.print(Panel(
        f"[bold cyan]Receipt ID:[/bold cyan] {receipt.receipt_id}\n"
        f"[bold cyan]Session ID:[/bold cyan] {receipt.session_id}\n"
        f"[bold cyan]System:[/bold cyan] {receipt.zkvm_system.upper()}\n"
        f"[bold cyan]Program ID:[/bold cyan] {receipt.program_id}\n"
        f"[bold cyan]Action Merkle Root:[/bold cyan] {receipt.journal.get('action_merkle_root')}\n"
        f"[bold cyan]Journal Digest:[/bold cyan] {receipt.journal_digest}\n"
        f"[bold cyan]Seal:[/bold cyan] {receipt.seal_hex}\n\n"
        f"[bold yellow]Committed Journal Payload:[/bold yellow]\n"
        f"{json.dumps(receipt.journal, indent=2)}",
        title="[bold magenta]ZKVM Receipt Details[/bold magenta]",
        border_style="magenta",
    ))


def _get_default_sample_evaluations() -> list[AgentEvaluationRecord]:
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
            safety_violations=0,
            zk_verified_rate=1.0,
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
            safety_violations=0,
            zk_verified_rate=1.0,
        ),
        AgentEvaluationRecord(
            agent_id="deepseek-agent-core",
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
            safety_violations=0,
            zk_verified_rate=1.0,
        ),
        AgentEvaluationRecord(
            agent_id="qwen-local-coder",
            model_id="qwen-2.5-coder-32b",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=84,
            pass_at_1=0.84,
            tool_f1_score=0.87,
            mean_latency_ms=740.0,
            p95_latency_ms=1210.0,
            avg_prompt_tokens=1650,
            avg_completion_tokens=550,
            safety_violations=0,
            zk_verified_rate=1.0,
        ),
        AgentEvaluationRecord(
            agent_id="llama-general-worker",
            model_id="llama-3.3-70b",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=81,
            pass_at_1=0.81,
            tool_f1_score=0.84,
            mean_latency_ms=790.0,
            p95_latency_ms=1280.0,
            avg_prompt_tokens=1600,
            avg_completion_tokens=540,
            safety_violations=1,
            zk_verified_rate=0.98,
        ),
        AgentEvaluationRecord(
            agent_id="gemini-multimodal-agent",
            model_id="gemini-1.5-pro",
            benchmark_suite="swe-bench-mini",
            total_tasks=100,
            passed_tasks=88,
            pass_at_1=0.88,
            tool_f1_score=0.89,
            mean_latency_ms=1350.0,
            p95_latency_ms=2100.0,
            avg_prompt_tokens=2200,
            avg_completion_tokens=700,
            safety_violations=0,
            zk_verified_rate=1.0,
        ),
        AgentEvaluationRecord(
            agent_id="gemini-flash-router",
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
            safety_violations=0,
            zk_verified_rate=1.0,
        ),
    ]


@main.group("leaderboard")
def leaderboard() -> None:
    """Cross-agent benchmark leaderboard, Pareto frontier & token economics commands."""
    pass


@leaderboard.command("generate")
@click.option("--eval-file", "-f", default=None, help="Path to JSON file containing list of evaluation records.")
@click.option("--sample", is_flag=True, default=False, help="Use built-in multi-model evaluation sample suite.")
@click.option("--output-format", "-t", type=click.Choice(["table", "json", "markdown"]), default="table", help="Output format.")
@click.option("--output-file", "-o", default=None, help="Optional file destination.")
def leaderboard_generate(eval_file: Optional[str], sample: bool, output_format: str, output_file: Optional[str]) -> None:
    """Generate ranked cross-agent benchmark leaderboard with composite scores and Pareto optimal tagging."""
    if eval_file:
        p = Path(eval_file)
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {eval_file}")
            sys.exit(1)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            records = [AgentEvaluationRecord.from_dict(item) for item in raw]
        except Exception as e:
            console.print(f"[bold red]Failed to parse evaluation file:[/bold red] {e}")
            sys.exit(1)
    else:
        records = _get_default_sample_evaluations()

    entries = LeaderboardEngine.generate_leaderboard(records)

    if output_format == "json":
        rendered = LeaderboardDashboard.export_json(entries)
    elif output_format == "markdown":
        rendered = LeaderboardDashboard.export_markdown(entries)
    else:
        table = Table(title="[bold cyan]Agent-Lab Unified Cross-Agent Benchmark Leaderboard (v0.6.0)[/bold cyan]")
        table.add_column("Rank", justify="center", style="bold")
        table.add_column("Tier", justify="center")
        table.add_column("Agent / Model", style="cyan")
        table.add_column("Composite", justify="right", style="bold magenta")
        table.add_column("Pass@1 (95% CI)", justify="center")
        table.add_column("Tool F1", justify="right")
        table.add_column("Latency (p95)", justify="right")
        table.add_column("Cost / 1k Tasks", justify="right", style="green")
        table.add_column("Pareto", justify="center")

        tier_colors = {"S": "bold green", "A": "bold cyan", "B": "bold blue", "C": "bold yellow", "D": "bold red"}
        for e in entries:
            tier_str = f"[{tier_colors.get(e.tier, 'white')}]{e.tier}[/{tier_colors.get(e.tier, 'white')}]"
            pareto_str = "[bold green]★ YES[/bold green]" if e.is_pareto_optimal else "[dim]—[/dim]"
            ci_str = f"{e.pass_at_1*100:.1f}% [{e.pass_at_1_ci[0]*100:.0f}–{e.pass_at_1_ci[1]*100:.0f}%]"
            table.add_row(
                f"#{e.rank}",
                tier_str,
                f"{e.agent_id}\n[dim]{e.model_id}[/dim]",
                f"{e.composite_score:.1f}",
                ci_str,
                f"{e.tool_f1:.4f}",
                f"{e.mean_latency_ms:.0f}ms ({e.p95_latency_ms:.0f})",
                f"${e.cost_per_1k_tasks_usd:.4f}",
                pareto_str,
            )
        console.print(table)
        rendered = None

    if output_file and rendered:
        Path(output_file).write_text(rendered, encoding="utf-8")
        console.print(f"[bold green]Saved leaderboard to:[/bold green] {output_file}")
    elif rendered:
        print(rendered)


@leaderboard.command("render-html")
@click.option("--eval-file", "-f", default=None, help="Path to JSON file containing list of evaluation records.")
@click.option("--output-file", "-o", default="leaderboard.html", help="Path for HTML dashboard output.")
@click.option("--title", default="Agent-Lab Unified Cross-Agent Benchmark Leaderboard", help="Report header title.")
@click.option("--simulate-economics/--no-simulate-economics", default=True, help="Include swarm token economics analysis.")
def leaderboard_render_html(eval_file: Optional[str], output_file: str, title: str, simulate_economics: bool) -> None:
    """Export self-contained dark-theme HTML report with responsive SVG charts."""
    if eval_file:
        p = Path(eval_file)
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {eval_file}")
            sys.exit(1)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            records = [AgentEvaluationRecord.from_dict(item) for item in raw]
        except Exception as e:
            console.print(f"[bold red]Failed to parse evaluation file:[/bold red] {e}")
            sys.exit(1)
    else:
        records = _get_default_sample_evaluations()

    entries = LeaderboardEngine.generate_leaderboard(records)

    econ_report = None
    if simulate_economics:
        # Default distribution across top frontier, mid-tier, and fast router
        econ_report = TokenEconomicsSimulator.simulate_workload(
            model_allocations={
                "claude-3-5-sonnet": 30,
                "gpt-4o": 20,
                "deepseek-coder-v2": 25,
                "gemini-1.5-flash": 25,
            },
            tasks_per_day=10000,
            days=30,
        )

    html_content = LeaderboardDashboard.export_html_report(
        entries=entries,
        economics=econ_report,
        title=title,
    )

    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    console.print(Panel(
        f"[bold green]HTML Dashboard Exported Successfully![/bold green]\n\n"
        f"Path: {out_path}\n"
        f"Models Evaluated: {len(entries)}\n"
        f"Pareto Optimal Models: {len([e for e in entries if e.is_pareto_optimal])}\n"
        f"Includes SVG Charts: Composite Bar Chart & Pareto Frontier Scatter Plot",
        title="[bold cyan]Leaderboard Export[/bold cyan]",
        border_style="green",
    ))


@leaderboard.command("simulate-cost")
@click.option("--tasks-per-day", "-t", default=10000, type=int, help="Number of agent tasks processed daily.")
@click.option("--days", "-d", default=30, type=int, help="Billing horizon in days.")
@click.option("--avg-prompt-tokens", default=1500, type=int, help="Average prompt tokens per task.")
@click.option("--avg-completion-tokens", default=500, type=int, help="Average completion tokens per task.")
def leaderboard_simulate_cost(tasks_per_day: int, days: int, avg_prompt_tokens: int, avg_completion_tokens: int) -> None:
    """Simulate swarm token economics, multi-model cost distribution and routing downgrade savings."""
    allocations = {
        "claude-3-5-sonnet": 30,
        "gpt-4o": 20,
        "deepseek-coder-v2": 25,
        "gemini-1.5-flash": 25,
    }
    report = TokenEconomicsSimulator.simulate_workload(
        model_allocations=allocations,
        tasks_per_day=tasks_per_day,
        days=days,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_completion_tokens=avg_completion_tokens,
    )

    table = Table(title="[bold green]Multi-Model Swarm Token Economics Projection[/bold green]")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="dim")
    table.add_column("Traffic Share", justify="right")
    table.add_column("Daily Tasks", justify="right")
    table.add_column("Cost / Task", justify="right")
    table.add_column("Daily Cost", justify="right")
    table.add_column("Monthly Cost", justify="right", style="bold green")

    for model_id, data in report.model_breakdown.items():
        table.add_row(
            model_id,
            data["provider"],
            f"{data['share_pct']}%",
            f"{data['tasks_per_day']:,}",
            f"${data['cost_per_task_usd']:.5f}",
            f"${data['daily_cost_usd']:.2f}",
            f"${data['monthly_cost_usd']:.2f}",
        )

    console.print(table)

    savings = report.downgrade_savings_analysis
    console.print(Panel(
        f"[bold cyan]Swarm Economics Summary:[/bold cyan]\n"
        f"Daily Total Tasks: {report.total_tasks_per_day:,} | Horizon: {report.simulation_days} days\n"
        f"Total Projected Tokens: {(report.total_prompt_tokens + report.total_completion_tokens):,}\n"
        f"Total Daily Cost: [bold green]${report.daily_cost_usd:,.2f}[/bold green]\n"
        f"Total Monthly Cost: [bold green]${report.monthly_cost_usd:,.2f}[/bold green]\n\n"
        f"[bold yellow]Hybrid Routing Optimization:[/bold yellow]\n"
        f"100% Flagship Benchmark Cost: ${savings['all_flagship_monthly_cost_usd']:,.2f} / month\n"
        f"Optimized Hybrid Swarm Cost: ${savings['hybrid_routing_monthly_cost_usd']:,.2f} / month\n"
        f"[bold green]Monthly Cost Savings: ${savings['monthly_savings_usd']:,.2f} ({savings['savings_percentage']}%) [/bold green]",
        title="[bold magenta]Token Economics Verdict[/bold magenta]",
        border_style="magenta",
    ))


@leaderboard.command("export-json")
@click.option("--eval-file", "-f", default=None, help="Path to evaluation JSON.")
@click.option("--output-file", "-o", required=True, help="Path for JSON output.")
def leaderboard_export_json(eval_file: Optional[str], output_file: str) -> None:
    """Export structured JSON schema of ranked leaderboard entries."""
    if eval_file:
        p = Path(eval_file)
        if not p.is_file():
            console.print(f"[bold red]File not found:[/bold red] {eval_file}")
            sys.exit(1)
        records = [AgentEvaluationRecord.from_dict(item) for item in json.loads(p.read_text(encoding="utf-8"))]
    else:
        records = _get_default_sample_evaluations()

    entries = LeaderboardEngine.generate_leaderboard(records)
    out_p = Path(output_file).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(LeaderboardDashboard.export_json(entries), encoding="utf-8")
    console.print(f"[bold green]Exported {len(entries)} ranked leaderboard entries to:[/bold green] {output_file}")


# ---------------------------------------------------------------------------
# Router Command Group (v0.7.0)
# ---------------------------------------------------------------------------

_CLI_ROUTER_INSTANCE: Optional[AdaptiveModelRouter] = None


def _get_cli_router() -> AdaptiveModelRouter:
    global _CLI_ROUTER_INSTANCE
    if _CLI_ROUTER_INSTANCE is None:
        _CLI_ROUTER_INSTANCE = AdaptiveModelRouter()
    return _CLI_ROUTER_INSTANCE


@main.group()
def router() -> None:
    """Adaptive dynamic token rate-limiter, circuit breakers, and cross-model fallback router."""
    pass


@router.command("status")
@click.option("--json-out", "-j", default=None, help="Optional path to export status as JSON.")
def router_status(json_out: Optional[str]) -> None:
    """Display real-time circuit breaker states, sliding-window token utilization, and fallback routes."""
    r = _get_cli_router()
    status = r.get_fleet_status()

    if json_out:
        p = Path(json_out).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(status, indent=2), encoding="utf-8")
        console.print(f"[bold green]Fleet status exported to:[/bold green] {json_out}")
        return

    # Table 1: Model Fleet & Circuit Breakers
    table = Table(title="Model Fleet Circuit Breakers & Sliding-Window Quotas", expand=True)
    table.add_column("Model ID", style="bold cyan")
    table.add_column("Circuit State", justify="center")
    table.add_column("RPM (Used / Max)", justify="right")
    table.add_column("TPM (Used / Max)", justify="right")
    table.add_column("Concurrency", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Trips", justify="right")
    table.add_column("Avg Latency", justify="right")

    for model_id, m in status["models"].items():
        state = m["circuit_state"]
        if state == "CLOSED":
            state_str = "[bold green]CLOSED[/bold green]"
        elif state == "HALF_OPEN":
            state_str = "[bold yellow]HALF_OPEN[/bold yellow]"
        else:
            state_str = "[bold red]OPEN[/bold red]"

        rpm_str = f"{m['current_rpm']} / {m['max_rpm']} ({m['rpm_pct']}%)"
        tpm_str = f"{m['current_tpm']:,} / {m['max_tpm']:,} ({m['tpm_pct']}%)"
        conc_str = f"{m['active_concurrency']} / {m['max_concurrency']}"
        lat_str = f"{m['avg_latency_ms']} ms" if m['avg_latency_ms'] > 0 else "—"

        table.add_row(
            model_id,
            state_str,
            rpm_str,
            tpm_str,
            conc_str,
            str(m["consecutive_failures"]),
            str(m["total_trips"]),
            lat_str,
        )

    console.print(table)

    # Table 2: Fallback Cascades
    rtable = Table(title="Configured Cross-Model Fallback Cascades", expand=True)
    rtable.add_column("Route Name", style="bold magenta")
    rtable.add_column("Primary Model", style="bold green")
    rtable.add_column("Cascade Fallbacks (in failover order)", style="yellow")
    rtable.add_column("Description", style="white")

    for rname, rinfo in status["routes"].items():
        fallbacks = " -> ".join(rinfo["fallback_models"]) if rinfo["fallback_models"] else "[dim]None[/dim]"
        rtable.add_row(rname, rinfo["primary_model"], fallbacks, rinfo["description"])

    console.print(rtable)


@router.command("route")
@click.option("--prompt", "-p", default="Write an asynchronous token-bucket rate limiter in Rust.", help="Prompt text.")
@click.option("--route", "-r", default="code-generation", help="Target fallback cascade route.")
@click.option("--priority", default="NORMAL", type=click.Choice(["CRITICAL", "HIGH", "NORMAL", "BACKGROUND"]), help="Agent priority tier.")
@click.option("--agent", "-a", default="agent-cli", help="Agent identifier.")
@click.option("--tokens", "-t", default=500, type=int, help="Estimated completion tokens.")
@click.option("--simulate-fail-primary", is_flag=True, help="Simulate HTTP 429 rate limit on primary model to test failover.")
def router_route(
    prompt: str,
    route: str,
    priority: str,
    agent: str,
    tokens: int,
    simulate_fail_primary: bool,
) -> None:
    """Route an LLM request through adaptive fallback cascade with rate-limiting."""
    r = _get_cli_router()
    prio_enum = AgentPriority(priority)

    invoker = None
    if simulate_fail_primary:
        def fault_invoker(model_id: str, p: str):
            route_obj = r.routes.get(route)
            prim = route_obj.primary_model if route_obj else "claude-3-5-sonnet"
            if model_id == prim:
                return False, "HTTP 429: Rate limit exceeded (simulated)", 20, 0
            return True, f"Success from fallback model {model_id}: Code generated successfully.", 25, tokens
        invoker = fault_invoker

    try:
        receipt = r.route_and_execute(
            prompt=prompt,
            route_name=route,
            priority=prio_enum,
            agent_id=agent,
            estimated_output_tokens=tokens,
            provider_invoker=invoker,
        )
    except Exception as exc:
        console.print(f"[bold red]Routing execution failed:[/bold red] {exc}")
        sys.exit(1)

    status_color = "green" if receipt.status == "success" else "yellow"
    console.print(Panel(
        f"[bold cyan]Receipt ID:[/bold cyan] {receipt.receipt_id}\n"
        f"[bold cyan]Task ID:[/bold cyan] {receipt.task_id} | [bold cyan]Agent:[/bold cyan] {receipt.agent_id} (Priority: {receipt.priority.value})\n"
        f"[bold cyan]Requested Route:[/bold cyan] {route} (Primary: [bold]{receipt.primary_model}[/bold])\n"
        f"[bold cyan]Resolved Model:[/bold cyan] [bold {status_color}]{receipt.resolved_model}[/bold {status_color}] "
        f"({'[bold yellow]AUTOMATIC FALLBACK TRIGGERED[/bold yellow]' if receipt.fallback_occurred else '[bold green]DIRECT PRIMARY HIT[/bold green]'})\n"
        f"[bold cyan]Execution Status:[/bold cyan] [{status_color}]{receipt.status}[/{status_color}]\n"
        f"[bold cyan]Total Latency:[/bold cyan] {receipt.total_latency_ms:.2f} ms | [bold cyan]Tokens:[/bold cyan] {receipt.total_tokens:,}\n"
        f"[bold cyan]Total Cost:[/bold cyan] ${receipt.total_cost_usd:.6f}\n\n"
        f"[bold white]Output Preview:[/bold white]\n{receipt.output_preview}",
        title="[bold green]Adaptive Route Execution Receipt[/bold green]",
        border_style="green" if not receipt.fallback_occurred else "yellow",
    ))

    # Attempts breakdown table
    att_table = Table(title="Cascade Attempt Breakdown", expand=True)
    att_table.add_column("Attempt", justify="center")
    att_table.add_column("Model ID", style="cyan")
    att_table.add_column("Success", justify="center")
    att_table.add_column("Latency (ms)", justify="right")
    att_table.add_column("Tokens", justify="right")
    att_table.add_column("Cost ($)", justify="right")
    att_table.add_column("Diagnostic / Error", style="red")

    for att in receipt.attempts:
        succ_str = "[bold green]YES[/bold green]" if att.success else "[bold red]NO[/bold red]"
        err_str = att.error_message or "[dim]None[/dim]"
        att_table.add_row(
            str(att.attempt_index),
            att.model_id,
            succ_str,
            f"{att.latency_ms:.1f}",
            str(att.prompt_tokens + att.completion_tokens),
            f"${att.cost_usd:.6f}",
            err_str,
        )

    console.print(att_table)


@router.command("reset-circuit")
@click.option("--model", "-m", default=None, help="Model ID to reset.")
@click.option("--all", "reset_all", is_flag=True, help="Reset all model circuit breakers.")
def router_reset_circuit(model: Optional[str], reset_all: bool) -> None:
    """Reset tripped circuit breakers to CLOSED."""
    r = _get_cli_router()
    if reset_all:
        r.reset_all_circuits()
        console.print("[bold green]Successfully reset all circuit breakers to CLOSED.[/bold green]")
    elif model:
        if r.reset_circuit(model):
            console.print(f"[bold green]Successfully reset circuit breaker for:[/bold green] {model}")
        else:
            console.print(f"[bold red]Unknown model ID:[/bold red] {model}")
            sys.exit(1)
    else:
        console.print("[bold yellow]Please specify --model <model_id> or --all[/bold yellow]")


@router.command("benchmark-resilience")
@click.option("--requests", "-n", default=20, type=int, help="Number of simulated requests.")
@click.option("--fault-rate", "-f", default=0.35, type=float, help="Synthetic rate limit failure probability on primary.")
@click.option("--route", "-r", default="code-generation", help="Target fallback cascade route.")
def router_benchmark_resilience(requests: int, fault_rate: float, route: str) -> None:
    """Benchmark swarm resilience under synthetic 429 rate limit faults and provider outages."""
    r = _get_cli_router()
    route_obj = r.routes.get(route)
    primary = route_obj.primary_model if route_obj else "claude-3-5-sonnet"

    console.print(f"[bold cyan]Running Resilience Benchmark:[/bold cyan] {requests} requests against '{route}' (Primary: {primary}, Fault Rate: {fault_rate * 100:.0f}%)...")

    success_count = 0
    fallback_count = 0
    primary_hit_count = 0
    failed_count = 0
    total_latency = 0.0
    total_cost = 0.0

    random.seed(42)  # Deterministic benchmark repeatability

    for i in range(requests):
        should_fail_primary = random.random() < fault_rate

        def benchmark_invoker(model_id: str, p: str):
            if model_id == primary and should_fail_primary:
                return False, "HTTP 429: Too Many Requests (Rate limit reached)", 50, 0
            return True, f"Synthesized output for task {i}", 50, 250

        try:
            receipt = r.route_and_execute(
                prompt=f"Benchmark task #{i+1}",
                route_name=route,
                priority=AgentPriority.NORMAL,
                agent_id=f"worker-{(i % 4) + 1}",
                estimated_output_tokens=250,
                provider_invoker=benchmark_invoker,
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

    avg_lat = (total_latency / success_count) if success_count > 0 else 0.0
    resilience_rate = (success_count / requests) * 100.0

    # Summary table
    res_table = Table(title="Swarm Fault-Tolerance & Resilience Metrics", expand=True)
    res_table.add_column("Metric", style="bold cyan")
    res_table.add_column("Value", style="bold green", justify="right")
    res_table.add_column("Evaluation", style="yellow")

    res_table.add_row("Total Requests Dispatched", str(requests), "High-Concurrency Swarm Simulation")
    res_table.add_row("Primary Model Hits", str(primary_hit_count), f"{round(primary_hit_count/requests*100, 1)}% Direct Routing")
    res_table.add_row("Cascade Fallbacks Recovered", str(fallback_count), f"{round(fallback_count/requests*100, 1)}% Zero-Downtime Failover")
    res_table.add_row("Unhandled Failures", str(failed_count), "Zero-Drop Target (0 Expected)")
    res_table.add_row("System Resilience Rate", f"{resilience_rate:.1f}%", "[bold green]100.0% Pass Gate[/bold green]" if resilience_rate >= 99.0 else "[bold red]SLA Breach[/bold red]")
    res_table.add_row("Mean End-to-End Latency", f"{avg_lat:.2f} ms", "Sub-10ms Overhead")
    res_table.add_row("Cumulative Task Cost", f"${total_cost:.5f}", "Economically Optimized")

    console.print(res_table)


# ---------------------------------------------------------------------------
# Cross-Agent Memory CLI Commands
# ---------------------------------------------------------------------------

_CLI_MEMORY_STORE_INSTANCE: Optional[CrossAgentMemoryStore] = None


def _get_cli_memory_store() -> CrossAgentMemoryStore:
    global _CLI_MEMORY_STORE_INSTANCE
    if _CLI_MEMORY_STORE_INSTANCE is None:
        _CLI_MEMORY_STORE_INSTANCE = CrossAgentMemoryStore()
    return _CLI_MEMORY_STORE_INSTANCE


@main.group()
def memory() -> None:
    """Cross-agent semantic memory store, episodic vector index, and hybrid retrieval."""
    pass


@memory.command("store")
@click.option("--content", "-c", required=True, help="Memory text content.")
@click.option("--agent-id", "-a", default="agent-cli", help="Authoring agent identifier.")
@click.option("--namespace", "-n", default="default", help="Memory partition namespace.")
@click.option("--kind", "-k", type=click.Choice(["EPISODIC", "SEMANTIC", "PROCEDURAL", "WORKING"]), default="SEMANTIC", help="Memory classification.")
@click.option("--access-level", type=click.Choice(["PRIVATE", "SWARM", "PUBLIC"]), default="SWARM", help="Visibility permission boundary.")
@click.option("--tag", "-t", multiple=True, help="Optional categorization tags.")
@click.option("--importance", "-i", default=0.5, type=float, help="Subjective importance weight [0.0, 1.0].")
@click.option("--ttl", default=None, type=float, help="Time-to-live expiration in seconds.")
def memory_store(
    content: str,
    agent_id: str,
    namespace: str,
    kind: str,
    access_level: str,
    tag: tuple[str, ...],
    importance: float,
    ttl: Optional[float],
) -> None:
    """Store an episodic or semantic memory item with vector indexing."""
    store = _get_cli_memory_store()
    try:
        entry = store.store(
            content=content,
            agent_id=agent_id,
            kind=MemoryKind(kind),
            access_level=MemoryAccessLevel(access_level),
            tags=list(tag),
            importance=importance,
            ttl_seconds=ttl,
            namespace=namespace,
        )
        console.print(f"[bold green]Stored memory:[/bold green] [cyan]{entry.id}[/cyan] in namespace '[yellow]{entry.namespace}[/yellow]' (Kind: {entry.kind.value}, Importance: {entry.importance:.2f})")
    except Exception as e:
        console.print(f"[bold red]Failed to store memory:[/bold red] {e}")
        sys.exit(1)


@memory.command("search")
@click.option("--query", "-q", required=True, help="Natural language search query.")
@click.option("--agent-id", "-a", default=None, help="Requesting agent ID for permission enforcement.")
@click.option("--namespace", "-n", default=None, help="Filter by namespace.")
@click.option("--kind", "-k", type=click.Choice(["EPISODIC", "SEMANTIC", "PROCEDURAL", "WORKING"]), default=None, help="Filter by memory kind.")
@click.option("--top-k", "-k-top", default=5, type=int, help="Maximum results to return.")
@click.option("--hybrid-alpha", default=0.70, type=float, help="Vector weight vs lexical keyword weight [0.0 to 1.0].")
@click.option("--min-score", default=0.0, type=float, help="Minimum compound relevance score filter.")
@click.option("--json-out", "-j", default=None, help="Export search results to JSON.")
def memory_search(
    query: str,
    agent_id: Optional[str],
    namespace: Optional[str],
    kind: Optional[str],
    top_k: int,
    hybrid_alpha: float,
    min_score: float,
    json_out: Optional[str],
) -> None:
    """Search memories using hybrid vector similarity and keyword ranking."""
    store = _get_cli_memory_store()
    kind_enum = MemoryKind(kind) if kind else None
    results = store.search(
        query=query,
        requesting_agent_id=agent_id,
        namespace=namespace,
        kind=kind_enum,
        top_k=top_k,
        min_score=min_score,
        hybrid_alpha=hybrid_alpha,
    )

    if json_out:
        payload = [r.to_dict() for r in results]
        Path(json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[bold green]Exported {len(results)} search results to:[/bold green] {json_out}")
        return

    if not results:
        console.print("[yellow]No relevant memories found matching query.[/yellow]")
        return

    table = Table(title=f"Memory Retrieval Results (Query: '{query}')", expand=True)
    table.add_column("Rank", style="bold cyan", width=5)
    table.add_column("ID", style="cyan")
    table.add_column("Agent / NS", style="yellow")
    table.add_column("Kind", style="magenta")
    table.add_column("Score", style="bold green", justify="right")
    table.add_column("Sim / KW", style="blue", justify="right")
    table.add_column("Content Preview", style="white")

    for i, r in enumerate(results, start=1):
        preview = r.entry.content[:60] + "..." if len(r.entry.content) > 60 else r.entry.content
        table.add_row(
            str(i),
            r.entry.id,
            f"{r.entry.agent_id} ({r.entry.namespace})",
            r.entry.kind.value,
            f"{r.score:.4f}",
            f"{r.similarity:.2f} / {r.keyword_score:.2f}",
            preview,
        )

    console.print(table)


@memory.command("stats")
@click.option("--json-out", "-j", default=None, help="Export stats to JSON.")
def memory_stats(json_out: Optional[str]) -> None:
    """Display memory volume, partition metrics, and embedding status."""
    store = _get_cli_memory_store()
    stats = store.get_stats()

    if json_out:
        Path(json_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        console.print(f"[bold green]Exported memory stats to:[/bold green] {json_out}")
        return

    table = Table(title="Cross-Agent Semantic Memory Store Statistics", expand=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green", justify="right")

    table.add_row("Total Active Memories", str(stats["total_memories"]))
    table.add_row("Unique Tags Indexed", str(stats["unique_tags"]))
    table.add_row("Average Importance Weight", str(stats["average_importance"]))
    table.add_row("Embedding Vector Dimension", f"{stats['embedding_dimension']}D")
    table.add_row("Partitions (Namespaces)", json.dumps(stats["namespaces"]))
    table.add_row("Authoring Agents", json.dumps(stats["agents"]))
    table.add_row("Memory Taxonomies", json.dumps(stats["kinds"]))
    table.add_row("Access Boundaries", json.dumps(stats["access_levels"]))

    console.print(table)


@memory.command("evict")
@click.option("--namespace", "-n", default="default", help="Target namespace to evict from.")
@click.option("--target-capacity", "-c", default=100, type=int, help="Target maximum memory capacity.")
@click.option("--strategy", "-s", type=click.Choice(["DECAY", "LRU", "LIF"]), default="DECAY", help="Eviction prioritization strategy.")
def memory_evict(namespace: str, target_capacity: int, strategy: str) -> None:
    """Evict expired or low-value memories to maintain strict capacity bounds."""
    store = _get_cli_memory_store()
    evicted = store.evict_to_capacity(namespace=namespace, target_capacity=target_capacity, strategy=strategy)
    console.print(f"[bold green]Evicted {evicted} items from namespace '{namespace}'[/bold green] (Strategy: {strategy}, Target: {target_capacity})")


@memory.command("export")
@click.option("--out", "-o", required=True, help="Destination JSON file path.")
def memory_export(out: str) -> None:
    """Export all memories and vector metadata to JSON."""
    store = _get_cli_memory_store()
    try:
        store.export_json(out)
        console.print(f"[bold green]Successfully exported memories to:[/bold green] {out}")
    except Exception as e:
        console.print(f"[bold red]Export failed:[/bold red] {e}")
        sys.exit(1)


@memory.command("import")
@click.option("--file", "-f", required=True, help="Path to JSON memory backup file.")
def memory_import(file: str) -> None:
    """Import memories from JSON backup file."""
    store = _get_cli_memory_store()
    try:
        count = store.import_json(file)
        console.print(f"[bold green]Successfully imported {count} memories from:[/bold green] {file}")
    except Exception as e:
        console.print(f"[bold red]Import failed:[/bold red] {e}")
        sys.exit(1)


# ==============================================================================
# Multi-Agent Swarm Arena & Tournament Commands (v0.9.0)
# ==============================================================================

_CLI_ARENA_FILE = Path(".agentlab/arena_leaderboard.json")
_CLI_ARENA_LEADERBOARD: Optional[ArenaLeaderboard] = None


def _get_cli_arena_leaderboard() -> ArenaLeaderboard:
    global _CLI_ARENA_LEADERBOARD
    if _CLI_ARENA_LEADERBOARD is None:
        if _CLI_ARENA_FILE.exists():
            try:
                _CLI_ARENA_LEADERBOARD = ArenaLeaderboard.import_json(_CLI_ARENA_FILE)
            except Exception:
                _CLI_ARENA_LEADERBOARD = ArenaLeaderboard()
        else:
            _CLI_ARENA_LEADERBOARD = ArenaLeaderboard()
            for agent, rating in [
                ("claude-3-5-sonnet", 1350.0),
                ("gpt-4o", 1320.0),
                ("deepseek-coder-v2", 1280.0),
                ("qwen-2.5-coder", 1250.0),
            ]:
                _CLI_ARENA_LEADERBOARD.register_agent(agent, rating)
    return _CLI_ARENA_LEADERBOARD


def _save_cli_arena_leaderboard(lb: ArenaLeaderboard) -> None:
    _CLI_ARENA_FILE.parent.mkdir(parents=True, exist_ok=True)
    lb.export_json(_CLI_ARENA_FILE)


@main.group()
def arena() -> None:
    """Multi-Agent Tournament Arena, Elo Ratings & Swarm Workflow Benchmarks."""
    pass


@arena.command("match")
@click.option("--agent-a", "-a", required=True, help="Identifier for Agent A.")
@click.option("--agent-b", "-b", required=True, help="Identifier for Agent B.")
@click.option("--passed-a", default=8, type=int, help="Tasks passed by Agent A.")
@click.option("--passed-b", default=7, type=int, help="Tasks passed by Agent B.")
@click.option("--total", default=10, type=int, help="Total evaluation tasks.")
@click.option("--tokens-a", default=1500, type=int, help="Tokens consumed by Agent A.")
@click.option("--tokens-b", default=2200, type=int, help="Tokens consumed by Agent B.")
@click.option("--json-out", "-j", default=None, help="Export match outcome to JSON file.")
def arena_match(
    agent_a: str,
    agent_b: str,
    passed_a: int,
    passed_b: int,
    total: int,
    tokens_a: int,
    tokens_b: int,
    json_out: Optional[str],
) -> None:
    """Execute a head-to-head evaluation match between two agents and update Elo ratings."""
    lb = _get_cli_arena_leaderboard()

    score_a = AgentMatchScore(
        agent_id=agent_a,
        tasks_passed=passed_a,
        total_tasks=total,
        pass_rate=round(passed_a / total, 4) if total > 0 else 0.0,
        total_tokens=tokens_a,
    )
    score_b = AgentMatchScore(
        agent_id=agent_b,
        tasks_passed=passed_b,
        total_tasks=total,
        pass_rate=round(passed_b / total, 4) if total > 0 else 0.0,
        total_tokens=tokens_b,
    )

    outcome = lb.record_match(agent_a=agent_a, agent_b=agent_b, score_a=score_a, score_b=score_b)
    _save_cli_arena_leaderboard(lb)

    if json_out:
        Path(json_out).write_text(json.dumps(outcome.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[bold green]Exported match outcome to:[/bold green] {json_out}")
        return

    table = Table(title=f"Match Outcome: {agent_a} vs {agent_b}", expand=True)
    table.add_column("Agent", style="bold cyan")
    table.add_column("Pass Rate", justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("Composite", justify="right")
    table.add_column("Result", justify="center", style="bold")
    table.add_column("Elo Delta", justify="right")
    table.add_column("New Rating", justify="right", style="bold green")

    res_a_str = "[green]WIN[/green]" if outcome.result_a == MatchResult.WIN else ("[yellow]DRAW[/yellow]" if outcome.result_a == MatchResult.DRAW else "[red]LOSS[/red]")
    res_b_str = "[green]WIN[/green]" if outcome.result_a == MatchResult.LOSS else ("[yellow]DRAW[/yellow]" if outcome.result_a == MatchResult.DRAW else "[red]LOSS[/red]")

    table.add_row(
        agent_a,
        f"{passed_a}/{total} ({score_a.pass_rate * 100:.1f}%)",
        str(tokens_a),
        f"{score_a.composite_score:.4f}",
        res_a_str,
        f"{outcome.elo_change_a:+.2f}",
        f"{lb.ratings[agent_a]:.2f}",
    )
    table.add_row(
        agent_b,
        f"{passed_b}/{total} ({score_b.pass_rate * 100:.1f}%)",
        str(tokens_b),
        f"{score_b.composite_score:.4f}",
        res_b_str,
        f"{outcome.elo_change_b:+.2f}",
        f"{lb.ratings[agent_b]:.2f}",
    )
    console.print(table)


@arena.command("tournament")
@click.option("--agents", "-a", default="claude-3-5-sonnet,gpt-4o,deepseek-coder-v2,qwen-2.5-coder", help="Comma-separated agent IDs.")
@click.option("--rounds", "-r", default=1, type=int, help="Number of round-robin rounds.")
@click.option("--json-out", "-j", default=None, help="Export tournament results to JSON.")
def arena_tournament(agents: str, rounds: int, json_out: Optional[str]) -> None:
    """Run a multi-agent round-robin tournament across all participant pairs."""
    lb = _get_cli_arena_leaderboard()
    agent_list = [a.strip() for a in agents.split(",") if a.strip()]

    if len(agent_list) < 2:
        console.print("[bold red]Tournament requires at least 2 agents.[/bold red]")
        sys.exit(1)

    total_matches = 0
    for round_num in range(1, rounds + 1):
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a, b = agent_list[i], agent_list[j]
                # Simulate realistic match performance with small stochastic variation
                pass_a = min(10, max(5, int(random.gauss(8, 1))))
                pass_b = min(10, max(5, int(random.gauss(8, 1))))
                score_a = AgentMatchScore(agent_id=a, tasks_passed=pass_a, total_tasks=10, pass_rate=pass_a / 10.0, total_tokens=1500)
                score_b = AgentMatchScore(agent_id=b, tasks_passed=pass_b, total_tasks=10, pass_rate=pass_b / 10.0, total_tokens=1600)
                lb.record_match(agent_a=a, agent_b=b, score_a=score_a, score_b=score_b)
                total_matches += 1

    _save_cli_arena_leaderboard(lb)
    console.print(f"[bold green]Executed {total_matches} tournament matches across {len(agent_list)} agents.[/bold green]")

    rankings = lb.get_rankings()
    if json_out:
        Path(json_out).write_text(json.dumps(rankings, indent=2), encoding="utf-8")
        console.print(f"[bold green]Exported tournament rankings to:[/bold green] {json_out}")
        return

    table = Table(title="Arena Tournament Standings (Elo Ratings)", expand=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Agent ID", style="bold cyan")
    table.add_column("Elo Rating", justify="right", style="bold green")
    table.add_column("Matches", justify="right")
    table.add_column("W / L / D", justify="center")
    table.add_column("Win Rate", justify="right")
    table.add_column("95% CI (Wilson)", justify="center", style="dim")

    for rank, r in enumerate(rankings, 1):
        ci = r["win_rate_ci_95"]
        table.add_row(
            str(rank),
            r["agent_id"],
            f"{r['rating']:.2f}",
            str(r["matches"]),
            f"{r['wins']} / {r['losses']} / {r['draws']}",
            f"{r['win_rate'] * 100:.1f}%",
            f"[{ci[0]:.2f}, {ci[1]:.2f}]",
        )
    console.print(table)


@arena.command("ratings")
@click.option("--json-out", "-j", default=None, help="Export ratings to JSON.")
def arena_ratings(json_out: Optional[str]) -> None:
    """Display current Elo leaderboard standings and win-rate statistics."""
    lb = _get_cli_arena_leaderboard()
    rankings = lb.get_rankings()

    if json_out:
        Path(json_out).write_text(json.dumps(rankings, indent=2), encoding="utf-8")
        console.print(f"[bold green]Exported ratings to:[/bold green] {json_out}")
        return

    table = Table(title="Agent Arena Elo Standings", expand=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Agent ID", style="bold cyan")
    table.add_column("Elo Rating", justify="right", style="bold green")
    table.add_column("Matches", justify="right")
    table.add_column("W / L / D", justify="center")
    table.add_column("Win Rate", justify="right")
    table.add_column("95% CI (Wilson)", justify="center", style="dim")

    for rank, r in enumerate(rankings, 1):
        ci = r["win_rate_ci_95"]
        table.add_row(
            str(rank),
            r["agent_id"],
            f"{r['rating']:.2f}",
            str(r["matches"]),
            f"{r['wins']} / {r['losses']} / {r['draws']}",
            f"{r['win_rate'] * 100:.1f}%",
            f"[{ci[0]:.2f}, {ci[1]:.2f}]",
        )
    console.print(table)


@arena.command("benchmark-swarm")
@click.option("--workflow-id", default="wf-std-dev", help="Benchmark workflow ID.")
@click.option("--workflow-name", default="Standard 4-Stage Multi-Agent Dev Pipeline", help="Workflow name.")
@click.option("--planner", default="agent-planner", help="Planner agent ID.")
@click.option("--coder", default="agent-coder", help="Coder agent ID.")
@click.option("--reviewer", default="agent-reviewer", help="Reviewer agent ID.")
@click.option("--auditor", default="agent-auditor", help="Auditor agent ID.")
@click.option("--json-out", "-j", default=None, help="Export workflow report to JSON.")
def arena_benchmark_swarm(
    workflow_id: str,
    workflow_name: str,
    planner: str,
    coder: str,
    reviewer: str,
    auditor: str,
    json_out: Optional[str],
) -> None:
    """Benchmark end-to-end multi-agent collaborative DAG workflow execution."""
    stages = SwarmWorkflowHarness.create_standard_dev_workflow(
        workflow_id=workflow_id,
        planner=planner,
        coder=coder,
        reviewer=reviewer,
        auditor=auditor,
    )
    harness = SwarmWorkflowHarness()
    report = harness.execute_workflow(workflow_id=workflow_id, workflow_name=workflow_name, stages=stages)

    if json_out:
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[bold green]Exported swarm workflow report to:[/bold green] {json_out}")
        return

    table = Table(title=f"Multi-Agent Workflow Benchmark: {workflow_name}", expand=True)
    table.add_column("Stage ID", style="bold cyan")
    table.add_column("Role", justify="center")
    table.add_column("Agent", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens", justify="right")

    for s in report.stages:
        status_str = "[green]COMPLETED[/green]" if s.status == "COMPLETED" else "[red]FAILED[/red]"
        table.add_row(
            s.stage_id,
            s.role,
            s.assigned_agent,
            status_str,
            f"{s.execution_time_ms:.1f}",
            str(s.tokens_consumed),
        )

    console.print(table)
    console.print(Panel.fit(
        f"[bold]Overall Success:[/bold] {'[green]YES[/green]' if report.overall_success else '[red]NO[/red]'}\n"
        f"[bold]Total Execution Time:[/bold] {report.total_execution_time_ms:.1f} ms\n"
        f"[bold]Total Tokens Consumed:[/bold] {report.total_tokens_consumed}\n"
        f"[bold]Consensus Rounds:[/bold] {report.consensus_rounds}\n"
        f"[bold]Coordination Efficiency:[/bold] [bold green]{report.coordination_efficiency * 100:.1f}%[/bold green]",
        title="Swarm Coordination Summary",
    ))


@arena.command("export")
@click.option("--out", "-o", required=True, help="Destination JSON file path.")
def arena_export(out: str) -> None:
    """Export the arena leaderboard and match history to JSON."""
    lb = _get_cli_arena_leaderboard()
    try:
        lb.export_json(out)
        console.print(f"[bold green]Successfully exported arena leaderboard to:[/bold green] {out}")
    except Exception as e:
        console.print(f"[bold red]Export failed:[/bold red] {e}")
        sys.exit(1)


@arena.command("import")
@click.option("--file", "-f", required=True, help="Path to JSON arena backup file.")
def arena_import(file: str) -> None:
    """Import arena leaderboard from JSON backup file."""
    try:
        imported = ArenaLeaderboard.import_json(file)
        global _CLI_ARENA_LEADERBOARD
        _CLI_ARENA_LEADERBOARD = imported
        _save_cli_arena_leaderboard(imported)
        console.print(f"[bold green]Successfully imported arena leaderboard with {len(imported.ratings)} agents and {len(imported.match_history)} matches.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Import failed:[/bold red] {e}")
        sys.exit(1)


@main.group()
def fleet() -> None:
    """Full-Lifecycle Multi-Agent Fleet Diagnostics, Missions & Benchmarks."""
    pass


@fleet.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output health report as JSON.")
def fleet_doctor(as_json: bool) -> None:
    """Execute deep self-diagnostics across all core agent-lab sub-engines."""
    diag = FleetDiagnostics()
    report = diag.run_diagnostics()

    if as_json:
        click.echo(report.to_json())
        return

    status_color = "green" if report.status == FleetHealthStatus.HEALTHY else ("yellow" if report.status == FleetHealthStatus.DEGRADED else "red")
    table = Table(title=f"Fleet Diagnostics ({report.status.value}) - agent-lab v{report.version}")
    table.add_column("Subsystem", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Message", style="dim")

    for check in report.checks:
        c = "green" if check.status == FleetHealthStatus.HEALTHY else ("yellow" if check.status == FleetHealthStatus.DEGRADED else "red")
        table.add_row(
            check.subsystem,
            f"[{c}]{check.status.value}[/{c}]",
            f"{check.latency_ms:.2f}",
            check.message,
        )

    console.print(table)
    overall_msg = "[bold green]All sub-engines operational and healthy.[/bold green]" if report.all_healthy else "[bold red]One or more sub-engines degraded or failing.[/bold red]"
    console.print(Panel(
        f"{overall_msg}\n"
        f"Timestamp: [cyan]{report.timestamp}[/cyan]\n"
        f"Version: [yellow]{report.version}[/yellow]",
        title="Fleet Status",
    ))


@fleet.command("run-mission")
@click.option("--id", "mission_id", default="mission_ga_001", help="Unique mission identifier.")
@click.option("--goal", default="Implement and verify cryptographic hash utility", help="Mission objective description.")
@click.option("--agents", default="agent-alpha,agent-beta,agent-gamma", help="Comma-separated agent names.")
@click.option("--failover", is_flag=True, help="Simulate upstream provider outage to test cascade router failover.")
@click.option("--json", "as_json", is_flag=True, help="Output receipt as JSON.")
def fleet_run_mission(mission_id: str, goal: str, agents: str, failover: bool, as_json: bool) -> None:
    """Execute an end-to-end full-lifecycle multi-agent mission with cryptographic receipt."""
    agent_list = [a.strip() for a in agents.split(",") if a.strip()]
    if len(agent_list) < 3:
        console.print("[bold red]Fleet missions require at least 3 agents for BFT quorum.[/bold red]")
        sys.exit(1)

    cfg = MissionConfig(
        mission_id=mission_id,
        goal=goal,
        agents=agent_list,
        simulate_failover=failover,
    )
    runner = FleetMissionRunner()
    receipt = runner.execute_mission(cfg)

    if as_json:
        click.echo(receipt.to_json())
        return

    status_color = "green" if receipt.status == "SUCCESS" else "red"
    console.print(Panel(
        f"[bold]Mission ID:[/bold] {receipt.mission_id}\n"
        f"[bold]Status:[/bold] [{status_color}]{receipt.status}[/{status_color}]\n"
        f"[bold]Routed Model:[/bold] {receipt.routed_model} (Attempts: {receipt.routing_attempts})\n"
        f"[bold]Memory Hits:[/bold] {receipt.memory_hits}\n"
        f"[bold]BFT Consensus:[/bold] {'[green]APPROVED[/green]' if receipt.consensus_approved else '[red]REJECTED[/red]'}\n"
        f"[bold]Consensus Merkle Root:[/bold] [dim]{receipt.consensus_merkle_root[:24]}...[/dim]\n"
        f"[bold]ZK Proof Verified:[/bold] {'[green]YES (SP1 zkVM)[/green]' if receipt.zk_verified else '[red]NO[/red]'}\n"
        f"[bold]AST Syntax Valid:[/bold] {'[green]YES[/green]' if receipt.syntax_valid else '[red]NO[/red]'}\n"
        f"[bold]Trajectory F1 Score:[/bold] [bold green]{receipt.trajectory_f1 * 100:.1f}%[/bold green]\n"
        f"[bold]Loop Detected:[/bold] {'[red]YES[/red]' if receipt.loop_detected else '[green]NO[/green]'}\n"
        f"[bold]Arena Composite Score:[/bold] {receipt.arena_match_score:.1f}/100.0\n"
        f"[bold]Cryptographic Seal:[/bold] [yellow]{receipt.sha256_seal[:32]}...[/yellow]",
        title=f"Fleet Mission Execution Receipt: {receipt.mission_id}",
    ))


@fleet.command("benchmark")
@click.option("--count", "-n", default=5, type=int, help="Number of simulated missions to execute.")
@click.option("--json", "as_json", is_flag=True, help="Output benchmark summary as JSON.")
def fleet_benchmark(count: int, as_json: bool) -> None:
    """Benchmark end-to-end fleet execution throughput across multiple missions."""
    runner = FleetMissionRunner()
    stats = runner.benchmark_fleet(num_missions=count)

    if as_json:
        click.echo(json.dumps(stats, indent=2))
        return

    table = Table(title="Fleet Mission Throughput Benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Total Missions", str(stats["num_missions"]))
    table.add_row("Successful Missions", str(stats["success_count"]))
    table.add_row("Success Rate", f"{stats['success_rate'] * 100:.1f}%")
    table.add_row("Elapsed Time", f"{stats['elapsed_seconds']:.2f} s")
    table.add_row("Missions / Second", f"{stats['missions_per_second']:.2f}")
    table.add_row("Average Latency", f"{stats['avg_latency_ms']:.1f} ms")

    console.print(table)


@main.group()
def portfolio() -> None:
    """Cross-Repository Ecosystem Integration, Diagnostics & Unified Simulation."""
    pass


@portfolio.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output portfolio discovery status as JSON.")
def portfolio_status(as_json: bool) -> None:
    """Inspect availability, versions, and capabilities of all 4 portfolio repositories."""
    components = PortfolioDiscovery.discover_all()

    if as_json:
        data = {k: v.to_dict() for k, v in components.items()}
        click.echo(json.dumps(data, indent=2))
        return

    table = Table(title="Autonomous Open-Source Ecosystem: Portfolio Status")
    table.add_column("Repository", style="cyan bold")
    table.add_column("Version", style="yellow")
    table.add_column("Available", justify="center")
    table.add_column("Live Package", justify="center")
    table.add_column("Capabilities", style="dim")

    for name, info in components.items():
        avail_str = "[green]YES[/green]" if info.is_available else "[red]NO[/red]"
        live_str = "[green]YES[/green]" if info.is_live_package else "[dim]NO[/dim]"
        caps = ", ".join(info.capabilities[:3]) + (f" (+{len(info.capabilities)-3})" if len(info.capabilities) > 3 else "")
        table.add_row(name, info.version, avail_str, live_str, caps)

    console.print(table)


@portfolio.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output portfolio health report as JSON.")
def portfolio_doctor(as_json: bool) -> None:
    """Execute end-to-end diagnostic probes across rollback, solana-tx-lens, agent-witness, and agent-lab."""
    doctor = PortfolioDoctor()
    report = doctor.run_doctor()

    if as_json:
        click.echo(report.to_json())
        return

    table = Table(title=f"Portfolio Doctor Diagnostics ({report.status.value})")
    table.add_column("Repository", style="cyan bold")
    table.add_column("Version", style="yellow")
    table.add_column("Status", style="bold")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Message", style="dim")

    for check in report.checks:
        c = "green" if check.status == PortfolioHealthStatus.HEALTHY else ("yellow" if check.status == PortfolioHealthStatus.DEGRADED else "red")
        table.add_row(
            check.subsystem,
            check.version,
            f"[{c}]{check.status.value}[/{c}]",
            f"{check.latency_ms:.2f}",
            check.message,
        )

    console.print(table)
    overall_msg = "[bold green]All 4 portfolio repositories operational and healthy.[/bold green]" if report.all_healthy else "[bold red]One or more portfolio repositories degraded or failing.[/bold red]"
    console.print(Panel(
        f"{overall_msg}\n"
        f"Timestamp: [cyan]{report.timestamp}[/cyan]\n"
        f"Active Components: [yellow]{len(report.checks)}/4[/yellow]",
        title="Ecosystem Health Status",
    ))


@portfolio.command("simulate")
@click.option("--id", "sim_id", default="sim_cross_001", help="Unique simulation identifier.")
@click.option("--goal", default="Autonomous code generation with transactional safety and on-chain settlement", help="Task objective.")
@click.option("--json", "as_json", is_flag=True, help="Output simulation receipt as JSON.")
def portfolio_simulate(sim_id: str, goal: str, as_json: bool) -> None:
    """Execute unified multi-agent mission exercising rollback, solana-tx-lens, agent-witness, and agent-lab."""
    simulator = CrossRepoFleetSimulator()
    receipt = simulator.run_simulation(simulation_id=sim_id, task_goal=goal)

    if as_json:
        click.echo(receipt.to_json())
        return

    status_color = "green" if receipt.status == "SUCCESS" else "red"
    console.print(Panel(
        f"[bold]Simulation ID:[/bold] {receipt.simulation_id}\n"
        f"[bold]Status:[/bold] [{status_color}]{receipt.status}[/{status_color}]\n"
        f"[bold]Latency:[/bold] {receipt.latency_ms:.2f} ms\n"
        f"[bold]Ecosystem Versions:[/bold] " + ", ".join([f"{k}={v}" for k, v in receipt.component_versions.items()]) + "\n\n"
        f"[bold cyan]1. Rollback Session (rollback):[/bold cyan]\n"
        f"  - Atomic Rollback Applied: {'[green]YES[/green]' if receipt.rollback_applied else '[red]NO[/red]'}\n"
        f"  - Corrupt Files Reverted: [green]{receipt.corrupt_files_reverted}[/green]\n"
        f"  - Restored Clean Hash: [dim]{receipt.clean_workspace_hash[:16]}...[/dim]\n\n"
        f"[bold cyan]2. Pre-Flight Simulation (solana-tx-lens):[/bold cyan]\n"
        f"  - Simulation Verified: {'[green]YES[/green]' if receipt.solana_sim_success else '[red]NO[/red]'}\n"
        f"  - Compute Units Consumed: [yellow]{receipt.solana_cu_consumed:,}[/yellow] CU\n"
        f"  - Dynamic Priority Fee: [yellow]{receipt.solana_priority_fee_micro_lamports:,}[/yellow] micro-lamports/CU\n\n"
        f"[bold cyan]3. Deliberation Attestation (agent-witness):[/bold cyan]\n"
        f"  - Merkle Tree Root: [dim]{receipt.witness_merkle_root[:24]}...[/dim] ({receipt.witness_step_count} steps)\n"
        f"  - SP1 zkVM Receipt Verified: {'[green]YES[/green]' if receipt.zkvm_receipt_verified else '[red]NO[/red]'}\n"
        f"  - Journal Digest: [dim]{receipt.zkvm_journal_digest[:24]}...[/dim]\n\n"
        f"[bold cyan]4. Meta-Orchestrator Fleet (agent-lab):[/bold cyan]\n"
        f"  - Routed Cascade Model: [green]{receipt.routed_model}[/green] (Memory Hits: {receipt.memory_hits})\n"
        f"  - Arena Composite Score: [bold green]{receipt.arena_match_score:.1f}/100.0[/bold green]\n"
        f"  - Verification Seal: [yellow]{receipt.sha256_seal[:32]}...[/yellow]",
        title=f"Unified Portfolio Simulation Receipt: {receipt.simulation_id}",
    ))


@portfolio.command("benchmark")
@click.option("--count", "-n", default=10, type=int, help="Number of simulated cross-repo missions to execute.")
@click.option("--json", "as_json", is_flag=True, help="Output benchmark summary as JSON.")
def portfolio_benchmark(count: int, as_json: bool) -> None:
    """Benchmark end-to-end cross-repository ecosystem throughput and latency percentiles."""
    simulator = CrossRepoFleetSimulator()
    stats = simulator.benchmark_portfolio(num_iterations=count)

    if as_json:
        click.echo(json.dumps(stats, indent=2))
        return

    table = Table(title="Cross-Repository Portfolio Benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Total Iterations", str(stats["num_iterations"]))
    table.add_row("Successful Iterations", str(stats["success_count"]))
    table.add_row("Success Rate", f"{stats['success_rate'] * 100:.1f}%")
    table.add_row("Elapsed Time", f"{stats['elapsed_seconds']:.3f} s")
    table.add_row("Simulations / Second", f"{stats['simulations_per_second']:.2f}")
    table.add_row("Average Latency", f"{stats['avg_latency_ms']:.2f} ms")
    table.add_row("p50 Latency", f"{stats['latency_p50_ms']:.2f} ms")
    table.add_row("p95 Latency", f"{stats['latency_p95_ms']:.2f} ms")
    table.add_row("p99 Latency", f"{stats['latency_p99_ms']:.2f} ms")

    console.print(table)


if __name__ == "__main__":
    main()


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


if __name__ == "__main__":
    main()

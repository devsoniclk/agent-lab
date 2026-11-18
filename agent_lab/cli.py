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


if __name__ == "__main__":
    main()

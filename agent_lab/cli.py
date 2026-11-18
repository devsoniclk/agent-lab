"""
agent_lab.cli: Command-Line Interface for Agent-Lab.
"""

from __future__ import annotations

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
from agent_lab.eval import PassAtKEstimator, ASTVerifier, RegressionTracker

console = Console()


@click.group()
@click.version_option(package_name="agent-lab")
def main() -> None:
    """agent-lab: Unified autonomous AI agent safety orchestrator & evaluation harness."""
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
    """Autonomous OSS evaluation and benchmarking tools."""
    pass


@eval.command("pass-at-k")
@click.option("-n", type=int, required=True, help="Total samples generated.")
@click.option("-c", type=int, required=True, help="Number of correct samples.")
@click.option("-k", type=int, required=True, help="Evaluations count (k).")
def eval_pass_at_k(n: int, c: int, k: int) -> None:
    """Compute unbiased Pass@k metric using Chen et al. formula."""
    try:
        score = PassAtKEstimator.estimate(n=n, c=c, k=k)
        console.print(f"[bold green]Pass@{k}:[/bold green] [cyan]{score:.4f}[/cyan] ({score*100:.2f}%)")
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


if __name__ == "__main__":
    main()

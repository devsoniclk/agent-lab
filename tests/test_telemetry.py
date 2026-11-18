"""
Unit tests for TelemetryCollector and performance dashboards.
"""

from __future__ import annotations

import json
from pathlib import Path
from rich.console import Console
from agent_lab.telemetry import TelemetryCollector


def test_telemetry_recording_flow(tmp_path: Path) -> None:
    db_file = tmp_path / "telemetry.json"
    collector = TelemetryCollector(storage_path=str(db_file))

    collector.record_session(agent_name="TestAgent", goal="Refactor API")
    collector.record_checkpoint(session_id="sess-1", label="pre-edit")
    collector.record_rollback(session_id="sess-1", checkpoint_id="cp-1", success=True)
    collector.record_consensus(approved=True, validators_count=4, slashed=0)
    collector.record_consensus(approved=False, validators_count=4, slashed=1)
    collector.record_solana_sim(success=True, units_consumed=25000)
    collector.record_eval(task_id="HumanEval/0", passed=True)
    collector.record_eval(task_id="HumanEval/1", passed=False)
    collector.record_trajectory_eval(f1_score=0.9, has_loop=False)
    collector.record_trajectory_eval(f1_score=0.5, has_loop=True)

    metrics = collector.get_metrics_dict()
    assert metrics["sessions_created"] == 1
    assert metrics["checkpoints_created"] == 1
    assert metrics["rollbacks_executed"] == 1
    assert metrics["consensus_rounds"] == 2
    assert metrics["consensus_approved"] == 1
    assert metrics["consensus_rejected"] == 1
    assert metrics["equivocation_slashed"] == 1
    assert metrics["solana_simulations"] == 1
    assert metrics["eval_tasks_run"] == 2
    assert metrics["eval_tasks_passed"] == 1
    assert metrics["eval_pass_rate"] == 0.5
    assert metrics["trajectories_evaluated"] == 2
    assert metrics["average_tool_f1"] == 0.7
    assert metrics["loops_detected"] == 1

    # Test Markdown formatting
    md = collector.to_markdown()
    assert "# Agent-Lab Telemetry" in md
    assert "- **Sessions Created:** 1" in md
    assert "Pass Rate: 50.0%" in md

    # Test save and reload
    saved_path = collector.save()
    assert Path(saved_path).exists()

    reloaded = TelemetryCollector.load(saved_path)
    r_metrics = reloaded.get_metrics_dict()
    assert r_metrics["sessions_created"] == 1
    assert r_metrics["consensus_rounds"] == 2
    assert r_metrics["equivocation_slashed"] == 1
    assert len(reloaded.events) == 10


def test_telemetry_dashboard_render() -> None:
    collector = TelemetryCollector()
    collector.record_session(agent_name="Bot", goal="Testing")
    collector.record_trajectory_eval(f1_score=1.0, has_loop=False)

    console = Console(record=True, width=100)
    collector.render_dashboard(console=console)
    output = console.export_text()

    assert "Agent Lifecycle & Safety Metrics" in output
    assert "Benchmark & Trajectory Evaluation" in output
    assert "Sessions Started" in output

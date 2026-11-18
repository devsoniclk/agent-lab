"""
tests/test_v9_coverage_edges.py: Edge cases and branch coverage for Arena, Elo, and Workflow Harness.
"""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

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
from agent_lab.cli import main
from agent_lab.mcp_server import (
    agentlab_arena_run_match,
    agentlab_arena_export_ratings,
)


def test_elo_extreme_rating_differences():
    elo = EloRatingSystem()
    # High difference positive
    exp_high = elo.expected_score(30000.0, 1000.0)
    assert exp_high == 1.0

    # High difference negative
    exp_low = elo.expected_score(1000.0, 30000.0)
    assert exp_low == 0.0


def test_arena_draw_outcome():
    lb = ArenaLeaderboard()
    score_a = AgentMatchScore(agent_id="agent-x", tasks_passed=5, total_tasks=10, pass_rate=0.5, total_tokens=1000)
    score_b = AgentMatchScore(agent_id="agent-y", tasks_passed=5, total_tasks=10, pass_rate=0.5, total_tokens=1000)

    outcome = lb.record_match(agent_a="agent-x", agent_b="agent-y", score_a=score_a, score_b=score_b)
    assert outcome.result_a == MatchResult.DRAW
    assert lb.draws_count["agent-x"] == 1
    assert lb.draws_count["agent-y"] == 1


def test_bradley_terry_edges():
    # Empty matrix
    assert BradleyTerryEstimator.estimate_skills({}) == {}

    # Single agent
    assert BradleyTerryEstimator.estimate_skills({"solo": {}}) == {"solo": 1.0}


def test_swarm_workflow_failing_stage():
    harness = SwarmWorkflowHarness()
    stages = [
        SwarmWorkflowStage(stage_id="s1", role="DEV", assigned_agent="a1"),
        SwarmWorkflowStage(stage_id="s2", role="DEV", assigned_agent="a2"),
    ]

    def failing_executor(stage: SwarmWorkflowStage) -> bool:
        if stage.stage_id == "s2":
            raise RuntimeError("Stage 2 synthetic error")
        return True

    report = harness.execute_workflow(
        workflow_id="wf-fail-01",
        workflow_name="Failing Workflow",
        stages=stages,
        stage_executor=failing_executor,
    )

    assert report.overall_success is False
    assert report.stages[0].status == "COMPLETED"
    assert report.stages[1].status == "FAILED"
    assert "synthetic error" in (report.stages[1].error or "")


def test_cli_tournament_insufficient_agents():
    runner = CliRunner()
    result = runner.invoke(main, ["arena", "tournament", "--agents", "solo-agent"])
    assert result.exit_code != 0
    assert "at least 2 agents" in result.output


def test_mcp_arena_in_memory_export():
    res_str = agentlab_arena_export_ratings(export_path="")
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert res["path"] == "in-memory"
    assert "data" in res

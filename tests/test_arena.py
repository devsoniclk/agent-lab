"""
tests/test_arena.py: Comprehensive test suite for Multi-Agent Tournament Arena & Workflow Harness.
"""

import json
import pytest
from pathlib import Path

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


def test_agent_match_score_composite():
    score = AgentMatchScore(
        agent_id="test-agent",
        tasks_passed=8,
        total_tasks=10,
        pass_rate=0.8,
        avg_latency_ms=120.0,
        total_tokens=2000,
        ast_score=1.0,
        trajectory_score=0.9,
    )
    comp = score.compute_composite()
    assert 0.0 <= comp <= 1.0
    assert score.composite_score == comp

    data = score.to_dict()
    assert data["agent_id"] == "test-agent"
    recovered = AgentMatchScore.from_dict(data)
    assert recovered.pass_rate == 0.8
    assert recovered.composite_score == comp


def test_elo_rating_system():
    elo = EloRatingSystem(default_rating=1200.0, k_provisional=32.0, k_established=16.0)

    # Equal ratings should give expected score of 0.5
    exp = elo.expected_score(1200.0, 1200.0)
    assert pytest.approx(exp, 0.001) == 0.5

    # Higher rated agent expected score > 0.5
    exp_high = elo.expected_score(1400.0, 1200.0)
    assert exp_high > 0.5

    # Update on win
    new_a, new_b, delta_a, delta_b = elo.update_ratings(
        rating_a=1200.0,
        rating_b=1200.0,
        actual_score_a=1.0,
        matches_a=0,
        matches_b=0,
    )
    assert delta_a > 0
    assert delta_b < 0
    assert new_a == 1200.0 + delta_a
    assert new_b == 1200.0 + delta_b

    # Established K-factor check
    k_prov = elo.get_k_factor(5)
    k_est = elo.get_k_factor(35)
    assert k_prov == 32.0
    assert k_est == 16.0


def test_bradley_terry_estimator():
    # Symmetric wins should yield equal skills
    matrix = {
        "agent-1": {"agent-2": 5.0, "agent-3": 5.0},
        "agent-2": {"agent-1": 5.0, "agent-3": 5.0},
        "agent-3": {"agent-1": 5.0, "agent-2": 5.0},
    }
    skills = BradleyTerryEstimator.estimate_skills(matrix)
    assert len(skills) == 3
    assert pytest.approx(skills["agent-1"], 0.05) == 1.0 / 3.0

    # Dominated agent should have lower skill
    matrix_asym = {
        "alpha": {"beta": 10.0},
        "beta": {"alpha": 1.0},
    }
    skills_asym = BradleyTerryEstimator.estimate_skills(matrix_asym)
    assert skills_asym["alpha"] > skills_asym["beta"]


def test_arena_leaderboard_lifecycle(tmp_path: Path):
    lb = ArenaLeaderboard()
    lb.register_agent("agent-a", 1300.0)
    lb.register_agent("agent-b", 1250.0)

    score_a = AgentMatchScore(agent_id="agent-a", tasks_passed=9, total_tasks=10, pass_rate=0.9)
    score_b = AgentMatchScore(agent_id="agent-b", tasks_passed=6, total_tasks=10, pass_rate=0.6)

    outcome = lb.record_match(agent_a="agent-a", agent_b="agent-b", score_a=score_a, score_b=score_b)

    assert outcome.result_a == MatchResult.WIN
    assert outcome.elo_change_a > 0
    assert outcome.elo_change_b < 0
    assert lb.wins_count["agent-a"] == 1
    assert lb.losses_count["agent-b"] == 1

    rankings = lb.get_rankings()
    assert len(rankings) == 2
    assert rankings[0]["agent_id"] == "agent-a"
    assert rankings[0]["rating"] > rankings[1]["rating"]

    # Export & Import JSON roundtrip
    export_file = tmp_path / "leaderboard.json"
    lb.export_json(export_file)
    assert export_file.exists()

    imported = ArenaLeaderboard.import_json(export_file)
    assert imported.ratings["agent-a"] == lb.ratings["agent-a"]
    assert len(imported.match_history) == 1


def test_swarm_workflow_harness():
    harness = SwarmWorkflowHarness()
    stages = SwarmWorkflowHarness.create_standard_dev_workflow(
        workflow_id="wf-test-01",
        planner="p1",
        coder="c1",
        reviewer="r1",
        auditor="a1",
    )
    assert len(stages) == 4

    report = harness.execute_workflow(
        workflow_id="wf-test-01",
        workflow_name="Test Dev Workflow",
        stages=stages,
    )

    assert report.overall_success is True
    assert len(report.stages) == 4
    assert all(s.status == "COMPLETED" for s in report.stages)
    assert report.total_execution_time_ms >= 0.0
    assert report.total_tokens_consumed > 0
    assert 0.0 <= report.coordination_efficiency <= 1.0

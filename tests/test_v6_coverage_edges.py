"""
Edge case and coverage tests for agent_lab v0.6.0:
- Empty leaderboard handling
- Single-item leaderboard Pareto optimality
- Equal-score tie breaking
- Zero task allocations in token economics
- Extreme latencies and violation penalties
- Telemetry inference recording and reporting
"""

from __future__ import annotations

from agent_lab.leaderboard import (
    AgentEvaluationRecord,
    LeaderboardDashboard,
    LeaderboardEngine,
    TokenEconomicsSimulator,
)
from agent_lab.telemetry import TelemetryCollector


def test_leaderboard_empty_list():
    entries = LeaderboardEngine.generate_leaderboard([])
    assert entries == []

    bar_svg = LeaderboardDashboard.generate_svg_bar_chart([])
    assert bar_svg == ""

    scatter_svg = LeaderboardDashboard.generate_svg_scatter_plot([])
    assert scatter_svg == ""


def test_leaderboard_single_item_always_pareto():
    rec = AgentEvaluationRecord(
        agent_id="solo-agent",
        model_id="claude-3-5-sonnet",
        benchmark_suite="eval",
        total_tasks=10,
        passed_tasks=8,
        pass_at_1=0.8,
        tool_f1_score=0.85,
        mean_latency_ms=1000.0,
        p95_latency_ms=1500.0,
    )
    entries = LeaderboardEngine.generate_leaderboard([rec])
    assert len(entries) == 1
    assert entries[0].rank == 1
    assert entries[0].is_pareto_optimal is True


def test_leaderboard_identical_scores_tie_breaker():
    rec1 = AgentEvaluationRecord(
        agent_id="agent-tie-1",
        model_id="gpt-4o",
        benchmark_suite="eval",
        total_tasks=10,
        passed_tasks=9,
        pass_at_1=0.9,
        tool_f1_score=0.9,
        mean_latency_ms=1000.0,
        p95_latency_ms=1500.0,
    )
    rec2 = AgentEvaluationRecord(
        agent_id="agent-tie-2",
        model_id="gpt-4o",
        benchmark_suite="eval",
        total_tasks=10,
        passed_tasks=9,
        pass_at_1=0.9,
        tool_f1_score=0.9,
        mean_latency_ms=1000.0,
        p95_latency_ms=1500.0,
    )
    entries = LeaderboardEngine.generate_leaderboard([rec1, rec2])
    assert len(entries) == 2
    assert entries[0].rank == 1
    assert entries[1].rank == 2


def test_token_economics_zero_tasks_or_weights():
    rep = TokenEconomicsSimulator.simulate_workload(
        model_allocations={"claude-3-5-sonnet": 0},
        tasks_per_day=0,
        days=0,
    )
    assert rep.total_tasks_per_day == 0
    assert rep.daily_cost_usd == 0.0
    assert rep.monthly_cost_usd == 0.0


def test_telemetry_record_inference_and_dashboard_display():
    collector = TelemetryCollector()
    collector.record_inference(
        model="claude-3-5-sonnet",
        prompt_tokens=2000,
        completion_tokens=500,
        latency_ms=1200.0,
        cost_usd=0.0135,
    )
    collector.record_inference(
        model="gemini-1.5-flash",
        prompt_tokens=1500,
        completion_tokens=400,
        latency_ms=400.0,
        cost_usd=0.00023,
    )

    metrics = collector.get_metrics_dict()
    assert metrics["total_inference_calls"] == 2
    assert metrics["total_prompt_tokens"] == 3500
    assert metrics["total_completion_tokens"] == 900
    assert metrics["avg_inference_latency_ms"] == 800.0
    assert metrics["total_inference_cost_usd"] > 0.013

    md = collector.to_markdown()
    assert "Model Inference Calls:" in md
    assert "Tokens:" in md


def test_extreme_penalties_and_custom_weights():
    rec = AgentEvaluationRecord(
        agent_id="penalized-agent",
        model_id="generic-model",
        benchmark_suite="eval",
        total_tasks=10,
        passed_tasks=5,
        pass_at_1=0.5,
        tool_f1_score=0.4,
        mean_latency_ms=10000.0,  # extreme latency
        p95_latency_ms=15000.0,
        safety_violations=20,     # extreme violations
        zk_verified_rate=0.0,     # failed zero-knowledge verification
    )
    custom_weights = {
        "accuracy": 0.2,
        "tool_f1": 0.2,
        "latency": 0.2,
        "cost": 0.2,
        "safety": 0.2,
    }
    score = LeaderboardEngine.calculate_composite_score(
        eval_record=rec,
        cost_per_1k=50.0,
        weights=custom_weights,
    )
    assert score < 30.0
    tier = LeaderboardEngine.assign_tier(score)
    assert tier == "D"

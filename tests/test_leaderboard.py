"""
Unit tests for agent_lab.leaderboard: Model pricing, evaluation records,
composite scoring, Pareto frontier, token economics simulator, and dashboard exports.
"""

from __future__ import annotations

import json

import pytest

from agent_lab.leaderboard import (
    AgentEvaluationRecord,
    LeaderboardDashboard,
    LeaderboardEngine,
    LeaderboardEntry,
    ModelPricing,
    SwarmEconomicsReport,
    TokenEconomicsSimulator,
)


def test_model_pricing_computation():
    pricing = ModelPricing(
        model_id="test-model",
        provider="test-org",
        input_cost_per_million=2.00,
        output_cost_per_million=8.00,
        context_window=64000,
        latency_sla_ms=900.0,
    )
    # 1,000,000 prompt tokens = $2.00, 500,000 completion tokens = $4.00 -> total $6.00
    cost = pricing.compute_task_cost(1_000_000, 500_000)
    assert cost == 6.0

    # 1,500 prompt tokens ($0.003) + 500 completion tokens ($0.004) = $0.007
    small_cost = pricing.compute_task_cost(1500, 500)
    assert round(small_cost, 5) == 0.007
    d = pricing.to_dict()
    assert d["model_id"] == "test-model"
    assert d["provider"] == "test-org"


def test_agent_evaluation_record_serialization():
    rec = AgentEvaluationRecord(
        agent_id="agent-01",
        model_id="claude-3-5-sonnet",
        benchmark_suite="swe-bench",
        total_tasks=50,
        passed_tasks=45,
        pass_at_1=0.90,
        pass_at_5=0.96,
        tool_f1_score=0.92,
        mean_latency_ms=1150.0,
        p95_latency_ms=1800.0,
        avg_prompt_tokens=1600,
        avg_completion_tokens=520,
        safety_violations=0,
        zk_verified_rate=1.0,
        metadata={"git_commit": "abcdef"},
    )
    d = rec.to_dict()
    assert d["agent_id"] == "agent-01"
    assert d["pass_at_1"] == 0.90
    assert d["metadata"]["git_commit"] == "abcdef"

    # Compute task cost via catalog
    task_cost = rec.compute_task_cost()
    assert task_cost > 0.0

    # Roundtrip from_dict
    rec2 = AgentEvaluationRecord.from_dict(d)
    assert rec2.agent_id == rec.agent_id
    assert rec2.model_id == rec.model_id
    assert rec2.pass_at_1 == rec.pass_at_1
    assert rec2.pass_at_5 == rec.pass_at_5


def test_composite_score_calculation():
    rec_high = AgentEvaluationRecord(
        agent_id="high-perf",
        model_id="claude-3-5-sonnet",
        benchmark_suite="test",
        total_tasks=10,
        passed_tasks=10,
        pass_at_1=1.0,
        tool_f1_score=1.0,
        mean_latency_ms=500.0,
        p95_latency_ms=800.0,
        avg_prompt_tokens=1000,
        avg_completion_tokens=300,
        safety_violations=0,
        zk_verified_rate=1.0,
    )
    score_high = LeaderboardEngine.calculate_composite_score(
        eval_record=rec_high,
        cost_per_1k=5.0,
        max_latency_ref_ms=3000.0,
        max_cost_ref_usd=25.0,
    )
    assert score_high >= 85.0

    rec_low = AgentEvaluationRecord(
        agent_id="low-perf",
        model_id="low-model",
        benchmark_suite="test",
        total_tasks=10,
        passed_tasks=2,
        pass_at_1=0.2,
        tool_f1_score=0.3,
        mean_latency_ms=2800.0,
        p95_latency_ms=3500.0,
        avg_prompt_tokens=1000,
        avg_completion_tokens=300,
        safety_violations=5,
        zk_verified_rate=0.5,
    )
    score_low = LeaderboardEngine.calculate_composite_score(
        eval_record=rec_low,
        cost_per_1k=20.0,
        max_latency_ref_ms=3000.0,
        max_cost_ref_usd=25.0,
    )
    assert score_low < score_high
    assert score_low < 40.0


def test_tier_assignment():
    assert LeaderboardEngine.assign_tier(95.0) == "S"
    assert LeaderboardEngine.assign_tier(90.0) == "S"
    assert LeaderboardEngine.assign_tier(85.0) == "A"
    assert LeaderboardEngine.assign_tier(80.0) == "A"
    assert LeaderboardEngine.assign_tier(75.0) == "B"
    assert LeaderboardEngine.assign_tier(70.0) == "B"
    assert LeaderboardEngine.assign_tier(65.0) == "C"
    assert LeaderboardEngine.assign_tier(60.0) == "C"
    assert LeaderboardEngine.assign_tier(55.0) == "D"
    assert LeaderboardEngine.assign_tier(0.0) == "D"


def test_pareto_frontier_logic():
    # Model A: High score (92), high cost ($12) -> Pareto optimal
    # Model B: Medium score (86), low cost ($2) -> Pareto optimal
    # Model C: Low score (80), medium cost ($5) -> Dominated by Model B (B has higher score and lower cost)
    # Model D: Very low score (70), very high cost ($15) -> Dominated by A and B
    entry_a = LeaderboardEntry(
        rank=1, agent_id="A", model_id="mod-a", tier="S", composite_score=92.0,
        pass_at_1=0.92, pass_at_1_ci=(0.85, 0.96), tool_f1=0.94, mean_latency_ms=1000.0,
        p95_latency_ms=1500.0, cost_per_1k_tasks_usd=12.0, safety_score=100.0, zk_rate=1.0,
    )
    entry_b = LeaderboardEntry(
        rank=2, agent_id="B", model_id="mod-b", tier="A", composite_score=86.0,
        pass_at_1=0.86, pass_at_1_ci=(0.78, 0.91), tool_f1=0.88, mean_latency_ms=800.0,
        p95_latency_ms=1200.0, cost_per_1k_tasks_usd=2.0, safety_score=100.0, zk_rate=1.0,
    )
    entry_c = LeaderboardEntry(
        rank=3, agent_id="C", model_id="mod-c", tier="A", composite_score=80.0,
        pass_at_1=0.80, pass_at_1_ci=(0.70, 0.87), tool_f1=0.82, mean_latency_ms=900.0,
        p95_latency_ms=1300.0, cost_per_1k_tasks_usd=5.0, safety_score=100.0, zk_rate=1.0,
    )
    entry_d = LeaderboardEntry(
        rank=4, agent_id="D", model_id="mod-d", tier="B", composite_score=70.0,
        pass_at_1=0.70, pass_at_1_ci=(0.60, 0.78), tool_f1=0.75, mean_latency_ms=1200.0,
        p95_latency_ms=1700.0, cost_per_1k_tasks_usd=15.0, safety_score=100.0, zk_rate=1.0,
    )

    entries = [entry_a, entry_b, entry_c, entry_d]
    LeaderboardEngine.compute_pareto_frontier(entries)

    assert entry_a.is_pareto_optimal is True
    assert entry_b.is_pareto_optimal is True
    assert entry_c.is_pareto_optimal is False  # B is cheaper ($2 < $5) and higher score (86 > 80)
    assert entry_d.is_pareto_optimal is False


def test_generate_leaderboard_ranking_and_ci():
    records = [
        AgentEvaluationRecord(
            agent_id="agent-01",
            model_id="claude-3-5-sonnet",
            benchmark_suite="unit-test",
            total_tasks=100,
            passed_tasks=90,
            pass_at_1=0.90,
            tool_f1_score=0.92,
            mean_latency_ms=1100.0,
            p95_latency_ms=1800.0,
        ),
        AgentEvaluationRecord(
            agent_id="agent-02",
            model_id="gemini-1.5-flash",
            benchmark_suite="unit-test",
            total_tasks=100,
            passed_tasks=75,
            pass_at_1=0.75,
            tool_f1_score=0.80,
            mean_latency_ms=450.0,
            p95_latency_ms=700.0,
        ),
    ]

    leaderboard = LeaderboardEngine.generate_leaderboard(records)
    assert len(leaderboard) == 2
    assert leaderboard[0].rank == 1
    assert leaderboard[1].rank == 2
    assert leaderboard[0].composite_score >= leaderboard[1].composite_score
    # Confidence intervals should be bounded between 0 and 1
    for item in leaderboard:
        assert 0.0 <= item.pass_at_1_ci[0] <= item.pass_at_1_ci[1] <= 1.0


def test_token_economics_simulator():
    allocations = {
        "claude-3-5-sonnet": 40,
        "deepseek-coder-v2": 40,
        "gemini-1.5-flash": 20,
    }
    report = TokenEconomicsSimulator.simulate_workload(
        model_allocations=allocations,
        tasks_per_day=5000,
        days=30,
        avg_prompt_tokens=1500,
        avg_completion_tokens=500,
    )

    assert isinstance(report, SwarmEconomicsReport)
    assert report.total_tasks_per_day == 5000
    assert report.simulation_days == 30
    assert report.daily_cost_usd > 0.0
    assert report.monthly_cost_usd == pytest.approx(report.daily_cost_usd * 30, rel=1e-2)
    assert len(report.model_breakdown) == 3
    assert "savings_percentage" in report.downgrade_savings_analysis
    assert report.downgrade_savings_analysis["monthly_savings_usd"] >= 0.0


def test_dashboard_svg_charts_generation():
    entry = LeaderboardEntry(
        rank=1,
        agent_id="AgentOne",
        model_id="claude-3-5-sonnet",
        tier="S",
        composite_score=94.5,
        pass_at_1=0.95,
        pass_at_1_ci=(0.88, 0.98),
        tool_f1=0.96,
        mean_latency_ms=1050.0,
        p95_latency_ms=1600.0,
        cost_per_1k_tasks_usd=12.5,
        safety_score=100.0,
        zk_rate=1.0,
        is_pareto_optimal=True,
    )
    bar_svg = LeaderboardDashboard.generate_svg_bar_chart([entry])
    assert "<svg" in bar_svg and "</svg>" in bar_svg
    assert "AgentOne" in bar_svg
    assert "94.5" in bar_svg

    scatter_svg = LeaderboardDashboard.generate_svg_scatter_plot([entry])
    assert "<svg" in scatter_svg and "</svg>" in scatter_svg
    assert "Pareto Frontier" in scatter_svg


def test_dashboard_html_and_markdown_export():
    entry = LeaderboardEntry(
        rank=1,
        agent_id="AgentOne",
        model_id="claude-3-5-sonnet",
        tier="S",
        composite_score=94.5,
        pass_at_1=0.95,
        pass_at_1_ci=(0.88, 0.98),
        tool_f1=0.96,
        mean_latency_ms=1050.0,
        p95_latency_ms=1600.0,
        cost_per_1k_tasks_usd=12.5,
        safety_score=100.0,
        zk_rate=1.0,
        is_pareto_optimal=True,
    )
    html_out = LeaderboardDashboard.export_html_report([entry], title="Custom Test Title")
    assert "<!DOCTYPE html>" in html_out
    assert "Custom Test Title" in html_out
    assert "badge-s" in html_out
    assert "AgentOne" in html_out
    assert "Pareto" in html_out

    md_out = LeaderboardDashboard.export_markdown([entry])
    assert "# Agent-Lab Unified Benchmark Leaderboard" in md_out
    assert "| #1 | **S** | AgentOne (`claude-3-5-sonnet`)" in md_out


def test_dashboard_json_export():
    entry = LeaderboardEntry(
        rank=1,
        agent_id="AgentOne",
        model_id="claude-3-5-sonnet",
        tier="S",
        composite_score=94.5,
        pass_at_1=0.95,
        pass_at_1_ci=(0.88, 0.98),
        tool_f1=0.96,
        mean_latency_ms=1050.0,
        p95_latency_ms=1600.0,
        cost_per_1k_tasks_usd=12.5,
        safety_score=100.0,
        zk_rate=1.0,
        is_pareto_optimal=True,
    )
    json_str = LeaderboardDashboard.export_json([entry])
    data = json.loads(json_str)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["agent_id"] == "AgentOne"
    assert data[0]["tier"] == "S"
    assert data[0]["is_pareto_optimal"] is True

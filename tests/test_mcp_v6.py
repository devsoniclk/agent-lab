"""
Integration tests for Agent-Lab FastMCP v0.6.0 tools:
- agentlab_leaderboard_generate
- agentlab_leaderboard_pareto_analysis
- agentlab_leaderboard_cost_simulator
- agentlab_leaderboard_export_html
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_lab.mcp_server import (
    agentlab_leaderboard_cost_simulator,
    agentlab_leaderboard_export_html,
    agentlab_leaderboard_generate,
    agentlab_leaderboard_pareto_analysis,
)


def test_mcp_leaderboard_generate_default():
    raw = agentlab_leaderboard_generate()
    data = json.loads(raw)
    assert data["status"] == "success"
    assert data["count"] >= 4
    board = data["leaderboard"]
    assert board[0]["rank"] == 1
    assert "composite_score" in board[0]
    assert "tier" in board[0]


def test_mcp_leaderboard_generate_custom_records():
    custom = [
        {
            "agent_id": "fast-solver",
            "model_id": "qwen-2.5-coder-32b",
            "total_tasks": 10,
            "passed_tasks": 9,
            "pass_at_1": 0.9,
            "tool_f1_score": 0.9,
        }
    ]
    raw = agentlab_leaderboard_generate(eval_records_json=json.dumps(custom))
    data = json.loads(raw)
    assert data["status"] == "success"
    assert data["count"] == 1
    assert data["leaderboard"][0]["agent_id"] == "fast-solver"


def test_mcp_leaderboard_pareto_analysis():
    raw = agentlab_leaderboard_pareto_analysis()
    data = json.loads(raw)
    assert data["status"] == "success"
    assert data["total_models"] >= 4
    assert data["pareto_optimal_count"] >= 1
    assert isinstance(data["pareto_optimal_entries"], list)
    assert isinstance(data["dominated_entries"], list)


def test_mcp_leaderboard_cost_simulator():
    allocations = json.dumps({
        "claude-3-5-sonnet": 30,
        "deepseek-coder-v2": 40,
        "gemini-1.5-flash": 30,
    })
    raw = agentlab_leaderboard_cost_simulator(
        model_allocations_json=allocations,
        tasks_per_day=5000,
        days=30,
    )
    data = json.loads(raw)
    assert data["status"] == "success"
    rep = data["report"]
    assert rep["total_tasks_per_day"] == 5000
    assert rep["daily_cost_usd"] > 0
    assert "savings_percentage" in rep["downgrade_savings_analysis"]


def test_mcp_leaderboard_export_html(tmp_path: Path):
    out_file = tmp_path / "mcp_board.html"
    raw = agentlab_leaderboard_export_html(
        output_path=str(out_file),
        title="MCP Swarm Report",
    )
    data = json.loads(raw)
    assert data["status"] == "success"
    assert out_file.is_file()
    assert data["models_ranked"] >= 4
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "MCP Swarm Report" in content


def test_mcp_leaderboard_error_handling():
    # Invalid JSON string passed
    raw = agentlab_leaderboard_generate(eval_records_json="invalid-json{{{")
    data = json.loads(raw)
    assert data["status"] == "error"
    assert "error" in data

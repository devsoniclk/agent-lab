"""
Integration tests for Agent-Lab CLI v0.6.0: Leaderboard, Pareto analysis, HTML render,
and token economics simulation commands.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from agent_lab.cli import main


def test_cli_leaderboard_help():
    runner = CliRunner()
    result = runner.invoke(main, ["leaderboard", "--help"])
    assert result.exit_code == 0
    assert "Cross-agent benchmark leaderboard" in result.output
    assert "generate" in result.output
    assert "render-html" in result.output
    assert "simulate-cost" in result.output
    assert "export-json" in result.output


def test_cli_leaderboard_generate_default():
    runner = CliRunner()
    result = runner.invoke(main, ["leaderboard", "generate"])
    assert result.exit_code == 0
    assert "Agent-Lab Unified Cross-Agent Benchmark Leaderboard" in result.output
    assert "gpt-4o" in result.output
    assert "Pareto" in result.output


def test_cli_leaderboard_generate_json_and_markdown():
    runner = CliRunner()
    # JSON output
    res_json = runner.invoke(main, ["leaderboard", "generate", "--output-format", "json"])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert isinstance(parsed, list)
    assert len(parsed) >= 5

    # Markdown output
    res_md = runner.invoke(main, ["leaderboard", "generate", "--output-format", "markdown"])
    assert res_md.exit_code == 0
    assert "# Agent-Lab Unified Benchmark Leaderboard" in res_md.output
    assert "| Rank | Tier |" in res_md.output


def test_cli_leaderboard_generate_with_custom_file(tmp_path: Path):
    custom_eval = [
        {
            "agent_id": "custom-agent-01",
            "model_id": "deepseek-coder-v2",
            "benchmark_suite": "swe-bench",
            "total_tasks": 20,
            "passed_tasks": 18,
            "pass_at_1": 0.90,
            "tool_f1_score": 0.92,
            "mean_latency_ms": 780.0,
            "p95_latency_ms": 1200.0,
        }
    ]
    eval_file = tmp_path / "custom_eval.json"
    eval_file.write_text(json.dumps(custom_eval), encoding="utf-8")

    out_file = tmp_path / "leaderboard.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "leaderboard", "generate",
        "-f", str(eval_file),
        "-t", "json",
        "-o", str(out_file),
    ])
    assert result.exit_code == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["agent_id"] == "custom-agent-01"


def test_cli_leaderboard_render_html(tmp_path: Path):
    out_html = tmp_path / "report.html"
    runner = CliRunner()
    result = runner.invoke(main, [
        "leaderboard", "render-html",
        "-o", str(out_html),
        "--title", "Test Swarm Evaluation",
    ])
    assert result.exit_code == 0
    assert out_html.is_file()
    content = out_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Test Swarm Evaluation" in content
    assert "<svg" in content


def test_cli_leaderboard_simulate_cost():
    runner = CliRunner()
    result = runner.invoke(main, [
        "leaderboard", "simulate-cost",
        "-t", "5000",
        "-d", "30",
    ])
    assert result.exit_code == 0
    assert "Multi-Model Swarm Token Economics Projection" in result.output
    assert "Hybrid Routing Optimization" in result.output
    assert "Monthly Cost Savings" in result.output


def test_cli_leaderboard_export_json_command(tmp_path: Path):
    out_json = tmp_path / "exported.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "leaderboard", "export-json",
        "-o", str(out_json),
    ])
    assert result.exit_code == 0
    assert out_json.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0


def test_cli_leaderboard_file_errors():
    runner = CliRunner()
    res = runner.invoke(main, ["leaderboard", "generate", "-f", "nonexistent_file.json"])
    assert res.exit_code != 0
    assert "File not found" in res.output

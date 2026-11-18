"""
tests/test_mcp_v9.py: Test suite for FastMCP Multi-Agent Arena Tools.
"""

import json
from pathlib import Path

from agent_lab.mcp_server import (
    agentlab_arena_run_match,
    agentlab_arena_get_ratings,
    agentlab_arena_benchmark_swarm,
    agentlab_arena_export_ratings,
)


def test_mcp_arena_run_match():
    res_str = agentlab_arena_run_match(
        agent_a="mcp-agent-a",
        agent_b="mcp-agent-b",
        tasks_passed_a=9,
        tasks_passed_b=6,
        total_tasks=10,
        tokens_a=1100,
        tokens_b=2100,
    )
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert "match" in res
    assert res["match"]["agent_a"] == "mcp-agent-a"
    assert res["match"]["result_a"] == "WIN"
    assert "new_ratings" in res


def test_mcp_arena_get_ratings():
    res_str = agentlab_arena_get_ratings()
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert res["total_agents"] >= 2
    assert "rankings" in res
    assert len(res["rankings"]) >= 2


def test_mcp_arena_benchmark_swarm():
    res_str = agentlab_arena_benchmark_swarm(
        workflow_id="wf-mcp-test",
        workflow_name="MCP Automated Pipeline",
    )
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert "report" in res
    assert res["report"]["workflow_id"] == "wf-mcp-test"
    assert res["report"]["overall_success"] is True
    assert len(res["report"]["stages"]) == 4


def test_mcp_arena_export_ratings(tmp_path: Path):
    out_file = tmp_path / "mcp_arena_export.json"
    res_str = agentlab_arena_export_ratings(export_path=str(out_file))
    res = json.loads(res_str)
    assert res["status"] == "success"
    assert out_file.exists()
    assert res["exported_agents"] >= 2

"""
Unit tests for distributed swarm MCP tools in Agent-Lab v0.3.0.
"""

from __future__ import annotations

import json
from agent_lab.mcp_server import (
    agentlab_swarm_status,
    agentlab_swarm_broadcast_proposal,
)


def test_mcp_swarm_status():
    raw = agentlab_swarm_status()
    res = json.loads(raw)
    assert res["status"] == "success"
    assert "committed_versions" in res
    assert "quorum_ratio" in res


def test_mcp_swarm_broadcast_proposal():
    raw = agentlab_swarm_broadcast_proposal(
        proposal_id="prop_mcp_01",
        proposer="agent-test",
        target_file="test_target.py",
        content="x = 10\n",
        description="Testing MCP proposal broadcast",
    )
    res = json.loads(raw)
    assert res["status"] == "success"
    assert res["committed"] is True
    assert "test_target.py" in res["applied_files"]

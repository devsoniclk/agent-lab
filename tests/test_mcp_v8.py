"""
Tests for agent-lab FastMCP memory tools (v0.8.0).
"""

import json
from agent_lab.mcp_server import (
    agentlab_memory_store,
    agentlab_memory_search,
    agentlab_memory_get_stats,
    agentlab_memory_clear_or_evict,
    _memory_store,
)


def setup_function():
    _memory_store.clear()


def test_mcp_memory_store_and_search_flow():
    # 1. Store memory
    raw_store = agentlab_memory_store(
        content="Cross-chain attestation dispatcher using LayerZero v2 OApp options",
        agent_id="agent-witness-dev",
        kind="PROCEDURAL",
        tags=["crosschain", "lz", "witness"],
        importance=0.9,
    )
    resp_store = json.loads(raw_store)
    assert resp_store["status"] == "success"
    mem_id = resp_store["memory"]["id"]
    assert mem_id.startswith("mem_")
    assert resp_store["memory"]["importance"] == 0.9

    # 2. Search memory
    raw_search = agentlab_memory_search(
        query="LayerZero crosschain attestation",
        top_k=3,
    )
    resp_search = json.loads(raw_search)
    assert resp_search["status"] == "success"
    assert resp_search["count"] == 1
    assert resp_search["results"][0]["entry"]["id"] == mem_id

    # 3. Get stats
    raw_stats = agentlab_memory_get_stats()
    resp_stats = json.loads(raw_stats)
    assert resp_stats["status"] == "success"
    assert resp_stats["stats"]["total_memories"] == 1

    # 4. Evict
    raw_evict = agentlab_memory_clear_or_evict(namespace="default", target_capacity=0)
    resp_evict = json.loads(raw_evict)
    assert resp_evict["status"] == "success"
    assert resp_evict["evicted_count"] == 1


def test_mcp_memory_error_paths():
    # Empty content error
    raw_err = agentlab_memory_store(content="", agent_id="agent-test")
    resp_err = json.loads(raw_err)
    assert resp_err["status"] == "error"
    assert "content cannot be empty" in resp_err["error"].lower()

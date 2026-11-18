"""
Tests for agent-lab CLI memory commands (v0.8.0).
"""

import json
import os
import tempfile
from click.testing import CliRunner
from agent_lab.cli import main, _CLI_MEMORY_STORE_INSTANCE
import agent_lab.cli as cli_mod


def setup_function():
    # Reset singleton before tests
    cli_mod._CLI_MEMORY_STORE_INSTANCE = None


def test_cli_memory_help():
    runner = CliRunner()
    result = runner.invoke(main, ["memory", "--help"])
    assert result.exit_code == 0
    assert "Cross-agent semantic memory store" in result.output
    assert "store" in result.output
    assert "search" in result.output
    assert "stats" in result.output
    assert "evict" in result.output


def test_cli_memory_store_and_search():
    runner = CliRunner()
    
    # Store
    res_store = runner.invoke(main, [
        "memory", "store",
        "--content", "Optimistic concurrency control with timeline DAG branches",
        "--agent-id", "test-agent",
        "--kind", "SEMANTIC",
        "--tag", "rollback",
        "--tag", "occ",
        "--importance", "0.85",
    ])
    assert res_store.exit_code == 0
    assert "Stored memory:" in res_store.output

    # Search
    res_search = runner.invoke(main, [
        "memory", "search",
        "--query", "concurrency control",
    ])
    assert res_search.exit_code == 0
    assert "Memory Retrieval Results" in res_search.output
    assert "test-agent" in res_search.output


def test_cli_memory_stats():
    runner = CliRunner()
    # Store one
    runner.invoke(main, ["memory", "store", "--content", "Stats test content", "--agent-id", "a1"])
    
    res_stats = runner.invoke(main, ["memory", "stats"])
    assert res_stats.exit_code == 0
    assert "Cross-Agent Semantic Memory Store Statistics" in res_stats.output
    assert "Total Active Memories" in res_stats.output


def test_cli_memory_evict_and_export_import():
    runner = CliRunner()
    runner.invoke(main, ["memory", "store", "--content", "Evict item 1", "--agent-id", "a1", "--importance", "0.1"])
    runner.invoke(main, ["memory", "store", "--content", "Evict item 2", "--agent-id", "a1", "--importance", "0.9"])

    # Evict down to 1
    res_evict = runner.invoke(main, ["memory", "evict", "--target-capacity", "1", "--strategy", "DECAY"])
    assert res_evict.exit_code == 0
    assert "Evicted" in res_evict.output

    # Export & Import
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "backup.json")
        res_exp = runner.invoke(main, ["memory", "export", "--out", json_file])
        assert res_exp.exit_code == 0
        assert os.path.exists(json_file)

        res_imp = runner.invoke(main, ["memory", "import", "--file", json_file])
        assert res_imp.exit_code == 0
        assert "Successfully imported" in res_imp.output

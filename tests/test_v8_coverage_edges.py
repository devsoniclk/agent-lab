"""
Tests for v0.8.0 coverage edge cases in memory and vector indexing.
"""

import json
import pytest
from agent_lab.memory import (
    CrossAgentMemoryStore,
    MemoryEntry,
    MemoryKind,
    MemoryAccessLevel,
    SemanticTextEmbedder,
    dot_product,
    cosine_similarity,
    euclidean_distance,
    normalize_vector,
)


def test_vector_extreme_edge_cases():
    # Identical zero vectors
    assert dot_product([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert euclidean_distance([0.0, 0.0], [0.0, 0.0]) == 0.0

    # Negative cosine similarity clamping
    sim = cosine_similarity([-1.0, 0.0], [1.0, 0.0])
    assert sim == -1.0


def test_memory_store_edge_cases():
    store = CrossAgentMemoryStore()

    # Empty search query
    assert store.search("") == []
    assert store.search("   ") == []

    # Search with no matching tags
    store.store("Content", agent_id="a1", tags=["alpha"])
    assert store.search("Content", tags=["beta"]) == []

    # Search with min_score filter higher than 1.0
    assert store.search("Content", min_score=1.5) == []

    # Update non-existent entry returns None
    assert store.update("non_existent", content="new") is None

    # Clear specific namespace vs all
    store.store("Content in ns1", agent_id="a1", namespace="ns1")
    store.store("Content in ns2", agent_id="a1", namespace="ns2")
    store.clear(namespace="ns1")
    stats = store.get_stats()
    assert "ns1" not in stats["namespaces"] or stats["namespaces"]["ns1"] == 0
    assert stats["namespaces"]["ns2"] == 1


def test_memory_entry_str_enum_conversion():
    entry = MemoryEntry(
        id="mem_1",
        agent_id="a1",
        content="Testing enum conversion",
        kind="PROCEDURAL",
        access_level="PUBLIC",
    )
    assert entry.kind == MemoryKind.PROCEDURAL
    assert entry.access_level == MemoryAccessLevel.PUBLIC

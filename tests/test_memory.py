"""
Unit tests for agent_lab.memory: Cross-Agent Semantic Memory Store & Vector Index.
"""

import math
import os
import tempfile
import time
import pytest

from agent_lab.memory import (
    MemoryKind,
    MemoryAccessLevel,
    MemoryEntry,
    MemorySearchResult,
    SemanticTextEmbedder,
    CrossAgentMemoryStore,
    dot_product,
    l2_norm,
    normalize_vector,
    cosine_similarity,
    euclidean_distance,
)


def test_vector_math_operations():
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]

    # Dot product
    assert dot_product(v1, v2) == 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0  # 32.0

    # L2 norm
    assert math.isclose(l2_norm([3.0, 4.0]), 5.0)

    # Normalize vector
    norm_v = normalize_vector([3.0, 4.0])
    assert math.isclose(norm_v[0], 0.6)
    assert math.isclose(norm_v[1], 0.8)
    assert math.isclose(l2_norm(norm_v), 1.0)

    # Zero vector normalization
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]

    # Cosine similarity
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    # Euclidean distance
    assert math.isclose(euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0)

    # Dimension mismatch
    with pytest.raises(ValueError):
        dot_product([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        euclidean_distance([1.0], [1.0, 2.0])


def test_semantic_text_embedder():
    embedder = SemanticTextEmbedder(dimension=64)
    assert embedder.dimension == 64

    with pytest.raises(ValueError):
        SemanticTextEmbedder(dimension=0)

    # Tokenization
    tokens = embedder.tokenize("Solana transaction lens v0.10.0!")
    assert "solana" in tokens
    assert "transaction" in tokens
    assert "lens" in tokens

    # Embedding determinism & normalization
    vec1 = embedder.embed("Solana zero-copy account layout")
    vec2 = embedder.embed("Solana zero-copy account layout")
    assert len(vec1) == 64
    assert vec1 == vec2
    assert math.isclose(l2_norm(vec1), 1.0, rel_tol=1e-5)

    # Empty string produces zero vector
    vec_empty = embedder.embed("")
    assert len(vec_empty) == 64
    assert all(x == 0.0 for x in vec_empty)

    # Semantic similarity: related vs unrelated
    sim_related = cosine_similarity(
        embedder.embed("Solana account memory alignment"),
        embedder.embed("Solana account layout alignment")
    )
    sim_unrelated = cosine_similarity(
        embedder.embed("Solana account memory alignment"),
        embedder.embed("Baking chocolate cake with strawberry frosting")
    )
    assert sim_related > sim_unrelated


def test_memory_entry_decay_and_scores():
    now = time.time()
    entry = MemoryEntry(
        id="mem_test",
        agent_id="agent-01",
        content="Test content",
        importance=0.8,
        ttl_seconds=60.0,
        decay_half_life_seconds=100.0,
        created_at=now - 30.0,
        last_accessed_at=now - 50.0,
    )

    # Expiration
    assert not entry.is_expired(now)
    assert entry.is_expired(now + 35.0)

    # Recency score calculation (at delta=50s with half_life=100s, recency = 2^(-0.5) ~= 0.707)
    recency = entry.compute_recency_score(now)
    assert 0.70 <= recency <= 0.72

    # Zero decay half life returns 1.0
    entry.decay_half_life_seconds = 0.0
    assert entry.compute_recency_score(now) == 1.0

    # Compound score
    entry.decay_half_life_seconds = 100.0
    score = entry.compute_compound_score(similarity=0.9, keyword_score=0.5, now=now)
    assert 0.0 <= score <= 1.0

    # Serialization roundtrip
    d = entry.to_dict()
    assert d["id"] == "mem_test"
    assert d["agent_id"] == "agent-01"
    reloaded = MemoryEntry.from_dict(d)
    assert reloaded.id == entry.id
    assert reloaded.importance == entry.importance


def test_cross_agent_memory_store_basic():
    store = CrossAgentMemoryStore()

    # Store memory
    e1 = store.store(
        content="Anchor PDA derivation for multisig registry",
        agent_id="agent-coder",
        kind=MemoryKind.SEMANTIC,
        tags=["solana", "anchor"],
        importance=0.9,
    )
    assert e1.id.startswith("mem_")
    assert e1.agent_id == "agent-coder"
    assert "solana" in e1.tags

    # Validation errors
    with pytest.raises(ValueError):
        store.store(content="", agent_id="agent-coder")
    with pytest.raises(ValueError):
        store.store(content="Valid", agent_id="")

    # Get memory
    fetched = store.get(e1.id)
    assert fetched is not None
    assert fetched.content == e1.content
    assert fetched.access_count == 2  # 1 on store + 1 on get

    # Update memory
    updated = store.update(e1.id, content="Updated Anchor PDA guide", tags=["solana", "anchor", "security"])
    assert updated is not None
    assert updated.content == "Updated Anchor PDA guide"
    assert "security" in updated.tags

    # Delete memory
    assert store.delete(e1.id)
    assert store.get(e1.id) is None
    assert not store.delete("non_existent")


def test_memory_access_control():
    store = CrossAgentMemoryStore()

    # Store PRIVATE memory
    priv = store.store(
        content="Secret agent reflection",
        agent_id="agent-alpha",
        access_level=MemoryAccessLevel.PRIVATE,
    )

    # Author can access
    assert store.get(priv.id, requesting_agent_id="agent-alpha") is not None

    # Other agent is denied
    assert store.get(priv.id, requesting_agent_id="agent-beta") is None

    # Search ignores private memories for other agents
    search_author = store.search("reflection", requesting_agent_id="agent-alpha")
    assert len(search_author) == 1

    search_other = store.search("reflection", requesting_agent_id="agent-beta")
    assert len(search_other) == 0

    # Unauthorized delete is rejected
    assert not store.delete(priv.id, requesting_agent_id="agent-beta")
    assert store.get(priv.id, requesting_agent_id="agent-alpha") is not None


def test_memory_hybrid_search_and_ranking():
    store = CrossAgentMemoryStore()

    store.store(
        content="Byzantine Fault Tolerant consensus engine with 2f+1 threshold",
        agent_id="agent-consensus",
        kind=MemoryKind.SEMANTIC,
        tags=["bft", "consensus"],
        importance=0.95,
    )
    store.store(
        content="Solana transaction compute unit optimization with access lists",
        agent_id="agent-solana",
        kind=MemoryKind.PROCEDURAL,
        tags=["solana", "cu"],
        importance=0.70,
    )
    store.store(
        content="Historical trajectory evaluation benchmark F1 scores",
        agent_id="agent-eval",
        kind=MemoryKind.EPISODIC,
        tags=["benchmark"],
        importance=0.50,
    )

    # Search for consensus
    res = store.search("BFT consensus 2f+1 quorum", top_k=2)
    assert len(res) >= 1
    assert "Byzantine" in res[0].entry.content
    assert res[0].score > 0.3

    # Filter by kind
    res_proc = store.search("optimization", kind=MemoryKind.PROCEDURAL)
    assert len(res_proc) == 1
    assert res_proc[0].entry.kind == MemoryKind.PROCEDURAL

    # Filter by tags
    res_tagged = store.search("Solana", tags=["solana"])
    assert len(res_tagged) == 1
    assert "compute unit" in res_tagged[0].entry.content


def test_memory_ttl_and_eviction():
    store = CrossAgentMemoryStore(max_capacity_per_namespace=3)

    # Store with TTL
    e_short = store.store(
        content="Ephemeral memory",
        agent_id="agent-temp",
        ttl_seconds=0.1,
    )
    time.sleep(0.15)
    assert e_short.is_expired()
    # Getting expired removes it
    assert store.get(e_short.id) is None

    # Test capacity eviction
    store.clear()
    store.store("Item 1", agent_id="a1", importance=0.2)
    store.store("Item 2", agent_id="a1", importance=0.9)
    store.store("Item 3", agent_id="a1", importance=0.8)
    
    # Adding 4th item should proactively evict the least valuable item (Item 1)
    e4 = store.store("Item 4", agent_id="a1", importance=0.7)
    stats = store.get_stats()
    assert stats["total_memories"] <= 3


def test_memory_export_import_roundtrip():
    store1 = CrossAgentMemoryStore()
    store1.store("Episodic log entry #1", agent_id="worker-1", tags=["log"])
    store1.store("Semantic fact #2", agent_id="worker-2", tags=["fact"])

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "memories.json")
        store1.export_json(export_path)
        assert os.path.exists(export_path)

        store2 = CrossAgentMemoryStore()
        imported = store2.import_json(export_path)
        assert imported == 2

        stats2 = store2.get_stats()
        assert stats2["total_memories"] == 2
        assert stats2["unique_tags"] == 2

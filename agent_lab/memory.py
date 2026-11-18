"""
agent_lab.memory: Cross-Agent Semantic Memory Store, Episodic Vector Index,
and Hybrid Retrieval Engine for Autonomous Agent Swarms.

Provides:
- MemoryKind: EPISODIC, SEMANTIC, PROCEDURAL, WORKING memory taxonomies.
- MemoryAccessLevel: PRIVATE, SWARM, PUBLIC access control boundaries.
- SemanticTextEmbedder: Pure-Python deterministic hashing-trick embedding generator.
- MemoryEntry: Decaying, access-weighted, importance-scored memory representation.
- CrossAgentMemoryStore: Thread-safe multi-namespace memory index with hybrid vector retrieval.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class MemoryKind(str, Enum):
    """Categorical classification of agent memories."""
    EPISODIC = "EPISODIC"       # Chronological experiences, execution trajectories, event histories
    SEMANTIC = "SEMANTIC"       # Factual knowledge, schemas, heuristics, environment invariants
    PROCEDURAL = "PROCEDURAL"   # Step-by-step algorithms, workflows, skill execution instructions
    WORKING = "WORKING"         # Ephemeral scratchpad, transient task context, intermediate states


class MemoryAccessLevel(str, Enum):
    """Access control permission boundary for memory visibility."""
    PRIVATE = "PRIVATE"  # Only visible to the authoring agent
    SWARM = "SWARM"      # Shared among all authenticated agents within the namespace/swarm
    PUBLIC = "PUBLIC"    # Globally readable across all swarms and namespaces


# ---------------------------------------------------------------------------
# Zero-Dependency Vector Mathematics & Normalization
# ---------------------------------------------------------------------------

def dot_product(v1: List[float], v2: List[float]) -> float:
    """Compute standard algebraic dot product of two real vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions mismatch: {len(v1)} vs {len(v2)}")
    return sum(a * b for a, b in zip(v1, v2))


def l2_norm(v: List[float]) -> float:
    """Compute Euclidean (L2) norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def normalize_vector(v: List[float]) -> List[float]:
    """Normalize vector to unit length (L2 norm = 1.0)."""
    norm = l2_norm(v)
    if norm < 1e-12:
        return [0.0] * len(v)
    return [x / norm for x in v]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors bounded in [-1.0, 1.0]."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions mismatch: {len(v1)} vs {len(v2)}")
    norm1 = l2_norm(v1)
    norm2 = l2_norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    sim = dot / (norm1 * norm2)
    return max(-1.0, min(1.0, sim))


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Compute standard Euclidean distance between two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions mismatch: {len(v1)} vs {len(v2)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# ---------------------------------------------------------------------------
# Pure-Python Deterministic Semantic Text Embedder
# ---------------------------------------------------------------------------

class SemanticTextEmbedder:
    """
    Deterministic zero-dependency text embedder using tokenization,
    multi-ngram feature hashing with sublinear term-frequency scaling,
    and sign projection into a fixed-dimensional unit hypersphere.
    """

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be strictly positive")
        self.dimension = dimension

    def tokenize(self, text: str) -> List[str]:
        """Extract alphanumeric word unigrams and character bigrams/trigrams."""
        clean = text.lower()
        words = re.findall(r"\b[a-z0-9_-]+\b", clean)
        tokens: List[str] = list(words)
        # Add character n-grams for subword morphological matching
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(f"tri:{word[i:i+3]}")
            if len(word) >= 2:
                for i in range(len(word) - 1):
                    tokens.append(f"bi:{word[i:i+2]}")
        return tokens

    def embed(self, text: str) -> List[float]:
        """Generate a normalized dense float vector from input string."""
        tokens = self.tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        counts = collections.Counter(tokens)
        vec = [0.0] * self.dimension

        for token, count in counts.items():
            # Sublinear TF weight: 1.0 + ln(count)
            weight = 1.0 + math.log(count)
            
            # Deterministic hash to bucket and sign
            h = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], byteorder="big") % self.dimension
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            
            vec[idx] += sign * weight

        return normalize_vector(vec)


# ---------------------------------------------------------------------------
# Memory Entry Data Representation
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """Individual unit of structured episodic or semantic knowledge."""
    id: str
    agent_id: str
    content: str
    namespace: str = "default"
    kind: MemoryKind = MemoryKind.SEMANTIC
    access_level: MemoryAccessLevel = MemoryAccessLevel.SWARM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5  # [0.0, 1.0]
    ttl_seconds: Optional[float] = None
    decay_half_life_seconds: float = 86400.0  # 24 hours nominal decay

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = MemoryKind(self.kind)
        if isinstance(self.access_level, str):
            self.access_level = MemoryAccessLevel(self.access_level)
        self.importance = max(0.0, min(1.0, float(self.importance)))

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if memory has surpassed its designated TTL."""
        if self.ttl_seconds is None:
            return False
        current_time = now if now is not None else time.time()
        return (current_time - self.created_at) > self.ttl_seconds

    def compute_recency_score(self, now: Optional[float] = None) -> float:
        """
        Compute continuous recency score in [0.0, 1.0] using exponential half-life decay.
        S_rec = e^(-lambda * delta_t), where lambda = ln(2) / half_life
        """
        current_time = now if now is not None else time.time()
        delta = max(0.0, current_time - self.last_accessed_at)
        if self.decay_half_life_seconds <= 0:
            return 1.0
        lam = math.log(2) / self.decay_half_life_seconds
        return math.exp(-lam * delta)

    def compute_compound_score(
        self,
        similarity: float,
        keyword_score: float = 0.0,
        now: Optional[float] = None,
        weight_sim: float = 0.50,
        weight_kw: float = 0.20,
        weight_rec: float = 0.15,
        weight_imp: float = 0.15,
    ) -> float:
        """
        Harmonize dense similarity, lexical match, temporal recency, and innate importance.
        Score = w_sim * Sim + w_kw * Kw + w_rec * Recency + w_imp * Importance
        """
        recency = self.compute_recency_score(now)
        # Normalized similarity in [0, 1]
        sim_norm = max(0.0, min(1.0, (similarity + 1.0) / 2.0 if similarity < 0.0 else similarity))
        kw_norm = max(0.0, min(1.0, keyword_score))
        
        score = (
            weight_sim * sim_norm
            + weight_kw * kw_norm
            + weight_rec * recency
            + weight_imp * self.importance
        )
        return round(score, 6)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory entry to canonical dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "kind": self.kind.value,
            "access_level": self.access_level.value,
            "content": self.content,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "embedding": self.embedding,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "importance": self.importance,
            "ttl_seconds": self.ttl_seconds,
            "decay_half_life_seconds": self.decay_half_life_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Deserialize memory entry from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            content=data["content"],
            namespace=data.get("namespace", "default"),
            kind=MemoryKind(data.get("kind", MemoryKind.SEMANTIC.value)),
            access_level=MemoryAccessLevel(data.get("access_level", MemoryAccessLevel.SWARM.value)),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", time.time()),
            last_accessed_at=data.get("last_accessed_at", time.time()),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            ttl_seconds=data.get("ttl_seconds"),
            decay_half_life_seconds=data.get("decay_half_life_seconds", 86400.0),
        )


@dataclass
class MemorySearchResult:
    """Ranked search result combining semantic vector match and metadata."""
    entry: MemoryEntry
    score: float
    similarity: float
    keyword_score: float
    recency_score: float
    importance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": round(self.score, 6),
            "similarity": round(self.similarity, 6),
            "keyword_score": round(self.keyword_score, 6),
            "recency_score": round(self.recency_score, 6),
            "importance": round(self.importance, 4),
        }


# ---------------------------------------------------------------------------
# Cross-Agent Semantic Memory Store Engine
# ---------------------------------------------------------------------------

class CrossAgentMemoryStore:
    """
    Thread-safe, multi-namespace cross-agent semantic memory store with
    vector indexing, hybrid lexical/dense retrieval, and memory eviction.
    """

    def __init__(
        self,
        embedder: Optional[SemanticTextEmbedder] = None,
        max_capacity_per_namespace: int = 1000,
        default_namespace: str = "default",
    ) -> None:
        self.embedder = embedder or SemanticTextEmbedder(dimension=128)
        self.max_capacity = max_capacity_per_namespace
        self.default_namespace = default_namespace
        self._lock = threading.RLock()

        # In-memory indices
        self._entries: Dict[str, MemoryEntry] = {}
        self._namespace_index: Dict[str, Set[str]] = collections.defaultdict(set)
        self._agent_index: Dict[str, Set[str]] = collections.defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = collections.defaultdict(set)

    def _generate_memory_id(self, agent_id: str, content: str) -> str:
        """Create deterministic content-hashed unique memory identifier."""
        ts = str(time.time_ns())
        raw = f"{agent_id}:{ts}:{content}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"mem_{digest}"

    def store(
        self,
        content: str,
        agent_id: str,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        access_level: MemoryAccessLevel = MemoryAccessLevel.SWARM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        ttl_seconds: Optional[float] = None,
        namespace: Optional[str] = None,
        decay_half_life_seconds: float = 86400.0,
    ) -> MemoryEntry:
        """Store an episodic or semantic memory item with vector indexing."""
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")
        if not agent_id:
            raise ValueError("agent_id must be provided")

        ns = namespace or self.default_namespace
        clean_tags = [t.strip().lower() for t in (tags or []) if t and t.strip()]

        with self._lock:
            # Capacity check and proactive eviction
            if len(self._namespace_index[ns]) >= self.max_capacity:
                self.evict_to_capacity(namespace=ns, target_capacity=self.max_capacity - 1)

            mem_id = self._generate_memory_id(agent_id, content)

            # Generate vector embedding if not explicitly provided
            if embedding is None:
                vec = self.embedder.embed(content)
            else:
                vec = normalize_vector(embedding)

            entry = MemoryEntry(
                id=mem_id,
                agent_id=agent_id,
                content=content.strip(),
                namespace=ns,
                kind=kind,
                access_level=access_level,
                tags=clean_tags,
                metadata=metadata or {},
                embedding=vec,
                created_at=time.time(),
                last_accessed_at=time.time(),
                access_count=1,
                importance=importance,
                ttl_seconds=ttl_seconds,
                decay_half_life_seconds=decay_half_life_seconds,
            )

            self._entries[mem_id] = entry
            self._namespace_index[ns].add(mem_id)
            self._agent_index[agent_id].add(mem_id)
            for tag in clean_tags:
                self._tag_index[tag].add(mem_id)

            return entry

    def get(self, memory_id: str, requesting_agent_id: Optional[str] = None) -> Optional[MemoryEntry]:
        """Retrieve memory item by ID enforcing access level permissions."""
        with self._lock:
            entry = self._entries.get(memory_id)
            if not entry:
                return None

            # Check expiration
            if entry.is_expired():
                self.delete(memory_id)
                return None

            # Access permission check
            if entry.access_level == MemoryAccessLevel.PRIVATE:
                if requesting_agent_id and requesting_agent_id != entry.agent_id:
                    return None  # Forbidden

            # Record access touch
            entry.last_accessed_at = time.time()
            entry.access_count += 1
            return entry

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        requesting_agent_id: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """Update fields of an existing memory entry."""
        with self._lock:
            entry = self.get(memory_id, requesting_agent_id=requesting_agent_id)
            if not entry:
                return None

            if content is not None and content.strip():
                entry.content = content.strip()
                entry.embedding = self.embedder.embed(entry.content)

            if tags is not None:
                # Remove old tags from tag index
                for t in entry.tags:
                    self._tag_index[t].discard(memory_id)
                new_tags = [t.strip().lower() for t in tags if t and t.strip()]
                entry.tags = new_tags
                for t in new_tags:
                    self._tag_index[t].add(memory_id)

            if metadata is not None:
                entry.metadata.update(metadata)

            if importance is not None:
                entry.importance = max(0.0, min(1.0, float(importance)))

            entry.last_accessed_at = time.time()
            return entry

    def delete(self, memory_id: str, requesting_agent_id: Optional[str] = None) -> bool:
        """Remove a memory entry and prune all associated indices."""
        with self._lock:
            entry = self._entries.get(memory_id)
            if not entry:
                return False

            if entry.access_level == MemoryAccessLevel.PRIVATE:
                if requesting_agent_id and requesting_agent_id != entry.agent_id:
                    return False  # Permission denied

            # Remove from entries
            del self._entries[memory_id]
            self._namespace_index[entry.namespace].discard(memory_id)
            self._agent_index[entry.agent_id].discard(memory_id)
            for t in entry.tags:
                self._tag_index[t].discard(memory_id)

            return True

    def search(
        self,
        query: str,
        requesting_agent_id: Optional[str] = None,
        namespace: Optional[str] = None,
        kind: Optional[MemoryKind] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        hybrid_alpha: float = 0.70,
        weight_sim: float = 0.50,
        weight_kw: float = 0.20,
        weight_rec: float = 0.15,
        weight_imp: float = 0.15,
    ) -> List[MemorySearchResult]:
        """
        Hybrid vector and lexical search across memories with recency and importance scoring.
        hybrid_alpha: 1.0 = Pure Vector, 0.0 = Pure Lexical Keyword.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        query_vec = self.embedder.embed(clean_query)
        query_tokens = set(self.embedder.tokenize(clean_query))

        target_tags = set(t.strip().lower() for t in (tags or []) if t and t.strip())
        now = time.time()

        with self._lock:
            candidates: List[MemoryEntry] = []
            
            # Select target namespaces
            if namespace:
                mem_ids = list(self._namespace_index.get(namespace, set()))
            else:
                mem_ids = list(self._entries.keys())

            for mid in mem_ids:
                entry = self._entries.get(mid)
                if not entry:
                    continue

                # Evict expired on sight
                if entry.is_expired(now):
                    self.delete(mid)
                    continue

                # Permission filter
                if entry.access_level == MemoryAccessLevel.PRIVATE:
                    if requesting_agent_id and requesting_agent_id != entry.agent_id:
                        continue

                # Kind filter
                if kind is not None and entry.kind != kind:
                    continue

                # Tag filter
                if target_tags and not target_tags.issubset(set(entry.tags)):
                    continue

                candidates.append(entry)

            results: List[MemorySearchResult] = []

            for entry in candidates:
                # 1. Dense vector cosine similarity
                if entry.embedding and query_vec:
                    sim = cosine_similarity(query_vec, entry.embedding)
                else:
                    sim = 0.0

                # 2. Lexical keyword match score (token overlap Jaccard / Overlap coefficient)
                entry_tokens = set(self.embedder.tokenize(entry.content))
                if entry_tokens and query_tokens:
                    overlap = len(query_tokens.intersection(entry_tokens))
                    kw_score = overlap / len(query_tokens)
                else:
                    kw_score = 0.0

                # Harmonize similarity and keyword via hybrid_alpha
                effective_sim = (hybrid_alpha * sim) + ((1.0 - hybrid_alpha) * kw_score)

                recency = entry.compute_recency_score(now)
                compound = entry.compute_compound_score(
                    similarity=effective_sim,
                    keyword_score=kw_score,
                    now=now,
                    weight_sim=weight_sim,
                    weight_kw=weight_kw,
                    weight_rec=weight_rec,
                    weight_imp=weight_imp,
                )

                if compound >= min_score:
                    results.append(
                        MemorySearchResult(
                            entry=entry,
                            score=compound,
                            similarity=sim,
                            keyword_score=kw_score,
                            recency_score=recency,
                            importance=entry.importance,
                        )
                    )

            # Sort descending by compound score
            results.sort(key=lambda r: r.score, reverse=True)
            top_results = results[:top_k]

            # Increment access counts for top retrieved items
            for res in top_results:
                res.entry.last_accessed_at = now
                res.entry.access_count += 1

            return top_results

    def evict_expired(self) -> int:
        """Evict all memories that have surpassed their TTL duration."""
        now = time.time()
        evicted = 0
        with self._lock:
            expired_ids = [mid for mid, entry in self._entries.items() if entry.is_expired(now)]
            for mid in expired_ids:
                if self.delete(mid):
                    evicted += 1
        return evicted

    def evict_to_capacity(
        self,
        namespace: Optional[str] = None,
        target_capacity: Optional[int] = None,
        strategy: str = "DECAY",
    ) -> int:
        """
        Evict least valuable memories until namespace reaches target_capacity.
        Strategies:
        - "DECAY": Lowest combined importance * recency score
        - "LRU": Least recently accessed
        - "LIF": Least important first
        """
        ns = namespace or self.default_namespace
        target = target_capacity if target_capacity is not None else self.max_capacity
        now = time.time()
        evicted_count = 0

        with self._lock:
            # First remove expired
            self.evict_expired()

            current_ids = list(self._namespace_index.get(ns, set()))
            if len(current_ids) <= target:
                return 0

            entries = [self._entries[mid] for mid in current_ids if mid in self._entries]

            if strategy.upper() == "LRU":
                entries.sort(key=lambda e: e.last_accessed_at)
            elif strategy.upper() == "LIF":
                entries.sort(key=lambda e: e.importance)
            else:  # DECAY
                entries.sort(key=lambda e: e.importance * e.compute_recency_score(now))

            to_remove = len(entries) - target
            for i in range(to_remove):
                if self.delete(entries[i].id):
                    evicted_count += 1

        return evicted_count

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve aggregated health and volume metrics across namespaces."""
        with self._lock:
            total_count = len(self._entries)
            by_namespace = {ns: len(ids) for ns, ids in self._namespace_index.items()}
            by_agent = {agent: len(ids) for agent, ids in self._agent_index.items()}
            by_kind = collections.Counter(e.kind.value for e in self._entries.values())
            by_access = collections.Counter(e.access_level.value for e in self._entries.values())

            avg_importance = (
                sum(e.importance for e in self._entries.values()) / total_count
                if total_count > 0 else 0.0
            )

            return {
                "total_memories": total_count,
                "namespaces": by_namespace,
                "agents": by_agent,
                "kinds": dict(by_kind),
                "access_levels": dict(by_access),
                "unique_tags": len(self._tag_index),
                "average_importance": round(avg_importance, 4),
                "embedding_dimension": self.embedder.dimension,
            }

    def export_json(self, path: Optional[str] = None) -> str:
        """Export all memories to a serialized JSON string or file."""
        with self._lock:
            payload = {
                "version": "1.0",
                "exported_at": time.time(),
                "memories": [e.to_dict() for e in self._entries.values()],
            }
            raw = json.dumps(payload, indent=2)
            if path:
                p = Path(path).resolve()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(raw, encoding="utf-8")
            return raw

    def import_json(self, data_or_path: str) -> int:
        """Import memories from serialized JSON string or file path."""
        p = Path(data_or_path)
        if p.exists() and p.is_file():
            raw = p.read_text(encoding="utf-8")
        else:
            raw = data_or_path

        payload = json.loads(raw)
        memories = payload.get("memories", [])
        imported = 0

        with self._lock:
            for item in memories:
                entry = MemoryEntry.from_dict(item)
                self._entries[entry.id] = entry
                self._namespace_index[entry.namespace].add(entry.id)
                self._agent_index[entry.agent_id].add(entry.id)
                for t in entry.tags:
                    self._tag_index[t].add(entry.id)
                imported += 1

        return imported

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear memories within a namespace or completely wipe the store."""
        with self._lock:
            if namespace:
                ids = list(self._namespace_index.get(namespace, set()))
                for mid in ids:
                    self.delete(mid)
            else:
                self._entries.clear()
                self._namespace_index.clear()
                self._agent_index.clear()
                self._tag_index.clear()

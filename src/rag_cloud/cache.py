from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable
from typing import Any

import numpy as np

from rag_cloud.clients import Clients
from rag_cloud.config import Settings


"""Redis-backed caching helpers.

The query pipeline uses three layers:
- exact cache for identical query + tenant matches
- embedding cache to avoid recomputing vectors
- semantic cache to reuse answers for similar queries
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_loads_redis_value(value: Any) -> Any | None:
    """Safely parse Redis values for static typing and runtime resilience."""

    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    if isinstance(value, Awaitable):
        return None
    return None


class CacheLayer:
    """Encapsulates all Redis key formats and cache lookup/write behavior."""

    def __init__(self, clients: Clients, settings: Settings):
        self.redis = clients.redis
        self.settings = settings

    def exact_get(self, query: str, tenant_id: str) -> dict[str, Any] | None:
        """Return a fully cached response for an exact query + tenant match."""

        # Tenant is part of the key so one user's cached answer is not reused
        # across another tenant boundary.
        key = f"rag:exact:{_sha256(query + '::' + tenant_id)}"
        parsed = _json_loads_redis_value(self.redis.get(key))
        return parsed if isinstance(parsed, dict) else None

    def exact_set(self, query: str, tenant_id: str, payload: dict[str, Any]) -> None:
        """Store a fully generated answer for exact reuse."""

        key = f"rag:exact:{_sha256(query + '::' + tenant_id)}"
        self.redis.setex(key, self.settings.cache_ttl_exact, json.dumps(payload))

    def embedding_get(self, text: str) -> list[float] | None:
        """Return a cached embedding vector if one exists."""

        key = f"rag:emb:{_sha256(text)}"
        parsed = _json_loads_redis_value(self.redis.get(key))
        return parsed if isinstance(parsed, list) else None

    def embedding_set(self, text: str, vector: list[float]) -> None:
        """Store an embedding vector for later query reuse."""

        key = f"rag:emb:{_sha256(text)}"
        self.redis.setex(key, self.settings.cache_ttl_embed, json.dumps(vector))

    def semantic_get(self, query_vector: list[float], tenant_id: str) -> dict[str, Any] | None:
        """Return a cached response whose stored query vector is close enough."""

        q = np.array(query_vector)
        best_score = 0.0
        best_payload = None

        # Learning-mode implementation: scan recent semantic entries and pick
        # the nearest cached vector by cosine similarity.
        for key in self.redis.scan_iter(match=f"rag:semantic:{tenant_id}:*", count=200):
            raw = self.redis.get(key)
            entry = _json_loads_redis_value(raw)
            if not entry:
                continue
            v = np.array(entry["vector"])
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) + 1e-9
            score = float(np.dot(q, v) / denom)
            if score > best_score:
                best_score = score
                best_payload = entry["response"]

        # Only reuse the cached response if similarity clears the configured
        # threshold; otherwise the pipeline continues to real retrieval.
        if best_score >= self.settings.similarity_threshold:
            return best_payload
        return None

    def semantic_set(self, query: str, tenant_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        """Store a query vector and its response for semantic reuse."""

        key = f"rag:semantic:{tenant_id}:{_sha256(query)}"
        value = {"vector": vector, "response": payload}
        self.redis.setex(key, self.settings.cache_ttl_semantic, json.dumps(value))

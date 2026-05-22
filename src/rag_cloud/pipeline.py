from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import cast
from typing import Any

import google.generativeai as genai

from rag_cloud.cache import CacheLayer
from rag_cloud.clients import Clients
from rag_cloud.config import Settings
from rag_cloud.metadata_store import MetadataStore


@dataclass
class QueryResponse:
    """Normalized response shape returned by the RAG pipeline."""

    answer: str
    sources: list[dict[str, Any]]
    cache: str


class RAGPipeline:
    """End-to-end query pipeline for the cloud-backed learning stack.

    Runtime flow:
    1. Check exact cache using query + tenant.
    2. Embed the query, using an embedding cache when available.
    3. Check semantic cache by vector similarity.
    4. Retrieve candidate chunks from Qdrant.
    5. Optionally rerank candidates with Cohere.
    6. Generate a grounded answer with Groq.
    7. Write exact and semantic cache entries for later reuse.
    """

    def __init__(self, clients: Clients, settings: Settings):
        self.clients = clients
        self.settings = settings
        self.cache = CacheLayer(clients, settings)
        self.metadata = MetadataStore(settings.database_url) if settings.database_url else None

    @staticmethod
    def _trace_span(trace: Any, name: str, payload: dict[str, Any] | None = None) -> Any:
        if not trace:
            return None
        try:
            return trace.span(name=name, input=payload) if payload is not None else trace.span(name=name)
        except Exception:
            return None

    @staticmethod
    def _span_end(span: Any, payload: dict[str, Any] | None = None) -> None:
        if not span:
            return
        try:
            if payload is not None:
                span.end(output=payload)
            else:
                span.end()
        except Exception:
            return

    @staticmethod
    def _trace_update(trace: Any, payload: dict[str, Any]) -> None:
        if not trace:
            return
        try:
            trace.update(output=payload)
        except Exception:
            return

    def _log_query_safe(self, **kwargs: Any) -> None:
        if not self.metadata:
            return
        try:
            self.metadata.log_query(**kwargs)
        except Exception:
            return

    def embed(self, text: str) -> list[float]:
        """Return a query embedding, preferring Redis before calling Gemini."""

        cached = self.cache.embedding_get(text)
        if cached is not None:
            return cached

        # Access through getattr to avoid static-export mismatch in SDK stubs.
        embed_fn = cast(Any, getattr(genai, "embed_content"))
        result = cast(
            dict[str, Any],
            embed_fn(
                model=self.settings.embed_model,
                content=text,
                task_type="retrieval_query",
                output_dimensionality=self.settings.embed_dim,
            ),
        )
        vector = result["embedding"]
        self.cache.embedding_set(text, vector)
        return vector

    def retrieve(self, query_vector: list[float]) -> list[dict[str, Any]]:
        """Fetch top-k candidate chunks from Qdrant using dense vector search."""

        # Qdrant is the first place where the pipeline leaves the cache path and
        # performs real retrieval work against indexed document chunks.
        # qdrant-client exposes dynamic response models; cast the call result so
        # static analysis can understand `.points` access.
        query_points_fn = cast(Any, getattr(self.clients.qdrant, "query_points"))
        search_result = cast(
            Any,
            query_points_fn(
                collection_name=self.settings.collection,
                query=query_vector,
                limit=self.settings.retrieve_top_k,
                with_payload=True,
            ),
        )
        hits = cast(list[Any], search_result.points)

        docs: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            docs.append(
                {
                    "id": str(hit.id),
                    "text": str(payload.get("text", "")),
                    "source": str(payload.get("source", "")),
                    "retrieval_score": float(hit.score or 0.0),
                }
            )
        return docs

    @staticmethod
    def should_rerank(query: str) -> bool:
        """Apply cheap guardrails before spending Cohere rerank calls.

        Current rule: only rerank when the query has at least 5 words.
        """

        return len(query.split()) >= 5

    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reorder retrieved documents by relevance when Cohere is configured.

        The fallback path is intentional for learning mode: if Cohere is not
        configured, or the query is very short, the pipeline keeps the original
        retrieval order and simply truncates to the final top-k.
        """

        if not docs:
            return []

        if not self.clients.cohere or not self.should_rerank(query):
            return docs[: self.settings.rerank_top_k]

        result = self.clients.cohere.rerank(
            model=self.settings.cohere_rerank_model,
            query=query,
            documents=[d["text"] for d in docs],
            top_n=self.settings.rerank_top_k,
        )

        reranked: list[dict[str, Any]] = []
        for item in result.results:
            original = docs[item.index]
            reranked.append(
                {
                    **original,
                    "rerank_score": float(item.relevance_score),
                }
            )
        return reranked

    def generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        """Build the final grounded prompt and send it to the Groq chat API."""

        context_text = "\n\n".join(
            [f"[{i+1}] source={c['source']}\n{c['text']}" for i, c in enumerate(contexts)]
        )
        prompt = (
            "Answer using only the provided context. "
            "If information is missing, say you do not have enough context. "
            "Cite claims as [1], [2], etc.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        completion = self.clients.groq.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=900,
        )
        return completion.choices[0].message.content or ""

    def query(self, user_query: str, tenant_id: str = "default") -> QueryResponse:
        """Execute the full cache-first RAG flow for one user query."""

        started = time.perf_counter()
        retrieval_count = 0
        selected_count = 0
        trace_id = None

        trace = None
        if self.clients.langfuse:
            try:
                trace = self.clients.langfuse.trace(
                    name="rag_query",
                    input={"query": user_query, "tenant_id": tenant_id},
                )
                trace_id = getattr(trace, "id", None)
            except Exception:
                trace = None
        try:
            exact_span = self._trace_span(trace, "exact_cache_lookup")

            # Exact cache is the cheapest path: same query + same tenant returns the
            # full stored answer without recomputing embeddings or retrieval.
            cached = self.cache.exact_get(user_query, tenant_id)
            if cached:
                self._span_end(exact_span, {"cache": "exact"})
                self._trace_update(trace, {"cache": "exact", "source_count": len(cached.get("sources", []))})
                latency_ms = int((time.perf_counter() - started) * 1000)
                source_names = [str(s.get("source", "")) for s in cached.get("sources", [])]
                self._log_query_safe(
                    tenant_id=tenant_id,
                    query_text=user_query,
                    cache_layer="exact",
                    retrieval_count=0,
                    selected_count=len(source_names),
                    answer_chars=len(cached.get("answer", "")),
                    latency_ms=latency_ms,
                    trace_id=trace_id,
                    source_names=source_names,
                )
                return QueryResponse(answer=cached["answer"], sources=cached["sources"], cache="exact")
            self._span_end(exact_span, {"cache": "miss"})

            embed_span = self._trace_span(trace, "embedding", {"model": self.settings.embed_model})

            query_vector = self.embed(user_query)
            self._span_end(embed_span, {"dim": len(query_vector)})

            semantic_span = self._trace_span(trace, "semantic_cache_lookup")

            # Semantic cache sits after embedding because it compares query vectors,
            # not raw strings. It can reuse answers for different phrasings that are
            # close enough in embedding space.
            sem_cached = self.cache.semantic_get(query_vector, tenant_id)
            if sem_cached:
                self._span_end(semantic_span, {"cache": "semantic"})
                self._trace_update(trace, {"cache": "semantic", "source_count": len(sem_cached.get("sources", []))})
                latency_ms = int((time.perf_counter() - started) * 1000)
                source_names = [str(s.get("source", "")) for s in sem_cached.get("sources", [])]
                self._log_query_safe(
                    tenant_id=tenant_id,
                    query_text=user_query,
                    cache_layer="semantic",
                    retrieval_count=0,
                    selected_count=len(source_names),
                    answer_chars=len(sem_cached.get("answer", "")),
                    latency_ms=latency_ms,
                    trace_id=trace_id,
                    source_names=source_names,
                )
                return QueryResponse(answer=sem_cached["answer"], sources=sem_cached["sources"], cache="semantic")
            self._span_end(semantic_span, {"cache": "miss"})

            retrieval_span = self._trace_span(
                trace,
                "retrieval",
                {"collection": self.settings.collection, "top_k": self.settings.retrieve_top_k},
            )

            # If both cache layers miss, the pipeline falls through to retrieval,
            # reranking, and final generation.
            retrieved = self.retrieve(query_vector)
            retrieval_count = len(retrieved)
            self._span_end(retrieval_span, {"candidate_count": len(retrieved)})

            rerank_span = self._trace_span(
                trace,
                "reranking",
                {
                    "enabled": bool(self.clients.cohere),
                    "top_k": self.settings.rerank_top_k,
                },
            )
            top_docs = self.rerank(user_query, retrieved)
            selected_count = len(top_docs)
            self._span_end(rerank_span, {"selected_count": len(top_docs)})

            generation_span = self._trace_span(trace, "generation", {"model": self.settings.llm_model})
            answer = self.generate(user_query, top_docs)
            self._span_end(generation_span, {"answer_length": len(answer)})

            payload = {"answer": answer, "sources": top_docs}
            self.cache.exact_set(user_query, tenant_id, payload)
            self.cache.semantic_set(user_query, tenant_id, query_vector, payload)

            latency_ms = int((time.perf_counter() - started) * 1000)
            source_names = [str(s.get("source", "")) for s in top_docs]
            self._log_query_safe(
                tenant_id=tenant_id,
                query_text=user_query,
                cache_layer="miss",
                retrieval_count=retrieval_count,
                selected_count=selected_count,
                answer_chars=len(answer),
                latency_ms=latency_ms,
                trace_id=trace_id,
                source_names=source_names,
            )

            self._trace_update(trace, {"cache": "miss", "source_count": len(top_docs)})
            return QueryResponse(answer=answer, sources=top_docs, cache="miss")
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log_query_safe(
                tenant_id=tenant_id,
                query_text=user_query,
                cache_layer="error",
                retrieval_count=retrieval_count,
                selected_count=selected_count,
                answer_chars=0,
                latency_ms=latency_ms,
                error_text=str(exc),
                trace_id=trace_id,
                source_names=[],
            )
            self._trace_update(trace, {"cache": "error", "error": str(exc)})
            raise
        finally:
            if self.clients.langfuse:
                try:
                    self.clients.langfuse.flush()
                except Exception:
                    pass

    def query_dict(self, user_query: str, tenant_id: str = "default") -> dict[str, Any]:
        """Convenience adapter used by CLI and FastAPI entrypoints."""

        return asdict(self.query(user_query, tenant_id=tenant_id))

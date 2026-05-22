"""Bootstrap script for first-time setup.

Purpose:
- verify required environment variables exist
- verify Qdrant connectivity
- create the target collection if needed
- verify Redis connectivity
- optionally verify Groq, Gemini, Cohere, Langfuse, and Neon
- run a small end-to-end smoke test after setup
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import redis

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
    console: Any = Console()
    rich_table_factory: Any = Table
except ImportError:
    HAS_RICH = False

    class FallbackConsole:
        def print(self, *args, **kwargs) -> None:
            print(*args)

        def rule(self, title: str) -> None:
            print(f"\n{'=' * 20} {title} {'=' * 20}")

    console = FallbackConsole()
    rich_table_factory = None

results: dict[str, tuple[str, str]] = {}


def fail(msg: str) -> NoReturn:
    console.print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(service: str, detail: str = "") -> None:
    results[service] = ("PASS", detail)
    console.print(f"[PASS] {service}" + (f" - {detail}" if detail else ""))


def warn(service: str, detail: str) -> None:
    results[service] = ("WARN", detail)
    console.print(f"[WARN] {service} - {detail}")


def error(service: str, detail: str) -> None:
    results[service] = ("FAIL", detail)
    console.print(f"[FAIL] {service} - {detail}")


def section(title: str) -> None:
    console.rule(title)


def check_qdrant(collection: str, embed_dim: int) -> bool:
    section("Qdrant")
    try:
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), check_version=False)
        qdrant.get_collections()

        if not qdrant.collection_exists(collection):
            qdrant.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            ok("Qdrant", f"created collection={collection}")
        else:
            ok("Qdrant", f"collection exists={collection}")
        return True
    except Exception as exc:
        error("Qdrant", str(exc))
        return False


def check_redis() -> bool:
    section("Redis")
    redis_url = os.getenv("REDIS_URL")
    if redis_url is None:
        fail("REDIS_URL is not set")

    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        key = "rag:bootstrap:test"
        client.setex(key, 30, "ok")
        if client.get(key) != "ok":
            raise RuntimeError("Redis read-back test failed")
        ok("Redis", "connection and write test passed")
        return True
    except Exception as exc:
        error("Redis", str(exc))
        return False


def check_metadata_db() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        warn("Metadata DB", "DATABASE_URL not set - skipping")
        return

    from rag_cloud.metadata_store import MetadataStore

    section("Metadata DB")
    try:
        MetadataStore(database_url)
        ok("Metadata DB", "schema is ready")
    except Exception as exc:
        error("Metadata DB", str(exc))


def check_gemini(embed_dim: int) -> bool:
    section("Gemini")
    try:
        configure_fn = getattr(genai, "configure")
        configure_fn(api_key=os.getenv("GOOGLE_API_KEY"))
        embed_fn = getattr(genai, "embed_content")
        result = embed_fn(
            model=os.getenv("EMBED_MODEL", "models/text-embedding-004"),
            content="bootstrap embedding probe",
            task_type="retrieval_document",
            output_dimensionality=embed_dim,
        )
        dim = len(result["embedding"])
        if dim != embed_dim:
            warn("Gemini", f"expected dim={embed_dim}, got dim={dim}")
        else:
            ok("Gemini", f"embedding probe passed, dim={dim}")
        return True
    except Exception as exc:
        error("Gemini", str(exc))
        return False


def check_groq() -> bool:
    section("Groq")
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly READY"}],
            max_tokens=10,
            temperature=0,
        )
        reply = (response.choices[0].message.content or "").strip()
        ok("Groq", f"model={model}, reply={reply}")
        return True
    except Exception as exc:
        error("Groq", str(exc))
        return False


def check_cohere() -> None:
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        warn("Cohere", "COHERE_API_KEY not set - skipping")
        return

    section("Cohere")
    try:
        from cohere import ClientV2 as CohereClient

        client = CohereClient(api_key=api_key)
        response = client.rerank(
            model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
            query="What is retrieval augmented generation?",
            documents=[
                "RAG combines retrieval with generation.",
                "The weather is sunny today.",
            ],
            top_n=1,
        )
        score = float(response.results[0].relevance_score)
        ok("Cohere", f"rerank probe passed, top_score={score:.4f}")
    except Exception as exc:
        error("Cohere", str(exc))


def check_langfuse() -> None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        warn("Langfuse", "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set - skipping")
        return

    section("Langfuse")
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        trace = client.trace(name="bootstrap_check", input={"source": "bootstrap_cloud.py"})
        trace.update(output={"status": "ok"})
        client.flush()
        ok("Langfuse", "trace probe sent")
    except Exception as exc:
        error("Langfuse", str(exc))


def smoke_test(collection: str) -> None:
    section("Smoke Test")

    if results.get("Qdrant", ("FAIL", ""))[0] != "PASS":
        warn("Smoke Test", "skipped because Qdrant is not ready")
        return
    if results.get("Redis", ("FAIL", ""))[0] != "PASS":
        warn("Smoke Test", "skipped because Redis is not ready")
        return
    if results.get("Gemini", ("FAIL", ""))[0] == "FAIL":
        warn("Smoke Test", "skipped because Gemini probe failed")
        return
    if results.get("Groq", ("FAIL", ""))[0] == "FAIL":
        warn("Smoke Test", "skipped because Groq probe failed")
        return

    try:
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), check_version=False)
        cache = redis.Redis.from_url(os.getenv("REDIS_URL", ""), decode_responses=True)
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        embed_fn = getattr(genai, "embed_content")

        started = time.time()
        embedded = embed_fn(
            model=os.getenv("EMBED_MODEL", "models/text-embedding-004"),
            content="What is retrieval augmented generation?",
            task_type="retrieval_query",
            output_dimensionality=int(os.getenv("EMBED_DIM", "768")),
        )
        vector = embedded["embedding"]

        query_points_fn = getattr(qdrant, "query_points")
        response = query_points_fn(
            collection_name=collection,
            query=vector,
            limit=5,
            with_payload=True,
        )
        hits = response.points

        completion = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": "In one sentence, what is RAG in AI?"}],
            max_tokens=60,
        )
        answer = (completion.choices[0].message.content or "").strip()

        cache.setex("rag:bootstrap:smoke", 300, json.dumps({"answer": answer}))
        total_ms = round((time.time() - started) * 1000)
        ok("Smoke Test", f"embed/search/generate/cache passed in {total_ms}ms, hits={len(hits)}")
    except Exception as exc:
        error("Smoke Test", str(exc))


def print_summary() -> None:
    section("Bootstrap Summary")
    if HAS_RICH:
        table = cast(Any, rich_table_factory)(show_header=True, header_style="bold")
        table.add_column("Service")
        table.add_column("Status")
        table.add_column("Detail")
        for service, (status, detail) in results.items():
            table.add_row(service, status, detail)
        console.print(table)
    else:
        for service, (status, detail) in results.items():
            console.print(f"{service}: {status} - {detail}")


def main() -> None:
    """Run lightweight connectivity and collection-creation checks."""

    load_dotenv()
    console.print(f"Bootstrap started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    required = [
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "REDIS_URL",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        fail(f"Missing required env vars: {', '.join(missing)}")

    collection = os.getenv("COLLECTION", "rag_docs")
    embed_dim = int(os.getenv("EMBED_DIM", "768"))

    check_qdrant(collection, embed_dim)
    check_redis()
    check_metadata_db()
    check_gemini(embed_dim)
    check_groq()
    check_cohere()
    check_langfuse()
    smoke_test(collection)
    print_summary()
    console.print("Cloud bootstrap finished.")


if __name__ == "__main__":
    main()

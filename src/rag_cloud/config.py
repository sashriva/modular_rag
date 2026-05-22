import os
from dataclasses import dataclass

from dotenv import load_dotenv


"""Configuration module for the RAG application.

This is the first file to read because every runtime decision flows from the
environment-backed Settings dataclass defined here.
"""


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """All environment-driven settings used across ingestion and query flows."""

    collection: str = os.getenv("COLLECTION", "rag_docs")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

    cache_ttl_exact: int = int(os.getenv("CACHE_TTL_EXACT", "86400"))
    cache_ttl_embed: int = int(os.getenv("CACHE_TTL_EMBED", "259200"))
    cache_ttl_semantic: int = int(os.getenv("CACHE_TTL_SEMANTIC", "21600"))

    qdrant_url: str = os.getenv("QDRANT_URL", "https://YOUR-CLUSTER.cloud.qdrant.io")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    redis_url: str = os.getenv("REDIS_URL", "rediss://default:YOUR_PASSWORD@YOUR_ENDPOINT.upstash.io:6379")

    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "models/text-embedding-004")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    cohere_rerank_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

    database_url: str = os.getenv("DATABASE_URL", "")

    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))


def get_settings() -> Settings:
    """Return a fresh Settings object populated from environment variables."""

    return Settings()

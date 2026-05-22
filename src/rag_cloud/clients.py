from __future__ import annotations

from typing import Any, cast

import google.generativeai as genai
import redis
from cohere import ClientV2 as CohereClient
from groq import Groq
from langfuse import Langfuse
from qdrant_client import QdrantClient

from rag_cloud.config import Settings


"""Service client factory.

This module is the boundary between local Python code and external providers.
Every higher-level module depends on this one for network access.
"""


class Clients:
    """Construct and hold reusable clients for all external services."""

    def __init__(self, settings: Settings):
        self.settings = settings

        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)

        self.qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.groq = Groq(api_key=settings.groq_api_key)
        cohere_client = cast(Any, CohereClient)
        self.cohere = cohere_client(api_key=settings.cohere_api_key) if settings.cohere_api_key else None
        self.langfuse = (
            Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            if settings.langfuse_public_key and settings.langfuse_secret_key
            else None
        )

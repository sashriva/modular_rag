"""FastAPI entrypoint.

This is the HTTP wrapper around the core pipeline. It contains almost no domain
logic by design; all retrieval and generation behavior lives in pipeline.py.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_cloud.clients import Clients
from rag_cloud.config import get_settings
from rag_cloud.pipeline import RAGPipeline


settings = get_settings()
clients = Clients(settings)
pipeline = RAGPipeline(clients, settings)

app = FastAPI(title="Cloud RAG (Free Tier)")


class QueryRequest(BaseModel):
    """Shape of one incoming HTTP query request."""

    query: str = Field(min_length=2)
    tenant_id: str = Field(default="default")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    """Forward one HTTP request into the shared RAG pipeline instance."""

    try:
        return pipeline.query_dict(user_query=req.query, tenant_id=req.tenant_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

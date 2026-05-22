from __future__ import annotations

"""Optional Postgres/Neon persistence for query telemetry and ingestion metadata."""

from typing import Any
from contextlib import contextmanager

from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import Json


class MetadataStore:
    """Thin Postgres wrapper used by runtime and ingestion scripts.

    The store is intentionally optional. If DATABASE_URL is missing, callers
    should skip constructing this class.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=self.database_url, sslmode="require")
        self.init_schema()

    @contextmanager
    def _connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def init_schema(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_query_logs (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        tenant_id TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        cache_layer TEXT NOT NULL,
                        retrieval_count INTEGER NOT NULL DEFAULT 0,
                        selected_count INTEGER NOT NULL DEFAULT 0,
                        answer_chars INTEGER NOT NULL DEFAULT 0,
                        latency_ms INTEGER NOT NULL,
                        error_text TEXT,
                        trace_id TEXT,
                        source_names JSONB NOT NULL DEFAULT '[]'::jsonb
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_ingested_files (
                        id BIGSERIAL PRIMARY KEY,
                        source_name TEXT NOT NULL,
                        chunks_indexed INTEGER NOT NULL,
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(source_name)
                    );
                    """
                )

    def log_query(
        self,
        *,
        tenant_id: str,
        query_text: str,
        cache_layer: str,
        retrieval_count: int,
        selected_count: int,
        answer_chars: int,
        latency_ms: int,
        error_text: str | None = None,
        trace_id: str | None = None,
        source_names: list[str] | None = None,
    ) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_query_logs (
                        tenant_id,
                        query_text,
                        cache_layer,
                        retrieval_count,
                        selected_count,
                        answer_chars,
                        latency_ms,
                        error_text,
                        trace_id,
                        source_names
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tenant_id,
                        query_text,
                        cache_layer,
                        retrieval_count,
                        selected_count,
                        answer_chars,
                        latency_ms,
                        error_text,
                        trace_id,
                        Json(source_names or []),
                    ),
                )

    def upsert_ingested_file(self, source_name: str, chunks_indexed: int) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_ingested_files (source_name, chunks_indexed)
                    VALUES (%s, %s)
                    ON CONFLICT (source_name)
                    DO UPDATE SET
                        chunks_indexed = EXCLUDED.chunks_indexed,
                        indexed_at = NOW()
                    """,
                    (source_name, chunks_indexed),
                )

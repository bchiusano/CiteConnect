from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from sqlalchemy import text


@dataclass(frozen=True)
class PostgresStatus:
    has_documents_table: bool
    has_chunks_table: bool
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class ElasticsearchStatus:
    ok: bool
    index: str
    indexed_count: int
    error: Optional[str] = None


def check_postgres_status(engine) -> PostgresStatus:
    """
    Check whether required tables exist and how many rows they contain.

    Returns a PostgresStatus object. Raises exceptions if the DB connection fails.
    (Let the caller decide how to handle errors; no prints here.)
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('documents', 'chunks')
        """)).fetchall()

        existing = {r[0] for r in rows}
        has_docs = "documents" in existing
        has_chunks = "chunks" in existing

        if not (has_docs and has_chunks):
            # Tables missing -> counts are zero by definition
            return PostgresStatus(
                has_documents_table=has_docs,
                has_chunks_table=has_chunks,
                document_count=0,
                chunk_count=0,
            )

        doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
        chunk_count = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0

        return PostgresStatus(
            has_documents_table=True,
            has_chunks_table=True,
            document_count=int(doc_count),
            chunk_count=int(chunk_count),
        )


def check_elasticsearch_status(
    es_url: str,
    index: str,
    *,
    timeout_s: int = 3
) -> ElasticsearchStatus:
    """
    Optional ES status check.

    - If the `elasticsearch` package is not installed, returns ok=False with error.
    - If ES is unreachable or index does not exist, returns ok=False with error.
    - No prints; returns structured status.

    Note: This is intentionally separate from the main RAG pipeline. Keep ES optional.
    """
    try:
        from elasticsearch import Elasticsearch  # optional dependency
    except Exception as e:
        return ElasticsearchStatus(
            ok=False,
            index=index,
            indexed_count=0,
            error=f"elasticsearch package not available: {e}",
        )

    try:
        es = Elasticsearch(es_url, request_timeout=timeout_s)

        if not es.ping():
            return ElasticsearchStatus(
                ok=False,
                index=index,
                indexed_count=0,
                error="ping failed",
            )

        if not es.indices.exists(index=index):
            return ElasticsearchStatus(
                ok=False,
                index=index,
                indexed_count=0,
                error="index does not exist",
            )

        count_resp: Dict[str, Any] = es.count(index=index)
        cnt = int(count_resp.get("count", 0))

        return ElasticsearchStatus(ok=True, index=index, indexed_count=cnt)

    except Exception as e:
        return ElasticsearchStatus(
            ok=False,
            index=index,
            indexed_count=0,
            error=str(e),
        )

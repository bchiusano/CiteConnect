from __future__ import annotations

from sqlalchemy import create_engine, text
from .config import PG_DSN

# If you ever change embedding model dimension, update this constant + re-embed.
EMBED_DIM = 1024


def get_engine():
    return create_engine(PG_DSN)


def setup_schema(engine) -> None:
    """
    Create tables + indexes for the "rich schema" pipeline.

    documents:
      - doc_id (string id, often ECLI or UUID) UNIQUE
      - source/doc_type/published_at/title/language/raw_metadata/text

    chunks:
      - chunk_id UNIQUE
      - doc_id FK -> documents(doc_id)
      - idx (chunk index within doc)
      - language/section_title/text/embedding
      - (optional) char_start/char_end (kept for compatibility; you can remove if unused)
    """
    with engine.begin() as conn:
        # pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # ---- documents ----
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                source TEXT,
                doc_type TEXT,
                published_at TIMESTAMPTZ,
                title TEXT,
                language TEXT,
                raw_metadata JSONB,
                text TEXT
            );
        """))

        # helpful index for frequent lookups by doc_id (unique already implies index, but explicit is fine)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_doc_id
            ON documents (doc_id);
        """))

        # ---- chunks ----
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                idx INTEGER,
                language TEXT,
                section_title TEXT,
                text TEXT,
                embedding VECTOR({EMBED_DIM}),
                char_start INTEGER,
                char_end INTEGER
            );
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
            ON chunks (doc_id);
        """))

        # Vector index (ivfflat) for cosine similarity
        # NOTE: ivfflat requires setting lists; defaults are OK to start.
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
            ON chunks
            USING ivfflat (embedding vector_cosine_ops);
        """))

        # Optional: full-text search index on chunk text (useful for hybrid retrieval)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
            ON chunks
            USING GIN (to_tsvector('simple', coalesce(text, '')));
        """))

from __future__ import annotations
import json
from typing import Dict, List
from sqlalchemy import text

from .embedding import embed_texts


def upsert_document(conn, doc: Dict) -> None:
    conn.execute(text("""
        INSERT INTO documents (doc_id, source, doc_type, published_at, title, language, raw_metadata, text)
        VALUES (:doc_id, :source, :doc_type, :published_at, :title, :language, CAST(:raw_metadata AS JSONB), :text)
        ON CONFLICT (doc_id) DO NOTHING
    """), {**doc, "raw_metadata": json.dumps(doc.get("raw_metadata", {}))})


def insert_chunks(conn, chunk_rows: List[Dict], *, embedder) -> None:
    texts = [c["text"] for c in chunk_rows]
    embs = embed_texts(embedder, texts)

    for c, v in zip(chunk_rows, embs):
        c["embedding"] = v

    stmt = text("""
        INSERT INTO chunks (chunk_id, doc_id, idx, language, section_title, text, embedding)
        VALUES (:chunk_id, :doc_id, :idx, :language, :section_title, :text, :embedding)
        ON CONFLICT (chunk_id) DO NOTHING
    """)

    # SQLAlchemy can insert many rows at once
    conn.execute(stmt, chunk_rows)



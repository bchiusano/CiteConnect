from __future__ import annotations
from typing import Optional, Tuple
from sqlalchemy import text


def clear_chunks(engine, *, confirm: bool = False) -> Optional[int]:
    if not confirm:
        return None
    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
        conn.execute(text("DELETE FROM chunks"))
        return int(n)


def clear_all(engine, *, confirm: bool = False) -> Optional[Tuple[int, int]]:
    if not confirm:
        return None
    with engine.begin() as conn:
        doc_n = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
        chunk_n = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
        conn.execute(text("DELETE FROM chunks"))
        conn.execute(text("DELETE FROM documents"))
        return int(doc_n), int(chunk_n)

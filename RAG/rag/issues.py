from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from .embedding import embed_texts


@dataclass(frozen=True)
class Issue:
    issue_id: str
    name: str
    description: str
    keywords: List[str]


DEFAULT_ISSUES: List[Issue] = [
    Issue(
        issue_id="procedural",
        name="Procedural irregularities",
        description="Complaints about enforcement procedure, notification, proportionality, due process.",
        keywords=["procedure", "procedural", "zorgvuldigheid", "motivering", "hoorplicht", "bezwaarprocedure",
                  "proportionaliteit", "evenredigheid", "kennisgeving", "termijn", "besluit"]
    ),
    Issue(
        issue_id="costs",
        name="Costs & reimbursement",
        description="Requests for reimbursement of costs, legal costs, fees, towing/storage costs.",
        keywords=["kosten", "proceskosten", "vergoeding", "griffierecht", "restitut", "vergoeding",
                  "sleepkosten", "stallingskosten", "kostenverhaal"]
    ),
    Issue(
        issue_id="substance",
        name="Substantive grounds",
        description="Substantive legal grounds: legality of removal, parking rules, factual dispute about the bicycle.",
        keywords=["fiets", "stalling", "parkeren", "verwijderd", "weesfiets", "verbod", "borden", "feitelijk"]
    ),
]


def upsert_issue_index(conn, issues: List[Issue], *, embedder) -> None:
    """Insert/update issues and their embeddings."""
    texts = [f"{it.name}\n{it.description}\nKeywords: {', '.join(it.keywords)}" for it in issues]
    embs = embed_texts(embedder, texts)

    rows = []
    for it, emb in zip(issues, embs):
        rows.append({
            "issue_id": it.issue_id,
            "name": it.name,
            "description": it.description,
            "keywords": json.dumps(it.keywords),
            "embedding": emb,
        })

    conn.execute(text("""
        INSERT INTO issue_index (issue_id, name, description, keywords, embedding)
        VALUES (:issue_id, :name, :description, CAST(:keywords AS JSONB), :embedding)
        ON CONFLICT (issue_id) DO UPDATE SET
          name = EXCLUDED.name,
          description = EXCLUDED.description,
          keywords = EXCLUDED.keywords,
          embedding = EXCLUDED.embedding
    """), rows)


def load_issue_index(conn) -> List[Dict]:
    return conn.execute(text("""
        SELECT issue_id, name, description, keywords, embedding
        FROM issue_index
    """)).mappings().all()


def predict_issues(conn, text_input: str, *, embedder, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Return [(issue_id, score), ...] score in [0,1] (rough).
    Hybrid: keyword hits + embedding similarity.
    """
    issues = load_issue_index(conn)
    if not issues:
        return []

    q_emb = embed_texts(embedder, [text_input])[0]

    # Dense similarity against issue embeddings using pgvector cosine distance (<=>),
    # then combine dense score with simple keyword hit count in Python.
    text_lower = text_input.lower()

    dense_scores: Dict[str, float] = {}
    for it in issues:
        # it["embedding"] is returned as list by pgvector driver
        # We'll approximate similarity by dot / norms here is expensive; instead do SQL is better.
        dense_scores[it["issue_id"]] = 0.0

    rows = conn.execute(text("""
        SELECT issue_id,
               1 - (embedding <=> (:qvec)::vector) AS sim
        FROM issue_index
        ORDER BY embedding <=> (:qvec)::vector
        LIMIT :k
    """), {"qvec": q_emb, "k": max(top_k, 5)}).mappings().all()

    for r in rows:
        dense_scores[r["issue_id"]] = float(r["sim"])

    # Keyword score
    keyword_scores: Dict[str, float] = {}
    for it in issues:
        kws = it["keywords"]
        if isinstance(kws, str):
            try:
                kws = json.loads(kws)
            except Exception:
                kws = []
        hits = sum(1 for kw in kws if kw and kw.lower() in text_lower)
        keyword_scores[it["issue_id"]] = min(1.0, hits / 5.0)  # 5 hits => 1.0

    # Combine
    combined: List[Tuple[str, float]] = []
    for it in issues:
        iid = it["issue_id"]
        sim = dense_scores.get(iid, 0.0)
        kw = keyword_scores.get(iid, 0.0)
        score = 0.7 * sim + 0.3 * kw
        combined.append((iid, float(score)))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]

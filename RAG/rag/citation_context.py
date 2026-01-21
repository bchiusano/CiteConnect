from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from sqlalchemy import text

from .embedding import embed_texts
from .utils import norm_ecli 


_ECLI_PATTERN = re.compile(r"ECLI:[A-Z]{2}:[A-Z0-9]{4}:[A-Z0-9]+", re.IGNORECASE)


def extract_citation_contexts(advice_text: str, ecli_list: List[str], *, window_chars: int = 800) -> List[Tuple[str, str]]:
    """
    Return [(ecli, context_text), ...]
    Strategy: find occurrences of each ECLI string in advice_text; take +/- window_chars.
    If not found, skip.
    """
    out = []
    if not advice_text:
        return out
    text_upper = advice_text.upper()

    for ecli in ecli_list:
        ne = norm_ecli(ecli)
        if not ne:
            continue
        idx = text_upper.find(ne)
        if idx == -1:
            continue
        start = max(0, idx - window_chars)
        end = min(len(advice_text), idx + len(ne) + window_chars)
        ctx = advice_text[start:end].strip()
        if ctx:
            out.append((ne, ctx))
    return out


def build_citation_context_prototypes(
    conn,
    *,
    embedder,
    issue_predictor,  # callable(conn, text, embedder=...) -> [(issue_id, score)]
    max_protos_per_ecli: int = 50,
    train_ids: Optional[Set[str]] = None,
) -> int:
    """
    Build prototypes from advice letters in DB, using ground-truth citations stored in documents.raw_metadata or separate table.
    
    **IMPORTANT**: To prevent data leakage, only use training set data.
    Pass train_ids to filter advice letters to training set only.
    
    Assumptions:
    - advice docs in documents where doc_type='advice'
    - their raw_metadata contains 'ECLI' (list or string) OR you have a helper to load ground truth.
    
    Parameters:
    -----------
    train_ids : Set[str], optional
        Set of training advice IDs. If provided, only builds prototypes from
        training set to prevent data leakage. If None, uses ALL data (⚠️ data leakage risk!)
    """
    # Build query - filter to training set if train_ids provided
    if train_ids is not None:
        advice_rows = conn.execute(text("""
            SELECT doc_id, text, raw_metadata
            FROM documents
            WHERE doc_type = 'advice'
              AND doc_id = ANY(:train_ids)
        """), {"train_ids": list(train_ids)}).mappings().all()
    else:
        advice_rows = conn.execute(text("""
            SELECT doc_id, text, raw_metadata
            FROM documents
            WHERE doc_type = 'advice'
        """)).mappings().all()

    inserted = 0

    for adv in advice_rows:
        adv_id = str(adv["doc_id"])
        adv_text = adv["text"] or ""
        meta = adv.get("raw_metadata") or {}

        # Try to get ecli list from meta['ECLI'] if exists
        ecli_list: List[str] = []
        if isinstance(meta, dict) and "ECLI" in meta and meta["ECLI"]:
            val = meta["ECLI"]
            if isinstance(val, list):
                ecli_list = [str(x) for x in val]
            elif isinstance(val, str):
                # Handle string representation of list (e.g., "['ECLI:NL:...']")
                import ast
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        ecli_list = [str(x) for x in parsed]
                    else:
                        ecli_list = [str(parsed)]
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single ECLI string
                    ecli_list = [str(val)]
            else:
                ecli_list = [str(val)]
        else:
            continue

        ecli_list = [norm_ecli(x) for x in ecli_list]
        ecli_list = [x for x in ecli_list if x]

        if not ecli_list:
            continue

        ctx_pairs = extract_citation_contexts(adv_text, ecli_list, window_chars=800)
        if not ctx_pairs:
            continue

        # Predict issue for each context (use top-1 issue)
        ctx_texts = [ctx for _, ctx in ctx_pairs]
        ctx_embs = embed_texts(embedder, ctx_texts)

        rows = []
        for (ecli, ctx), emb in zip(ctx_pairs, ctx_embs):
            issues = issue_predictor(conn, ctx, embedder=embedder, top_k=1)
            issue_id = issues[0][0] if issues else None

            rows.append({
                "ecli": ecli,
                "source_doc_id": adv_id,
                "issue_id": issue_id,
                "text": ctx,
                "embedding": emb,
            })

        # Limit per ECLI
        conn.execute(text("""
            INSERT INTO citation_context_prototypes (ecli, source_doc_id, issue_id, text, embedding)
            VALUES (:ecli, :source_doc_id, :issue_id, :text, :embedding)
        """), rows)
        inserted += len(rows)

    return inserted


def search_citation_context(
    conn,
    query_text: str,
    *,
    embedder,
    k: int = 50
) -> List[Dict]:
    """
    Return top prototypes by similarity.
    Output rows contain: ecli, issue_id, sim, text
    """
    q_emb = embed_texts(embedder, [query_text])[0]
    rows = conn.execute(text("""
        SELECT proto_id, ecli, issue_id, text,
               1 - (embedding <=> (:qvec)::vector) AS sim
        FROM citation_context_prototypes
        ORDER BY embedding <=> (:qvec)::vector
        LIMIT :k
    """), {"qvec": q_emb, "k": k}).mappings().all()

    return [{
        "proto_id": r["proto_id"],
        "ecli": r["ecli"],
        "issue_id": r["issue_id"],
        "sim": float(r["sim"]),
        "text": r["text"],
    } for r in rows]


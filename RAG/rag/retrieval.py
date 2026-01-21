from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import text

from .embedding import embed_texts
from .citation_context import search_citation_context as _search_citation_context
from .issues import predict_issues
from .prior import get_popular_ecli
from .config import USE_CITATION_CONTEXT


def _maybe_search_citation_context(conn, advice_text: str, *, embedder, k: int):
    """Wrapper that can disable citation-context retrieval via config.

    When USE_CITATION_CONTEXT=0, this returns an empty list so that
    all downstream citation-based boosts are effectively turned off
    without touching the rest of the retrieval logic.
    """
    if not USE_CITATION_CONTEXT:
        return []
    return _search_citation_context(conn, advice_text, embedder=embedder, k=k)


def _rrf_fuse(rank_a: Dict[str, int], rank_b: Dict[str, int], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion:
    score(doc) = sum(1 / (k + rank_i(doc)))
    ranks are 1-based
    """
    scores = defaultdict(float)
    for cid, r in rank_a.items():
        scores[cid] += 1.0 / (k + r)
    for cid, r in rank_b.items():
        scores[cid] += 1.0 / (k + r)
    return dict(scores)


def dense_search_chunks(conn, query_text: str, *, embedder, k: int = 200) -> List[Dict]:
    q_emb = embed_texts(embedder, [query_text])[0]
    rows = conn.execute(text("""
        SELECT c.chunk_id, c.doc_id AS ecli, c.text,
               1 - (c.embedding <=> (:qvec)::vector) AS sim
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE d.doc_type = 'ecli'
        ORDER BY c.embedding <=> (:qvec)::vector
        LIMIT :k
    """), {"qvec": q_emb, "k": k}).mappings().all()

    return [{
        "chunk_id": r["chunk_id"],
        "ecli": r["ecli"],
        "text": r["text"],
        "score_dense": float(r["sim"]),
    } for r in rows]


def bm25_search_chunks(conn, query_text: str, *, k: int = 200) -> List[Dict]:
    # Dutch FTS: plainto_tsquery is robust; if you want phrase matching, use phraseto_tsquery.
    rows = conn.execute(text("""
        SELECT c.chunk_id, c.doc_id AS ecli, c.text,
               ts_rank_cd(c.text_tsv, plainto_tsquery('dutch', :q)) AS bm25
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE d.doc_type = 'ecli'
          AND c.text_tsv @@ plainto_tsquery('dutch', :q)
        ORDER BY bm25 DESC
        LIMIT :k
    """), {"q": query_text, "k": k}).mappings().all()

    return [{
        "chunk_id": r["chunk_id"],
        "ecli": r["ecli"],
        "text": r["text"],
        "score_bm25": float(r["bm25"]),
    } for r in rows]


def hybrid_retrieve_chunks(
    conn,
    query_text: str,
    *,
    embedder,
    k_dense: int = 200,
    k_bm25: int = 200,
    k_final: int = 300,
) -> List[Dict]:
    """
    Hybrid retrieval over chunks.

    NOTE: BM25 / FTS requires an additional `text_tsv` column + index on `chunks`.
    To keep the system robust when that column is missing, we currently fall back
    to dense-only retrieval.
    """
    dense = dense_search_chunks(conn, query_text, embedder=embedder, k=k_dense)

    # Try BM25; fall back gracefully if the FTS column/index is missing.
    try:
        bm25 = bm25_search_chunks(conn, query_text, k=k_bm25)
    except Exception:
        # Important: if the BM25 query fails (e.g. missing text_tsv column),
        # PostgreSQL marks the current transaction as aborted. We must roll
        # back here, otherwise any later queries on this connection (such as
        # citation_context or issues) will see InFailedSqlTransaction.
        try:
            conn.rollback()
        except Exception:
            pass
        bm25 = []

    if not bm25:
        # Dense-only mode: keep API compatible by setting score_hybrid from dense score.
        for r in dense:
            r["score_hybrid"] = r.get("score_dense", 0.0)
        dense.sort(key=lambda x: x["score_hybrid"], reverse=True)
        return dense[:k_final]

    # build ranks
    rank_dense = {r["chunk_id"]: i + 1 for i, r in enumerate(dense)}
    rank_bm25 = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25)}

    fused = _rrf_fuse(rank_dense, rank_bm25, k=60)
    # merge metadata
    by_id: Dict[str, Dict] = {}
    for r in dense:
        by_id[r["chunk_id"]] = {**r}
    for r in bm25:
        if r["chunk_id"] not in by_id:
            by_id[r["chunk_id"]] = {**r}
        else:
            by_id[r["chunk_id"]].update(r)

    out = []
    for cid, s in fused.items():
        rec = by_id.get(cid, {"chunk_id": cid})
        rec["score_hybrid"] = float(s)
        out.append(rec)

    out.sort(key=lambda x: x["score_hybrid"], reverse=True)
    return out[:k_final]


def aggregate_best_evidence_by_ecli(chunk_hits: List[Dict], *, score_key: str = "score") -> List[Dict]:
    best: Dict[str, Dict] = {}
    for h in chunk_hits:
        ecli = h.get("ecli")
        if not ecli:
            continue
        if ecli not in best or h.get(score_key, 0.0) > best[ecli].get(score_key, 0.0):
            best[ecli] = h
    out = list(best.values())
    out.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)
    return out


def retrieve_ecli(
    conn,
    advice_text: str,
    *,
    embedder,
    reranker=None,         # FlagReranker or None
    top_ecli: int = 10,
    retrieve_chunks: int = 300,
    proto_k: int = 50,
    train_ids: Optional[Set[str]] = None,  # For data leakage prevention
) -> List[Dict]:
    """
    End-to-end:
    - hybrid retrieve chunks
    - add citation-context boost
    - issue-aware boost
    - (optional) rerank chunks
    - aggregate by ECLI
    """
    # 1) Issue prediction for the whole letter (optional)
    try:
        issue_scores = predict_issues(conn, advice_text, embedder=embedder, top_k=3)  # [(issue_id, score)]
        issue_score_map = {iid: s for iid, s in issue_scores}
    except Exception:
        # If the issue_index table or related infra is missing, skip issue-aware boosts.
        # Important: roll back the failed transaction so later queries can run.
        try:
            conn.rollback()
        except Exception:
            pass
        issue_score_map = {}

    # 2) Hybrid retrieve chunks
    chunk_hits = hybrid_retrieve_chunks(
        conn, advice_text, embedder=embedder, k_dense=200, k_bm25=200, k_final=retrieve_chunks
    )

    # Start with score = score_hybrid (RRF). We'll add boosts on top.
    for h in chunk_hits:
        h["score"] = float(h.get("score_hybrid", 0.0))

    # 3) Citation-context prototypes retrieval (optional)
    try:
        proto_hits = _maybe_search_citation_context(conn, advice_text, embedder=embedder, k=proto_k)
    except Exception:
        # If citation_context_prototypes table is missing, skip this boost.
        # Roll back the failed transaction to keep the connection usable.
        try:
            conn.rollback()
        except Exception:
            pass
        proto_hits = []

    # Convert proto hits into ECLI boosts
    # idea: if prototype very similar, boost that ECLI
    proto_best_sim: Dict[str, float] = {}
    proto_issue: Dict[str, str] = {}
    proto_best_text: Dict[str, str] = {}
    for p in proto_hits:
        ecli = p["ecli"]
        sim = float(p["sim"])
        if ecli not in proto_best_sim or sim > proto_best_sim[ecli]:
            proto_best_sim[ecli] = sim
            proto_issue[ecli] = p.get("issue_id")
            txt = p.get("text") or ""
            proto_best_text[ecli] = txt

    # Check which ECLIs from prototypes are NOT in current chunk_hits
    existing_eclis = {h.get("ecli") for h in chunk_hits if h.get("ecli")}
    missing_eclis = {ecli: sim for ecli, sim in proto_best_sim.items() 
                     if ecli not in existing_eclis and sim > 0.5}  # Only add if similarity is decent
    
    # For missing ECLIs with good prototype similarity, fetch their best chunk
    if missing_eclis:
        q_emb = embed_texts(embedder, [advice_text])[0]
        for ecli, proto_sim in missing_eclis.items():
            # Get best chunk for this ECLI
            try:
                rows = conn.execute(text("""
                    SELECT c.chunk_id, c.doc_id AS ecli, c.text,
                           1 - (c.embedding <=> (:qvec)::vector) AS sim
                    FROM chunks c
                    WHERE c.doc_id = :ecli
                    ORDER BY c.embedding <=> (:qvec)::vector
                    LIMIT 1
                """), {"qvec": q_emb, "ecli": ecli}).mappings().all()
                
                if rows:
                    r = rows[0]
                    base_score = float(r["sim"])
                    chunk_hit = {
                        "chunk_id": r["chunk_id"],
                        "ecli": r["ecli"],
                        "text": r["text"],
                        "score_dense": base_score,
                        "score_hybrid": base_score,  # Use dense score as hybrid score
                        "score": base_score,  # Start with base score, will be boosted below
                    }
                    chunk_hits.append(chunk_hit)
            except Exception as e:
                # If query fails, skip this ECLI
                # Silently continue to avoid breaking the retrieval
                continue

    # Boost existing candidates (and newly added ones)
    for h in chunk_hits:
        ecli = h.get("ecli")
        if not ecli:
            continue
        if ecli in proto_best_sim:
            # boost strength: increased to 0.5 for better visibility of citation contexts
            # This helps ECLIs that are frequently cited but have lower semantic similarity
            # Higher boost ensures they can compete with high semantic similarity matches
            # For very high similarity (>0.75), apply even stronger boost
            sim = proto_best_sim[ecli]
            if sim > 0.75:
                h["score"] += 0.65 * sim  # Very strong boost for high citation similarity (increased from 0.6)
            else:
                h["score"] += 0.5 * sim  # Strong boost for moderate citation similarity

    # 4) Issue-aware boost using prototype issue alignment
    # If the ECLI prototype issue matches predicted top issues -> boost
    for h in chunk_hits:
        ecli = h.get("ecli")
        if not ecli:
            continue
        iid = proto_issue.get(ecli)
        if iid and iid in issue_score_map:
            h["score"] += 0.10 * issue_score_map[iid]  # small prior

    # 5) Optional rerank (cross-encoder) on top N chunks
    if reranker is not None:
        # choose top 80 chunks for rerank
        candidates = sorted(chunk_hits, key=lambda x: x["score"], reverse=True)[:80]
        
        citation_candidates = []
        normal_candidates = []
        citation_indices = []
        normal_indices = []
        
        for i, c in enumerate(candidates):
            ecli = c.get("ecli")
            if ecli and ecli in proto_best_sim and proto_best_sim[ecli] > 0.75:
                citation_candidates.append(c)
                citation_indices.append(i)
            else:
                normal_candidates.append(c)
                normal_indices.append(i)
        
        # Only rerank normal candidates
        if normal_candidates:
            pairs = []
            for c in normal_candidates:
                doc_text = c.get("text", "") or ""
                ecli_cand = c.get("ecli")
                ctx = None
                if ecli_cand and ecli_cand in proto_best_text:
                    ctx = proto_best_text[ecli_cand]
                if ctx:
                    doc_text = (
                        "[CITATION CONTEXT]\n"
                        + str(ctx).strip()
                        + "\n\n[DECISION CHUNK]\n"
                        + doc_text
                    )
                doc_text = doc_text[:2000]
                pairs.append([advice_text, doc_text])

            try:
                rr = reranker.compute_score(pairs, normalize=True)
                if not isinstance(rr, list):
                    rr = [float(rr)]
                for c, s in zip(normal_candidates, rr):
                    # combine: keep original score, add rerank strong signal
                    c["score_rerank"] = float(s)
                    ecli = c.get("ecli")
                    if ecli and ecli in proto_best_sim and proto_best_sim[ecli] > 0.7:
                        # High citation similarity: 70% original, 30% rerank
                        c["score"] = 0.7 * c["score"] + 0.3 * float(s)
                    else:
                        # Normal: 30% original, 70% rerank
                        c["score"] = 0.3 * c["score"] + 0.7 * float(s)
            except Exception:
                # fail silently
                pass
        
        # Citation-boosted candidates keep their original scores (no reranking)
        # This preserves the citation boost advantage

        # write back (candidates are dict refs in chunk_hits)
        chunk_hits.sort(key=lambda x: x["score"], reverse=True)

    # 6) Aggregate by ECLI (best evidence per ecli)
    ecli_hits = aggregate_best_evidence_by_ecli(chunk_hits, score_key="score")

    # 6a) Popularity prior: use ground-truth citation frequency as a small prior
    # so that very frequently cited ECLI get a gentle boost everywhere.
    # IMPORTANT: Only use training set to prevent data leakage
    try:
        popular_list = get_popular_ecli(min_citations=5, train_ids=train_ids)
        pop_counts: Dict[str, int] = {e: c for e, c in popular_list}
        max_pop = max(pop_counts.values()) if pop_counts else 0
    except Exception:
        pop_counts = {}
        max_pop = 0

    if max_pop > 0:
        for h in ecli_hits:
            ecli = h.get("ecli")
            if not ecli:
                continue
            cnt = pop_counts.get(ecli, 0)
            if cnt <= 0:
                continue
            # Normalize and apply a small, concave prior
            # pop_norm in [0,1]; use sqrt to reduce dominance of extremely popular ECLI
            pop_norm = (cnt / max_pop) ** 0.5
            h["score"] += 0.12 * pop_norm  # up to ~0.12 extra for the most popular ones

    # 6b) Fallback: ensure strong citation-prototype ECLI can enter the list
    #
    # If an ECLI has a high citation-context similarity but no chunk made it
    # into the hybrid candidates (even after the injection above, e.g. due to
    # exceptions or very low dense similarity), we still want it to have a
    # chance to appear in the final ranking – especially in cases like
    # ECLI:NL:GHDHA:2014:2134 which is heavily cited.
    existing_eclis_final = {h.get("ecli") for h in ecli_hits if h.get("ecli")}
    fallback_records: List[Dict] = []
    for ecli, sim in proto_best_sim.items():
        if ecli in existing_eclis_final:
            continue
        # Only consider fairly strong prototype matches
        if sim < 0.7:
            continue

        # Use prototype similarity + popularity as a prior-style score. We scale it so that
        # these fallback entries are competitive but do not completely dominate
        # truly high-scoring semantic matches.
        base_prior = 0.75 * float(sim)

        # Add a small popularity prior if available
        pop_prior = 0.0
        if max_pop > 0 and ecli in pop_counts:
            cnt = pop_counts[ecli]
            pop_norm = (cnt / max_pop) ** 0.5
            pop_prior = 0.10 * pop_norm  # up to 0.10 extra

        prior_score = base_prior + pop_prior

        # We don't strictly need a chunk here for evaluation that only cares
        # about the ECLI; however, try to provide a text snippet if possible.
        snippet = ""
        try:
            row = conn.execute(
                text(
                    """
                    SELECT text
                    FROM documents
                    WHERE doc_id = :ecli
                    """
                ),
                {"ecli": ecli},
            ).scalar()
            if row:
                snippet = str(row)[:300]
        except Exception:
            # If anything goes wrong, we still keep the fallback entry without text.
            snippet = ""

        fallback_records.append(
            {
                "ecli": ecli,
                "score": prior_score,
                "text": snippet,
                "chunk_id": None,
                "score_dense": None,
                "score_bm25": None,
                "score_hybrid": None,
                "score_rerank": None,
            }
        )

    if fallback_records:
        # Merge fallback records, but guarantee that high-sim citation-based
        # ECLIs are present in the final top_ecli list, even if their score is
        # slightly lower than the weakest purely semantic hits.
        ecli_hits.extend(fallback_records)

        # Sort all by score descending
        ecli_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        fallback_ids = {r["ecli"] for r in fallback_records}
        fallback_list = [r for r in ecli_hits if r.get("ecli") in fallback_ids]
        non_fallback_list = [r for r in ecli_hits if r.get("ecli") not in fallback_ids]

        # Keep as many non-fallback hits as possible while always including
        # all fallback ECLIs (bounded by top_ecli).
        keep_non = max(0, top_ecli - len(fallback_list))
        trimmed_non_fallback = non_fallback_list[:keep_non]

        combined = trimmed_non_fallback + fallback_list
        combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        ecli_hits = combined[:top_ecli]
    else:
        # No fallback: just truncate as usual
        ecli_hits = ecli_hits[:top_ecli]

    # Output format
    results = []
    for r in ecli_hits:
        results.append({
            "ecli_number": r.get("ecli"),
            "score": float(r.get("score", 0.0)),
            "text_snippet": (r.get("text") or "")[:300],
            "chunk_id": r.get("chunk_id"),
            "score_dense": float(r.get("score_dense", 0.0)) if r.get("score_dense") is not None else None,
            "score_bm25": float(r.get("score_bm25", 0.0)) if r.get("score_bm25") is not None else None,
            "score_hybrid": float(r.get("score_hybrid", 0.0)) if r.get("score_hybrid") is not None else None,
            "score_rerank": float(r.get("score_rerank", 0.0)) if r.get("score_rerank") is not None else None,
        })
    return results

# rag/eval.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .ground_truth import GroundTruth, norm_ecli
from .metrics import PerQueryMetrics, aggregate, compute_per_query_metrics


@dataclass
class EvalRow:
    doc_id: str
    advice_text: str
    raw_metadata: Any


def _extract_zaaknummer_like(raw_metadata: Any) -> Optional[str]:
    """
    Try to extract zaaknummer from raw_metadata (dict or JSON string).
    Returns string or None.
    """
    if raw_metadata is None:
        return None

    meta = raw_metadata
    if isinstance(meta, str):
        s = meta.strip()
        if not s:
            return None
        try:
            meta = json.loads(s)
        except Exception:
            return None

    if isinstance(meta, dict):
        v = meta.get("Octopus zaaknummer") or meta.get("zaaknummer") or meta.get("Zaaknummer")
        if v is None:
            return None
        vv = str(v).strip()
        return vv if vv and vv.lower() not in {"nan", "none"} else None

    return None


def _expected_for_doc(gt: GroundTruth, doc_id: str, raw_metadata: Any) -> Optional[List[str]]:
    """
    Match ground truth by:
    1) doc_id
    2) zaaknummer extracted from raw_metadata
    """
    exp = gt.get(doc_id)
    if exp:
        return exp

    zk = _extract_zaaknummer_like(raw_metadata)
    if zk:
        exp = gt.get(zk)
        if exp:
            return exp

    return None


def fetch_advice_docs(engine: Engine, *, limit: Optional[int] = None) -> List[EvalRow]:
    """
    Fetch advice docs from DB.
    Assumes documents table has: doc_id, text, raw_metadata, doc_type.
    """
    q = """
        SELECT doc_id, text, raw_metadata
        FROM documents
        WHERE doc_type = 'advice'
        ORDER BY doc_id
    """
    if limit is not None:
        q += " LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(q), {"limit": limit} if limit is not None else {}).mappings().all()

    out: List[EvalRow] = []
    for r in rows:
        out.append(EvalRow(
            doc_id=str(r["doc_id"]),
            advice_text=r["text"] or "",
            raw_metadata=r.get("raw_metadata"),
        ))
    return out


RetrieveFn = Callable[[str, int], List[Dict[str, Any]]]


def _predicted_ecli_numbers(results: Sequence[Dict[str, Any]], *, top_k: int) -> List[str]:
    """
    Extract normalized ECLI identifiers from retrieval results.
    Expects dicts with key 'ecli_number' (preferred) or 'doc_id' as fallback.
    Preserves order, drops duplicates.
    """
    seen = set()
    out: List[str] = []

    for r in results[:top_k]:
        e = r.get("ecli_number") or r.get("doc_id")
        ne = norm_ecli(e)
        if ne and ne not in seen:
            seen.add(ne)
            out.append(ne)
    return out


def evaluate_all(
    *,
    engine: Engine,
    ground_truth: GroundTruth,
    retrieve_fn: RetrieveFn,
    top_k: int = 10,
    limit: Optional[int] = None,
    additional_ks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval function on advice letters that have ground truth.

    retrieve_fn signature: retrieve_fn(advice_text: str, top_k: int) -> List[dict]
    dict must contain 'ecli_number' (or 'doc_id') + 'score' optional.

    Args:
        top_k: Primary k value for evaluation (default: 10)
        additional_ks: Additional k values to evaluate (e.g., [5] to also compute @5)

    Returns:
      {
        "total_advice": int,
        "evaluated": int,
        "with_results": int,
        "precision": float,
        "recall": float,
        "f1": float,
        "mrr": float,
        "hit_rate_at_k": float,
        "detailed": [...],
        "metrics_at_k": {k: {...}}  # Additional metrics for other k values
      }
    """
    docs = fetch_advice_docs(engine, limit=limit)
    
    # Collect all k values to evaluate
    all_ks = [top_k]
    if additional_ks:
        all_ks.extend([k for k in additional_ks if k != top_k])
    max_k = max(all_ks)

    per_query: List[PerQueryMetrics] = []
    per_query_by_k: Dict[int, List[PerQueryMetrics]] = {k: [] for k in all_ks}
    detailed: List[Dict[str, Any]] = []
    with_results = 0
    evaluated = 0

    for d in docs:
        expected = _expected_for_doc(ground_truth, d.doc_id, d.raw_metadata)
        if not expected:
            continue

        expected_norm = {norm_ecli(x) for x in expected if norm_ecli(x)}
        if not expected_norm:
            continue

        # Retrieve once with max_k, then evaluate at different k values
        results = retrieve_fn(d.advice_text, max_k)
        predicted_ranked = _predicted_ecli_numbers(results, top_k=max_k)

        if predicted_ranked:
            with_results += 1

        # Compute metrics for primary top_k
        m = compute_per_query_metrics(expected_norm, predicted_ranked, k=top_k)
        per_query.append(m)
        evaluated += 1

        # Compute metrics for all k values
        for k in all_ks:
            m_k = compute_per_query_metrics(expected_norm, predicted_ranked, k=k)
            per_query_by_k[k].append(m_k)

        detailed.append({
            "doc_id": d.doc_id,
            "expected": sorted(expected_norm),
            "predicted": predicted_ranked[:top_k],  # Store top_k predictions
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "reciprocal_rank": m.reciprocal_rank,
            "hit_at_k": m.hit_at_k,
        })

    agg = aggregate(per_query)
    result = {
        "total_advice": len(docs),
        "evaluated": evaluated,
        "with_results": with_results,
        **agg,
        "detailed": detailed,
    }
    
    # Add metrics for additional k values
    if additional_ks:
        metrics_at_k = {}
        for k in additional_ks:
            if k != top_k and per_query_by_k[k]:
                metrics_at_k[k] = aggregate(per_query_by_k[k])
        if metrics_at_k:
            result["metrics_at_k"] = metrics_at_k
    
    return result
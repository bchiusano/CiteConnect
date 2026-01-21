# rag/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass
class PerQueryMetrics:
    precision: float
    recall: float
    f1: float
    reciprocal_rank: float
    hit_at_k: float


def precision_recall_f1(expected: Set[str], predicted: Set[str]) -> Tuple[float, float, float]:
    if not predicted:
        precision = 0.0
    else:
        precision = len(expected & predicted) / len(predicted)

    if not expected:
        recall = 0.0
    else:
        recall = len(expected & predicted) / len(expected)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def reciprocal_rank(expected: Set[str], predicted_ranked: Sequence[str]) -> float:
    """
    MRR contribution for a single query: 1/rank of first relevant result.
    """
    for i, item in enumerate(predicted_ranked, 1):
        if item in expected:
            return 1.0 / i
    return 0.0


def hit_at_k(expected: Set[str], predicted_ranked: Sequence[str], k: int) -> float:
    """
    Hit@K for a single query: 1 if any relevant in top K, else 0.
    """
    top = predicted_ranked[:k]
    return 1.0 if any(x in expected for x in top) else 0.0


def compute_per_query_metrics(expected: Set[str], predicted_ranked: Sequence[str], k: int) -> PerQueryMetrics:
    predicted_set = set(predicted_ranked[:k])  # evaluated set is top-k (dedup implicit)
    p, r, f1 = precision_recall_f1(expected, predicted_set)
    rr = reciprocal_rank(expected, predicted_ranked[:k])
    hit = hit_at_k(expected, predicted_ranked, k)
    return PerQueryMetrics(precision=p, recall=r, f1=f1, reciprocal_rank=rr, hit_at_k=hit)


def aggregate(per_query: Iterable[PerQueryMetrics]) -> Dict[str, float]:
    """
    Average metrics across queries.
    """
    items = list(per_query)
    if not items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0, "hit_rate_at_k": 0.0}

    n = len(items)
    return {
        "precision": sum(x.precision for x in items) / n,
        "recall": sum(x.recall for x in items) / n,
        "f1": sum(x.f1 for x in items) / n,
        "mrr": sum(x.reciprocal_rank for x in items) / n,
        "hit_rate_at_k": sum(x.hit_at_k for x in items) / n,
    }

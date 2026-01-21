# scripts/evaluate.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path so we can import rag module
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from rag.db import get_engine
from rag.embedding import load_embedder
from rag.eval import evaluate_all
from rag.ground_truth import (
    load_ground_truth_from_advice_excel,
    split_train_test,
)


# Plug in your own retrieval function here.
# Convention: return List[dict], each dict should contain at least:
#   - "ecli_number" (or "doc_id")
#   - "score" (optional)
#   - "text_snippet" (optional)

def make_retrieve_fn(engine, embedder, train_ids=None):
    """
    Return a closure retrieve_fn(advice_text, top_k) that uses your existing retrieval.
    
    Args:
        train_ids: Set of training advice IDs to prevent data leakage
    """
    from rag.retrieval import retrieve_ecli
    from rag.embedding import load_reranker
    from rag.config import USE_RERANKER
    
    reranker = load_reranker() if USE_RERANKER else None

    def retrieve_fn(advice_text: str, top_k: int) -> List[Dict[str, Any]]:
        # Open a short-lived connection for each evaluation query
        with engine.connect() as conn:
            results = retrieve_ecli(
                conn,
                advice_text,
                embedder=embedder,
                reranker=reranker,
                top_ecli=top_k,
                train_ids=train_ids,  # Pass train_ids to prevent data leakage
            )
            return results

    return retrieve_fn


def main():
    engine = get_engine()
    embedder = load_embedder()

    base = Path.cwd()
    advice_excel = base / "Dataset Advice letters on objections towing of bicycles.xlsx"

    gt = load_ground_truth_from_advice_excel(advice_excel)
    
    # Split into train/test once (same split is used for:
    # - building priors (popular ECLI, citation prototypes) -> train only
    # - computing a test-only evaluation)
    train_gt, test_gt = split_train_test(gt, test_ratio=0.2, random_seed=42)
    train_ids = set(train_gt.keys())
    print(f"Data split: {len(train_ids)} training, {len(test_gt)} test (total {len(gt)})")

    # Retrieval uses only training IDs for any statistics to avoid leakage
    retrieve_fn = make_retrieve_fn(engine, embedder, train_ids=train_ids)

    # 1) Evaluate on ALL advice with ground truth
    metrics_10_all = evaluate_all(
        engine=engine,
        ground_truth=gt,
        retrieve_fn=retrieve_fn,
        top_k=10,
        additional_ks=[5],  # Also compute @5 metrics
        limit=None,
    )

    metrics_5_all = metrics_10_all.get("metrics_at_k", {}).get(5, {})

    # 2) Evaluate only on TEST 20% (same split as above)
    metrics_10_test = evaluate_all(
        engine=engine,
        ground_truth=test_gt,
        retrieve_fn=retrieve_fn,
        top_k=10,
        additional_ks=[5],
        limit=None,
    )
    metrics_5_test = metrics_10_test.get("metrics_at_k", {}).get(5, {})

    print("=" * 70)
    print("RAG Evaluation Results (All vs Test-Only)")
    print("=" * 70)

    print(">>> Overall (ALL advice with ground truth)")
    print(f"Total advice in DB: {metrics_10_all['total_advice']}")
    print(f"Evaluated (has GT): {metrics_10_all['evaluated']}")
    print(f"Queries with results: {metrics_10_all['with_results']}")
    print()
    print("-" * 70)
    print("Metrics @5 (Top-5) - ALL")
    print("-" * 70)
    if metrics_5_all:
        print(f"HitRate@5:  {metrics_5_all['hit_rate_at_k']:.4f}")
        print(f"Precision@5: {metrics_5_all['precision']:.4f}")
        print(f"Recall@5:   {metrics_5_all['recall']:.4f}")
        print(f"F1@5:       {metrics_5_all['f1']:.4f}")
        print(f"MRR@5:      {metrics_5_all['mrr']:.4f}")
    else:
        print("Metrics @5 not available for ALL")
    print()
    print("-" * 70)
    print("Metrics @10 (Top-10) - ALL")
    print("-" * 70)
    print(f"HitRate@10:  {metrics_10_all['hit_rate_at_k']:.4f}")
    print(f"Precision@10: {metrics_10_all['precision']:.4f}")
    print(f"Recall@10:   {metrics_10_all['recall']:.4f}")
    print(f"F1@10:       {metrics_10_all['f1']:.4f}")
    print(f"MRR@10:      {metrics_10_all['mrr']:.4f}")

    print()
    print("-" * 70)
    print("Metrics @5 (Top-5) - TEST 20%")
    print("-" * 70)
    if metrics_5_test:
        print(f"HitRate@5:  {metrics_5_test['hit_rate_at_k']:.4f}")
        print(f"Precision@5: {metrics_5_test['precision']:.4f}")
        print(f"Recall@5:   {metrics_5_test['recall']:.4f}")
        print(f"F1@5:       {metrics_5_test['f1']:.4f}")
        print(f"MRR@5:      {metrics_5_test['mrr']:.4f}")
    else:
        print("Metrics @5 not available for TEST")
    print()
    print("-" * 70)
    print("Metrics @10 (Top-10) - TEST 20%")
    print("-" * 70)
    print(f"HitRate@10:  {metrics_10_test['hit_rate_at_k']:.4f}")
    print(f"Precision@10: {metrics_10_test['precision']:.4f}")
    print(f"Recall@10:   {metrics_10_test['recall']:.4f}")
    print(f"F1@10:       {metrics_10_test['f1']:.4f}")
    print(f"MRR@10:      {metrics_10_test['mrr']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

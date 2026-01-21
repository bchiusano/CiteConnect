from __future__ import annotations
from typing import List, Optional

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from .config import EMBED_MODEL_NAME, USE_FP16, RERANK_MODEL_NAME, USE_RERANKER, EMBED_BATCH_SIZE, EMBED_MAX_LENGTH


def load_embedder() -> BGEM3FlagModel:
    return BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=USE_FP16)


def load_reranker() -> Optional[FlagReranker]:
    if not USE_RERANKER:
        return None
    return FlagReranker(RERANK_MODEL_NAME, use_fp16=USE_FP16)


def embed_texts(model: BGEM3FlagModel, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    out = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        max_length=EMBED_MAX_LENGTH,
        return_dense=True
    )["dense_vecs"]
    return [list(map(float, v)) for v in out]

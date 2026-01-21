EMBED_MODEL_NAME = "BAAI/bge-m3"
TOP_K_RETRIEVE = 200
TOP_K_RERANK = 50
FINAL_K = 5

import os

PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg2://postgres:postgres@localhost:5432/ecli")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
USE_FP16 = True

USE_RERANKER = True
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")

# Toggle for using citation-context prototypes during retrieval.
# Set environment variable USE_CITATION_CONTEXT=0 to completely disable
# citation-based boosts and fallbacks (for ablation experiments).
USE_CITATION_CONTEXT = os.getenv("USE_CITATION_CONTEXT", "1") == "1"

EMBED_BATCH_SIZE = 32
EMBED_MAX_LENGTH = 8192

# scripts/build_issue_index.py
from __future__ import annotations

import sys
from pathlib import Path

# Add parent project directory to path so we can import rag module
# __file__ = .../dsp/RAG/scripts/built_issue_index.py
# parent_dir = .../dsp  (this is where the rag package is installed from)
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from rag.db import get_engine, setup_schema
from rag.embedding import load_embedder
from rag.issues import upsert_issue_index, DEFAULT_ISSUES

engine = get_engine()
setup_schema(engine)
embedder = load_embedder()

with engine.begin() as conn:
    upsert_issue_index(conn, DEFAULT_ISSUES, embedder=embedder)
print("Issue index built.")

# scripts/build_citation_context.py
from pathlib import Path
from rag.db import get_engine, setup_schema
from rag.embedding import load_embedder
from rag.issues import predict_issues
from rag.citation_contest import build_citation_context_prototypes
from rag.ground_truth import load_ground_truth_from_advice_excel, get_train_ids

engine = get_engine()
setup_schema(engine)
embedder = load_embedder()

# Load ground truth and get training IDs to prevent data leakage
base = Path.cwd()
advice_excel = base / "Dataset Advice letters on objections towing of bicycles.xlsx"
gt = load_ground_truth_from_advice_excel(advice_excel)
train_ids = get_train_ids(gt, test_ratio=0.2, random_seed=42)
print(f"Building citation context prototypes using {len(train_ids)} training documents (excluding {len(gt) - len(train_ids)} test documents)")

with engine.begin() as conn:
    n = build_citation_context_prototypes(
        conn,
        embedder=embedder,
        issue_predictor=predict_issues,
        train_ids=train_ids,  # Only use training set
    )
print(f"Inserted {n} citation-context prototypes (from training set only).")

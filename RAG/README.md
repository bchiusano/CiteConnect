## ECLI Citation Retrieval (RAG)

This repository implements a **Retrieval-Augmented Generation (RAG)** system for retrieving relevant **ECLI case law** for Dutch municipal **advice letters** (e.g. objections to bicycle towing).
The system ingests ECLI decisions and advice letters, stores dense embeddings in **PostgreSQL + pgvector**, and retrieves ECLI identifiers via **hybrid semantic + citation-context + issue-aware** search.
Evaluation is performed against ground-truth ECLI annotations from the advice letter dataset, with a clean train/test split to avoid data leakage.

---

## Project Structure

```text
.
├── rag/                       # Core Python package
│   ├── db.py                  # Database engine and schema helpers
│   ├── embedding.py           # Load embedding and (optional) reranker models
│   ├── loaders.py             # Load documents from Excel / folders
│   ├── utils.py               # Chunking and text utilities
│   ├── ingest.py              # Insert documents and chunks into DB
│   ├── chunking.py            # Sentence-based text chunking
│   ├── retrieval.py           # Hybrid + citation-context + issue-aware retrieval
│   ├── citation_context.py    # Build / query citation-context prototypes
│   ├── ground_truth.py        # Load & split ground-truth ECLI labels
│   ├── issues.py              # Issue index + issue prediction
│   ├── metrics.py             # Precision / Recall / F1 / MRR utilities
│   ├── eval.py                # Evaluation loop
│   └── config.py              # Model + DB configuration, feature flags
│
├── scripts/
│   ├── ingest.py              # Run ingestion from Excel into Postgres
│   ├── built_issue_index.py   # Build issue index + citation-context prototypes
│   ├── evaluate.py            # Run evaluation (ALL vs TEST 20%)
│   └── search.py              # CLI for querying a single advice letter
│
├── DATA ecli_nummers juni 2025 v1 (version 1).xlsx       # ECLI decisions
├── Dataset Advice letters on objections towing of bicycles.xlsx  # Advice + ECLI labels
├── DATA_LEAKAGE_REPORT.md      # Notes on data leakage and fixes
├── ENVIRONMENT.md              # Optional environment notes
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python environment

- **Python**: 3.8+
- Install dependencies and the package:

```bash
cd /media/990/xue_gnn/dsp/RAG
pip install -r requirements.txt
pip install -e .          # install the rag package in editable mode
```

### 2. PostgreSQL + pgvector

- PostgreSQL 12+ with `vector` extension:

```sql
CREATE EXTENSION vector;
```

- Create a database (default name: `ecli`).
- Configure the DSN via environment variable or `rag/config.py`:

```bash
export PG_DSN="postgresql+psycopg2://postgres:postgres@localhost:5432/ecli"
```

Or edit `PG_DSN` in `rag/config.py`.

---

## Data Ingestion

Place the two Excel files in the project root (as shown above), then run:

```bash
cd /media/990/xue_gnn/dsp/RAG
python scripts/ingest.py
```

This will:

- Load ECLI decisions and advice letters from Excel.
- Split documents into overlapping chunks.
- Embed all chunks with **BGE-M3**.
- Insert documents and chunk embeddings into PostgreSQL.

---

## Building Indices (Issues + Citation Context)

To enable issue-aware retrieval and citation-context boosts, build the indices using only the **training 80%** of advice letters (to avoid data leakage):

```bash
cd /media/990/xue_gnn/dsp/RAG
python scripts/built_issue_index.py
```

This script:

- Builds/updates the **issue index** (`issue_index` table).
- Builds **citation-context prototypes** (`citation_context_prototypes` table) from training advice letters only.

---

## Evaluation (ALL vs TEST 20%)

Run:

```bash
cd /media/990/xue_gnn/dsp/RAG
python scripts/evaluate.py
```

The script will:

1. Load ground truth from  
   `Dataset Advice letters on objections towing of bicycles.xlsx`.
2. Perform an **80/20 train/test split** (fixed seed, reproducible).
   - Training IDs are used **only** for:
     - Popular ECLI prior (`rag/prior.py`).
     - Citation-context prototypes (`rag/citation_context.py`).
   - No test labels are used to build any priors (no data leakage).
3. Evaluate the same retrieval pipeline on:
   - **ALL** advice with ground truth.
   - **TEST 20%** only (strict hold-out).

For both ALL and TEST, it reports:

- HitRate@K  
- Precision@K  
- Recall@K  
- F1@K  
- MRR@K  

---

## Ablation: Disable Citation Context

You can disable citation-context prototypes via an environment variable, without changing code:

```bash
cd /media/990/xue_gnn/dsp/RAG

# With citation context (default)
python scripts/evaluate.py

# Without citation context
USE_CITATION_CONTEXT=0 python scripts/evaluate.py
```

This toggles `_maybe_search_citation_context` in `rag/retrieval.py`. Comparing the two runs shows the contribution of citation-context prototypes to Hit/Recall/MRR.

---

## Retrieving ECLI for a New Advice Letter

Once the DB and indices are built, you can retrieve ECLI numbers for a **single new advice letter** using `scripts/search.py`.

### From a `.txt` file

```bash
cd /media/990/xue_gnn/dsp/RAG
python scripts/search.py --txt /path/to/new_advice.txt --top 10
```

### From a `.docx` file

```bash
cd /media/990/xue_gnn/dsp/RAG
python scripts/search.py --docx /path/to/new_advice.docx --top 10
```

### From inline text

```bash
cd /media/990/xue_gnn/dsp/RAG
python scripts/search.py --text "Paste the advice letter text here..." --top 10
```

The script will:

- Load the DB engine, embedder and optional reranker.
- Call `rag.retrieval.retrieve_ecli(...)` with:
  - Hybrid dense + BM25 retrieval over ECLI chunks.
  - Citation-context prototype boosts (if enabled).
  - Issue-aware boosts.
  - Optional reranking.
- Print top-K ECLI candidates with scores and short snippets.

---

## Notes on Data Leakage

- Ground-truth ECLI labels are used **only** for:
  - Computing **popular ECLI priors** (`rag/prior.py`), restricted to **training IDs**.
  - Building **citation-context prototypes** from **training advice letters only**.
- Evaluation uses:
  - **ALL** advice (for an overall picture).
  - **TEST 20%** only (strict hold-out), using the same split as index construction.
- No test labels are used to build any index or prior.  
 


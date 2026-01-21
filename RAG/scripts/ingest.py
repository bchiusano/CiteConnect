"""
Ingest script for RAG system.

Make sure the rag package is installed:
    pip install -e .
"""
import sys
from pathlib import Path

# Add parent directory to path if package not installed
script_path = Path(__file__).resolve()
parent_dir = script_path.parent.parent  # Go up from scripts/ to dsp/
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from rag.db import get_engine, setup_schema
from rag.embedding import load_embedder
from rag.ingest_runner import ingest_both_excels


if __name__ == "__main__":
    engine = get_engine()
    setup_schema(engine)
    embedder = load_embedder()

    base = Path.cwd()

    ecli_count, advice_count = ingest_both_excels(
        engine=engine,
        embedder=embedder,
        base_path=base,
        ecli_filename="DATA ecli_nummers juni 2025 v1 (version 1).xlsx",
        advice_filename="Dataset Advice letters on objections towing of bicycles.xlsx",
    )

    print(f"Ingest done: ecli={ecli_count}, advice={advice_count}, total={ecli_count + advice_count}")

"""
Initialize the RAG database schema.

Run this script from the parent directory (dsp/) with:
    python -m rag.run_rag
    
Or from the RAG directory with:
    PYTHONPATH=/media/990/xue_gnn/dsp python run_rag.py
"""
import sys
from pathlib import Path

# Add parent directory to Python path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from rag.db import get_engine, setup_schema

engine = get_engine()
setup_schema(engine)
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Dict, Iterator, Optional, List

import pandas as pd

from .utils import detect_language_stub, extract_ecli_from_text, normalize_ecli


def load_docs_from_folder(folder: str = "data") -> Iterator[Dict]:
    paths = list(Path(folder).glob("**/*.txt"))
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        meta_path = p.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        doc_id = meta.get("doc_id") or str(uuid.uuid4())
        yield {
            "doc_id": doc_id,
            "title": meta.get("title") or p.stem,
            "source": meta.get("source") or str(p),
            "doc_type": meta.get("doc_type") or "unknown",
            "published_at": meta.get("published_at"),
            "language": detect_language_stub(text),
            "raw_metadata": meta,
            "text": text,
        }


def load_docs_from_excel(
    excel_path: str,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    title_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
) -> Iterator[Dict]:
    df = pd.read_excel(excel_path)

    if text_column is None:
        raise ValueError("text_column must be provided (to avoid guessing the wrong column).")

    for idx, row in df.iterrows():
        raw_text = row.get(text_column, "")
        if pd.isna(raw_text) or str(raw_text).strip() == "":
            continue

        text = str(raw_text)
        ecli_in_text = extract_ecli_from_text(text)
        doc_id = None

        if id_column and (id_column in df.columns) and not pd.isna(row.get(id_column)):
            doc_id = str(row[id_column]).strip()
        elif ecli_in_text:
            doc_id = normalize_ecli(ecli_in_text)
        else:
            doc_id = str(uuid.uuid4())

        title = str(row[title_column]).strip() if (title_column and not pd.isna(row.get(title_column))) else f"Document {idx+1}"

        if metadata_columns:
            meta = {col: row[col] for col in metadata_columns if col in df.columns and not pd.isna(row.get(col))}
        else:
            meta = {col: row[col] for col in df.columns if col != text_column and not pd.isna(row.get(col))}

        if ecli_in_text:
            meta["ecli"] = normalize_ecli(ecli_in_text)

        yield {
            "doc_id": doc_id,
            "title": title,
            "source": str(excel_path),
            "doc_type": "ecli" if (str(doc_id).upper().startswith("ECLI:")) else "advice",
            "published_at": meta.get("published_at") or meta.get("date") or meta.get("datum"),
            "language": detect_language_stub(text),
            "raw_metadata": meta,
            "text": text,
        }

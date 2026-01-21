from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional

from .chunking import chunk_text


_ECLI_PATTERN = re.compile(r"ECLI:[A-Z]{2}:[A-Z0-9]{4}:[A-Z0-9]+", re.IGNORECASE)


def normalize_ecli(ecli: str) -> str:
    if ecli is None:
        return ""
    s = str(ecli).strip()
    # normalize full-width colon etc. (minimal)
    s = s.replace("：", ":")
    # ensure prefix
    if not s.upper().startswith("ECLI:") and ":" in s:
        # if user stored NL:XXXX:... then leave as-is; you can tighten later
        pass
    return s.upper()


def norm_ecli(ecli: Optional[str]) -> str:
    """
    Public helper so other modules can import `norm_ecli` from `rag.utils`.
    """
    return normalize_ecli(ecli) if ecli is not None else ""


def extract_ecli_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ECLI_PATTERN.search(text)
    return normalize_ecli(m.group(0)) if m else None


def detect_language_stub(text: str) -> str:
    # You currently assume Dutch; keep it explicit as a stub.
    return "nl"


def build_chunks(
    doc: Dict,
    *,
    target_tokens: int = 350,
    overlap_tokens: int = 80
) -> List[Dict]:
    """
    Convert a document dict into chunk dicts.
    Offsets are intentionally omitted to keep ingestion simple and robust.
    """
    chunks = chunk_text(doc["text"], target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    out: List[Dict] = []
    raw_meta = doc.get("raw_metadata") or {}

    for i, ch in enumerate(chunks):
        out.append({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc["doc_id"],
            "idx": i,  # Changed from chunk_index to idx to match SQL parameter name
            "language": doc.get("language"),
            "section_title": raw_meta.get("section_title") if isinstance(raw_meta, dict) else None,
            "text": ch,
        })

    return out

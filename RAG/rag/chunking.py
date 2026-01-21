from __future__ import annotations
from functools import lru_cache
from typing import List
import spacy


@lru_cache(maxsize=1)
def get_nlp(model_name: str = "nl_core_news_md"):
    return spacy.load(model_name)


def chunk_text(text: str, target_tokens: int = 350, overlap_tokens: int = 80) -> List[str]:
    """
    Sentence-aware chunking using spaCy sentences.
    Chunks are approximately `target_tokens` spaCy tokens, with `overlap_tokens` token overlap.
    """
    nlp = get_nlp()
    doc = nlp(text)

    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for sent_text in sents:
        sent_doc = nlp(sent_text)
        sent_tokens = len(sent_doc)

        if cur and (cur_tokens + sent_tokens > target_tokens):
            chunk = " ".join(cur).strip()
            chunks.append(chunk)

            if overlap_tokens > 0:
                # Take overlap from the end of the chunk by spaCy tokens (approx).
                tail_doc = nlp(chunk)
                tail_text = " ".join([t.text for t in tail_doc[-overlap_tokens:]]).strip()
                cur = [tail_text] if tail_text else []
                cur_tokens = len(nlp(tail_text)) if tail_text else 0
            else:
                cur = []
                cur_tokens = 0

        cur.append(sent_text)
        cur_tokens += sent_tokens

    if cur:
        chunks.append(" ".join(cur).strip())

    return chunks

# scripts/search.py
from __future__ import annotations

import argparse
from pathlib import Path

from rag.db import get_engine, setup_schema
from rag.embedding import load_embedder  
from rag.retrieval import retrieve_ecli  # hybrid+citation-context+issue

# optional: docx
def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def main():
    ap = argparse.ArgumentParser(description="Search relevant ECLI for an advice letter (hybrid + prototypes + issues).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Advice letter text (inline).")
    src.add_argument("--txt", type=str, help="Path to a .txt file containing advice letter text.")
    src.add_argument("--docx", type=str, help="Path to a .docx file containing advice letter text.")
    src.add_argument("--from-db", type=str, help="Use an existing advice doc from DB by doc_id.")

    ap.add_argument("--top", type=int, default=10, help="Number of ECLI to return (default: 10).")
    ap.add_argument("--chunks", type=int, default=300, help="How many chunks to retrieve before aggregation (default: 300).")
    ap.add_argument("--proto-k", type=int, default=50, help="How many prototypes to use for boosting (default: 50).")
    ap.add_argument("--no-rerank", action="store_true", help="Disable reranker (if you have one).")

    args = ap.parse_args()

    # 1) Load resources
    engine = get_engine()
    setup_schema(engine)  # safe to call repeatedly
    embedder = load_embedder()

    # Optional reranker: only if you have load_reranker implemented
    reranker = None
    if not args.no_rerank:
        try:
            from rag.embedding import load_reranker  # optional
            reranker = load_reranker()
        except Exception:
            reranker = None

    # 2) Get advice text
    advice_text = ""
    if args.text:
        advice_text = args.text
    elif args.txt:
        advice_text = read_text_file(args.txt)
    elif args.docx:
        advice_text = read_docx(args.docx)
    elif args.from_db:
        with engine.begin() as conn:
            row = conn.execute(
                # assumes documents(doc_id, text, doc_type)
                # doc_id is your advice doc id
                __import__("sqlalchemy").text("SELECT text FROM documents WHERE doc_type='advice' AND doc_id=:id"),
                {"id": args.from_db},
            ).mappings().first()
            if not row:
                raise SystemExit(f"No advice doc found in DB for doc_id={args.from_db}")
            advice_text = row["text"] or ""

    advice_text = (advice_text or "").strip()
    if not advice_text:
        raise SystemExit("Empty advice text. Provide --text/--txt/--docx/--from-db.")

    # 3) Search
    with engine.begin() as conn:
        results = retrieve_ecli(
            conn,
            advice_text,
            embedder=embedder,
            reranker=reranker,
            top_ecli=args.top,
            retrieve_chunks=args.chunks,
            proto_k=args.proto_k,
        )

    # 4) Pretty print (minimal)
    print(f"\nFound {len(results)} ECLI:\n")
    for i, r in enumerate(results, 1):
        ecli = r.get("ecli_number")
        score = r.get("score", 0.0)
        print(f"{i:>2}. {ecli}  score={score:.4f}")
        snip = (r.get("text_snippet") or "").replace("\n", " ").strip()
        if snip:
            print(f"    {snip}...")
        # Uncomment for debugging signals
        # print(f"    dense={r.get('score_dense')} bm25={r.get('score_bm25')} hybrid={r.get('score_hybrid')} rerank={r.get('score_rerank')}")
        print("")


if __name__ == "__main__":
    main()

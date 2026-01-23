# RAG Enhancements

Extra retrieval features: citation context search/boost, issue boost, popularity prior, fallback (fallback is mainly a safety net and is rarely triggered in practice).

## Architecture

Uses two separate Chroma collections:
1. **Main DB** (`chroma_db_infloat_multilingual`): Case chunks for hybrid retrieval
2. **Citation DB** (`chroma_citation_prototypes`): Citation contexts from training data

Citation DB is built automatically on first run from training documents.

## Usage

```python
rag = LegalRAGSystem()
# Citation DB is initialized in run_evaluation() after train/test split
rag.run_evaluation(mode="sample")
```

Or manually:
```python
rag = LegalRAGSystem()
train_ids = get_train_ids(ground_truth, test_ratio=0.2)
rag.init_citation_db(train_ids, force_rebuild=False)
results = rag.get_top_10_for_letter(letter_text, train_ids=train_ids, use_enhancements=True)
```

## Files

- `rag_enhancements.py`: Enhancement functions + citation DB builder
- `resources/chroma_citation_prototypes/`: Citation prototype collection (auto-generated)

## Notes

- Citation context search uses separate collection (no domain filtering)
- Set `force_rebuild=True` to regenerate citation prototypes
- Requires train_ids for data leakage prevention

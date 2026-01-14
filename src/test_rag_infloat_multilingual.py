import os
import pickle
import re
import pandas as pd
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

# --- DYNAMIC CONFIGURATION ---
from rag_pipeline_infloat_multilingual import PERSIST_DIR
from rag_index_infloat_multilingual import BM25_INDEX_PATH, CORPUS_PATH

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# --- PARAMETERS ---
SEARCH_K = 300       
RERANK_TOP_N = 60    
CANDIDATE_LIMIT = 100 

def clean_ecli(text):
    if pd.isna(text) or str(text).lower() == 'nan': return ""
    text = str(text).upper().strip().replace("ECLI:", "")
    for char in ["[", "]", "'", '"', " ", "\n", "\r", ";"]:
        text = text.replace(char, "")
    return text

def rrf_fusion(vector_hits, bm25_hits, k=60):
    fused_scores = {}
    for rank, doc in enumerate(vector_hits):
        doc_id = doc.metadata.get('ecli_nummer')
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 / (rank + k))
    for rank, doc_dict in enumerate(bm25_hits):
        doc_id = doc_dict['metadata'].get('ecli_nummer')
        # Lexical boost for exact keyword matches
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.2 * (1 / (rank + k))
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def main():
    print(f"--- STARTING FULL EVALUATION (k={SEARCH_K}) ---")
    
    # 1. LOAD ENGINES
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'mps'})
    db = Chroma(collection_name="legal_rag", persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    print("Loading BM25 index and corpus...")
    with open(BM25_INDEX_PATH, "rb") as f: bm25_model = pickle.load(f)
    with open(CORPUS_PATH, "rb") as f: legal_corpus = pickle.load(f)
    
    reranker = FlashrankRerank(model="ms-marco-MultiBERT-L-12", top_n=RERANK_TOP_N)

    # 2. LOAD DATA
    df = pd.read_excel("../data/Dataset Advice letters on objections towing of bicycles.xlsx")
    test_df = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
    print(f"Total Rows to Evaluate: {len(test_df)}")

    results_data = []
    tp, total_targets = 0, 0
    start_time = time.time()

    # 3. EVALUATION LOOP
    for idx, row in test_df.iterrows():
        letter_text = str(row['geanonimiseerd_doc_inhoud'])
        actual_list = [clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if clean_ecli(e)]
        total_targets += len(actual_list)
        
        # Paragraph Extraction (Robust Logic)
        all_segments = [p.strip() for p in re.split(r'\n+', letter_text) if len(p.strip()) > 50]
        claims = all_segments[2:12] if len(all_segments) > 5 else all_segments
        
        ecli_scores_tracker = {} 

        for claim in claims:
            try:
                # Stage 1: Hybrid Retrieval
                vector_hits = db.similarity_search(f"query: {claim}", k=SEARCH_K)
                query_tokens = re.findall(r'\w+', claim.lower())
                bm25_hits = bm25_model.get_top_n(query_tokens, legal_corpus, n=SEARCH_K)
                
                # Stage 2: RRF Fusion
                fused_results = rrf_fusion(vector_hits, bm25_hits)
                candidate_map = {doc.metadata['ecli_nummer']: doc for doc in vector_hits}
                for d in bm25_hits:
                    eid = d['metadata']['ecli_nummer']
                    if eid not in candidate_map:
                        candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])

                top_candidates = [candidate_map[eid] for eid, _ in fused_results[:CANDIDATE_LIMIT] if eid in candidate_map]
                
                # Stage 3: Neural Reranking (Cross-Encoder)
                refined = reranker.compress_documents(top_candidates, claim)
                for r in refined:
                    ecli = clean_ecli(r.metadata.get('ecli_nummer', ''))
                    score = r.metadata.get('relevance_score', 0)
                    # Track the best score found for this ECLI across all claims
                    if ecli not in ecli_scores_tracker or score > ecli_scores_tracker[ecli]:
                        ecli_scores_tracker[ecli] = score
            except Exception: continue

        # --- RANKING & SORTING ---
        sorted_found = sorted(ecli_scores_tracker.items(), key=lambda x: x[1], reverse=True)
        top_10 = [item[0] for item in sorted_found[:10]]
        
        hits = set(actual_list).intersection(set(ecli_scores_tracker.keys()))
        tp += len(hits)

        # Calculate Rank of first correct hit (for MRR evaluation)
        first_hit_rank = 0
        for rank, ecli in enumerate(top_10, 1):
            if ecli in actual_list:
                first_hit_rank = rank
                break

        # --- LOGGING TO CSV STRUCTURE ---
        results_data.append({
            "row_id": idx,
            "target_eclis": "; ".join(actual_list),
            "top_10_found": "; ".join(top_10),
            "hit_count": len(hits),
            "total_expected": len(actual_list),
            "first_hit_rank": first_hit_rank,
            "success_rate": len(hits) / len(actual_list) if actual_list else 0
        })

        # Progress reporting
        if len(results_data) % 5 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"[{elapsed:.1f} min] Done {len(results_data)}/{len(test_df)} | Recall: {tp/total_targets:.2%}")
            pd.DataFrame(results_data).to_csv("interim_results.csv", index=False)

    # FINAL EXPORT
    final_df = pd.DataFrame(results_data)
    final_df.to_csv("final_recall_results.csv", index=False)
    
    print(f"\n" + "="*40)
    print(f"EVALUATION COMPLETE")
    print(f"Final Recall: {tp/total_targets:.2%}")
    print(f"Results saved to final_recall_results.csv")
    print("="*40)

if __name__ == "__main__":
    main()
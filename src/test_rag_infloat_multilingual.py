import os
import sys
import pickle
import re
import time
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIGURATION ---
from rag_pipeline_infloat_multilingual import PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME, letters_path
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
from rag_index_infloat_multilingual import BM25_INDEX_PATH, CORPUS_PATH
RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

# --- PARAMETERS ---
NUM_TEST_ROWS = 30 
RANDOM_SEED = 40
SEARCH_K = 400       
RERANK_TOP_N = 60    
CANDIDATE_LIMIT = 100 

class LegalRAGSystem:
    def __init__(self):
        print("Initializing Engines...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'mps'})
        self.db = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR, embedding_function=self.embeddings)
        
        with open(BM25_INDEX_PATH, "rb") as f: self.bm25_model = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f: self.legal_corpus = pickle.load(f)
        
        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'flashrank_cache'))
        self.reranker = FlashrankRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)

    def clean_ecli(self, text):
        """Standardizes ECLI strings for comparison."""
        if pd.isna(text) or str(text).lower() == 'nan': return ""
        text = str(text).upper().strip().replace("ECLI:", "")
        for char in ["[", "]", "'", '"', " ", "\n", "\r", ";"]:
            text = text.replace(char, "")
        return text

    def get_top_10_for_letter(self, letter_text, domain="bicycle"):
        """
        UI METHOD: Takes a single letter, splits into 5 issues, and returns best 10 ECLIs.
        """
        # 1. Prepare Domain Anchors
        keywords = DOMAIN_MAP.get(domain, {}).get("keywords", [])
        anchor = " ".join(keywords)

        # 2. Split FULL letter into exactly 5 claims/issues
        # First, clean text and split into sentences (using a basic split for logic consistency)
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', letter_text) if len(s.strip()) > 10]
        
        num_issues = 5
        if len(sentences) < num_issues:
            claims = sentences
        else:
            # Divide sentences into 5 roughly equal chunks
            chunk_size = len(sentences) // num_issues
            claims = [" ".join(sentences[i * chunk_size : (i + 1) * chunk_size]) for i in range(num_issues)]
        
        ecli_scores = {} 

        for claim in claims:
            try:
                # Query Expansion with anchor terms
                enhanced_query = f"{anchor} {claim}"
                
                # Retrieval using SEARCH_K parameter
                v_hits = self.db.similarity_search(f"query: {enhanced_query}", k=SEARCH_K)
                tokens = re.findall(r'\w+', enhanced_query.lower())
                b_hits = self.bm25_model.get_top_n(tokens, self.legal_corpus, n=SEARCH_K)
                
                # Hybrid Fusion (RRF)
                fused_results = self._rrf_fusion(v_hits, b_hits)
                
                # Prepare candidates for Reranker
                candidate_map = {doc.metadata['ecli_nummer']: doc for doc in v_hits}
                for d in b_hits:
                    eid = d['metadata']['ecli_nummer']
                    if eid not in candidate_map:
                        candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])
                
                # Using CANDIDATE_LIMIT parameter
                top_cands = [candidate_map[eid] for eid, _ in fused_results[:CANDIDATE_LIMIT] if eid in candidate_map]
                
                # Neural Rerank
                refined = self.reranker.compress_documents(top_cands, enhanced_query)
                for r in refined:
                    ecli = self.clean_ecli(r.metadata.get('ecli_nummer', ''))
                    score = r.metadata.get('relevance_score', 0)
                    # Keep the best score for an ECLI found across the 5 claims
                    if ecli not in ecli_scores or score > ecli_scores[ecli]:
                        ecli_scores[ecli] = score
            except: continue

        # Sort and return top 10
        sorted_eclis = sorted(ecli_scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_eclis[:10]]

    def _rrf_fusion(self, v_hits, b_hits, k=60):
        scores = {}
        for r, d in enumerate(v_hits):
            eid = d.metadata.get('ecli_nummer')
            scores[eid] = scores.get(eid, 0) + (1 / (r + k))
        for r, d in enumerate(b_hits):
            eid = d['metadata'].get('ecli_nummer')
            # 1.2 boost for lexical matching consistency
            scores[eid] = scores.get(eid, 0) + 1.2 * (1 / (r + k))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def run_evaluation(self, mode="full"):
        """
        Runs evaluation and calculates MRR and Recall@10.
        """
        print(f"--- STARTING EVALUATION MODE: {mode.upper()} (k={SEARCH_K}) ---")
        df = pd.read_excel(letters_path)
        data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
        
        if mode == "sample":
            data = data.sample(n=NUM_TEST_ROWS, random_state=RANDOM_SEED)
        
        results = []
        metrics = {"reciprocal_ranks": [], "hits_at_10": 0, "total_targets": 0}
        
        start_time = time.time()
        for idx, row in data.iterrows():
            targets = [self.clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if self.clean_ecli(e)]
            top_10 = self.get_top_10_for_letter(str(row['geanonimiseerd_doc_inhoud']), ACTIVE_DOMAIN)
            
            # --- METRIC CALCULATIONS ---
            hits = set(targets).intersection(set(top_10))
            metrics["hits_at_10"] += len(hits)
            metrics["total_targets"] += len(targets)
            
            # MRR (Mean Reciprocal Rank) calculation
            rank_score = 0
            for i, ecli in enumerate(top_10, 1):
                if ecli in targets:
                    rank_score = 1/i
                    break
            metrics["reciprocal_ranks"].append(rank_score)

            # --- DETAILED PRINT FOR MONITORING ---
            print(f"\nRow ID: {idx}")
            print(f"Target ECLIs:  {targets}")
            print(f"Top 10 Found:  {top_10}")
            print(f"Result:        {len(hits)}/{len(targets)} hits | Rank Score: {rank_score:.4f}")

            results.append({
                "row_id": idx,
                "targets": "; ".join(targets),
                "top_10": "; ".join(top_10),
                "recall_at_10": len(hits) / len(targets) if targets else 0,
                "mrr": rank_score
            })
            
            if len(results) % 5 == 0:
                current_recall = metrics['hits_at_10'] / metrics['total_targets'] if metrics['total_targets'] > 0 else 0
                print(f"\n>>> PROGRESS: {len(results)}/{len(data)} | Current Recall@10: {current_recall:.2%}")

        # Final Summary
        final_recall = metrics["hits_at_10"] / metrics["total_targets"] if metrics['total_targets'] > 0 else 0
        final_mrr = np.mean(metrics["reciprocal_ranks"])
        
        print("\n" + "="*50)
        print(f"FINAL EVALUATION METRICS ({mode.upper()})")
        print(f"Total Targets:  {metrics['total_targets']}")
        print(f"Total Hits@10:  {metrics['hits_at_10']}")
        print(f"Final Recall@10: {final_recall:.4f}")
        print(f"Final MRR:       {final_mrr:.4f}")
        print(f"Total Time:     {(time.time()-start_time)/60:.2f} mins")
        print("="*50)
        
        pd.DataFrame(results).to_csv(f"eval_{mode}_results.csv", index=False)

# --- EXECUTION ---
if __name__ == "__main__":
    rag = LegalRAGSystem()
    
    # Verify with sample first, then switch to 'full'
    rag.run_evaluation(mode="sample") 
    # rag.run_evaluation(mode="full")
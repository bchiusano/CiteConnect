import os
import pickle
import re
import time
import sys
import spacy
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIGURATION ---
from rag_pipeline_infloat_multilingual import PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME, letters_path
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
from rag_index_infloat_multilingual import BM25_INDEX_PATH, CORPUS_PATH
from text_summarizer import BulkInferenceEngine, GroqSummarizer
RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

# --- PARAMETERS ---
NUM_TEST_ROWS = 30    
RANDOM_SEED = 40
SEARCH_K = 200       
RERANK_TOP_N = 50     
CANDIDATE_LIMIT = 50  
MIN_SCORE = 0.3       
RERANK_WEIGHT = 0.8

# --- SPACY INITIALIZATION ---
try:
    nlp = spacy.load("nl_core_news_md")
except:
    print("nl_core_news_md not found, falling back to small model.")
    nlp = spacy.load("nl_core_news_sm")

class LegalRAGSystem:
    def __init__(self, initialize_summarizer=False, use_groq=False):
        print("Initializing Engines...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'mps'}
        )
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )

        with open(BM25_INDEX_PATH, "rb") as f: self.bm25_model = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f: self.legal_corpus = pickle.load(f)

        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'flashrank_cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.reranker = FlashrankRerank(
            model=RERANK_MODEL, top_n=RERANK_TOP_N
        )

        # Initialize summarizer if needed (for dynamic summary generation)
        self.summarizer = None
        if initialize_summarizer:
            if use_groq:
                print("Initializing Groq Summarizer...")
                self.summarizer = GroqSummarizer()
            else:
                print("Initializing Local Summary Engine...")
                self.summarizer = BulkInferenceEngine()

    def clean_ecli(self, text):
        if pd.isna(text) or str(text).lower() == 'nan': return ""
        cleaned = str(text).upper().replace("ECLI:", "").strip()
        cleaned = re.sub(r'[^A-Z0-9:]', '', cleaned)
        return cleaned

    def _search_and_rerank(self, search_text, domain="bicycle"):
        """
        Generic search + reranking method.
        Returns top 10 ECLI matches for a given search text.
        """
        keywords = DOMAIN_MAP.get(domain, {}).get("keywords", [])
        anchor = " ".join(keywords)
        
        ecli_best_chunks = {}
        
        try:
            # Parallel Search (Vector + Keyword)
            v_hits = self.db.similarity_search(f"query: {search_text}", k=SEARCH_K)
            tokens = re.findall(r'\w+', (f"{anchor} {search_text}").lower())
            b_hits = self.bm25_model.get_top_n(tokens, self.legal_corpus, n=SEARCH_K)

            # Hybrid Fusion (0.7 / 0.3)
            fused = self._rrf_fusion(v_hits, b_hits)
            
            candidate_map = {self.clean_ecli(d.metadata['ecli_nummer']): d for d in v_hits}
            for d in b_hits:
                eid = self.clean_ecli(d['metadata'].get('ecli_nummer'))
                if eid not in candidate_map:
                    candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])
            
            # Neural Reranking (Weighted 0.2 / 0.8)
            rerank_candidates = []
            for eid, hybrid_score in fused[:RERANK_TOP_N]:
                if eid in candidate_map:
                    rerank_candidates.append((candidate_map[eid], hybrid_score))
            
            if rerank_candidates:
                docs_to_rerank = [item[0] for item in rerank_candidates]
                refined = self.reranker.compress_documents(docs_to_rerank, search_text)
                
                for r in refined:
                    ecli = self.clean_ecli(r.metadata.get('ecli_nummer', ''))
                    rerank_score = r.metadata.get('relevance_score', 0)
                    
                    # Find original hybrid score
                    original_hybrid_score = next((score for eid, score in fused if eid == ecli), 0)
                    
                    # Apply Weighting: 20% Original, 80% Reranker
                    final_score = ((1 - RERANK_WEIGHT) * original_hybrid_score) + (RERANK_WEIGHT * rerank_score)
                    
                    if final_score >= MIN_SCORE:
                        if ecli not in ecli_best_chunks or final_score > ecli_best_chunks[ecli]:
                            ecli_best_chunks[ecli] = final_score
        except Exception as e:
            print(f"Error in search: {e}")
        
        return [eid for eid, score in sorted(ecli_best_chunks.items(), key=lambda x: x[1], reverse=True)[:10]]

    def get_top_10_split_issues(self, letter_text, domain="bicycle"):
        """
        APPROACH 1: Split letter into 5 issues and search each separately.
        (Original logic - preserved)
        """
        doc = nlp(letter_text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        num_issues = 5
        chunk_size = max(1, len(sentences) // num_issues)
        issues = [" ".join(sentences[i * chunk_size : (i + 1) * chunk_size]) for i in range(num_issues)]
        
        ecli_best_chunks = {}
        
        for issue_text in issues:
            results = self._search_and_rerank(issue_text, domain)
            for ecli in results:
                # Aggregate across issues
                if ecli not in ecli_best_chunks:
                    ecli_best_chunks[ecli] = 0
                ecli_best_chunks[ecli] += 1
        
        return [eid for eid, _ in sorted(ecli_best_chunks.items(), key=lambda x: x[1], reverse=True)[:10]]

    def get_top_10_with_summary(self, letter_text, summary_text, domain="bicycle"):
        """
        APPROACH 2: Use the summary (not full letter) for searching.
        Note: summary_text should be pre-generated from text_summarizer.py
        """
        return self._search_and_rerank(summary_text, domain)

    def get_top_10_mixed(self, letter_text, summary_text, domain="bicycle"):
        """
        APPROACH 3 (FUTURE): Mix both split issues + summary.
        Returns aggregated top 10 from both approaches.
        """
        split_results = self.get_top_10_split_issues(letter_text, domain)
        summary_results = self.get_top_10_with_summary(letter_text, summary_text, domain)
        
        # Combine with weights (50/50 for now)
        combined = {}
        for i, ecli in enumerate(split_results):
            combined[ecli] = combined.get(ecli, 0) + (1 / (i + 1)) * 0.5
        for i, ecli in enumerate(summary_results):
            combined[ecli] = combined.get(ecli, 0) + (1 / (i + 1)) * 0.5
        
        return [eid for eid, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]]

    def get_top_10_for_letter(self, letter_text, domain="bicycle", approach="split", summary_text=None):
        """
        Main wrapper to choose which search approach to use.
        
        Args:
            letter_text: Full letter content
            domain: Domain type (e.g., "bicycle")
            approach: One of ["split", "summary", "mixed"]
            summary_text: Pre-generated summary (required for "summary" and "mixed" approaches)
        
        Returns:
            List of top 10 ECLI codes
        """
        if approach == "split":
            return self.get_top_10_split_issues(letter_text, domain)
        elif approach == "summary":
            if not summary_text:
                raise ValueError("summary_text is required for 'summary' approach")
            return self.get_top_10_with_summary(letter_text, summary_text, domain)
        elif approach == "mixed":
            if not summary_text:
                raise ValueError("summary_text is required for 'mixed' approach")
            return self.get_top_10_mixed(letter_text, summary_text, domain)
        else:
            raise ValueError(f"Unknown approach: {approach}. Use 'split', 'summary', or 'mixed'.")

    def query_single_letter(self, letter_text, approach="split", summary_text=None, return_details=False):
        """
        REAL-TIME QUERY: Get top 10 matches for a single advice letter.
        
        Args:
            letter_text: Full text of the advice letter
            approach: "split", "summary", or "mixed"
            summary_text: Pre-generated summary (only needed for "summary" or "mixed")
            return_details: If True, returns (eclis, scores, documents). If False, returns just ECLIs
        
        Returns:
            List of top 10 ECLI codes (or with details if return_details=True)
        """
        print(f"\n{'='*50}")
        print(f"REAL-TIME QUERY: {approach.upper()} approach")
        print(f"Input letter length: {len(letter_text)} chars")
        print(f"{'='*50}\n")
        
        # Get top 10 ECLIs
        if approach == "split":
            print("Strategy: Splitting letter into 5 issues...")
            top_10_eclis = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="split")
        
        elif approach == "summary":
            if not summary_text:
                if not self.summarizer:
                    raise ValueError("summary_text required, or initialize_summarizer=True")
                print("Generating summary...")
                summary_start = time.time()
                summary_text = self.summarizer.generate_query(letter_text)
                summary_time = time.time() - summary_start
                print(f"Summary generated in {summary_time:.2f}s:")
                print(f"  {summary_text}\n")
            else:
                print("Using provided summary...")
                print(f"  {summary_text}\n")
            
            top_10_eclis = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, 
                                                       approach="summary", summary_text=summary_text)
        
        elif approach == "mixed":
            if not summary_text:
                if not self.summarizer:
                    raise ValueError("summary_text required, or initialize_summarizer=True")
                print("Generating summary for mixed approach...")
                summary_start = time.time()
                summary_text = self.summarizer.generate_query(letter_text)
                summary_time = time.time() - summary_start
                print(f"Summary generated in {summary_time:.2f}s:")
                print(f"  {summary_text}\n")
            
            top_10_eclis = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, 
                                                       approach="mixed", summary_text=summary_text)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        # Display results
        print(f"TOP 10 MATCHING CASES:")
        print(f"-" * 50)
        for i, ecli in enumerate(top_10_eclis, 1):
            print(f"{i:2d}. {ecli}")
        print(f"{'='*50}\n")
        
        return top_10_eclis

    def _rrf_fusion(self, v_hits, b_hits, k=60):
        scores = {}
        # 0.7 Semantic / 0.3 Keyword split
        for r, d in enumerate(v_hits):
            eid = self.clean_ecli(d.metadata.get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.7 * (1 / (r + k))
        for r, d in enumerate(b_hits):
            eid = self.clean_ecli(d['metadata'].get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.3 * (1 / (r + k))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def run_evaluation(self, mode="full", search_approach="split", summary_file=None, num_rows=NUM_TEST_ROWS, random_seed=RANDOM_SEED):
        """
        EVALUATION MODE: Uses pre-loaded summaries from CSV file.
        Best for comparing different search approaches on fixed summaries.
        
        Args:
            mode: "full" or "sample"
            search_approach: "split", "summary", or "mixed"
            summary_file: Path to pre-generated summaries CSV (required for "summary" and "mixed")
            num_rows: Number of rows for sample mode (uses NUM_TEST_ROWS if None)
            random_seed: Random seed for reproducibility (uses RANDOM_SEED if None)
        """
        
        print(f"--- EVALUATION MODE ---")
        print(f"Mode: {mode.upper()} | Approach: {search_approach.upper()}")
        if mode == "sample":
            print(f"Sample Size: {num_rows} | Random Seed: {random_seed}")
        if summary_file:
            print(f"Summary File: {summary_file}")
        print("-" * 50)
        
        # Load data and prepare evaluation set
        df = pd.read_excel(letters_path)
        data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
        
        if mode == "sample":
            data = data.sample(n=num_rows, random_state=random_seed)
        
        # Load summaries if using summary-based or mixed approach
        summaries_df = None
        if search_approach in ["summary", "mixed"] and summary_file:
            summaries_df = pd.read_csv(summary_file)
            print(f"Loaded {len(summaries_df)} summaries from {summary_file}")
        
        results = []
        metrics = {
            "reciprocal_ranks": [], 
            "hits_at_10": 0, 
            "total_targets": 0,
            "precisions": []
        }
        
        start_time = time.time()
        
        for idx, row in data.iterrows():
            # 1. Prepare standardized targets
            targets = [self.clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if self.clean_ecli(e)]
            
            # 2. Retrieval using selected approach
            letter_text = str(row['geanonimiseerd_doc_inhoud'])
            summary_text = None
            
            # Try to get summary if needed
            if search_approach in ["summary", "mixed"] and summaries_df is not None:
                matching_summary = summaries_df[summaries_df['Original_Index'] == idx]
                if not matching_summary.empty:
                    summary_text = matching_summary.iloc[0]['Generated_Summary']
            
            if search_approach == "split":
                found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="split")
            elif search_approach == "summary":
                if summary_text:
                    found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="summary", summary_text=summary_text)
                else:
                    print(f"  ⚠️ No summary for idx {idx}, falling back to split approach")
                    found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="split")
            elif search_approach == "mixed":
                if summary_text:
                    found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="mixed", summary_text=summary_text)
                else:
                    print(f"  ⚠️ No summary for idx {idx}, falling back to split approach")
                    found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="split")
            else:
                raise ValueError(f"Unknown approach: {search_approach}")
            
            top_10 = [self.clean_ecli(f) for f in found_raw]
            
            # 3. Calculate Row Metrics
            hits = [t for t in targets if t in top_10]
            row_recall = len(hits) / len(targets) if len(targets) > 0 else 0
            row_precision = len(hits) / len(top_10) if len(top_10) > 0 else 0
            
            # Accumulate for global metrics
            metrics["hits_at_10"] += len(hits)
            metrics["total_targets"] += len(targets)
            metrics["precisions"].append(row_precision)
            
            # 4. Calculate Reciprocal Rank for MRR
            rank_score = 0
            for i, ecli in enumerate(top_10, 1):
                if ecli in targets:
                    rank_score = 1/i
                    break
            metrics["reciprocal_ranks"].append(rank_score)

            # --- DETAILED OUTPUT PER ROW ---
            print(f"\nRow ID: {idx}")
            print(f"Target ECLIs:  {targets}")
            print(f"Top 10 Found:  {top_10}")
            print(f"Result:        {len(hits)}/{len(targets)} hits")
            print(f"Recall:        {row_recall:.4f}")
            print(f"Precision:     {row_precision:.4f}")
            print(f"MRR Rank Score: {rank_score:.4f}")
            print("-" * 30)

            # Store results for CSV export
            results.append({
                "row_id": idx, 
                "targets": "; ".join(targets), 
                "top_10": "; ".join(top_10),
                "recall_at_10": row_recall, 
                "precision_at_10": row_precision,
                "mrr": rank_score,
                "approach": search_approach
            })
            
            # Progress tracker
            if len(results) % 5 == 0:
                current_recall = metrics['hits_at_10'] / metrics['total_targets'] if metrics['total_targets'] > 0 else 0
                print(f"\n>>> PROGRESS: {len(results)}/{len(data)} | Current Recall@10: {current_recall:.2%}")

        # --- FINAL SUMMARY STATISTICS ---
        final_recall = metrics["hits_at_10"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_precision = np.mean(metrics["precisions"])
        final_mrr = np.mean(metrics["reciprocal_ranks"])
        
        print("\n" + "="*50)
        print(f"FINAL EVALUATION METRICS (EVALUATION MODE)")
        print(f"Mode: {mode.upper()} | Approach: {search_approach.upper()}")
        if mode == "sample":
            print(f"Sample: {num_rows} rows | Seed: {random_seed}")
        print(f"Recall@10 (Accuracy): {final_recall:.4f} ({final_recall*100:.2f}%)")
        print(f"Precision@10:         {final_precision:.4f} ({final_precision*100:.2f}%)")
        print(f"MRR:                  {final_mrr:.4f}")
        print(f"Total Evaluation Time: {(time.time()-start_time)/60:.2f} mins")
        print("="*50)
        
        # Save detailed results to CSV
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        eval_dir = os.path.join(root_dir, 'data', 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

        if mode == "sample":
            filename = f"eval_{mode}_seed{random_seed}_{search_approach}_results.csv"
        else:
            filename = f"eval_{mode}_{search_approach}_results.csv"
            
        save_path = os.path.join(eval_dir, filename)
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")

    def run_evaluation_with_live_summaries(self, mode="full", search_approach="summary", num_rows=NUM_TEST_ROWS, random_seed=RANDOM_SEED):
        """
        ACTUAL RUN MODE: Generates summaries dynamically during evaluation.
        Requires summarizer to be initialized.
        Best for real-world performance testing.
        
        Args:
            mode: "full" or "sample"
            search_approach: "summary" or "mixed"
            num_rows: Number of rows for sample mode (uses NUM_TEST_ROWS if None)
            random_seed: Random seed for reproducibility (uses RANDOM_SEED if None)
        """
        
        if not self.summarizer:
            raise RuntimeError("Summarizer not initialized. Pass initialize_summarizer=True to __init__")
        
        print(f"--- ACTUAL RUN MODE (LIVE SUMMARIES) ---")
        print(f"Mode: {mode.upper()} | Approach: {search_approach.upper()}")
        if mode == "sample":
            print(f"Sample Size: {num_rows} | Random Seed: {random_seed}")
        print("-" * 50)
        
        # Load data and prepare evaluation set
        df = pd.read_excel(letters_path)
        data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
        
        if mode == "sample":
            data = data.sample(n=num_rows, random_state=random_seed)
        
        results = []
        metrics = {
            "reciprocal_ranks": [], 
            "hits_at_10": 0, 
            "total_targets": 0,
            "precisions": [],
            "summarization_times": []
        }
        
        start_time = time.time()
        
        for idx, row in data.iterrows():
            # 1. Prepare standardized targets
            targets = [self.clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if self.clean_ecli(e)]
            
            # 2. Generate summary on-the-fly
            letter_text = str(row['geanonimiseerd_doc_inhoud'])
            summary_start = time.time()
            summary_text = self.summarizer.generate_query(letter_text)
            summary_time = time.time() - summary_start
            metrics["summarization_times"].append(summary_time)
            
            # 3. Retrieval using selected approach
            if search_approach == "summary":
                found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="summary", summary_text=summary_text)
            elif search_approach == "mixed":
                found_raw = self.get_top_10_for_letter(letter_text, ACTIVE_DOMAIN, approach="mixed", summary_text=summary_text)
            else:
                raise ValueError(f"For live summaries, use 'summary' or 'mixed' approach, not '{search_approach}'")
            
            top_10 = [self.clean_ecli(f) for f in found_raw]
            
            # 4. Calculate Row Metrics
            hits = [t for t in targets if t in top_10]
            row_recall = len(hits) / len(targets) if len(targets) > 0 else 0
            row_precision = len(hits) / len(top_10) if len(top_10) > 0 else 0
            
            # Accumulate for global metrics
            metrics["hits_at_10"] += len(hits)
            metrics["total_targets"] += len(targets)
            metrics["precisions"].append(row_precision)
            
            # 5. Calculate Reciprocal Rank for MRR
            rank_score = 0
            for i, ecli in enumerate(top_10, 1):
                if ecli in targets:
                    rank_score = 1/i
                    break
            metrics["reciprocal_ranks"].append(rank_score)

            # --- DETAILED OUTPUT PER ROW ---
            print(f"\nRow ID: {idx}")
            print(f"Summary Gen Time: {summary_time:.2f}s")
            print(f"Summary: {summary_text[:100]}...")
            print(f"Target ECLIs:  {targets}")
            print(f"Top 10 Found:  {top_10}")
            print(f"Result:        {len(hits)}/{len(targets)} hits")
            print(f"Recall:        {row_recall:.4f}")
            print(f"Precision:     {row_precision:.4f}")
            print(f"MRR Rank Score: {rank_score:.4f}")
            print("-" * 30)

            # Store results for CSV export
            results.append({
                "row_id": idx, 
                "summary": summary_text,
                "targets": "; ".join(targets), 
                "top_10": "; ".join(top_10),
                "recall_at_10": row_recall, 
                "precision_at_10": row_precision,
                "mrr": rank_score,
                "summary_time": round(summary_time, 2),
                "approach": search_approach
            })
            
            # Progress tracker
            if len(results) % 5 == 0:
                current_recall = metrics['hits_at_10'] / metrics['total_targets'] if metrics['total_targets'] > 0 else 0
                avg_summary_time = np.mean(metrics["summarization_times"])
                print(f"\n>>> PROGRESS: {len(results)}/{len(data)} | Recall: {current_recall:.2%} | Avg Summary Time: {avg_summary_time:.2f}s")

        # --- FINAL SUMMARY STATISTICS ---
        final_recall = metrics["hits_at_10"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_precision = np.mean(metrics["precisions"])
        final_mrr = np.mean(metrics["reciprocal_ranks"])
        avg_summary_time = np.mean(metrics["summarization_times"])
        total_time = time.time() - start_time
        
        print("\n" + "="*50)
        print(f"FINAL METRICS (ACTUAL RUN MODE - LIVE SUMMARIES)")
        print(f"Mode: {mode.upper()} | Approach: {search_approach.upper()}")
        if mode == "sample":
            print(f"Sample: {num_rows} rows | Seed: {random_seed}")
        print(f"Recall@10 (Accuracy): {final_recall:.4f} ({final_recall*100:.2f}%)")
        print(f"Precision@10:         {final_precision:.4f} ({final_precision*100:.2f}%)")
        print(f"MRR:                  {final_mrr:.4f}")
        print(f"Avg Summarization Time: {avg_summary_time:.2f}s per document")
        print(f"Total Evaluation Time: {total_time/60:.2f} mins")
        print("="*50)
        
        # Save detailed results to CSV
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        eval_dir = os.path.join(root_dir, 'data', 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

        if mode == "sample":
            filename = f"eval_{mode}_seed{random_seed}_{search_approach}_live_results.csv"
        else:
            filename = f"eval_{mode}_{search_approach}_live_results.csv"
            
        save_path = os.path.join(eval_dir, filename)
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")

# --- EXECUTION ---
if __name__ == "__main__":
    
    # ============================================
    # SCENARIO 1: Batch testing on 567 rows (pre-generated summaries)
    # ============================================
    
    # rag = LegalRAGSystem(initialize_summarizer=False)

    # Option A: Split approach only (baseline)
    # rag.run_evaluation(mode="full", search_approach="split")
    
    # Option B: Sample with different random seeds
    # rag.run_evaluation(mode="sample", search_approach="split", num_rows=NUM_TEST_ROWS, random_seed=RANDOM_SEED)
      
    # Option C: Compare all approaches (pre-generated summaries)
    # rag.run_evaluation(mode="sample", search_approach="split")
    # rag.run_evaluation(mode="sample", search_approach="summary", num_rows=NUM_TEST_ROWS, random_seed=RANDOM_SEED,
    #                    summary_file="../data/evaluation/Summaries_Results.csv")
    # rag.run_evaluation(mode="sample", search_approach="mixed", num_rows=NUM_TEST_ROWS, random_seed=RANDOM_SEED,
    #                    summary_file="../data/evaluation/Summaries_Results.csv")
    
    
    # ============================================
    # SCENARIO 2: Batch testing - Live summary generation
    # ============================================
    
    rag = LegalRAGSystem(initialize_summarizer=True)

    # Option A: Use config parameters (NUM_TEST_ROWS=30, RANDOM_SEED=40)
    rag.run_evaluation_with_live_summaries(mode="sample", search_approach="summary")
    
    # Option B: Compare approaches with live summaries
    # rag.run_evaluation_with_live_summaries(mode="sample", search_approach="summary")
    # rag.run_evaluation_with_live_summaries(mode="sample", search_approach="mixed")
    
    # Option C: Full 567 rows with live summaries (slow, ~40+ hours)
    # rag.run_evaluation_with_live_summaries(mode="full", search_approach="mixed")


    # ============================================
    # SCENARIO 3: Real-time single letter query
    # ============================================
    
    # rag = LegalRAGSystem(initialize_summarizer=True)
    # sample_letter = """
    # Dit is een voorbeeld juridische brief over het bezwaar tegen de vordering van een borgsom 
    # voor het parkeren van fietsen. De eigenaar betoogt dat de voorschriften niet correct zijn 
    # toegepast en dat de locatie niet onder de regulering valt.
    # """
    # top_10 = rag.query_single_letter(sample_letter, approach="split")
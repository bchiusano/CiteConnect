import os
import sys
import pickle
import re
import time
import sys
import spacy
import platform
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

#RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# --- PARAMETERS ---
NUM_TEST_ROWS = 30
RANDOM_SEED = 40
SEARCH_K = 200
RERANK_TOP_N = 50
CANDIDATE_LIMIT = 50
MIN_SCORE = 0.3
RERANK_WEIGHT = 0.8


def get_device():
    """Automatically detect the best available device for the current platform."""
    if platform.system() == "Darwin":  # macOS
        return "mps"
    elif platform.system() == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


# --- SPACY INITIALIZATION ---
try:
    nlp = spacy.load("nl_core_news_md")
except:
    print("nl_core_news_md not found, falling back to small model.")
    nlp = spacy.load("nl_core_news_sm")


def clean_ecli(text):
    if pd.isna(text) or str(text).lower() == 'nan': return ""
    cleaned = str(text).upper().replace("ECLI:", "").strip()
    cleaned = re.sub(r'[^A-Z0-9:]', '', cleaned)
    return cleaned


class LegalRAGSystem:
    def __init__(self):
        print("Initializing Engines...")
        device = get_device()
        print(f"Using device: {device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device}
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

    def get_top_10_for_letter(self, letter_text, domain="bicycle"):
        keywords = DOMAIN_MAP.get(domain, {}).get("keywords", [])
        anchor = " ".join(keywords)

        # 1. Split into 5 issues
        doc = nlp(letter_text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        num_issues = 5
        chunk_size = max(1, len(sentences) // num_issues)
        issues = [" ".join(sentences[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_issues)]

        ecli_best_chunks = {}

        for issue_text in issues:
            try:
                # 2. Parallel Search
                v_hits = self.db.similarity_search(f"query: {issue_text}", k=SEARCH_K)
                tokens = re.findall(r'\w+', (f"{anchor} {issue_text}").lower())
                b_hits = self.bm25_model.get_top_n(tokens, self.legal_corpus, n=SEARCH_K)

                # 3. Hybrid Fusion (0.7 / 0.3)
                fused = self._rrf_fusion(v_hits, b_hits)

                candidate_map = {clean_ecli(d.metadata['ecli_nummer']): d for d in v_hits}
                for d in b_hits:
                    eid = clean_ecli(d['metadata'].get('ecli_nummer'))
                    if eid not in candidate_map:
                        candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])

                # 4. Neural Reranking (Weighted 0.2 / 0.8)
                rerank_candidates = []
                for eid, hybrid_score in fused[:RERANK_TOP_N]:
                    if eid in candidate_map:
                        rerank_candidates.append((candidate_map[eid], hybrid_score))

                if rerank_candidates:
                    docs_to_rerank = [item[0] for item in rerank_candidates]
                    refined = self.reranker.compress_documents(docs_to_rerank, issue_text)

                    for r in refined:
                        ecli = clean_ecli(r.metadata.get('ecli_nummer', ''))
                        rerank_score = r.metadata.get('relevance_score', 0)

                        # Find original hybrid score
                        original_hybrid_score = next((score for eid, score in fused if eid == ecli), 0)

                        # Apply Weighting: 20% Original, 80% Reranker
                        final_score = ((1 - RERANK_WEIGHT) * original_hybrid_score) + (RERANK_WEIGHT * rerank_score)

                        # Aggregation: Keep the best result across all 5 segments
                        if final_score >= MIN_SCORE:
                            if ecli not in ecli_best_chunks or final_score > ecli_best_chunks[ecli]:
                                ecli_best_chunks[ecli] = final_score
            except:
                continue
        print()
        return [(eid, score) for eid, score in sorted(ecli_best_chunks.items(), key=lambda x: x[1], reverse=True)[:10]]

    def _rrf_fusion(self, v_hits, b_hits, k=60):
        scores = {}
        # 0.7 Semantic / 0.3 Keyword split
        for r, d in enumerate(v_hits):
            eid = clean_ecli(d.metadata.get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.7 * (1 / (r + k))
        for r, d in enumerate(b_hits):
            eid = clean_ecli(d['metadata'].get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.3 * (1 / (r + k))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def run_evaluation(self, mode="full"):
        print(f"--- STARTING EVALUATION MODE: {mode.upper()} ---")

        # Load data and prepare evaluation set
        df = pd.read_excel(letters_path)
        data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])

        if mode == "sample":
            # Using the new seed to validate consistency
            data = data.sample(n=NUM_TEST_ROWS, random_state=RANDOM_SEED)

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
            targets = [clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if clean_ecli(e)]

            # 2. Retrieval using the Issues-Based strategy
            found_raw = self.get_top_10_for_letter(str(row['geanonimiseerd_doc_inhoud']), ACTIVE_DOMAIN)
            # TODO: fix because output for top 10 changed
            top_10 = [clean_ecli(f) for f in found_raw]

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
                    rank_score = 1 / i
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
                "mrr": rank_score
            })

            # Progress tracker
            if len(results) % 5 == 0:
                current_recall = metrics['hits_at_10'] / metrics['total_targets'] if metrics['total_targets'] > 0 else 0
                print(f"\n>>> PROGRESS: {len(results)}/{len(data)} | Current Recall@10: {current_recall:.2%}")

        # --- FINAL SUMMARY STATISTICS ---
        final_recall = metrics["hits_at_10"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_precision = np.mean(metrics["precisions"])
        final_mrr = np.mean(metrics["reciprocal_ranks"])

        print("\n" + "=" * 50)
        print(f"FINAL EVALUATION METRICS (SEED: {RANDOM_SEED})")
        print(f"Recall@10 (Accuracy): {final_recall:.4f} ({final_recall * 100:.2f}%)")
        print(f"Precision@10:         {final_precision:.4f} ({final_precision * 100:.2f}%)")
        print(f"MRR:                  {final_mrr:.4f}")
        print(f"Total Evaluation Time: {(time.time() - start_time) / 60:.2f} mins")
        print("=" * 50)

        # Save detailed results to CSV
        # 1. Get root directory (one level up from /src)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # 2. Define the evaluation directory
        eval_dir = os.path.join(root_dir, 'data', 'evaluation')
        # 3. Create the folder if it doesn't exist
        os.makedirs(eval_dir, exist_ok=True)

        # 4. Construct filename
        if mode == "sample":
            filename = f"eval_{mode}_seed{RANDOM_SEED}_results.csv"
        else:
            filename = f"eval_{mode}_results.csv"

        # 5. Save to the specific folder
        save_path = os.path.join(eval_dir, filename)
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")


# --- EXECUTION ---
if __name__ == "__main__":
    rag = LegalRAGSystem()
    rag.run_evaluation(mode="sample")
    # rag.run_evaluation(mode="full")

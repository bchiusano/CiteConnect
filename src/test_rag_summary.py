import os
import pickle
import re
import time
import sys
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

# --- PATH FIX ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# --- CONFIGURATION ---
from rag_pipeline_infloat_multilingual import PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME, letters_path
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
from rag_index_infloat_multilingual import BM25_INDEX_PATH, CORPUS_PATH

# Load summaries from data/summary/summaryTexts.py
import importlib.util
_summary_path = os.path.join(ROOT_DIR, 'data', 'summary', 'summaryTexts.py')
_spec = importlib.util.spec_from_file_location("summaryTexts", _summary_path)
_summary_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summary_module)
summaries_0_24 = _summary_module.summaries_0_24
summaries_25_49 = _summary_module.summaries_25_49
summaries_50_74 = _summary_module.summaries_50_74
summaries_75_99 = _summary_module.summaries_75_99

RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

# --- PARAMETERS ---
SEARCH_K = 200
RERANK_TOP_N = 50
CANDIDATE_LIMIT = 50
MIN_SCORE = 0.3
RERANK_WEIGHT = 0.8

# Merge all summaries into a single dict (keys 0-99)
SUMMARIES = {}
SUMMARIES.update(summaries_0_24)
SUMMARIES.update(summaries_25_49)
SUMMARIES.update(summaries_50_74)
SUMMARIES.update(summaries_75_99)


class SummaryRAGEvaluator:
    def __init__(self):
        print("Initializing RAG engines...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'mps'}
        )
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )

        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25_model = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f:
            self.legal_corpus = pickle.load(f)

        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'flashrank_cache'))
        os.makedirs(cache_path, exist_ok=True)

        self.reranker = FlashrankRerank(
            model=RERANK_MODEL, top_n=RERANK_TOP_N
        )
        print("Engines ready.")

    def clean_ecli(self, text):
        if pd.isna(text) or str(text).lower() == 'nan':
            return ""
        cleaned = str(text).upper().replace("ECLI:", "").strip()
        cleaned = re.sub(r'[^A-Z0-9:]', '', cleaned)
        return cleaned

    def _rrf_fusion(self, v_hits, b_hits, k=60):
        scores = {}
        for r, d in enumerate(v_hits):
            eid = self.clean_ecli(d.metadata.get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.7 * (1 / (r + k))
        for r, d in enumerate(b_hits):
            eid = self.clean_ecli(d['metadata'].get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.3 * (1 / (r + k))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _search_and_rerank(self, search_text, domain="bicycle"):
        keywords = DOMAIN_MAP.get(domain, {}).get("keywords", [])
        anchor = " ".join(keywords)

        ecli_best_chunks = {}

        try:
            v_hits = self.db.similarity_search(f"query: {search_text}", k=SEARCH_K)
            tokens = re.findall(r'\w+', (f"{anchor} {search_text}").lower())
            b_hits = self.bm25_model.get_top_n(tokens, self.legal_corpus, n=SEARCH_K)

            fused = self._rrf_fusion(v_hits, b_hits)

            candidate_map = {self.clean_ecli(d.metadata['ecli_nummer']): d for d in v_hits}
            for d in b_hits:
                eid = self.clean_ecli(d['metadata'].get('ecli_nummer'))
                if eid not in candidate_map:
                    candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])

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
                    original_hybrid_score = next((score for eid, score in fused if eid == ecli), 0)
                    final_score = ((1 - RERANK_WEIGHT) * original_hybrid_score) + (RERANK_WEIGHT * rerank_score)

                    if final_score >= MIN_SCORE:
                        if ecli not in ecli_best_chunks or final_score > ecli_best_chunks[ecli]:
                            ecli_best_chunks[ecli] = final_score
        except Exception as e:
            print(f"Error in search: {e}")

        return [eid for eid, score in sorted(ecli_best_chunks.items(), key=lambda x: x[1], reverse=True)[:10]]

    def run_evaluation(self, num_rows=100):
        """
        Evaluate using pre-computed summaries (indices 0-99) as search input.
        Compares retrieved ECLIs against ground-truth ECLIs from the dataset.
        """
        print(f"\n{'='*60}")
        print(f"SUMMARY-BASED RAG EVALUATION")
        print(f"Using {len(SUMMARIES)} pre-computed summaries (rows 0-{max(SUMMARIES.keys())})")
        print(f"Rows to evaluate: {num_rows}")
        print(f"{'='*60}\n")

        df = pd.read_excel(letters_path)
        # Use the first num_rows rows (matching summary indices)
        data = df.head(num_rows)

        results = []
        metrics = {
            "reciprocal_ranks": [],
            "hits_at_10": 0,
            "total_targets": 0,
            "precisions": [],
            "rows_skipped": 0
        }

        start_time = time.time()

        for idx, row in data.iterrows():
            # Skip rows without ground-truth ECLI
            ecli_val = row.get('ECLI')
            if pd.isna(ecli_val) or str(ecli_val).strip() == '':
                metrics["rows_skipped"] += 1
                continue

            # Skip rows without a summary
            if idx not in SUMMARIES:
                metrics["rows_skipped"] += 1
                continue

            summary_text = SUMMARIES[idx]

            # Prepare ground-truth targets
            targets = [self.clean_ecli(e) for e in str(ecli_val).replace(';', ',').split(',') if self.clean_ecli(e)]
            if not targets:
                metrics["rows_skipped"] += 1
                continue

            # Search using summary
            found_raw = self._search_and_rerank(summary_text, ACTIVE_DOMAIN)
            top_10 = [self.clean_ecli(f) for f in found_raw]

            # Calculate metrics
            hits = [t for t in targets if t in top_10]
            row_recall = len(hits) / len(targets) if targets else 0
            row_precision = len(hits) / len(top_10) if top_10 else 0

            metrics["hits_at_10"] += len(hits)
            metrics["total_targets"] += len(targets)
            metrics["precisions"].append(row_precision)

            # MRR
            rank_score = 0
            for i, ecli in enumerate(top_10, 1):
                if ecli in targets:
                    rank_score = 1 / i
                    break
            metrics["reciprocal_ranks"].append(rank_score)

            # Per-row output
            print(f"Row {idx:3d} | Hits: {len(hits)}/{len(targets)} | Recall: {row_recall:.4f} | MRR: {rank_score:.4f}")
            print(f"         Summary: {summary_text[:80]}...")
            print(f"         Targets: {targets}")
            print(f"         Found:   {top_10}")
            print(f"         {'-'*50}")

            results.append({
                "row_id": idx,
                "summary": summary_text,
                "targets": "; ".join(targets),
                "top_10": "; ".join(top_10),
                "hits": len(hits),
                "recall_at_10": row_recall,
                "precision_at_10": row_precision,
                "mrr": rank_score
            })

            # Progress
            if len(results) % 10 == 0:
                current_recall = metrics['hits_at_10'] / metrics['total_targets'] if metrics['total_targets'] > 0 else 0
                print(f"\n>>> PROGRESS: {len(results)} evaluated | Recall@10: {current_recall:.2%}\n")

        # Final metrics
        total_time = time.time() - start_time
        final_recall = metrics["hits_at_10"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_precision = np.mean(metrics["precisions"]) if metrics["precisions"] else 0
        final_mrr = np.mean(metrics["reciprocal_ranks"]) if metrics["reciprocal_ranks"] else 0

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - SUMMARY APPROACH")
        print(f"{'='*60}")
        print(f"Rows evaluated:       {len(results)}")
        print(f"Rows skipped:         {metrics['rows_skipped']}")
        print(f"Recall@10:            {final_recall:.4f} ({final_recall*100:.2f}%)")
        print(f"Precision@10:         {final_precision:.4f} ({final_precision*100:.2f}%)")
        print(f"MRR:                  {final_mrr:.4f}")
        print(f"Total hits:           {metrics['hits_at_10']}/{metrics['total_targets']}")
        print(f"Evaluation time:      {total_time/60:.2f} mins")
        print(f"{'='*60}")

        # Save results
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        eval_dir = os.path.join(root_dir, 'data', 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

        save_path = os.path.join(eval_dir, "eval_summary_approach_results.csv")
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")

        return {
            "recall_at_10": final_recall,
            "precision_at_10": final_precision,
            "mrr": final_mrr,
            "total_evaluated": len(results),
            "total_time_mins": total_time / 60
        }


    def run_evaluation_live(self, num_rows=100):
        """
        Evaluate using Groq API to generate summaries on-the-fly.
        Not limited to pre-computed summaries — works on any number of rows.
        Requires GROQ_API_KEY environment variable.
        """
        from text_summarizer import GroqSummarizer
        summarizer = GroqSummarizer()

        print(f"\n{'='*60}")
        print(f"LIVE SUMMARY RAG EVALUATION (Groq API)")
        print(f"Rows to evaluate: {num_rows}")
        print(f"{'='*60}\n")

        df = pd.read_excel(letters_path)
        data = df.head(num_rows)

        results = []
        metrics = {
            "reciprocal_ranks": [],
            "hits_at_10": 0,
            "total_targets": 0,
            "precisions": [],
            "rows_skipped": 0,
            "summarization_times": []
        }

        start_time = time.time()

        for idx, row in data.iterrows():
            ecli_val = row.get('ECLI')
            if pd.isna(ecli_val) or str(ecli_val).strip() == '':
                metrics["rows_skipped"] += 1
                continue

            targets = [self.clean_ecli(e) for e in str(ecli_val).replace(';', ',').split(',') if self.clean_ecli(e)]
            if not targets:
                metrics["rows_skipped"] += 1
                continue

            letter_text = str(row['geanonimiseerd_doc_inhoud'])

            # Generate summary via Groq
            sum_start = time.time()
            summary_text = summarizer.generate_query(letter_text)
            sum_time = time.time() - sum_start
            metrics["summarization_times"].append(sum_time)

            # Search using summary
            found_raw = self._search_and_rerank(summary_text, ACTIVE_DOMAIN)
            top_10 = [self.clean_ecli(f) for f in found_raw]

            # Metrics
            hits = [t for t in targets if t in top_10]
            row_recall = len(hits) / len(targets) if targets else 0
            row_precision = len(hits) / len(top_10) if top_10 else 0

            metrics["hits_at_10"] += len(hits)
            metrics["total_targets"] += len(targets)
            metrics["precisions"].append(row_precision)

            rank_score = 0
            for i, ecli in enumerate(top_10, 1):
                if ecli in targets:
                    rank_score = 1 / i
                    break
            metrics["reciprocal_ranks"].append(rank_score)

            print(f"Row {idx:3d} | Hits: {len(hits)}/{len(targets)} | Recall: {row_recall:.4f} | MRR: {rank_score:.4f} | Summary: {sum_time:.1f}s")
            print(f"         {summary_text[:100]}...")
            print(f"         {'-'*50}")

            results.append({
                "row_id": idx,
                "summary": summary_text,
                "targets": "; ".join(targets),
                "top_10": "; ".join(top_10),
                "hits": len(hits),
                "recall_at_10": row_recall,
                "precision_at_10": row_precision,
                "mrr": rank_score,
                "summary_time": round(sum_time, 2)
            })

            if len(results) % 10 == 0:
                current_recall = metrics['hits_at_10'] / metrics['total_targets'] if metrics['total_targets'] > 0 else 0
                avg_sum_time = np.mean(metrics["summarization_times"])
                print(f"\n>>> PROGRESS: {len(results)} evaluated | Recall@10: {current_recall:.2%} | Avg summary: {avg_sum_time:.2f}s\n")

        # Final metrics
        total_time = time.time() - start_time
        final_recall = metrics["hits_at_10"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_precision = np.mean(metrics["precisions"]) if metrics["precisions"] else 0
        final_mrr = np.mean(metrics["reciprocal_ranks"]) if metrics["reciprocal_ranks"] else 0
        avg_sum_time = np.mean(metrics["summarization_times"]) if metrics["summarization_times"] else 0

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - LIVE SUMMARY APPROACH (Groq)")
        print(f"{'='*60}")
        print(f"Rows evaluated:       {len(results)}")
        print(f"Rows skipped:         {metrics['rows_skipped']}")
        print(f"Recall@10:            {final_recall:.4f} ({final_recall*100:.2f}%)")
        print(f"Precision@10:         {final_precision:.4f} ({final_precision*100:.2f}%)")
        print(f"MRR:                  {final_mrr:.4f}")
        print(f"Avg summary time:     {avg_sum_time:.2f}s")
        print(f"Total hits:           {metrics['hits_at_10']}/{metrics['total_targets']}")
        print(f"Evaluation time:      {total_time/60:.2f} mins")
        print(f"{'='*60}")

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        eval_dir = os.path.join(root_dir, 'data', 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)

        save_path = os.path.join(eval_dir, "eval_summary_live_groq_results.csv")
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")

        return {
            "recall_at_10": final_recall,
            "precision_at_10": final_precision,
            "mrr": final_mrr,
            "total_evaluated": len(results),
            "avg_summary_time": avg_sum_time,
            "total_time_mins": total_time / 60
        }


if __name__ == "__main__":
    evaluator = SummaryRAGEvaluator()

    # Option 1: Pre-computed summaries (first 100 rows, no API needed)
    # evaluator.run_evaluation(num_rows=100)

    # Option 2: Live Groq summaries (requires GROQ_API_KEY env var)
    evaluator.run_evaluation_live(num_rows=100)
